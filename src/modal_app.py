import json
import os
import sys
import time
from typing import List, Optional, Union

import modal
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

SRC_DIR = "/root/src"
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import TinyLM

APP_NAME = "tinylm-openai"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "numpy",
        "pydantic",
        "requests",
        "tiktoken",
        "torch",
        "wandb",
    )
)
if os.path.isdir("src"):
    image = image.add_local_dir("src", remote_path=SRC_DIR, copy=True)
if os.path.isfile("checkpoints/best.pt"):
    image = image.add_local_file(
        "checkpoints/best.pt", remote_path="/root/checkpoints/best.pt", copy=True
    )
elif os.path.isdir("checkpoints"):
    image = image.add_local_dir(
        "checkpoints", remote_path="/root/checkpoints", copy=True
    )

app = modal.App(APP_NAME)



class CompletionRequest(BaseModel):
    model: str = Field(default="tinylm")
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="tinylm")
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    stream: bool = False


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for {name}: {value}") from exc


def _load_checkpoint(checkpoint_path: str) -> dict:
    if checkpoint_path.startswith("http://") or checkpoint_path.startswith("https://"):
        response = requests.get(checkpoint_path, timeout=60)
        response.raise_for_status()
        local_path = "/tmp/tinylm-checkpoint.pt"
        with open(local_path, "wb") as handle:
            handle.write(response.content)
        checkpoint_path = local_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return torch.load(checkpoint_path, map_location="cpu")


def _build_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for message in messages:
        role = message.role.lower().strip()
        lines.append(f"{role}: {message.content.strip()}")
    lines.append("assistant:")
    return "\n".join(lines)


@app.function(
    image=image,
    gpu=os.getenv("MODAL_GPU", "A10G"),
    scaledown_window=300,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    api = FastAPI()

    vocab_size = _get_env_int("TINYLM_VOCAB_SIZE", 50257)
    emb_dim = _get_env_int("TINYLM_EMB_DIM", 768)
    n_layers = _get_env_int("TINYLM_N_LAYERS", 12)
    n_heads = _get_env_int("TINYLM_N_HEADS", 12)
    att_dim = _get_env_int("TINYLM_ATT_DIM", 64)
    max_seq_len = _get_env_int("TINYLM_MAX_SEQ_LEN", 512)

    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH", "/root/checkpoints/best.pt")
    if not os.path.exists(checkpoint_path):
        try:
            candidates = [
                name
                for name in os.listdir("/root/checkpoints")
                if name.endswith(".pt")
            ]
        except FileNotFoundError:
            candidates = []
        if candidates:
            checkpoint_path = os.path.join("/root/checkpoints", sorted(candidates)[0])
        elif os.path.exists("/root/best.pt"):
            checkpoint_path = "/root/best.pt"

    model = TinyLM(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        att_dim=att_dim,
        max_seq_len=max_seq_len,
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    def generate_text(prompt: str, max_tokens: int, temperature: float) -> str:
        input_tokens = tokenizer.encode_ordinary(prompt)
        idx = torch.tensor([input_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
        generated = output[0].tolist()
        return tokenizer.decode(generated)

    end_token_text = "<|endoftext|>"
    end_token_id = tokenizer.encode(
        end_token_text, allowed_special={end_token_text}
    )[-1]

    def generate_stream(prompt: str, max_tokens: int, temperature: float):
        input_tokens = tokenizer.encode_ordinary(prompt)
        idx = torch.tensor([input_tokens], dtype=torch.long, device=device)
        temp = max(temperature, 1e-6)

        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len :]
                logits = model(idx_cond)
                logits = logits[:, -1, :] / temp
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
                token_id = idx_next.item()
                if token_id == end_token_id:
                    break
                token_text = tokenizer.decode([token_id])
                if end_token_text in token_text:
                    before, _sep, _after = token_text.partition(end_token_text)
                    if before:
                        yield before
                    break
                yield token_text

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode_ordinary(text))

    @api.get("/health")
    def health():
        return {"status": "ok", "model": "tinylm"}

    @api.post("/v1/completions")
    def completions(request: CompletionRequest):
        if request.stream:
            prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
            if len(prompts) != 1:
                raise HTTPException(status_code=400, detail="stream=true only supports a single prompt.")
            prompt = prompts[0]
            created = int(time.time())

            def event_stream():
                for delta in generate_stream(prompt, request.max_tokens, request.temperature):
                    payload = {
                        "id": f"cmpl-{created}",
                        "object": "text_completion",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "text": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                payload = {
                    "id": f"cmpl-{created}",
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        created = int(time.time())
        choices = []

        for idx, prompt in enumerate(prompts):
            text = generate_text(prompt, request.max_tokens, request.temperature)
            choices.append(
                {
                    "index": idx,
                    "text": text[len(prompt) :],
                    "finish_reason": "stop",
                }
            )

        usage = {
            "prompt_tokens": sum(count_tokens(p) for p in prompts),
            "completion_tokens": sum(count_tokens(choice["text"]) for choice in choices),
            "total_tokens": sum(count_tokens(p) for p in prompts)
            + sum(count_tokens(choice["text"]) for choice in choices),
        }

        return {
            "id": f"cmpl-{created}",
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": choices,
            "usage": usage,
        }

    @api.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        if request.stream:
            created = int(time.time())
            prompt = _build_prompt(request.messages)

            def event_stream():
                role_payload = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(role_payload)}\n\n"

                for delta in generate_stream(prompt, request.max_tokens, request.temperature):
                    payload = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                payload = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        prompt = _build_prompt(request.messages)
        created = int(time.time())
        completion = generate_text(prompt, request.max_tokens, request.temperature)
        assistant_text = completion[len(prompt) :]

        usage = {
            "prompt_tokens": count_tokens(prompt),
            "completion_tokens": count_tokens(assistant_text),
            "total_tokens": count_tokens(prompt) + count_tokens(assistant_text),
        }

        return {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    return api
