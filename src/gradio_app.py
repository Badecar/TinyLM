import os
import sys
import time
from typing import Optional

import modal

APP_NAME = "tinylm-gradio"
SRC_DIR = "/root/src"
CHECKPOINT_DIR = "/root/checkpoints"

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "gradio",
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
        "checkpoints/best.pt", remote_path=f"{CHECKPOINT_DIR}/best.pt", copy=True
    )
elif os.path.isdir("checkpoints"):
    image = image.add_local_dir(
        "checkpoints", remote_path=CHECKPOINT_DIR, copy=True
    )

app = modal.App(APP_NAME)

_MODEL = None
_TOKENIZER = None


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for {name}: {value}") from exc


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)

    from model import TinyLM
    import torch
    import tiktoken

    vocab_size = _get_env_int("TINYLM_VOCAB_SIZE", 50257)
    emb_dim = _get_env_int("TINYLM_EMB_DIM", 768)
    n_layers = _get_env_int("TINYLM_N_LAYERS", 12)
    n_heads = _get_env_int("TINYLM_N_HEADS", 12)
    att_dim = _get_env_int("TINYLM_ATT_DIM", 64)
    max_seq_len = _get_env_int("TINYLM_MAX_SEQ_LEN", 512)

    checkpoint_path = os.getenv(
        "MODEL_CHECKPOINT_PATH", f"{CHECKPOINT_DIR}/best.pt"
    )
    if not os.path.exists(checkpoint_path):
        candidates = [
            name for name in os.listdir(CHECKPOINT_DIR) if name.endswith(".pt")
        ]
        if candidates:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, sorted(candidates)[0])
        elif os.path.exists("/root/best.pt"):
            checkpoint_path = "/root/best.pt"
        else:
            raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    model = TinyLM(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        att_dim=att_dim,
        max_seq_len=max_seq_len,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    _MODEL = model
    _TOKENIZER = tokenizer
    return _MODEL, _TOKENIZER


def _generate_text(prompt: str, max_tokens: int, temperature: float) -> str:
    import torch

    model, tokenizer = _load_model()
    device = next(model.parameters()).device
    input_tokens = tokenizer.encode_ordinary(prompt)
    idx = torch.tensor([input_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    generated = output[0].tolist()
    return tokenizer.decode(generated)


def _generate_text_stream(prompt: str, max_tokens: int, temperature: float):
    import torch

    model, tokenizer = _load_model()
    device = next(model.parameters()).device
    input_tokens = tokenizer.encode_ordinary(prompt)
    end_token_text = "<|endoftext|>"
    end_token_id = tokenizer.encode(
        end_token_text, allowed_special={end_token_text}
    )[-1]
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


@app.function(
    image=image,
    gpu=os.getenv("MODAL_GPU", "A10G"),
    scaledown_window=300,
    timeout=600,
)
def predict(text: str, max_tokens: int, temperature: float) -> str:
    completion = _generate_text(text, max_tokens=max_tokens, temperature=temperature)
    return completion[len(text) :]


@app.function(
    image=image,
    gpu=os.getenv("MODAL_GPU", "A10G"),
    scaledown_window=300,
    timeout=600,
    is_generator=True,
)
def predict_stream(text: str, max_tokens: int, temperature: float):
    for delta in _generate_text_stream(text, max_tokens=max_tokens, temperature=temperature):
        yield delta


@app.function(
    image=image,
    scaledown_window=300,
    min_containers=1,
    max_containers=1,
)
@modal.asgi_app()
def gradio_app():
    import gradio as gr
    from fastapi import FastAPI

    def ui_fn(text: str, max_tokens: int, temperature: float):
        output = ""
        for delta in predict_stream.remote_gen(text, max_tokens, temperature):
            output += delta
            yield output

    demo = gr.Interface(
        fn=ui_fn,
        inputs=[
            gr.Textbox(label="Input", lines=4, placeholder="Skriv noget..."),
            gr.Slider(1, 256, value=64, step=1, label="Max tokens"),
            gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature"),
        ],
        outputs=gr.Textbox(label="Output", lines=8),
        title="TinyLM on Modal",
        description=f"UI → inference function ({APP_NAME})",
        flagging_mode="never",
    ).queue()

    return gr.mount_gradio_app(FastAPI(), demo, path="/")
