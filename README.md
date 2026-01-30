# TinyLM

## Modal OpenAI-compatible API

This repo includes two Modal options:

- `src/modal_app.py`: native TinyLM FastAPI server (works with `checkpoints/best.pt`).
- `src/vllm_app.py`: vLLM OpenAI-compatible server (requires HF-compatible weights).
- `src/gradio_app.py`: Gradio UI that calls the TinyLM inference function.

### Deploy

1. Set the checkpoint path (local path or URL):

   - Local: put your file in `checkpoints/` and use `MODEL_CHECKPOINT_PATH=/root/checkpoints/best.pt`
   - URL: `MODEL_CHECKPOINT_PATH=https://.../best.pt`

2. Optional model config overrides:

   - `TINYLM_VOCAB_SIZE` (default 50257)
   - `TINYLM_EMB_DIM` (default 768)
   - `TINYLM_N_LAYERS` (default 12)
   - `TINYLM_N_HEADS` (default 12)
   - `TINYLM_ATT_DIM` (default 64)
   - `TINYLM_MAX_SEQ_LEN` (default 512)

3. Deploy:

   - Native TinyLM: `modal deploy src/modal_app.py`
   - vLLM (HF weights): `modal deploy src/vllm_app.py`
   - Gradio UI: `modal deploy src/gradio_app.py`

### Endpoints

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /health`

### vLLM notes

`vllm` expects a Hugging Face-compatible model (config + weights). If you want to serve
TinyLM via vLLM, you’ll need to export your model checkpoint to a HF format and set:

- `MODEL_NAME` (HF repo or path)
- Optional `MODEL_REVISION`

Example:

```bash
MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507-FP8" modal deploy src/vllm_app.py
```

### Example requests

```bash
curl -X POST "$MODAL_ENDPOINT/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinylm",
    "prompt": "Once upon a time",
    "max_tokens": 64,
    "temperature": 0.8
  }'
```

```bash
curl -X POST "$MODAL_ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinylm",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a short poem about the ocean."}
    ],
    "max_tokens": 64,
    "temperature": 0.8
  }'
```