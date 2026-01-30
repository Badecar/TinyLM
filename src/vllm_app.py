import os
import subprocess

import modal

APP_NAME = "tinylm-vllm"

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Thinking-2507-FP8")
MODEL_REVISION = os.getenv("MODEL_REVISION")
FAST_BOOT = os.getenv("FAST_BOOT", "true").lower() == "true"
N_GPU = int(os.getenv("VLLM_N_GPU", "1"))
VLLM_PORT = 8000

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App(APP_NAME)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=f"A10G:{N_GPU}",
    scaledown_window=15 * 60,
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        "tinylm",
    ]

    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    subprocess.Popen(" ".join(cmd), shell=True)
