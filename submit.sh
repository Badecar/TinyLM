#!/bin/bash
### DTU HPC Job Submission Script for TinyLM ###
#BSUB -J tinylm_elite
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 15:00
#BSUB -B
#BSUB -N

# 1. High-Speed Local Workspace Setup
# Use $TMPDIR if set by LSF, otherwise fall back to /tmp with unique job directory
WORK_DIR="${TMPDIR:-/tmp/${USER}_tinylm_$$}"
mkdir -p "$WORK_DIR"
NODE_VENV="$WORK_DIR/venv"
NODE_DATA="$WORK_DIR/data"
mkdir -p "$NODE_DATA"
mkdir -p logs

echo "Using work directory: $WORK_DIR"

# Load essential CUDA module
module load cuda/12.1

# 2. Ephemeral Environment with 'uv'
# We install uv and create the venv on the local SSD to save Home space.
curl -LsSf https://astral.sh/uv/install.sh | BINDIR="$WORK_DIR/bin" sh
export PATH="$WORK_DIR/bin:$PATH"

echo "Building local venv on SSD at $NODE_VENV..."
uv venv "$NODE_VENV" --python 3.12
source "$NODE_VENV/bin/activate"

# Install dependencies into the local venv. 
# --no-cache-dir is used to prevent filling up your Home quota (~/.cache)
echo "Installing dependencies to local SSD..."
uv pip install torch wandb tiktoken datasets tqdm pyyaml certifi --no-cache-dir

# 3. Data Staging
PROJECT_ROOT="${LS_SUBCWD:-$(pwd)}"
DATA_DIR="$PROJECT_ROOT/slm_data"
CONFIG_PATH="$PROJECT_ROOT/configs/train.yaml"
DATA_CONFIG="$PROJECT_ROOT/configs/data.yaml"

# Safely check dataset kind using the newly installed local python
DATASET_KIND="$(python - <<PY
import yaml
with open("$DATA_CONFIG", "r") as f:
    print((yaml.safe_load(f) or {}).get("dataset_kind", "elite").strip())
PY
)"

echo "Staging $DATASET_KIND data to Local SSD ($NODE_DATA)..."
if [ "$DATASET_KIND" = "elite" ]; then
    cp "$DATA_DIR"/*.bin "$NODE_DATA/"
else
    cp "$DATA_DIR"/train.bin "$NODE_DATA/"
fi

# 4. Run Training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training at: $(date)"
# We run using the python interpreter inside the node's local venv
python src/main.py --config "$CONFIG_PATH" --data_config "$DATA_CONFIG" --data_dir "$NODE_DATA"

# 5. Cleanup
echo "Cleaning up local SSD..."
rm -rf "$WORK_DIR"
# If TMPDIR was set by LSF, it would be wiped automatically anyway

echo "Job completed at: $(date)"