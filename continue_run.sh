#!/bin/bash
### DTU HPC Job Submission Script for TinyLM (Resume) ###
#BSUB -J tinylm_resume
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 18:00
#BSUB -B
#BSUB -N

# 1. Environment Setup
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

# Load essential modules for GPU training
module load cuda/12.1

# 2. Data Staging (The "HPC Pro" Workaround)
NODE_DATA="/tmp/${USER}_tinylm_data"
mkdir -p "$NODE_DATA"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATASET_KIND="${DATASET_KIND:-elite}" # elite | tinystories
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/slm_data}"
if [ "$DATASET_KIND" = "elite" ]; then
  CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/train_elite_resume.yaml}"
elif [ "$DATASET_KIND" = "tinystories" ]; then
  CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/train_tinystories_resume.yaml}"
else
  CONFIG_PATH="${CONFIG_PATH:-}"
fi

echo "Staging data: ${DATA_DIR} -> Local SSD ($NODE_DATA)..."
if [ "$DATASET_KIND" = "elite" ]; then
  cp "$DATA_DIR"/*.bin "$NODE_DATA/"
elif [ "$DATASET_KIND" = "tinystories" ]; then
  cp "$DATA_DIR"/train.bin "$NODE_DATA/"
else
  echo "Unknown DATASET_KIND: $DATASET_KIND (expected elite or tinystories)"
  exit 1
fi

# 3. UV Virtual Environment Logic
echo "Syncing dependencies..."
uv sync

# 4. Run the Training Burn (Resume)
echo "Starting resume training at: $(date)"
CHECKPOINT_ROOT="$HOME/checkpoints"
CHECKPOINT_VERSION="v1"
if [ "$DATASET_KIND" = "elite" ]; then
  uv run src/main.py \
    --config "$CONFIG_PATH" \
    --data_dir "$NODE_DATA" \
    --checkpoint_dir "$CHECKPOINT_ROOT/$CHECKPOINT_VERSION"
else
  uv run src/train.py \
    --config "$CONFIG_PATH" \
    --data_path "$NODE_DATA/train.bin" \
    --checkpoint_dir "$CHECKPOINT_ROOT/$CHECKPOINT_VERSION"
fi

# 5. Cleanup
echo "Cleaning up local SSD..."
rm -rf "$NODE_DATA"

echo "Job completed at: $(date)"
