#!/bin/bash
### DTU HPC Job Submission Script for TinyLM ###
#BSUB -J tinylm_elite
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 8
#BSUB -R "rusage[mem=16GB]"   # Increased for staging overhead
#BSUB -R "span[hosts=1]"
#BSUB -W 20:00
#BSUB -B
#BSUB -N

## THIS ASSUMES THAT .VENV IS IN THE HOME DIRECTORY ##

# 1. Environment Setup
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

# Load essential modules for GPU training
module load cuda/12.1

# 2. Data Staging (The "HPC Pro" Workaround)
# We move the data from your Home directory to the Local SSD of the GPU node.
# This ensures the GPU never waits for the network.
NODE_DATA="/tmp/${USER}_tinylm_data"
mkdir -p "$NODE_DATA"

PROJECT_ROOT="${LS_SUBCWD:-$(pwd)}"
DATASET_KIND="$(python3 - <<PY
import yaml
with open("$PROJECT_ROOT/configs/data.yaml", "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
print((data.get("dataset_kind") or "elite").strip())
PY
)"
DATA_DIR="$PROJECT_ROOT/slm_data"
CONFIG_PATH="$PROJECT_ROOT/configs/train.yaml"
DATA_CONFIG="$PROJECT_ROOT/configs/data.yaml"

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
# We sync before running. UV is fast enough to do this every time.
echo "Syncing dependencies..."
uv sync
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Run the Training Burn
# We pass the LOCAL_DATA path to your script
echo "Starting training at: $(date)"
uv run src/main.py --config "$CONFIG_PATH" --data_config "$DATA_CONFIG" --data_dir "$NODE_DATA"

# 5. Cleanup
# Removing the data from the local node to keep the cluster healthy
echo "Cleaning up local SSD..."
rm -rf "$NODE_DATA"

echo "Job completed at: $(date)"