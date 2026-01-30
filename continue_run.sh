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
export WANDB_API_KEY="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"
export WANDB_ENTITY="badecar-danmarks-tekniske-universitet-dtu"
export WANDB_PROJECT="TinyLM"

# Load essential modules for GPU training
module load cuda/12.1

# 2. Data Staging (The "HPC Pro" Workaround)
NODE_DATA="/tmp/${USER}_tinylm_data"
mkdir -p "$NODE_DATA"

echo "Staging data: Home -> Local SSD ($NODE_DATA)..."
cp ~/slm_data/train.bin "$NODE_DATA/"

# 3. UV Virtual Environment Logic
echo "Syncing dependencies..."
uv sync

# 4. Run the Training Burn (Resume)
echo "Starting resume training at: $(date)"
CHECKPOINT_ROOT="$HOME/checkpoints"
CHECKPOINT_VERSION="v1"
uv run src/main.py \
  --data_dir "$NODE_DATA" \
  --batch_size 64 \
  --resume_latest \
  --checkpoint_dir "$CHECKPOINT_ROOT/$CHECKPOINT_VERSION" \
  --max_steps 160000

# 5. Cleanup
echo "Cleaning up local SSD..."
rm -rf "$NODE_DATA"

echo "Job completed at: $(date)"
