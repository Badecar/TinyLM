#!/bin/bash
### DTU HPC Job Submission Script for TinyLM ###
#BSUB -J tinylm_burn
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 8
#BSUB -R "rusage[mem=16GB]"   # Increased for staging overhead
#BSUB -R "span[hosts=1]"
#BSUB -W 0:10
#BSUB -B
#BSUB -N

# 1. Environment Setup
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"
# Yes, this is the wandb API key. Yes, it is freely visible here on github. No, I am not worried about it.
export WANDB_API_KEY="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"
export WANDB_ENTITY="badecar-danmarks-tekniske-universitet-dtu"
export WANDB_PROJECT="TinyLM"

# Load essential modules for GPU training
module load cuda/12.1

# 2. Data Staging (The "HPC Pro" Workaround)
# We move the data from your Home directory to the Local SSD of the GPU node.
# This ensures the GPU never waits for the network.
NODE_DATA="/tmp/${USER}_tinylm_data"
mkdir -p "$NODE_DATA"

echo "Staging data: Home -> Local SSD ($NODE_DATA)..."
# Copy refined bins to local SSD
cp ~/slm_data/*.bin "$NODE_DATA/"

# 3. UV Virtual Environment Logic
# We sync before running. UV is fast enough to do this every time.
echo "Syncing dependencies..."
uv sync

# 4. Run the Training Burn
# We pass the LOCAL_DATA path to your script
echo "Starting training at: $(date)"
uv run src/main.py --data_dir "$NODE_DATA" --batch_size 64

# 5. Cleanup
# Removing the data from the local node to keep the cluster healthy
echo "Cleaning up local SSD..."
rm -rf "$NODE_DATA"

echo "Job completed at: $(date)"