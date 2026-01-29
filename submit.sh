#!/bin/bash
### DTU HPC Job Submission Script for TinyLM ###
#BSUB -J tinylm_burn
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"   # Increased for staging overhead
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -B
#BSUB -N

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

echo "Staging data: Home -> Local SSD ($NODE_DATA)..."
# Replace '~/slm_data/train.bin' with your actual path in Home
cp ~/slm_data/train.bin "$NODE_DATA/"

# 3. UV Virtual Environment Logic
# We sync before running. UV is fast enough to do this every time.
echo "Syncing dependencies..."
uv sync

# 4. Run the Training Burn
# We pass the LOCAL_DATA path to your script
echo "Starting training at: $(date)"
uv run src/main.py --data_path "$NODE_DATA/train.bin" --batch_size 64

# 5. Cleanup
# Removing the data from the local node to keep the cluster healthy
echo "Cleaning up local SSD..."
rm -rf "$NODE_DATA"

echo "Job completed at: $(date)"