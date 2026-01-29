#!/bin/sh
### DTU HPC Job Submission Script for TinyLM ###
#BSUB -J tinylm
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
###-- send notification at start --
#BSUB -B
###-- send notification at completion --
#BSUB -N
### end of BSUB options ###

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Starting at: $(date)"
echo "Running on: $(hostname)"
echo "=========================================="

# Load UV (assuming it's installed in your home directory)
# If UV is available as a module, use: module load uv
export PATH="$HOME/.local/bin:$PATH"

# Verify UV is available
which uv || { echo "ERROR: UV not found. Please install UV first."; exit 1; }

# Show UV version
echo "UV version: $(uv --version)"

# Sync dependencies (creates venv and installs packages from pyproject.toml)
echo "Syncing dependencies with UV..."
uv sync

# Run the main script
echo "Running TinyLM..."
uv run src/main.py

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

