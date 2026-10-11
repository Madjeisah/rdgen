#!/bin/bash
#SBATCH --job-name=rdgen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Usage: sbatch --array=1-N scripts/slurm_array.sh jobs.txt
set -euo pipefail
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "[$(date -u +%FT%TZ)] $CMD"
eval "$CMD"
