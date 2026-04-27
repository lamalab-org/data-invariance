#!/bin/bash
#SBATCH --job-name=inv-gpu
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2

# GPU experiments: Waterbirds, CelebA, CivilComments
# Submit: sbatch slurm/run_gpu.sh

set -e
export PATH="$HOME/.local/bin:$PATH"
cd /vast/lo45pic/data-invariance

DATASETS=("waterbirds" "celeba" "civilcomments")
SEEDS=("42,123,789,2024,7" "42,123,789,2024,7" "42,123,789")
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SD=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== $DS | seeds=$SD | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "Start: $(date)"

mkdir -p logs

uv run python scripts/run_experiment.py \
    --dataset $DS --seeds $SD --device cuda --methods erm,jtt,ours \
    > logs/final_${DS}.log 2>&1

echo "Done: $(date)"
tail -5 logs/final_${DS}.log
