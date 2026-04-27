#!/bin/bash
#SBATCH --job-name=inv-v2-gpu
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_v2gpu_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_v2gpu_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3

# V2 GPU sweep with V-REx candidate instrumentation.
# Submit: sbatch slurm/run_v2_gpu.sh

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=("waterbirds" "celeba" "civilcomments" "multinli")
SEEDS=("42,123,789,2024,7" "42,123,789,2024,7" "42,123,789" "42,123,789")
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SD=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== v2 $DS | seeds=$SD | $(hostname) ==="
echo "Start: $(date)"

mkdir -p logs

python scripts/run_experiment.py \
    --dataset $DS --seeds $SD --device cuda --methods erm,jtt,ours \
    > logs/v2_${DS}.log 2>&1

echo "Done: $(date)"
tail -20 logs/v2_${DS}.log
