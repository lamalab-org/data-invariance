#!/bin/bash
#SBATCH --job-name=inv-v2-cpu
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_v2cpu_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_v2cpu_%j_%a.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-9

# V2 CPU sweep with V-REx candidate instrumentation.
# Submit: sbatch slurm/run_v2_cpu.sh

set -e
export PATH="$HOME/.local/bin:$PATH"
cd /vast/lo45pic/data-invariance

DATASETS=(cmnist multi_cmnist tadf mof_thermal mof_solvent perovskite battery bace bbbp hiv)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== v2 $DS | $(hostname) ==="
echo "Start: $(date)"

mkdir -p logs

uv run python scripts/run_experiment.py \
    --dataset $DS --seeds 42,123,789,2024,7 --device cpu --methods erm,jtt,ours \
    > logs/v2_${DS}.log 2>&1

echo "Done: $(date)"
tail -20 logs/v2_${DS}.log
