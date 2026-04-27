#!/bin/bash
#SBATCH --job-name=inv-v2-batt
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_v2batt_%j.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_v2batt_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Rerun battery with instrumentation (previous attempt produced empty log).
set -e
export PATH="$HOME/.local/bin:$PATH"
cd /vast/lo45pic/data-invariance

echo "=== v2 battery rerun | $(hostname) | $(date) ==="
mkdir -p logs

uv run python scripts/run_experiment.py \
    --dataset battery --seeds 42,123,789,2024,7 --device cpu --methods erm,jtt,ours \
    > logs/v2_battery.log 2>&1

echo "Done: $(date)"
tail -25 logs/v2_battery.log
