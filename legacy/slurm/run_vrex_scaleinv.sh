#!/bin/bash
#SBATCH --job-name=inv-vrex-si
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_vrex_si_%j.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_vrex_si_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00

# Waterbirds with scale-invariant V-REx penalty (CV^2, fixed lam=1).
# Submit: sbatch slurm/run_vrex_scaleinv.sh

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

echo "=== vrex-scale-invariant waterbirds | $(date) ==="

python scripts/run_experiment.py \
    --dataset waterbirds --seeds 42,123,789,2024,7 --device cuda --methods erm,jtt,ours \
    > logs/vrex_si_waterbirds.log 2>&1

echo "Done: $(date)"
tail -30 logs/vrex_si_waterbirds.log
