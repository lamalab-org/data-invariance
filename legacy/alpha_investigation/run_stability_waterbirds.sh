#!/bin/bash
#SBATCH --job-name=inv-stab-wb
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_stab_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_stab_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-2

# Partition-sensitivity uncertainty on Waterbirds.
# 3 array tasks, one per mode (run in parallel):
#   0: erm         (1 full-data model, single seed)
#   1: ensemble    (K=3 full-data models, different inits)
#   2: partition   (K=2 disjoint-half models)
# All use data_seed=42. If results are promising we extend to 3 seeds.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

MODES=("erm" "ensemble" "partition")
KS=(1 3 2)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}

echo "=== stab waterbirds mode=$MODE K=$K | $(hostname) | $(date) ==="

mkdir -p logs

python scripts/partition_pair_train.py \
    --dataset waterbirds --data_seeds 42 --K $K --mode $MODE \
    --output_dir outputs/stability \
    > logs/stab_waterbirds_${MODE}.log 2>&1

echo "Done: $(date)"
tail -15 logs/stab_waterbirds_${MODE}.log
