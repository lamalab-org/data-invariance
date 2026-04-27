#!/bin/bash
#SBATCH --job-name=inv-wb-cs
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_wbcs_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_wbcs_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

# Cross-sample protocol on Waterbirds: 4 array tasks running in parallel.
# 0: erm           - 5 seeds × 1 model
# 1: bagging_K2    - 5 seeds × 2 models
# 2: bagging_K5    - 5 seeds × 5 models
# 3: twin_indep    - 5 seeds × 2 models, λ=300 frozen from BACE dev

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

SEEDS="1,2,3,4,5"
MODES=(erm bagging bagging twin_indep)
KS=(1 2 5 2)
EXTRA_ARGS=("" "--K 2" "--K 5" "--lam 300.0")
TAGS=(erm bagging_K2 bagging_K5 twin_indep_lam300)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
EXTRA=${EXTRA_ARGS[$SLURM_ARRAY_TASK_ID]}

echo "=== waterbirds  $TAG  $(hostname) | $(date) ==="

mkdir -p logs/cross_sample

python scripts/cross_sample_train.py \
    --dataset waterbirds --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode "$MODE" $EXTRA \
    > "logs/cross_sample/waterbirds_${TAG}.log" 2>&1

echo "Done: $(date)"
tail -10 "logs/cross_sample/waterbirds_${TAG}.log"
