#!/bin/bash
#SBATCH --job-name=inv-cb-scope
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_cb_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_cb_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-11

# Cross-sample protocol on ChemBERTa-fine-tuned chemistry datasets.
# 6 datasets × 2 modes (erm, twin_indep λ=300) = 12 array tasks.
# Each task runs 5 seeds for one (dataset, mode) combo.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

SEEDS="1,2,3,4,5"
DATASETS=(bace_chemberta bbbp_chemberta pgp_broccatelli_chemberta \
          bbb_martins_chemberta ames_chemberta dili_chemberta)
MODES=(erm twin_indep)
EXTRA_ARGS=("" "--lam 300.0")
TAGS=(erm twin_indep_lam300)

DS_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
MD_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))
DS=${DATASETS[$DS_IDX]}
MODE=${MODES[$MD_IDX]}
TAG=${TAGS[$MD_IDX]}
EXTRA=${EXTRA_ARGS[$MD_IDX]}

echo "=== $DS  $TAG  $(hostname) | $(date) ==="
mkdir -p logs/cross_sample

python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode "$MODE" $EXTRA \
    > "logs/cross_sample/${DS}_${TAG}.log" 2>&1

echo "Done: $(date)"
tail -15 "logs/cross_sample/${DS}_${TAG}.log"
