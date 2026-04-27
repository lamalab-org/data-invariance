#!/bin/bash
#SBATCH --job-name=inv-bsl
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_bsl_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_bsl_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-23

# Three new baselines on 8 chemistry datasets:
#   codistillation        — Anil 2018, disjoint shards + KL agreement
#   distillation_anchor   — Jiang 2022, fresh student + KL to frozen ERM anchor
#   twin_indep_shared     — ablation: same bootstrap to both twin networks
#
# 8 datasets × 3 modes = 24 array tasks. 10 seeds per (mode, dataset).

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=(bace dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames)
MODES=(codistillation distillation_anchor twin_indep_shared)
SEEDS="1,2,3,4,5,6,7,8,9,10"

DS_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
MD_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
DS=${DATASETS[$DS_IDX]}
MODE=${MODES[$MD_IDX]}

echo "=== $DS  $MODE  $(hostname) | $(date) ==="
mkdir -p logs/baselines

python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode "$MODE" --lam 300.0 \
    > "logs/baselines/${DS}_${MODE}.log" 2>&1

echo "Done: $(date)"
tail -10 "logs/baselines/${DS}_${MODE}.log"
