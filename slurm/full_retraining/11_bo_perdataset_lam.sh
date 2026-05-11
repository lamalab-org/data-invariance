#!/bin/bash
#SBATCH --job-name=inv-bo-perds
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/bo_perds_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/bo_perds_%A_%a.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-89%12

# Per-dataset BO follow-up: train twin-indep at the median lambda the
# cross-sample BO sweep selected for each dataset.  All 10 train_seeds
# per dataset use the SAME lambda, so cross-sample churn between any
# two trainings is fixed-lambda variance (apples-to-apples vs. the
# frozen lambda=300 baseline in outputs/cross_sample/).
#
# The medians below were computed from
# outputs/cross_sample_bayes_cross/<dataset>/twin_indep_bayes_train*_lam*.npz
# after job 8201607 finished:
#   python scripts/...  (median across the 10 BO-selected lambdas)

set -e

cd /vast/lo45pic/data-invariance
PY="$HOME/.local/bin/uv run --no-sync python"

DATASETS=(bace dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames cyp2d6_substrate)
LAMS=(93.03 4.78 84.66 362.28 373.72 74.60 14.91 424.08 256.28)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

DS_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
DS=${DATASETS[$DS_IDX]}
LAM=${LAMS[$DS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CANONICAL_DATA_SEED=${CANONICAL_DATA_SEED:-99}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/cross_sample_bayes_perdataset}

mkdir -p logs/bo_perdataset
LOG="logs/bo_perdataset/${DS}_seed${SEED}_lam${LAM}.log"

echo "=== $DS twin_indep seed=$SEED lam=$LAM out=$OUTPUT_DIR $(hostname) $(date) ===" > "$LOG"

$PY scripts/cross_sample_train.py \
  --dataset "$DS" \
  --canonical_data_seed "$CANONICAL_DATA_SEED" \
  --train_seeds "$SEED" \
  --mode twin_indep \
  --lam "$LAM" \
  --output_dir "$OUTPUT_DIR" \
  >> "$LOG" 2>&1

echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
