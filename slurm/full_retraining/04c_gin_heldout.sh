#!/bin/bash
#SBATCH --job-name=inv-gin-ho
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/gin_heldout_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/gin_heldout_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-17%6
# %6 caps concurrent running tasks to stay under the per-user QOS GRES
# limit (signal 53 kills if exceeded).

# GIN held-out chemistry datasets transfer test:
#   6 datasets x 3 methods = 18 array tasks
#   datasets: bbbp, dili, cyp2d6_substrate, pgp_broccatelli, bbb_martins, ames
#   methods : ERM, Bagging-K=5, Twin-bootstrap lam=10 (rule-selected on GIN)
# All 10 train_seeds per (dataset, method).

set -e
cd /vast/lo45pic/data-invariance

PY="$HOME/.local/bin/uv run --no-sync python"

DATASETS=("bbbp_gin" "dili_gin" "cyp2d6_substrate_gin" "pgp_broccatelli_gin" "bbb_martins_gin" "ames_gin")
METHODS=("erm|0|0" "bagging|5|0" "twin_indep|0|10.0")
SEEDS="1,2,3,4,5,6,7,8,9,10"

# Canonical-data seed.  Pass via env: CANON=7 sbatch slurm/full_retraining/04c_gin_heldout.sh
CANON=${CANON:-99}
if [ "$CANON" = "99" ]; then
  OUT_DIR="outputs/cross_sample"
  LOG_TAG=""
else
  OUT_DIR="outputs/cross_sample_seed${CANON}"
  LOG_TAG="_seed${CANON}"
fi

# Compute (dataset_idx, method_idx) from SLURM_ARRAY_TASK_ID.
# Layout: dataset_idx = task / 3, method_idx = task % 3.
DS_IDX=$((SLURM_ARRAY_TASK_ID / 3))
M_IDX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET=${DATASETS[$DS_IDX]}
SPEC=${METHODS[$M_IDX]}
MODE=${SPEC%%|*}
REST=${SPEC#*|}
K=${REST%%|*}
LAM=${REST#*|}

mkdir -p "logs/gin_heldout${LOG_TAG}"
LOG="logs/gin_heldout${LOG_TAG}/${DATASET}_${MODE}_K${K}_lam${LAM}.log"

echo "=== ${DATASET}  ${MODE}  K=${K}  lam=${LAM}  canon=${CANON}  $(hostname)  $(date) ===" > "$LOG"
if [ "$MODE" = "twin_indep" ]; then
  $PY scripts/cross_sample_train.py \
    --dataset "$DATASET" --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
    --mode "$MODE" --lam "$LAM" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
elif [ "$MODE" = "bagging" ]; then
  $PY scripts/cross_sample_train.py \
    --dataset "$DATASET" --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
    --mode "$MODE" --K "$K" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
else
  $PY scripts/cross_sample_train.py \
    --dataset "$DATASET" --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
    --mode "$MODE" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
fi
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
