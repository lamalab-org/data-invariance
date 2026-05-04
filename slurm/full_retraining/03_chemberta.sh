#!/bin/bash
#SBATCH --job-name=inv-cberta
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/chemberta_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/chemberta_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-17

# ChemBERTa scope: 6 datasets × 3 methods (ERM, twin lam=300, twin lam=10).
# 10 seeds per cell (the user explicitly asked for 10 to match headline).

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=(bace_chemberta bbbp_chemberta pgp_broccatelli_chemberta bbb_martins_chemberta ames_chemberta dili_chemberta)
# (mode, lam) pairs; for ERM lam is unused.
METHODS=("erm|0" "twin_indep|300.0" "twin_indep|10.0")
SEEDS="1,2,3,4,5,6,7,8,9,10"

# Canonical-data seed.  Pass via env to dispatch the seed-sensitivity sweep:
#   CANON=7 sbatch slurm/full_retraining/03_chemberta.sh
# When CANON=99, NPZs go to the canonical outputs/cross_sample/ tree;
# otherwise to outputs/cross_sample_seed${CANON}/ so the seed-99 NPZs
# are not overwritten.
CANON=${CANON:-99}
if [ "$CANON" = "99" ]; then
  OUT_DIR="outputs/cross_sample"
  LOG_TAG=""
else
  OUT_DIR="outputs/cross_sample_seed${CANON}"
  LOG_TAG="_seed${CANON}"
fi

DS_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
M_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
DS=${DATASETS[$DS_IDX]}
SPEC=${METHODS[$M_IDX]}
MODE=${SPEC%%|*}
LAM=${SPEC#*|}

mkdir -p "logs/chemberta${LOG_TAG}"
LOG="logs/chemberta${LOG_TAG}/${DS}_${MODE}_lam${LAM}.log"

echo "=== $DS  $MODE  lam=$LAM  canon=$CANON  $(hostname)  $(date) ===" > "$LOG"
if [ "$MODE" = "twin_indep" ]; then
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
    --mode "$MODE" --lam "$LAM" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
else
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
    --mode "$MODE" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
fi
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
