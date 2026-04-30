#!/bin/bash
#SBATCH --job-name=inv-reg
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/reg_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/reg_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-11

# Regression: 3 datasets × 4 methods × 10 seeds.
# (regression uses lam=3 for twin per app:regression rule.)

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=(esol_reg freesolv_reg lipo_reg)
METHODS=("erm|0|0" "bagging|2|0" "bagging|5|0" "twin_indep|0|3.0")
SEEDS="1,2,3,4,5,6,7,8,9,10"

DS_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
M_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
DS=${DATASETS[$DS_IDX]}
SPEC=${METHODS[$M_IDX]}
MODE=${SPEC%%|*}
REST=${SPEC#*|}
K=${REST%%|*}
LAM=${REST#*|}

mkdir -p logs/regression
LOG="logs/regression/${DS}_${MODE}_K${K}_lam${LAM}.log"

echo "=== $DS  $MODE  K=$K  lam=$LAM  $(hostname)  $(date) ===" > "$LOG"
if [ "$MODE" = "twin_indep" ]; then
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --lam "$LAM" >> "$LOG" 2>&1
elif [ "$MODE" = "bagging" ]; then
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --K "$K" >> "$LOG" 2>&1
else
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" >> "$LOG" 2>&1
fi
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
