#!/bin/bash
#SBATCH --job-name=inv-headline
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/headline_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/headline_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-53

# Headline classification: 9 datasets × 6 methods, 10 seeds per cell.
# Each array task runs one (dataset, method) tuple with all 10 seeds in
# a single python invocation.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=(bace dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames cyp2d6_substrate)
# 6 method specs encoded as "mode|K|lam" (use 0 for unused).
METHODS=(
  "erm|0|0"
  "mc_dropout|20|0"
  "bagging|2|0"
  "bagging|5|0"
  "deep_ensemble|5|0"
  "twin_indep|0|300.0"
)
SEEDS="1,2,3,4,5,6,7,8,9,10"

DS_IDX=$(( SLURM_ARRAY_TASK_ID / ${#METHODS[@]} ))
M_IDX=$(( SLURM_ARRAY_TASK_ID % ${#METHODS[@]} ))
DS=${DATASETS[$DS_IDX]}
SPEC=${METHODS[$M_IDX]}
MODE=${SPEC%%|*}
REST=${SPEC#*|}
K=${REST%%|*}
LAM=${REST#*|}

mkdir -p logs/headline
LOG="logs/headline/${DS}_${MODE}_K${K}_lam${LAM}.log"

echo "=== $DS  $MODE  K=$K  lam=$LAM  $(hostname)  $(date) ===" > "$LOG"

if [ "$MODE" = "twin_indep" ]; then
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --lam "$LAM" >> "$LOG" 2>&1
elif [ "$MODE" = "mc_dropout" ]; then
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --K "$K" >> "$LOG" 2>&1
elif [ "$MODE" = "erm" ]; then
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" >> "$LOG" 2>&1
else
  python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --K "$K" >> "$LOG" 2>&1
fi

echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
