#!/bin/bash
#SBATCH --job-name=inv-gin
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/gin_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/gin_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-3

# GIN architecture cross-check on BACE: 4 methods × 10 seeds.
#   0: ERM
#   1: Bagging-K=5 (independent of lambda)
#   2: Twin-bootstrap lam=300 (failed transfer)
#   3: Twin-bootstrap lam=10 (rule-selected on GIN)

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

METHODS=("erm|0|0" "bagging|5|0" "twin_indep|0|300.0" "twin_indep|0|10.0")
SEEDS="1,2,3,4,5,6,7,8,9,10"

SPEC=${METHODS[$SLURM_ARRAY_TASK_ID]}
MODE=${SPEC%%|*}
REST=${SPEC#*|}
K=${REST%%|*}
LAM=${REST#*|}

mkdir -p logs/gin
LOG="logs/gin/bace_gin_${MODE}_K${K}_lam${LAM}.log"

echo "=== bace_gin  $MODE  K=$K  lam=$LAM  $(hostname)  $(date) ===" > "$LOG"
if [ "$MODE" = "twin_indep" ]; then
  python scripts/cross_sample_train.py \
    --dataset bace_gin --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --lam "$LAM" >> "$LOG" 2>&1
elif [ "$MODE" = "bagging" ]; then
  python scripts/cross_sample_train.py \
    --dataset bace_gin --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --K "$K" >> "$LOG" 2>&1
else
  python scripts/cross_sample_train.py \
    --dataset bace_gin --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" >> "$LOG" 2>&1
fi
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
