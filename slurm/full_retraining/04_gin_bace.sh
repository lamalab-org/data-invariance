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
cd /vast/lo45pic/data-invariance

# Use uv-managed venv (on /vast, reliably visible to compute nodes) instead
# of conda activate (lives in $HOME/miniconda3/, NFS attr-cache delays
# can hide newly-installed packages on compute nodes -- bit us on the
# torch_geometric install in the previous resubmit).
PY="$HOME/.local/bin/uv run --no-sync python"

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
  $PY scripts/cross_sample_train.py \
    --dataset bace_gin --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --lam "$LAM" >> "$LOG" 2>&1
elif [ "$MODE" = "bagging" ]; then
  $PY scripts/cross_sample_train.py \
    --dataset bace_gin --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" --K "$K" >> "$LOG" 2>&1
else
  $PY scripts/cross_sample_train.py \
    --dataset bace_gin --canonical_data_seed 99 --train_seeds "$SEEDS" \
    --mode "$MODE" >> "$LOG" 2>&1
fi
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
