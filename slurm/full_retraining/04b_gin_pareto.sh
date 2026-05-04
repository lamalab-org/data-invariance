#!/bin/bash
#SBATCH --job-name=inv-gin-par
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/gin_pareto_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/gin_pareto_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%6

# BACE-GIN lambda Pareto sweep at additional canonical seeds (7, 42).
# Covers lam in {1, 3, 30, 100} -- lam=10 and lam=300 are already in
# 04_gin_bace.sh.  Used by scripts/analyze_gin_lambda.py to verify
# the rule picks lambda=10 on every canonical seed independently.

set -e
cd /vast/lo45pic/data-invariance

PY="$HOME/.local/bin/uv run --no-sync python"

LAMBDAS=(1.0 3.0 30.0 100.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
SEEDS="1,2,3,4,5,6,7,8,9,10"

CANON=${CANON:-7}
OUT_DIR="outputs/cross_sample_seed${CANON}"
LOG_TAG="_seed${CANON}"

mkdir -p "logs/gin${LOG_TAG}"
LOG="logs/gin${LOG_TAG}/bace_gin_twin_indep_lam${LAM}.log"

echo "=== bace_gin twin_indep lam=$LAM canon=$CANON $(hostname) $(date) ===" > "$LOG"
$PY scripts/cross_sample_train.py \
  --dataset bace_gin --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
  --mode twin_indep --lam "$LAM" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
