#!/bin/bash
#SBATCH --job-name=inv-water
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/water_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/water_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-2

# Waterbirds (ImageNet-pretrained ResNet-50): 3 methods × 10 seeds.
#   0: ERM
#   1: Twin-bootstrap lam=300 (failed transfer demonstration)
#   2: Twin-bootstrap lam=10 (rule-selected)

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

METHODS=("erm|0" "twin_indep|300.0" "twin_indep|10.0")
SEEDS="1,2,3,4,5,6,7,8,9,10"

# Canonical-data seed.  Pass via env: CANON=7 sbatch slurm/full_retraining/05_waterbirds.sh
CANON=${CANON:-99}
if [ "$CANON" = "99" ]; then
  OUT_DIR="outputs/cross_sample"
  LOG_TAG=""
else
  OUT_DIR="outputs/cross_sample_seed${CANON}"
  LOG_TAG="_seed${CANON}"
fi

SPEC=${METHODS[$SLURM_ARRAY_TASK_ID]}
MODE=${SPEC%%|*}
LAM=${SPEC#*|}

mkdir -p "logs/waterbirds${LOG_TAG}"
LOG="logs/waterbirds${LOG_TAG}/${MODE}_lam${LAM}.log"

echo "=== waterbirds  $MODE  lam=$LAM  canon=$CANON  $(hostname)  $(date) ===" > "$LOG"
if [ "$MODE" = "twin_indep" ]; then
  python scripts/cross_sample_train.py \
    --dataset waterbirds --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
    --mode "$MODE" --lam "$LAM" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
else
  python scripts/cross_sample_train.py \
    --dataset waterbirds --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
    --mode "$MODE" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
fi
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
