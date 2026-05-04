#!/bin/bash
#SBATCH --job-name=inv-cberta-par
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/chemberta_pareto_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/chemberta_pareto_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%6

# BACE-ChemBERTa lambda Pareto sweep at the additional canonical seeds
# (7 and 42).  The sweep covers lam in {1, 3, 30, 100} -- lam=10 and
# lam=300 are already captured by the headline 03_chemberta.sh
# dispatch, so this script only fills in the gaps.  Used by
# scripts/analyze_chemberta_lambda.py to verify the rule picks
# lambda=10 on every canonical seed independently.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

LAMBDAS=(1.0 3.0 30.0 100.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
SEEDS="1,2,3,4,5,6,7,8,9,10"

CANON=${CANON:-7}
OUT_DIR="outputs/cross_sample_seed${CANON}"
LOG_TAG="_seed${CANON}"

mkdir -p "logs/chemberta${LOG_TAG}"
LOG="logs/chemberta${LOG_TAG}/bace_chemberta_twin_indep_lam${LAM}.log"

echo "=== bace_chemberta twin_indep lam=$LAM canon=$CANON $(hostname) $(date) ===" > "$LOG"
python scripts/cross_sample_train.py \
  --dataset bace_chemberta --canonical_data_seed "$CANON" --train_seeds "$SEEDS" \
  --mode twin_indep --lam "$LAM" --output_dir "$OUT_DIR" >> "$LOG" 2>&1
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
