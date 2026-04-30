#!/bin/bash
#SBATCH --job-name=inv-pareto
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/pareto_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/pareto_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-5

# Pareto curve on BACE: twin-bootstrap at lam in {1,3,10,30,100,300},
# 10 seeds per lambda.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

LAMBDAS=(1.0 3.0 10.0 30.0 100.0 300.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/pareto
LOG="logs/pareto/bace_lam${LAM}.log"

echo "=== bace twin_indep lam=$LAM  $(hostname)  $(date) ===" > "$LOG"
python scripts/cross_sample_train.py \
  --dataset bace --canonical_data_seed 99 --train_seeds "$SEEDS" \
  --mode twin_indep --lam "$LAM" >> "$LOG" 2>&1
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
