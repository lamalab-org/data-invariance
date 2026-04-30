#!/bin/bash
#SBATCH --job-name=inv-cb-lam
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_cb_lam_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_cb_lam_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-4

# BACE-ChemBERTa lambda sweep for the rule-selected lambda story.
# 5 lambda values, 5 seeds each, on bace_chemberta only.
# After this finishes, pick the largest lambda satisfying the
# 0.02 accuracy tolerance and run on held-out ChemBERTa datasets.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

SEEDS="1,2,3,4,5"
LAMBDAS=(1.0 3.0 10.0 30.0 100.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "=== bace_chemberta twin_indep lam=$LAM  $(hostname) | $(date) ==="
mkdir -p logs/cross_sample

python scripts/cross_sample_train.py \
    --dataset bace_chemberta --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode twin_indep --lam "$LAM" \
    > "logs/cross_sample/bace_chemberta_twin_lam${LAM}.log" 2>&1

echo "Done: $(date)"
tail -15 "logs/cross_sample/bace_chemberta_twin_lam${LAM}.log"
