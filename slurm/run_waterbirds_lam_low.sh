#!/bin/bash
#SBATCH --job-name=inv-wb-lam
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_wb_lam_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_wb_lam_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-2

# Waterbirds lambda sweep at the small end (lambda in {1, 3, 10}).
# Existing runs cover lambda in {30, 60, 100, 300}. After this finishes,
# all lambdas in {1, 3, 10, 30, 60, 100, 300} are available; pick the
# largest lambda satisfying the 0.02 id-acc tolerance to close the
# pretrained-backbone scope loop on the Waterbirds vision modality.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

SEEDS="1,2,3,4,5"
LAMBDAS=(1.0 3.0 10.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "=== waterbirds twin_indep lam=$LAM  $(hostname) | $(date) ==="
mkdir -p logs/cross_sample

python scripts/cross_sample_train.py \
    --dataset waterbirds --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode twin_indep --lam "$LAM" \
    > "logs/cross_sample/waterbirds_twin_lam${LAM}.log" 2>&1

echo "Done: $(date)"
tail -15 "logs/cross_sample/waterbirds_twin_lam${LAM}.log"
