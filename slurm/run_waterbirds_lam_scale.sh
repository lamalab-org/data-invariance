#!/bin/bash
#SBATCH --job-name=inv-wb-scale
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_wbsc_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_wbsc_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-2

# Twin_indep λ-scaling validation on Waterbirds.
# Scaling rule: λ_N = λ_BACE × (N_BACE/N) = 300 × (968/4795) ≈ 60.
# We test λ ∈ {30, 60, 100} to bracket the predicted optimum.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

LAMS=(30.0 60.0 100.0)
LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}

echo "=== twin_indep waterbirds  λ=$LAM  $(date) ==="

mkdir -p logs/cross_sample

python scripts/cross_sample_train.py \
    --dataset waterbirds --canonical_data_seed 99 \
    --train_seeds 1,2,3,4,5 --mode twin_indep --lam "$LAM" \
    > "logs/cross_sample/waterbirds_twin_indep_lam${LAM}.log" 2>&1

echo "Done: $(date)"
tail -10 "logs/cross_sample/waterbirds_twin_indep_lam${LAM}.log"
