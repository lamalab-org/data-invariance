#!/bin/bash
#SBATCH --job-name=inv-par
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_par_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_par_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-5

# Pareto curve on BBBP: tests whether the BACE-frozen λ=300 generalises
# to a second held-out dataset. We sweep λ ∈ {1, 3, 10, 30, 100, 300}
# with 10 seeds each. If BBBP's optimum is also at the high-λ extreme of
# the rule (id-acc >= ERM-id-acc - 0.02), the BACE-frozen choice is robust;
# if it's at a different λ, the dev rule is dataset-specific.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

LAMS=(1.0 3.0 10.0 30.0 100.0 300.0)
LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}
SEEDS="1,2,3,4,5,6,7,8,9,10"

echo "=== twin_indep bbbp  λ=$LAM  $(hostname) | $(date) ==="
mkdir -p logs/pareto_bbbp

python scripts/cross_sample_train.py \
    --dataset bbbp --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode twin_indep --lam "$LAM" \
    > "logs/pareto_bbbp/lam${LAM}.log" 2>&1

echo "Done: $(date)"
tail -10 "logs/pareto_bbbp/lam${LAM}.log"
