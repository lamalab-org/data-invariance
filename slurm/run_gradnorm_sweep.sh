#!/bin/bash
#SBATCH --job-name=inv-gn
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_gn_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_gn_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# GradNorm-balanced lambda (twin_gradnorm) on 4 chemistry datasets, 10 seeds
# each, to test whether the BACE failure mode (CI on Δ id_churn vs ERM
# includes zero) is robust across datasets or BACE-specific.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=(bbbp tadf mof_thermal ames)
SEEDS="1,2,3,4,5,6,7,8,9,10"
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== twin_gradnorm $DS  $(hostname) | $(date) ==="
mkdir -p logs/gradnorm

python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode twin_gradnorm --target_ratio 1.0 \
    > "logs/gradnorm/${DS}.log" 2>&1

echo "Done: $(date)"
tail -15 "logs/gradnorm/${DS}.log"
