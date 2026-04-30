#!/bin/bash
#SBATCH --job-name=inv-cb-lam10
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_cb_lam10_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_cb_lam10_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-4

# Twin-indep on the 5 held-out ChemBERTa datasets at the rule-selected
# lambda=10 (selected on BACE-ChemBERTa under the 0.02 tolerance rule).
# Closes the pretrained-backbone scope loop the way GIN closed the
# architecture loop: the rule transfers, the lambda value does not.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

SEEDS="1,2,3,4,5"
DATASETS=(bbbp_chemberta pgp_broccatelli_chemberta \
          bbb_martins_chemberta ames_chemberta dili_chemberta)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=== $DS twin_indep lam=10  $(hostname) | $(date) ==="
mkdir -p logs/cross_sample

python scripts/cross_sample_train.py \
    --dataset "$DS" --canonical_data_seed 99 \
    --train_seeds "$SEEDS" --mode twin_indep --lam 10.0 \
    > "logs/cross_sample/${DS}_twin_lam10.log" 2>&1

echo "Done: $(date)"
tail -15 "logs/cross_sample/${DS}_twin_lam10.log"
