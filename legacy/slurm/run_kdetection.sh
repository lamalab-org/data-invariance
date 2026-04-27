#!/bin/bash
#SBATCH --job-name=inv-kdetect
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_kdetect_%j.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_kdetect_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# K-detection diagnostic across all chemistry + MoleculeNet datasets.
# Submit: sbatch slurm/run_kdetection.sh

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance
mkdir -p logs

echo "=== K-detection across all datasets ==="
for DS in cmnist multi_cmnist tadf mof_thermal mof_solvent perovskite battery bace bbbp hiv; do
    echo ">>> $DS"
    python scripts/k_detection.py --dataset $DS 2>&1 | grep -E "loss dist|Elbow|Best K|Oracle|group [0-9]"
    echo
done > logs/kdetection_all.log 2>&1

echo "Done: $(date)"
