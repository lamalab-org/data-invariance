#!/bin/bash
#SBATCH --job-name=inv-excl
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/excluded_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/excluded_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-4

# Excluded datasets: 5 chemistry datasets × ERM × 10 seeds.
# Populates the filter-outcomes table (Appendix~\ref{tab:filter-outcomes})
# documenting why each excluded dataset fails the +5pp ERM-vs-majority filter.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=(bioavailability_ma mof_solvent cyp2c9_substrate cyp3a4_substrate clintox)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/excluded
LOG="logs/excluded/${DS}_erm.log"

echo "=== $DS  erm  $(hostname)  $(date) ===" > "$LOG"
python scripts/cross_sample_train.py \
  --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
  --mode erm >> "$LOG" 2>&1
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
