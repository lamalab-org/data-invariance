#!/bin/bash
#SBATCH --job-name=inv-bord
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/borderline_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/borderline_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2

# Borderline datasets: 3 chemistry datasets × ERM × 10 seeds.
# Closes the prose claim that borderline magnitudes appear in
# Table~\ref{tab:fragility-magnitudes} (bottom group).

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

DATASETS=(skin_reaction herg hia_hou)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/borderline
LOG="logs/borderline/${DS}_erm.log"

echo "=== $DS  erm  $(hostname)  $(date) ===" > "$LOG"
python scripts/cross_sample_train.py \
  --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
  --mode erm >> "$LOG" 2>&1
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
