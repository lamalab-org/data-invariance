#!/bin/bash
#SBATCH --job-name=inv-nscale
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/nscale_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/nscale_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-4

# Within-dataset N-scaling on BACE: subsample the canonical pool to
# M in {200,400,600,800,968} and run ERM at each M with 10 seeds.

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance

SIZES=(200 400 600 800 968)
M=${SIZES[$SLURM_ARRAY_TASK_ID]}
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/nscaling
LOG="logs/nscaling/bace_M${M}.log"

echo "=== bace-N$M  erm  $(hostname)  $(date) ===" > "$LOG"
python scripts/cross_sample_train.py \
  --dataset bace --canonical_data_seed 99 --train_seeds "$SEEDS" \
  --mode erm --subsample_size "$M" \
  --output_dir outputs/cross_sample_nscaling/M${M} >> "$LOG" 2>&1
echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
