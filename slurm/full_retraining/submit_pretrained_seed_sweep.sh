#!/bin/bash
# Submit the pretrained-backbone canonical-seed sensitivity sweep.
#
# For each additional canonical_data_seed (default 7 and 42), dispatches:
#   03_chemberta.sh   (6 datasets x 3 methods x 10 train_seeds, GPU)
#   04_gin_bace.sh    (4 methods x 10 train_seeds, GPU)
#   05_waterbirds.sh  (3 methods x 10 train_seeds, GPU)
#
# Output goes to outputs/cross_sample_seed${CANON}/<dataset>/ so the
# canonical_data_seed=99 NPZs are not overwritten.  Job IDs are
# appended to logs/pretrained_seed_sweep_jobs.txt.
#
# After completion, run on the workstation:
#   uv run python scripts/aggregate_seed_sensitivity.py
# to combine the per-seed outputs into outputs/<table>.csv with
# *_seed_lo / *_seed_hi spread columns.
#
# Usage from the cluster:
#   bash slurm/full_retraining/submit_pretrained_seed_sweep.sh           # seeds 7 and 42
#   CANON_SEEDS="7"   bash slurm/full_retraining/submit_pretrained_seed_sweep.sh
#   CANON_SEEDS="42"  bash slurm/full_retraining/submit_pretrained_seed_sweep.sh

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

CANON_SEEDS=${CANON_SEEDS:-"7 42"}
JOBS_FILE=logs/pretrained_seed_sweep_jobs.txt

echo "Pretrained-backbone seed sweep: $(date)" | tee -a "$JOBS_FILE"
for CANON in $CANON_SEEDS; do
  echo "--- canonical_data_seed=${CANON} ---" | tee -a "$JOBS_FILE"
  for SCRIPT in 03_chemberta.sh 04_gin_bace.sh 05_waterbirds.sh; do
    JOB=$(CANON=$CANON sbatch --parsable slurm/full_retraining/$SCRIPT)
    echo "  $JOB  $SCRIPT  CANON=$CANON" | tee -a "$JOBS_FILE"
  done
done

echo
echo "Submitted.  Monitor with:"
echo "  squeue --jobs \$(awk '/CANON=/{print \$1}' $JOBS_FILE | paste -sd,) -u \$USER"
