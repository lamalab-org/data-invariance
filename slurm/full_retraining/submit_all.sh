#!/bin/bash
# Master dispatcher: submit every block of the full paper retraining sweep.
#
# Usage (from repo root on draco):
#   bash slurm/full_retraining/submit_all.sh
#
# Each `sbatch` returns a job id; we record them all to
# `logs/full_retraining_jobs.txt` so progress can be tracked with
# `squeue --jobs $(paste -sd, logs/full_retraining_jobs.txt)`.

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs/slurm

dispatch() {
  local script="$1"
  local jobid
  jobid=$(sbatch --parsable "$script")
  echo "$jobid  $script"
  echo "$jobid" >> logs/full_retraining_jobs.txt
}

: > logs/full_retraining_jobs.txt

dispatch slurm/full_retraining/01_headline_cls.sh
dispatch slurm/full_retraining/02_pareto_bace.sh
dispatch slurm/full_retraining/03_chemberta.sh
dispatch slurm/full_retraining/04_gin_bace.sh
dispatch slurm/full_retraining/05_waterbirds.sh
dispatch slurm/full_retraining/06_regression.sh
dispatch slurm/full_retraining/07_nscaling_bace.sh
dispatch slurm/full_retraining/08_borderline.sh
dispatch slurm/full_retraining/09_excluded.sh

echo
echo "Submitted; ids in logs/full_retraining_jobs.txt"
echo "Monitor:  squeue --jobs \$(paste -sd, logs/full_retraining_jobs.txt) -u \$USER"
