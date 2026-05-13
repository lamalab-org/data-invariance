#!/bin/bash
#SBATCH --job-name=inv-bo-perds
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/bo_perds_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/bo_perds_%A_%a.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-89%12

# Stage 2 of the per-dataset BO protocol (Appendix~\ref{app:bayes_twin}).
#
# Stage 1: cross_sample_train_bayes.py runs BO per (dataset, train_seed)
#          and writes one NPZ per cell at the seed-level BO-selected
#          lambda.  Outputs go to outputs/cross_sample_bayes_cross/.
# Bridge:  scripts/select_perdataset_lambda.py reads those NPZs and
#          writes the per-dataset median lambda to
#          outputs/bo_perdataset_lambdas.csv.
# Stage 2 (this script): retrain twin-bootstrap at the dataset's median
#          lambda for every train_seed, so cross-sample churn between
#          any two retrainings within a dataset is fixed-lambda variance
#          on both arms of the comparison vs. the frozen lambda=300
#          baseline.
#
# Run stage 1 first, then:
#   uv run python scripts/select_perdataset_lambda.py \
#       --source outputs/cross_sample_bayes_cross \
#       --csv    outputs/bo_perdataset_lambdas.csv
#   sbatch slurm/full_retraining/11_bo_perdataset_lam.sh

set -e

cd /vast/lo45pic/data-invariance
PY="$HOME/.local/bin/uv run --no-sync python"

LAMBDA_CSV=${LAMBDA_CSV:-outputs/bo_perdataset_lambdas.csv}
if [ ! -f "$LAMBDA_CSV" ]; then
  echo "ERROR: lambda CSV not found at $LAMBDA_CSV"
  echo "Run scripts/select_perdataset_lambda.py first." >&2
  exit 2
fi

# Read (dataset, median_lambda) pairs from the CSV, skipping the header.
# awk preserves the order they appear in the file.
mapfile -t DATASETS < <(awk -F, 'NR>1 {print $1}' "$LAMBDA_CSV")
mapfile -t LAMS     < <(awk -F, 'NR>1 {print $2}' "$LAMBDA_CSV")
SEEDS=(1 2 3 4 5 6 7 8 9 10)

DS_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
DS=${DATASETS[$DS_IDX]}
LAM=${LAMS[$DS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CANONICAL_DATA_SEED=${CANONICAL_DATA_SEED:-99}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/cross_sample_bayes_perdataset}

mkdir -p logs/bo_perdataset
LOG="logs/bo_perdataset/${DS}_seed${SEED}_lam${LAM}.log"

echo "=== $DS twin_indep seed=$SEED lam=$LAM out=$OUTPUT_DIR $(hostname) $(date) ===" > "$LOG"

$PY scripts/cross_sample_train.py \
  --dataset "$DS" \
  --canonical_data_seed "$CANONICAL_DATA_SEED" \
  --train_seeds "$SEED" \
  --mode twin_indep \
  --lam "$LAM" \
  --output_dir "$OUTPUT_DIR" \
  >> "$LOG" 2>&1

echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
