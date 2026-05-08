#!/bin/bash
#SBATCH --job-name=inv-bayes-twin
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/bayes_twin_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/bayes_twin_%A_%a.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-89%12

# Bayesian lambda search for headline classification datasets.
# Each array task runs one (dataset, train_seed) tuple.  Each seed
# performs BAYES_TRIALS twin-indep trainings plus one final retraining
# at the selected lambda.
#
# Default protocol (rigorous, used for preprint+revision):
#   --objective_split val --val_frac 0.2
# Held-out validation is carved from the canonical training pool with
# seed=$CANONICAL_DATA_SEED before bootstrapping; id_test and ood_test
# never enter selection.  Outputs land in
# outputs/cross_sample_bayes_val/ to keep them separate from the v1
# id_test-objective runs in outputs/cross_sample_bayes/.
#
# Optional overrides:
#   BAYES_TRIALS=8 sbatch slurm/full_retraining/10_bayes_twin_headline.sh
#   OBJECTIVE_SPLIT=id_test OUTPUT_DIR=outputs/cross_sample_bayes \
#     sbatch slurm/full_retraining/10_bayes_twin_headline.sh
#   INITIAL_LAMS=1.0,10.0,100.0,300.0 sbatch ...
#   INCLUDE_ZERO_TRIAL=1 sbatch ...

set -e

cd /vast/lo45pic/data-invariance
PY="$HOME/.local/bin/uv run --no-sync python"

DATASETS=(bace dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames cyp2d6_substrate)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

DS_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
DS=${DATASETS[$DS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CANONICAL_DATA_SEED=${CANONICAL_DATA_SEED:-99}
BAYES_TRIALS=${BAYES_TRIALS:-50}
BAYES_INIT_TRIALS=${BAYES_INIT_TRIALS:-4}
BAYES_SEED=${BAYES_SEED:-0}
LAM_MIN=${LAM_MIN:-1e-3}
LAM_MAX=${LAM_MAX:-3e2}
OBJECTIVE_SPLIT=${OBJECTIVE_SPLIT:-val}
VAL_FRAC=${VAL_FRAC:-0.2}
CV_FOLDS=${CV_FOLDS:-1}
# Default output dir splits hold-out (cv=1) from k-fold runs so they
# don't clobber each other.
if [ "$CV_FOLDS" -ge 2 ]; then
  OUTPUT_DIR=${OUTPUT_DIR:-outputs/cross_sample_bayes_kfold${CV_FOLDS}}
else
  OUTPUT_DIR=${OUTPUT_DIR:-outputs/cross_sample_bayes_val}
fi

mkdir -p logs/bayes_twin_headline
LOG="logs/bayes_twin_headline/${DS}_seed${SEED}_trials${BAYES_TRIALS}_${OBJECTIVE_SPLIT}_cv${CV_FOLDS}.log"

echo "=== $DS twin_indep_bayes seed=$SEED trials=$BAYES_TRIALS lam=[$LAM_MIN,$LAM_MAX] objective=$OBJECTIVE_SPLIT cv_folds=$CV_FOLDS val_frac=$VAL_FRAC out=$OUTPUT_DIR $(hostname) $(date) ===" > "$LOG"

CMD=(
  $PY scripts/cross_sample_train_bayes.py
  --dataset "$DS"
  --canonical_data_seed "$CANONICAL_DATA_SEED"
  --train_seeds "$SEED"
  --bayes_trials "$BAYES_TRIALS"
  --bayes_init_trials "$BAYES_INIT_TRIALS"
  --bayes_seed "$BAYES_SEED"
  --lam_min "$LAM_MIN"
  --lam_max "$LAM_MAX"
  --objective_split "$OBJECTIVE_SPLIT"
  --cv_folds "$CV_FOLDS"
  --output_dir "$OUTPUT_DIR"
)

if [ "$OBJECTIVE_SPLIT" = "val" ] && [ "$CV_FOLDS" -lt 2 ]; then
  CMD+=(--val_frac "$VAL_FRAC")
fi

if [ -n "${INITIAL_LAMS:-}" ]; then
  CMD+=(--initial_lams "$INITIAL_LAMS")
fi

if [ "${INCLUDE_ZERO_TRIAL:-0}" = "1" ]; then
  CMD+=(--include_zero_trial)
fi

"${CMD[@]}" >> "$LOG" 2>&1

echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
