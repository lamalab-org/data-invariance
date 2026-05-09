#!/bin/bash
#SBATCH --job-name=inv-bayes-twin
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/bayes_twin_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/bayes_twin_%A_%a.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-89%12

# Per-dataset Bayesian optimisation of lambda for twin-bootstrap
# (App.~bayes_twin).  Each array task runs one (dataset, train_seed)
# tuple: BAYES_TRIALS twin-indep trainings plus one final retraining
# at the BO-selected lambda.  Four overridable axes:
#
#   OBJECTIVE_SPLIT ∈ {val, id_test, ood_test}   default val
#       The "val" path carves a held-out fraction (or k folds) off the
#       canonical training pool; id_test / ood_test paths are oracle
#       baselines for ablation.
#
#   CV_FOLDS = N (>=1)                            default 1
#       1 = hold-out (val_frac of the pool), >=2 = k-fold CV (averaged
#       over folds; final retrain on full pool).
#
#   BO_OBJECTIVE ∈ {acc, churn_constrained, cross_sample_constrained}
#                                                 default acc
#       'acc' maximises val accuracy.  'churn_constrained' maximises
#       (-val_churn) -- inter-network disagreement on val (a lower-bound
#       proxy for cross-sample churn).  'cross_sample_constrained'
#       maximises (-val_cross_churn) -- argmax disagreement between two
#       ensembles trained at train_seed and train_seed+SHADOW_SEED_OFFSET
#       on the same fold-train pool, the val-side analogue of the
#       cross-sample churn the paper reports on test.  Costs ~2x per
#       trial.  See the python script's docstring for the exact formula.
#
#   SELECTION_RULE ∈ {best_score, rule_largest_lam}  default best_score
#       'rule_largest_lam' is the paper's pre-registered rule applied
#       per-dataset.
#
# Other useful overrides:
#   BAYES_TRIALS, BAYES_INIT_TRIALS, BAYES_SEED,
#   LAM_MIN, LAM_MAX,
#   VAL_FRAC, PREREG_TOLERANCE, CHURN_PENALTY,
#   INITIAL_LAMS, INCLUDE_ZERO_TRIAL,
#   OUTPUT_DIR.
#
# Default OUTPUT_DIR is split by mode so distinct runs don't clobber
# each other; you can override.

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
BO_OBJECTIVE=${BO_OBJECTIVE:-acc}
CHURN_PENALTY=${CHURN_PENALTY:-100.0}
SELECTION_RULE=${SELECTION_RULE:-best_score}
PREREG_TOLERANCE=${PREREG_TOLERANCE:-0.02}
SHADOW_SEED_OFFSET=${SHADOW_SEED_OFFSET:-1000}

# Per-mode default output dirs so hold-out / k-fold / churn-BO /
# cross-sample-BO sweeps don't clobber each other.  Explicit OUTPUT_DIR=
# overrides.
if [ -z "${OUTPUT_DIR:-}" ]; then
  if [ "$BO_OBJECTIVE" = "cross_sample_constrained" ]; then
    OUTPUT_DIR="outputs/cross_sample_bayes_cross"
  elif [ "$BO_OBJECTIVE" = "churn_constrained" ]; then
    OUTPUT_DIR="outputs/cross_sample_bayes_churn"
  elif [ "$CV_FOLDS" -ge 2 ]; then
    OUTPUT_DIR="outputs/cross_sample_bayes_kfold${CV_FOLDS}"
  else
    OUTPUT_DIR="outputs/cross_sample_bayes_val"
  fi
fi

mkdir -p logs/bayes_twin_headline
LOG="logs/bayes_twin_headline/${DS}_seed${SEED}_${BO_OBJECTIVE}_${OBJECTIVE_SPLIT}_cv${CV_FOLDS}.log"

echo "=== $DS twin_indep_bayes seed=$SEED trials=$BAYES_TRIALS lam=[$LAM_MIN,$LAM_MAX] objective=$OBJECTIVE_SPLIT cv_folds=$CV_FOLDS bo_objective=$BO_OBJECTIVE selection=$SELECTION_RULE out=$OUTPUT_DIR $(hostname) $(date) ===" > "$LOG"

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
  --bo_objective "$BO_OBJECTIVE"
  --churn_penalty "$CHURN_PENALTY"
  --selection_rule "$SELECTION_RULE"
  --prereg_tolerance "$PREREG_TOLERANCE"
  --shadow_seed_offset "$SHADOW_SEED_OFFSET"
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
