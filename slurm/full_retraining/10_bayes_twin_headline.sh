#!/bin/bash
#SBATCH --job-name=inv-bayes-twin
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm/bayes_twin_%A_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm/bayes_twin_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-89

# Bayesian lambda search for headline classification datasets.
# Each array task runs one (dataset, train_seed) tuple.  This keeps tasks
# bounded because each seed performs BAYES_TRIALS twin-indep trainings plus
# one final retraining at the selected lambda.
#
# Optional overrides:
#   BAYES_TRIALS=8 sbatch slurm/full_retraining/10_bayes_twin_headline.sh
#   LAM_MIN=1e-2 LAM_MAX=1e3 OBJECTIVE_SPLIT=ood_test sbatch ...
#   INITIAL_LAMS=1.0,10.0,100.0,300.0 sbatch ...
#   INCLUDE_ZERO_TRIAL=1 sbatch ...

# set -e
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate invariance
# cd /vast/lo45pic/data-invariance

DATASETS=(bace dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames cyp2d6_substrate)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

DS_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
DS=${DATASETS[$DS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

BAYES_TRIALS=${BAYES_TRIALS:-50}
BAYES_INIT_TRIALS=${BAYES_INIT_TRIALS:-4}
BAYES_SEED=${BAYES_SEED:-0}
LAM_MIN=${LAM_MIN:-1e-3}
LAM_MAX=${LAM_MAX:-3e2}
OBJECTIVE_SPLIT=${OBJECTIVE_SPLIT:-id_test}

mkdir -p logs/bayes_twin_headline
LOG="logs/bayes_twin_headline/${DS}_seed${SEED}_trials${BAYES_TRIALS}.log"

echo "=== $DS twin_indep_bayes seed=$SEED trials=$BAYES_TRIALS lam=[$LAM_MIN,$LAM_MAX] objective=$OBJECTIVE_SPLIT $(hostname) $(date) ===" > "$LOG"

CMD=(
  python scripts/cross_sample_train_bayes.py
  --dataset "$DS"
  --canonical_data_seed 99
  --train_seeds "$SEED"
  --bayes_trials "$BAYES_TRIALS"
  --bayes_init_trials "$BAYES_INIT_TRIALS"
  --bayes_seed "$BAYES_SEED"
  --lam_min "$LAM_MIN"
  --lam_max "$LAM_MAX"
  --objective_split "$OBJECTIVE_SPLIT"
)

if [ -n "${INITIAL_LAMS:-}" ]; then
  CMD+=(--initial_lams "$INITIAL_LAMS")
fi

if [ "${INCLUDE_ZERO_TRIAL:-0}" = "1" ]; then
  CMD+=(--include_zero_trial)
fi

"${CMD[@]}" >> "$LOG" 2>&1

echo "=== Done: $(date) ===" >> "$LOG"
tail -5 "$LOG"
