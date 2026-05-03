#!/bin/bash
# Re-run the full main-table sweep at one additional canonical seed for
# the canonical-seed sensitivity analysis.  Writes to
# outputs/cross_sample_seed{seed}/<dataset>/.  The original outputs/
# cross_sample/ tree (canonical_seed=99) is untouched.
#
# Usage:
#   bash scripts/run_seed_sweep.sh 7   > logs/seed_sweep_7.log  2>&1 &
#   bash scripts/run_seed_sweep.sh 42  > logs/seed_sweep_42.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

CANON="${1:?usage: $0 <canonical_data_seed>}"
mkdir -p "logs/seed_sweep_${CANON}"
LOG_DIR="logs/seed_sweep_${CANON}"
SEEDS="1,2,3,4,5,6,7,8,9,10"
LAM=300.0
OUT="outputs/cross_sample_seed${CANON}"

DATASETS=(bace dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames cyp2d6_substrate)
PARETO_LAMS=(1.0 3.0 10.0 30.0 100.0 300.0)

stamp() { date +"%H:%M:%S"; }

run() {
  local tag="$1"; shift
  local out="$LOG_DIR/$tag.log"
  echo "[$(stamp)] start  $tag"
  if uv run python scripts/cross_sample_train.py "$@" >"$out" 2>&1; then
    echo "[$(stamp)] done   $tag"
  else
    echo "[$(stamp)] FAILED $tag (see $out)"
  fi
}

echo "=== canonical_data_seed=${CANON} ($(stamp)) ==="
for DS in "${DATASETS[@]}"; do
  run "${DS}_erm"     --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode erm                  --output_dir "$OUT"
  run "${DS}_mcd"     --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode mc_dropout --K 20    --output_dir "$OUT"
  run "${DS}_swa"     --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode swa                  --output_dir "$OUT"
  run "${DS}_bagK2"   --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 2        --output_dir "$OUT"
  run "${DS}_bagK5"   --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 5        --output_dir "$OUT"
  run "${DS}_deepens" --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode deep_ensemble --K 5  --output_dir "$OUT"
  run "${DS}_twin"    --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode twin_indep --lam $LAM --output_dir "$OUT"
done
# BACE Pareto sweep at the new seed for the lambda-selection-rule
# robustness check.
for L in "${PARETO_LAMS[@]}"; do
  run "bace_lam${L}"  --dataset bace --canonical_data_seed $CANON --train_seeds $SEEDS --mode twin_indep --lam $L --output_dir "$OUT"
done

echo "=== seed=${CANON} sweep done $(stamp) ==="
