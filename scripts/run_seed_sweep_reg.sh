#!/bin/bash
# Regression sweep at one additional canonical seed.  Run alongside
# run_seed_sweep.sh once that has finished its (heavier) classification
# load.  Outputs go to outputs/cross_sample_seed{seed}/<dataset>_reg/.
set -e
cd "$(dirname "$0")/.."
CANON="${1:?usage: $0 <canonical_data_seed>}"
mkdir -p "logs/seed_sweep_reg_${CANON}"
LOG_DIR="logs/seed_sweep_reg_${CANON}"
SEEDS="1,2,3,4,5,6,7,8,9,10"
OUT="outputs/cross_sample_seed${CANON}"
DATASETS=(esol_reg freesolv_reg lipo_reg)
LAM_REG=3.0
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
echo "=== regression seed=${CANON} ($(stamp)) ==="
for DS in "${DATASETS[@]}"; do
  run "${DS}_erm"   --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode erm                     --output_dir "$OUT"
  run "${DS}_bagK2" --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 2           --output_dir "$OUT"
  run "${DS}_bagK5" --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 5           --output_dir "$OUT"
  run "${DS}_twin"  --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode twin_indep --lam $LAM_REG --output_dir "$OUT"
done
echo "=== reg seed=${CANON} done $(stamp) ==="
