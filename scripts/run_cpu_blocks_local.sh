#!/bin/bash
# Run every CPU block of the full paper retraining sweep on the local
# machine.  Mirrors the slurm dispatch in slurm/full_retraining/ but
# inline (no slurm; sequential).
#
# Blocks run:
#   01 headline classification (9 datasets x 6 methods x 10 seeds)
#   02 Pareto on BACE          (6 lambdas x 10)
#   06 regression              (3 datasets x 4 methods x 10)
#   07 N-scaling on BACE       (5 sizes x 10)
#   08 borderline ERM          (3 datasets x 10)
#   09 excluded ERM            (5 datasets x 10)
#
# The GPU blocks (03 ChemBERTa, 04 GIN, 05 Waterbirds) are NOT run
# here -- they need GPU and will be dispatched on draco when the
# cluster is back.
#
# Usage (from repo root):
#   bash scripts/run_cpu_blocks_local.sh > logs/local_cpu_sweep.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

mkdir -p logs/local_sweep
LOG_DIR=logs/local_sweep
SEEDS="1,2,3,4,5,6,7,8,9,10"
CANON=99
LAM=300.0

DATASETS_HEADLINE=(bace dili pgp_broccatelli bbb_martins bbbp tadf mof_thermal ames cyp2d6_substrate)
DATASETS_REG=(esol_reg freesolv_reg lipo_reg)
DATASETS_BORDER=(skin_reaction herg hia_hou)
DATASETS_EXCL=(bioavailability_ma mof_solvent cyp2c9_substrate cyp3a4_substrate clintox)
PARETO_LAMS=(1.0 3.0 10.0 30.0 100.0 300.0)
NSCALE_SIZES=(200 400 600 800 968)

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

echo "=== Block 01: headline classification ($(stamp)) ==="
for DS in "${DATASETS_HEADLINE[@]}"; do
  run "01_${DS}_erm"             --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode erm
  run "01_${DS}_mcd"             --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode mc_dropout --K 20
  run "01_${DS}_bagK2"           --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 2
  run "01_${DS}_bagK5"           --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 5
  run "01_${DS}_deepens"         --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode deep_ensemble --K 5
  run "01_${DS}_twin"            --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode twin_indep --lam $LAM
done

echo "=== Block 02: Pareto on BACE ($(stamp)) ==="
for L in "${PARETO_LAMS[@]}"; do
  run "02_bace_lam${L}"          --dataset bace --canonical_data_seed $CANON --train_seeds $SEEDS --mode twin_indep --lam $L
done

echo "=== Block 06: regression ($(stamp)) ==="
for DS in "${DATASETS_REG[@]}"; do
  run "06_${DS}_erm"             --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode erm
  run "06_${DS}_bagK2"           --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 2
  run "06_${DS}_bagK5"           --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode bagging --K 5
  run "06_${DS}_twin"            --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode twin_indep --lam 3.0
done

echo "=== Block 07: N-scaling on BACE ($(stamp)) ==="
for M in "${NSCALE_SIZES[@]}"; do
  mkdir -p "outputs/cross_sample_nscaling/M${M}"
  run "07_bace_M${M}"            --dataset bace --canonical_data_seed $CANON --train_seeds $SEEDS --mode erm \
                                   --subsample_size $M --output_dir "outputs/cross_sample_nscaling/M${M}"
done

echo "=== Block 08: borderline ERM ($(stamp)) ==="
for DS in "${DATASETS_BORDER[@]}"; do
  run "08_${DS}_erm"             --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode erm
done

echo "=== Block 09: excluded ERM ($(stamp)) ==="
for DS in "${DATASETS_EXCL[@]}"; do
  run "09_${DS}_erm"             --dataset "$DS" --canonical_data_seed $CANON --train_seeds $SEEDS --mode erm
done

echo "=== All CPU blocks done ($(stamp)) ==="
