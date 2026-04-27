#!/bin/bash
# Cross-sample sweep on TDC ADME / Tox benchmarks.
# 8 datasets × {erm, bagging K=2, bagging K=5, deep_ensemble K=5, twin_indep λ=300} × 10 train_seeds.

set -e
cd "$(dirname "$0")/.."

DATASETS=(hia_hou bioavailability_ma pgp_broccatelli bbb_martins
          herg dili ames skin_reaction)
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/tdc

for DS in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "  $DS  $(date)"
    echo "=========================================="

    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode erm \
        > "logs/tdc/${DS}_erm.log" 2>&1
    echo "  $DS erm done"

    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode bagging --K 2 \
        > "logs/tdc/${DS}_bagging_K2.log" 2>&1
    echo "  $DS bagging K=2 done"

    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode bagging --K 5 \
        > "logs/tdc/${DS}_bagging_K5.log" 2>&1
    echo "  $DS bagging K=5 done"

    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode deep_ensemble --K 5 \
        > "logs/tdc/${DS}_deep_ensemble.log" 2>&1
    echo "  $DS deep ensemble done"

    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode twin_indep --lam 300.0 \
        > "logs/tdc/${DS}_twin_indep_lam300.log" 2>&1
    echo "  $DS twin_indep done"
done

echo
echo "TDC sweep complete: $(date)"
