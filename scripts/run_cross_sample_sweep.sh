#!/bin/bash
# Replicate the BACE positive result across small-N scientific datasets.
# Each (dataset, mode) gets 5 train_seeds on the canonical (data_seed=99) test set.

set -e
cd "$(dirname "$0")/.."

DATASETS=(bbbp tadf mof_thermal mof_solvent)
SEEDS="1,2,3,4,5"

mkdir -p logs/cross_sample

for DS in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "  $DS  $(date)"
    echo "=========================================="
    # ERM baseline
    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode erm \
        > "logs/cross_sample/${DS}_erm.log" 2>&1
    # Twin at λ=10
    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode twin --lam 10.0 \
        > "logs/cross_sample/${DS}_twin_lam10.log" 2>&1
    # Twin at λ=100
    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode twin --lam 100.0 \
        > "logs/cross_sample/${DS}_twin_lam100.log" 2>&1
    echo "  done $DS"
done

echo
echo "Sweep complete: $(date)"
