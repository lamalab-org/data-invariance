#!/bin/bash
# Adaptive λ (gradient-norm balanced) sweep across 5 small-N datasets, 10 seeds.

set -e
cd "$(dirname "$0")/.."

DATASETS=(bace bbbp tadf mof_thermal mof_solvent)
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/gradnorm

for DS in "${DATASETS[@]}"; do
    echo "  $DS  $(date)"
    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode twin_gradnorm --target_ratio 1.0 \
        > "logs/gradnorm/${DS}.log" 2>&1
done

echo "Sweep complete: $(date)"
