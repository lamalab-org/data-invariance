#!/bin/bash
# Lambda Pareto sweep: 5 datasets × {ERM, Twin λ ∈ [1, 3, 10, 30, 100, 300]} × 10 train seeds.
# All models on canonical (data_seed=99) test set.

set -e
cd "$(dirname "$0")/.."

DATASETS=(bace bbbp tadf mof_thermal mof_solvent)
LAMS=(1.0 3.0 10.0 30.0 100.0 300.0)
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/lambda_pareto

for DS in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "  $DS  $(date)"
    echo "=========================================="

    # ERM with 10 seeds.
    uv run python scripts/cross_sample_train.py \
        --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
        --mode erm \
        > "logs/lambda_pareto/${DS}_erm.log" 2>&1
    echo "  $DS erm done"

    # Twin at each λ.
    for LAM in "${LAMS[@]}"; do
        uv run python scripts/cross_sample_train.py \
            --dataset "$DS" --canonical_data_seed 99 --train_seeds "$SEEDS" \
            --mode twin --lam "$LAM" \
            > "logs/lambda_pareto/${DS}_twin_lam${LAM}.log" 2>&1
        echo "  $DS twin λ=$LAM done"
    done
done

echo
echo "Pareto sweep complete: $(date)"
