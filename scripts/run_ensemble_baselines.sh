#!/bin/bash
# Honest ensemble baselines under the cross-sample protocol:
#   deep_ensemble K=5: parameter variance, same data
#   bagging K=5:       parameter + data variance, no consistency loss
# 5 datasets × 10 train_seeds × 2 modes = 100 ensemble trainings (each = K=5 models).

set -e
cd "$(dirname "$0")/.."

DATASETS=(bace bbbp tadf mof_thermal mof_solvent)
SEEDS="1,2,3,4,5,6,7,8,9,10"

mkdir -p logs/ensemble

for DS in "${DATASETS[@]}"; do
    for MODE in deep_ensemble bagging; do
        echo "  $DS $MODE  $(date)"
        uv run python scripts/cross_sample_train.py \
            --dataset "$DS" --canonical_data_seed 99 \
            --train_seeds "$SEEDS" --mode "$MODE" --K 5 \
            > "logs/ensemble/${DS}_${MODE}.log" 2>&1
    done
done
echo "Sweep complete: $(date)"
