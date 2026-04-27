#!/bin/bash
# Regularization sweep: does weight_decay reduce fragility?
# 3 datasets × 5 weight_decay values × {erm, partition} × 3 seeds.
# Default is 1e-4. We sweep 5 orders of magnitude around it.

set -e
cd "$(dirname "$0")/.."

DATASETS=(bace bbbp tadf)
WDS=(1e-5 1e-4 1e-3 1e-2 1e-1)
SEEDS="42,123,789"
OUT=outputs/fragility_wd

mkdir -p $OUT logs/wd

for DS in "${DATASETS[@]}"; do
    for WD in "${WDS[@]}"; do
        TAG=${DS}_wd${WD}
        echo "=== $TAG  $(date) ==="

        # ERM at this WD (for accuracy).
        uv run python scripts/partition_pair_train.py \
            --dataset "$DS" --data_seeds "$SEEDS" --K 1 --mode erm \
            --weight_decay "$WD" --output_dir "$OUT/wd${WD}" \
            > "logs/wd/${TAG}_erm.log" 2>&1

        # Partition pair at this WD (for fragility).
        uv run python scripts/partition_pair_train.py \
            --dataset "$DS" --data_seeds "$SEEDS" --K 2 --mode partition \
            --weight_decay "$WD" --output_dir "$OUT/wd${WD}" \
            > "logs/wd/${TAG}_partition.log" 2>&1
    done
done

echo
echo "Sweep complete: $(date)"
ls $OUT/
