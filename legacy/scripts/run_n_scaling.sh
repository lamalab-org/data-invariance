#!/bin/bash
# Within-dataset N-scaling experiment on BACE.
# Subsamples the 968 training examples to M ∈ {200, 400, 600, 800, 968}
# and measures fragility at each size. Theory predicts fragility ~ 1/M.

set -e
cd "$(dirname "$0")/.."

SIZES=(200 400 600 800 968)
SEEDS="42,123,789"
OUT=outputs/fragility_nscaling

mkdir -p $OUT logs/n_scaling

for M in "${SIZES[@]}"; do
    echo "=========================================="
    echo "  BACE  subsample=$M  $(date)"
    echo "=========================================="

    # Partition pair at this size.
    LOG=logs/n_scaling/bace_M${M}_partition.log
    uv run python scripts/partition_pair_train.py \
        --dataset bace --data_seeds "$SEEDS" --K 2 --mode partition \
        --subsample_size "$M" \
        --output_dir "$OUT/M${M}" \
        > "$LOG" 2>&1

    # ERM at this size (for accuracy reference).
    LOG=logs/n_scaling/bace_M${M}_erm.log
    uv run python scripts/partition_pair_train.py \
        --dataset bace --data_seeds "$SEEDS" --K 1 --mode erm \
        --subsample_size "$M" \
        --output_dir "$OUT/M${M}" \
        > "$LOG" 2>&1

    echo "  done M=$M"
done

echo
echo "N-scaling sweep complete: $(date)"
ls -la $OUT/
