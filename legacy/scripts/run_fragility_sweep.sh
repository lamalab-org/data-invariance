#!/bin/bash
# Fragility sweep on small-N scientific datasets.
# 7 datasets × {erm, ensemble(K=5), partition(K=2)} × 3 seeds = ~168 model trainings.
# All CPU-fast (MLP on tabular features); expect ~1–2h wall-clock locally.

set -e
cd "$(dirname "$0")/.."

DATASETS=(bace bbbp tadf mof_thermal mof_solvent battery perovskite)
SEEDS="42,123,789"
OUT=outputs/fragility

mkdir -p $OUT logs/fragility

for DS in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "  $DS  $(date)"
    echo "=========================================="

    for MODE_SPEC in "erm:1" "ensemble:5" "partition:2"; do
        MODE=${MODE_SPEC%%:*}
        K=${MODE_SPEC##*:}
        LOG=logs/fragility/${DS}_${MODE}.log
        echo ">> $DS / $MODE / K=$K  -> $LOG"
        uv run python scripts/partition_pair_train.py \
            --dataset "$DS" --data_seeds "$SEEDS" --K "$K" --mode "$MODE" \
            --save_train_preds --output_dir "$OUT" \
            > "$LOG" 2>&1
        tail -1 "$LOG" || true
    done
done

echo
echo "Sweep done: $(date)"
ls -la $OUT/
