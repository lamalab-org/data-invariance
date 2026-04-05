#!/bin/bash
# Multi-seed Waterbirds comparison: ERM, JTT, Group DRO, DFR, Ours (adaptive).
# 5 seeds, validation-based model selection, consistent V-REx code.
#
# Usage: bash scripts/multiseed_waterbirds.sh
# Output: results printed to stdout, one line per (method, seed) pair.

set -e

SEEDS="42 123 456 789 1337"
COMMON="dataset=waterbirds training.epochs=15 training.batch_size=64 training.lr=1e-4 wandb.enabled=false"

for SEED in $SEEDS; do
    echo "===== SEED=$SEED ====="

    echo "--- ERM ---"
    uv run python run.py $COMMON method=erm training.seed=$SEED 2>&1 | grep '^\[epoch' | tail -1

    echo "--- JTT ---"
    uv run python run.py $COMMON method=jtt training.seed=$SEED training.discovery_epochs=1 training.discovery_criterion=loss training.discovery_upweight=50.0 2>&1 | grep '^\[epoch' | tail -1

    echo "--- Group DRO ---"
    uv run python run.py $COMMON method=group_dro training.seed=$SEED 2>&1 | grep '^\[epoch' | tail -1

    echo "--- DFR ---"
    uv run python run.py $COMMON method=dfr training.seed=$SEED 2>&1 | grep '^\[epoch' | tail -1

    echo "--- Ours (adaptive) ---"
    uv run python run.py $COMMON method=discovered_split training.seed=$SEED training.discovery_epochs=5 training.lambda_disagree=10.0 training.discovery_criterion=loss training.discovery_upweight=50.0 training.early_stop_patience=5 2>&1 | grep '^\[epoch' | tail -1

    echo
done
