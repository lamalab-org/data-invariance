#!/bin/bash
# Collect results from all completed jobs into one summary.
#
# Usage:
#   bash slurm/collect_results.sh

cd "$(dirname "$0")/.."

echo "=== COMPLETE RESULTS ==="
echo "Generated: $(date)"
echo

for DS in waterbirds celeba civilcomments cmnist multi_cmnist tadf \
          mof_thermal mof_solvent perovskite battery bace bbbp hiv; do
    if [ -f "logs/final_${DS}.log" ]; then
        echo "--- $DS ---"
        grep "^Method\|^erm \|^jtt \|^ours " "logs/final_${DS}.log"
        echo
    else
        echo "--- $DS --- (not yet available)"
        echo
    fi
done
