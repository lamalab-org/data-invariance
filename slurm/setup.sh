#!/bin/bash
# One-time setup on Draco cluster.
# Run interactively: bash slurm/setup.sh
set -e

WORKDIR=/vast/lo45pic/data-invariance
echo "=== Setting up in $WORKDIR ==="

# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd $WORKDIR
uv sync

# Quick GPU test (run from login node or interactive session)
uv run python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
from train import make_dataloaders, evaluate, discover_environments, train_vrex
print('All imports OK')
"

echo "=== Setup complete ==="
