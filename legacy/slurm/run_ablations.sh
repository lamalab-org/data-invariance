#!/bin/bash
#SBATCH --job-name=inv-ablation
#SBATCH --output=/vast/lo45pic/data-invariance/logs/slurm_ablation_%j_%a.out
#SBATCH --error=/vast/lo45pic/data-invariance/logs/slurm_ablation_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

# Ablation experiments on Waterbirds (5 seeds each).
# Submit: sbatch slurm/run_ablations.sh
#
# Array indices:
#   0 = Upweight sensitivity: α ∈ {10, 20, 50, 100}
#   1 = λ sensitivity: 0.5x, 1x, 2x, 5x auto-λ
#   2 = Ingredient ablation: ERM, +upweight, +VREx, +SWA, full
#   3 = Resampling-OOD: 10 subsamples

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate invariance
cd /vast/lo45pic/data-invariance
mkdir -p logs

SEEDS="42,123,789,2024,7"

case $SLURM_ARRAY_TASK_ID in

0) # Upweight sensitivity
echo "=== Upweight sensitivity on Waterbirds ==="
for ALPHA in 10 20 50 100; do
    echo ">>> alpha=$ALPHA"
    python -c "
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from scripts.run_experiment import build_cfg, HPARAMS, run_method
from train import make_dataloaders, discover_environments, auto_lambda
from utils import set_seed

cfg = build_cfg('waterbirds')
epochs = HPARAMS['waterbirds']['epochs']
device = torch.device('cuda')
cfg.training.discovery_upweight = $ALPHA

results = []
for seed in [42, 123, 789, 2024, 7]:
    cfg.training.seed = seed; set_seed(seed)
    loaders = make_dataloaders(cfg)
    set_seed(seed)
    a, w, dm = discover_environments(cfg, loaders, device)
    lam = auto_lambda(dm, cfg)
    r = run_method('ours', cfg, loaders, device, seed, epochs,
                   assignment=a, weights=w, disc_m=dm, lam=lam)
    results.append(r['swa_free'])

print(f'alpha={$ALPHA}  SWA+Free={np.mean(results):.4f}±{np.std(results):.4f}')
" 2>&1
done > logs/ablation_upweight.log 2>&1
;;

1) # Lambda sensitivity
echo "=== Lambda sensitivity on Waterbirds ==="
for MULT in 0.5 1.0 2.0 5.0; do
    echo ">>> lambda_mult=$MULT"
    python -c "
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from scripts.run_experiment import build_cfg, HPARAMS, run_method
from train import make_dataloaders, discover_environments, auto_lambda
from utils import set_seed

cfg = build_cfg('waterbirds')
epochs = HPARAMS['waterbirds']['epochs']
device = torch.device('cuda')

results = []
for seed in [42, 123, 789, 2024, 7]:
    cfg.training.seed = seed; set_seed(seed)
    loaders = make_dataloaders(cfg)
    set_seed(seed)
    a, w, dm = discover_environments(cfg, loaders, device)
    lam = auto_lambda(dm, cfg) * $MULT
    r = run_method('ours', cfg, loaders, device, seed, epochs,
                   assignment=a, weights=w, disc_m=dm, lam=lam)
    results.append(r['swa_free'])

print(f'lambda_mult={$MULT}  SWA+Free={np.mean(results):.4f}±{np.std(results):.4f}')
" 2>&1
done > logs/ablation_lambda.log 2>&1
;;

2) # Ingredient ablation
echo "=== Ingredient ablation on Waterbirds ==="
python -c "
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from scripts.run_experiment import build_cfg, HPARAMS, run_method
from train import make_dataloaders, discover_environments, auto_lambda
from utils import set_seed

cfg = build_cfg('waterbirds')
epochs = HPARAMS['waterbirds']['epochs']
device = torch.device('cuda')

# A) Plain ERM (no discovery, no upweight, no VREx)
print('--- ERM ---')
r_erm = []
for seed in [42, 123, 789, 2024, 7]:
    cfg.training.seed = seed; set_seed(seed)
    loaders = make_dataloaders(cfg)
    r = run_method('erm', cfg, loaders, device, seed, epochs)
    r_erm.append(r['swa_free'])
print(f'ERM: {np.mean(r_erm):.4f}±{np.std(r_erm):.4f}')

# B) Discovery + upweight only (= JTT, lambda=0)
print('--- Discovery + upweight (no VREx) ---')
r_jtt = []
for seed in [42, 123, 789, 2024, 7]:
    cfg.training.seed = seed; set_seed(seed)
    loaders = make_dataloaders(cfg)
    set_seed(seed)
    a, w, dm = discover_environments(cfg, loaders, device)
    r = run_method('ours', cfg, loaders, device, seed, epochs,
                   assignment=a, weights=w, disc_m=dm, lam=0.0)
    r_jtt.append(r['swa_free'])
print(f'Upweight only: {np.mean(r_jtt):.4f}±{np.std(r_jtt):.4f}')

# C) Discovery + VREx (no upweight)
print('--- Discovery + VREx (no upweight) ---')
cfg.training.discovery_upweight = 0.0
r_vrex_only = []
for seed in [42, 123, 789, 2024, 7]:
    cfg.training.seed = seed; set_seed(seed)
    loaders = make_dataloaders(cfg)
    set_seed(seed)
    a, w, dm = discover_environments(cfg, loaders, device)
    lam = auto_lambda(dm, cfg)
    r = run_method('ours', cfg, loaders, device, seed, epochs,
                   assignment=a, weights=w, disc_m=dm, lam=lam)
    r_vrex_only.append(r['swa_free'])
cfg.training.discovery_upweight = 50.0
print(f'VREx only: {np.mean(r_vrex_only):.4f}±{np.std(r_vrex_only):.4f}')

# D) Full method (upweight + VREx + auto-lambda)
print('--- Full method ---')
r_full = []
for seed in [42, 123, 789, 2024, 7]:
    cfg.training.seed = seed; set_seed(seed)
    loaders = make_dataloaders(cfg)
    set_seed(seed)
    a, w, dm = discover_environments(cfg, loaders, device)
    lam = auto_lambda(dm, cfg)
    r = run_method('ours', cfg, loaders, device, seed, epochs,
                   assignment=a, weights=w, disc_m=dm, lam=lam)
    r_full.append(r['swa_free'])
print(f'Full: {np.mean(r_full):.4f}±{np.std(r_full):.4f}')
" > logs/ablation_ingredients.log 2>&1
;;

3) # Resampling-OOD
echo "=== Resampling-OOD on Waterbirds ==="
python scripts/resampling_stability_test.py \
    --dataset waterbirds --device cuda --n_subsamples 10 \
    > logs/ablation_resampling_waterbirds.log 2>&1
;;

esac

echo "Done: $(date)"
