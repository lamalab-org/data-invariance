"""Pure Group DRO on K=2 loss-discovered environments — any dataset.

Usage:
    uv run python scripts/dro_discovered.py --dataset cmnist
    uv run python scripts/dro_discovered.py --dataset waterbirds --seeds 42,123,789
    uv run python scripts/dro_discovered.py --dataset multi_cmnist --device cpu

This is the unified entry point for our main method across all datasets. Same
discovery (loss scoring, median split, upweight=50), same aggregator (pure
Group DRO), same SWA model selection. Dataset-specific training config
(lr, batch_size, epochs) is loaded from an in-file registry so each run is
reproducible from a single command.

Prints per-epoch metrics (line-buffered) and a final summary with both the
best-by-val checkpoint and the SWA (last-N-epoch weight average).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MLP, make_resnet_backbone  # noqa: E402
from train import (  # noqa: E402
    _ModelSelector,
    _val_score,
    discover_environments,
    evaluate,
    make_dataloaders,
)
from utils import set_seed  # noqa: E402

DEFAULT_SEEDS = [42, 123, 789]

def _to_device(x, device):
    """Move input to device, handling both tensors and dicts (for DistilBERT)."""
    if isinstance(x, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    return x.to(device)

DEFAULT_DRO_STEP_SIZE = 0.01  # exponentiated-gradient step for group weights (Sagawa default)
SWA_WINDOW = 5

# Per-dataset training config. Each entry matches the hyper-parameters we used
# for V-REx so the DRO numbers are a direct head-to-head comparison.
DATASET_TRAINING = {
    # Epoch counts match the V-REx configs used for the prior Waterbirds/
    # CMNIST/TADF results so the DRO numbers are directly comparable. Longer
    # training hurts DRO on the CMNIST family — it over-rotates group weights
    # and overfits the minority environment.
    "cmnist": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 10,
        "discovery_epochs": 5,
    },
    "multi_cmnist": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 15,
        "discovery_epochs": 5,
    },
    "tadf": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 20,
        "discovery_epochs": 5,
    },
    "waterbirds": {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 15,
        "discovery_epochs": 5,
    },
    # MOF thermal stability: medium chemistry dataset (N=3132).
    "mof_solvent": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 20,
        "discovery_epochs": 5,
    },
    # Battery capacity: large chemistry dataset (N=39504).
    "battery": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 15,
        "discovery_epochs": 5,
    },
    "mof_thermal": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 20,
        "discovery_epochs": 5,
    },
    # Perovskite solar cells: large chemistry dataset (N=48380).
    "perovskite": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 15,
        "discovery_epochs": 5,
    },
    # MoleculeNet: scaffold-split distribution shift benchmarks.
    # Morgan fingerprint features (2048 bits) + MLP.
    "bace": {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "bbbp": {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "epochs": 30, "discovery_epochs": 5},
    "hiv": {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256, "epochs": 20, "discovery_epochs": 5},
    # CivilComments: text toxicity with DistilBERT backbone.
    # Standard benchmark from WILDS / JTT / CnC / DFR.
    "civilcomments": {
        "lr": 2e-5,  # standard for fine-tuning DistilBERT
        "weight_decay": 1e-2,
        "batch_size": 32,
        "epochs": 5,
        "discovery_epochs": 2,
    },
    # CelebA: Sagawa et al. 2020 setup. Standard hparams from the DFR/DRO
    # literature: AdamW lr=1e-4, batch_size=128, ~10 epochs is enough on
    # ResNet-50 because the dataset is much larger than Waterbirds (162k
    # train) so we get many more gradient steps per epoch.
    "celeba": {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 10,
        "discovery_epochs": 3,  # large dataset → fewer discovery epochs needed
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def build_cfg(dataset_name: str) -> OmegaConf:
    """Build an OmegaConf matching what run.py constructs via Hydra.

    Loads the dataset YAML to get dataset-specific fields (correlation, paths).
    """
    repo_root = Path(__file__).resolve().parent.parent
    dataset_cfg = OmegaConf.load(repo_root / f"configs/dataset/{dataset_name}.yaml")
    tr = DATASET_TRAINING[dataset_name]

    return OmegaConf.create(
        {
            "dataset": OmegaConf.to_container(dataset_cfg),
            "model": {"hidden_dim": 256},
            "training": {
                "lr": tr["lr"],
                "weight_decay": tr["weight_decay"],
                "batch_size": tr["batch_size"],
                "epochs": tr["epochs"],
                "seed": 42,
                "discovery_epochs": tr["discovery_epochs"],
                "discovery_criterion": "loss",
                "discovery_quantile": 0.5,
                "discovery_upweight": 50.0,
                "discovery_reweight": 0.0,
                "discovery_rounds": 1,
                "num_discovery_envs": 2,
                # V-REx fields — unused by DRO but required by discover_environments
                "lambda_disagree": 10.0,
                "lambda_anneal_factor": 1.0,
                "adv_lr": 1e-2,
                "early_stop_patience": 5,
                "freeze_backbone": False,
                "balanced_sampling": False,
                "env_mixup": 0.0,
                "training_noise": 0.0,
                "adv_init": "zeros",
                "adv_init_scale": 1.0,
                "head_noise": 0.0,
                "adv_warmup_epochs": 0,
                "adv_steps_per_model_step": 1,
                "lambda_warmup_epochs": 0,
                "adv_entropy_bonus": 0.0,
                "lambda_threshold": 0.0,
                "lambda_ramp_range": 0.0,
                "adv_mode": "task_loss",
            },
            "method": {"name": "discovered_split"},
            "wandb": {"enabled": False},
        }
    )


def build_model(cfg, device: torch.device):
    """Build a fresh model matching the dataset's architecture."""
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    # MLP backbone — needs to know input_dim from the dataset
    return None  # caller will build it after instantiating loaders


def run_seed(cfg, seed: int, device: torch.device, epochs: int, dro_step_size: float) -> dict:
    log(f"\n=== SEED {seed} ===")
    t0 = time.time()

    cfg.training.seed = seed
    set_seed(seed)
    loaders = make_dataloaders(cfg)

    log(f"[seed={seed}] discovering environments ...")
    assignment, weights, disc_m = discover_environments(cfg, loaders, device)
    n_a = int((assignment == 0).sum().item())
    n_b = int((assignment == 1).sum().item())
    log(
        f"[seed={seed}] discovery: n_A={n_a} n_B={n_b} "
        f"signal_ratio={disc_m.get('adaptive/signal_ratio', float('nan')):.1f}"
    )

    # Fresh model for DRO training
    set_seed(seed)
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    elif cfg.dataset.arch == "distilbert":
        from models import make_distilbert_backbone
        backbone, out_dim = make_distilbert_backbone()
        model = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    else:
        input_dim = loaders["train"].dataset.input_dim
        model = MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    assignment_dev = assignment.to(device)
    weights_dev = weights.to(device)
    K = int(assignment.max().item()) + 1
    group_weights = torch.ones(K, device=device) / K

    selector = _ModelSelector()
    # Store ALL epoch snapshots on CPU. At the end we take the SWA window
    # ending at the best-by-val epoch (not at the end of training) so that
    # degradation after the peak cannot hurt SWA.
    all_states: list = []

    for epoch in range(epochs):
        model.train()
        ep_loss_sum = 0.0
        ep_n = 0
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            ce = F.cross_entropy(model(x), y, reduction="none")
            a = assignment_dev[idx]
            w = weights_dev[idx]

            env_losses = []
            env_present = []
            for k in range(K):
                mask = a == k
                if mask.any():
                    wk = w[mask]
                    env_losses.append((wk * ce[mask]).sum() / wk.sum())
                    env_present.append(k)

            if len(env_losses) >= 2:
                env_t = torch.stack(env_losses)
                gw = group_weights[env_present].detach()
                loss = (gw * env_t).sum()
            else:
                loss = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if len(env_losses) >= 2:
                with torch.no_grad():
                    group_weights[env_present] *= torch.exp(
                        dro_step_size * env_t.detach()
                    )
                    group_weights /= group_weights.sum()

            ep_loss_sum += loss.item() * x.size(0)
            ep_n += x.size(0)

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        val_score = _val_score(id_m)
        val_wga = id_m.get("worst_group_acc", id_m.get("acc", 0.0))
        ood_wga = ood_m.get("worst_group_acc", ood_m.get("acc", 0.0))
        ood_acc = ood_m.get("acc", 0.0)
        train_loss = ep_loss_sum / max(ep_n, 1)
        gw_str = " ".join(f"{w:.3f}" for w in group_weights.tolist())

        selector.update(
            val_score,
            model,
            {
                "epoch": epoch,
                "val_wga": val_wga,
                "val_score": val_score,
                "ood_wga": ood_wga,
                "ood_acc": ood_acc,
            },
        )
        all_states.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )

        log(
            f"[seed={seed}] epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  "
            f"val={val_wga:.4f}  "
            f"ood_wga={ood_wga:.4f}  "
            f"ood_acc={ood_acc:.4f}  "
            f"gw=[{gw_str}]"
        )

    # --- SWA: weight average of a SWA_WINDOW window anchored at the best-by-val epoch ---
    # Window is [best_epoch - (SWA_WINDOW - 1), best_epoch] clipped to [0, epoch].
    # This makes SWA robust to degradation after the peak: if the model gets
    # worse after epoch 12, we don't average in epochs 25-29.
    best_epoch = selector.best_metrics.get("epoch", len(all_states) - 1)
    swa_end = best_epoch + 1  # exclusive
    swa_start = max(0, swa_end - SWA_WINDOW)
    swa_window_states = all_states[swa_start:swa_end]
    log(
        f"[seed={seed}] SWA window: epochs [{swa_start}, {swa_end - 1}] "
        f"(anchored at best epoch {best_epoch})"
    )

    swa_state = {}
    for key in swa_window_states[-1]:
        tensors = [s[key] for s in swa_window_states]
        if tensors[0].is_floating_point():
            swa_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            swa_state[key] = tensors[-1]

    model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})

    # Only ResNet has BN running stats worth re-estimating; MLP has no BN.
    if cfg.dataset.arch == "resnet":
        model.train()
        with torch.no_grad():
            for batch in loaders["train"]:
                model(_to_device(batch["image"], device))

    swa_id_m = evaluate(model, loaders["id_test"], device)
    swa_ood_m = evaluate(model, loaders["ood_test"], device)
    swa_val_wga = swa_id_m.get("worst_group_acc", swa_id_m.get("acc", 0.0))
    swa_ood_wga = swa_ood_m.get("worst_group_acc", swa_ood_m.get("acc", 0.0))
    swa_ood_acc = swa_ood_m.get("acc", 0.0)

    best = selector.restore(model)
    best["swa_val_wga"] = swa_val_wga
    best["swa_ood_wga"] = swa_ood_wga
    best["swa_ood_acc"] = swa_ood_acc

    dt = time.time() - t0
    log(
        f"[seed={seed}] BEST ep={best.get('epoch', -1):2d}  "
        f"val={best.get('val_wga', 0.0):.4f}  "
        f"ood_wga={best.get('ood_wga', 0.0):.4f}  "
        f"ood_acc={best.get('ood_acc', 0.0):.4f}"
    )
    log(
        f"[seed={seed}] SWA (last {SWA_WINDOW})  "
        f"val={swa_val_wga:.4f}  "
        f"ood_wga={swa_ood_wga:.4f}  "
        f"ood_acc={swa_ood_acc:.4f}  "
        f"({dt/60:.1f} min)"
    )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_TRAINING.keys()))
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument(
        "--dro_step_size",
        type=float,
        default=DEFAULT_DRO_STEP_SIZE,
        help="Exponentiated-gradient step size for DRO group weights",
    )
    ap.add_argument(
        "--num_envs",
        type=int,
        default=2,
        help="Number of discovered environments K (default 2 → median split)",
    )
    args = ap.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = build_cfg(args.dataset)
    cfg.training.num_discovery_envs = args.num_envs
    epochs = cfg.training.epochs

    log(
        f"dataset={args.dataset}  device={device}  seeds={seeds}  "
        f"epochs={epochs}  dro_step_size={args.dro_step_size}  K={args.num_envs}"
    )

    results = {}
    for seed in seeds:
        results[seed] = run_seed(cfg, seed, device, epochs, args.dro_step_size)

    log(f"\n=== FINAL ({args.dataset}, DRO K=2 loss-discovery, upweight=50) ===")
    best_wgas = [r["ood_wga"] for r in results.values()]
    best_accs = [r["ood_acc"] for r in results.values()]
    swa_wgas = [r["swa_ood_wga"] for r in results.values()]
    swa_accs = [r["swa_ood_acc"] for r in results.values()]
    for seed, r in results.items():
        log(
            f"seed={seed}  "
            f"BEST ood_wga={r['ood_wga']:.4f} ood_acc={r['ood_acc']:.4f} "
            f"(ep{r.get('epoch', -1):2d})  "
            f"SWA ood_wga={r['swa_ood_wga']:.4f} ood_acc={r['swa_ood_acc']:.4f}"
        )
    log(
        f"\nBEST  MEAN ood_wga = {np.mean(best_wgas):.4f} ± {np.std(best_wgas):.4f}   "
        f"MEAN ood_acc = {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}"
    )
    log(
        f"SWA   MEAN ood_wga = {np.mean(swa_wgas):.4f} ± {np.std(swa_wgas):.4f}   "
        f"MEAN ood_acc = {np.mean(swa_accs):.4f} ± {np.std(swa_accs):.4f}"
    )


if __name__ == "__main__":
    main()
