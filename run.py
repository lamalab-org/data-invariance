"""Entry point for training via Hydra config.

Usage:
    python run.py method=erm dataset=waterbirds
    python run.py method=discovered_split dataset=waterbirds training.epochs=15
    python run.py method=jtt dataset=cmnist
"""
from __future__ import annotations

from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from train import (
    _build_model,
    discover_environments,
    discover_jtt_weights,
    make_dataloaders,
    train_erm,
    train_jtt,
    train_vrex,
)
from utils import get_device, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.dataset.name}_{cfg.method.name}_{datetime.now():%m%d-%H%M}",
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        mode="online" if cfg.wandb.enabled else "disabled",
    )

    set_seed(cfg.training.seed)
    device = get_device()
    loaders = make_dataloaders(cfg)
    method = cfg.method.name

    if method == "erm":
        model = _build_model(cfg, loaders, device)
        train_erm(cfg, model, loaders, device, run)

    elif method == "discovered_split":
        assignment, weights, disc_metrics = discover_environments(cfg, loaders, device)
        set_seed(cfg.training.seed)
        model = _build_model(cfg, loaders, device)
        train_vrex(cfg, model, loaders, device, run, assignment, weights, disc_metrics)

    elif method == "jtt":
        weights, disc_metrics = discover_jtt_weights(cfg, loaders, device)
        set_seed(cfg.training.seed)
        model = _build_model(cfg, loaders, device)
        train_jtt(cfg, model, loaders, device, run, weights, disc_metrics)

    else:
        raise NotImplementedError(f"Unknown method: {method}")

    run.finish()


if __name__ == "__main__":
    main()
