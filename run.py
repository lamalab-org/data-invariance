from __future__ import annotations

from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from models import MLP, MultiHeadMLP, SplitMLP
from train import make_dataloaders, train_adversarial_split, train_adversarial_split_multi, train_erm, train_oracle_split, train_random_split
from utils import get_device, set_seed


def make_run_name(cfg: DictConfig) -> str:
    """Compose a human-readable run name from method and timestamp.

    Format: `{method}_{MMDD-HHMM}`, e.g. `adversarial_split_0330-1542`.
    Short enough to read in the wandb sidebar; unique enough to avoid collisions.
    """
    ts = datetime.now().strftime("%m%d-%H%M")
    return f"{cfg.method.name}_{ts}"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert to a plain dict so wandb can log it — OmegaConf objects aren't serialisable.
    # resolve=True expands any variable interpolations in the YAML.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=cfg.wandb.project,
        name=make_run_name(cfg),
        tags=list(cfg.wandb.tags),
        config=cfg_dict,
        # Hydra already saves outputs to outputs/<date>/<time>/; point wandb there
        # too so artefacts, logs, and configs are co-located.
        dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        mode="online" if cfg.wandb.enabled else "disabled",
    )

    set_seed(cfg.training.seed)
    device = get_device()
    loaders = make_dataloaders(cfg)

    # input_dim derived from the dataset so the model is not coupled to image shape
    input_dim = loaders["train"].dataset.images.shape[1:].numel()  # 3*28*28 = 2352

    if cfg.method.name == "erm":
        model = MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim).to(device)
        train_erm(cfg, model, loaders, device, run)

    elif cfg.method.name == "random_split":
        model = SplitMLP(
            input_dim=input_dim,
            hidden_dim=cfg.model.hidden_dim,
            separate_backbones=cfg.model.separate_backbones,
        ).to(device)
        train_random_split(cfg, model, loaders, device, run)

    elif cfg.method.name == "oracle_split":
        model = SplitMLP(
            input_dim=input_dim,
            hidden_dim=cfg.model.hidden_dim,
            separate_backbones=cfg.model.separate_backbones,
        ).to(device)
        train_oracle_split(cfg, model, loaders, device, run)

    elif cfg.method.name == "adversarial_split":
        if cfg.model.num_heads == 2:
            model = SplitMLP(
                input_dim=input_dim,
                hidden_dim=cfg.model.hidden_dim,
                separate_backbones=cfg.model.separate_backbones,
            ).to(device)
            train_adversarial_split(cfg, model, loaders, device, run)
        else:
            model = MultiHeadMLP(
                input_dim=input_dim,
                hidden_dim=cfg.model.hidden_dim,
                num_heads=cfg.model.num_heads,
                separate_backbones=cfg.model.separate_backbones,
            ).to(device)
            train_adversarial_split_multi(cfg, model, loaders, device, run)

    else:
        raise NotImplementedError(f"method '{cfg.method.name}' not yet implemented")

    run.finish()


if __name__ == "__main__":
    main()
