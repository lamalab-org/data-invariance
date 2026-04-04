from __future__ import annotations

from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from models import MLP, MultiHeadMLP, SplitMLP, make_resnet_backbone
from train import discover_environments, make_dataloaders, train_adversarial_split, train_adversarial_split_multi, train_discovered_split, train_erm, train_group_dro, train_jtt, train_oracle_split, train_random_split, train_resampling
from utils import get_device, set_seed


def make_run_name(cfg: DictConfig) -> str:
    """Compose a human-readable run name from dataset, method, and timestamp."""
    ts = datetime.now().strftime("%m%d-%H%M")
    return f"{cfg.dataset.name}_{cfg.method.name}_{ts}"


def _make_model(cfg: DictConfig, loaders, method_name: str, device):
    """Construct the right model for the dataset architecture and training method.

    MLP datasets (CMNIST): use input_dim from the dataset.
    ResNet datasets (Waterbirds): use a pretrained ResNet-50 backbone.

    For single-head methods (erm, discovered_split): returns MLP.
    For split methods: returns SplitMLP or MultiHeadMLP.
    """
    arch = cfg.dataset.arch

    if arch == "mlp":
        input_dim = loaders["train"].dataset.input_dim
        if method_name in ("erm", "discovered_split"):
            return MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim).to(device)
        elif method_name in ("random_split", "oracle_split", "resampling"):
            return SplitMLP(
                input_dim=input_dim,
                hidden_dim=cfg.model.hidden_dim,
                separate_backbones=cfg.model.separate_backbones,
            ).to(device)
        elif method_name == "adversarial_split":
            if cfg.model.num_heads == 2:
                return SplitMLP(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.hidden_dim,
                    separate_backbones=cfg.model.separate_backbones,
                ).to(device)
            else:
                return MultiHeadMLP(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.hidden_dim,
                    num_heads=cfg.model.num_heads,
                    separate_backbones=cfg.model.separate_backbones,
                ).to(device)

    elif arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        if method_name in ("erm", "discovered_split"):
            return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
        elif method_name in ("random_split", "oracle_split", "resampling"):
            return SplitMLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
        elif method_name == "adversarial_split":
            if cfg.model.num_heads == 2:
                return SplitMLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
            else:
                return MultiHeadMLP(backbone=backbone, backbone_out_dim=out_dim, num_heads=cfg.model.num_heads).to(device)

    raise ValueError(f"Unknown arch={arch} or method={method_name}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=cfg.wandb.project,
        name=make_run_name(cfg),
        tags=list(cfg.wandb.tags),
        config=cfg_dict,
        dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        mode="online" if cfg.wandb.enabled else "disabled",
    )

    set_seed(cfg.training.seed)
    device = get_device()
    loaders = make_dataloaders(cfg)

    method = cfg.method.name

    if method == "erm":
        model = _make_model(cfg, loaders, method, device)
        train_erm(cfg, model, loaders, device, run)

    elif method == "random_split":
        model = _make_model(cfg, loaders, method, device)
        train_random_split(cfg, model, loaders, device, run)

    elif method == "oracle_split":
        model = _make_model(cfg, loaders, method, device)
        train_oracle_split(cfg, model, loaders, device, run)

    elif method == "adversarial_split":
        model = _make_model(cfg, loaders, method, device)
        if cfg.model.num_heads == 2:
            train_adversarial_split(cfg, model, loaders, device, run)
        else:
            train_adversarial_split_multi(cfg, model, loaders, device, run)

    elif method == "resampling":
        model = _make_model(cfg, loaders, method, device)
        train_resampling(cfg, model, loaders, device, run)

    elif method == "discovered_split":
        n_rounds = getattr(cfg.training, "discovery_rounds", 1)
        model = None
        for round_i in range(n_rounds):
            # Round 0: discover from fresh ERM.
            # Round 1+: discover from previous round's trained model.
            assignment, weights, discovery_metrics = discover_environments(
                cfg, loaders, device, existing_model=model,
            )
            set_seed(cfg.training.seed + round_i)
            model = _make_model(cfg, loaders, method, device)
            train_discovered_split(cfg, model, loaders, device, run, assignment, weights, discovery_metrics)

    elif method == "jtt":
        # JTT (Just Train Twice): train discovery ERM, identify misclassified
        # examples, upweight them with a fixed factor in a second round.
        # Faithful to Liu et al. 2021: binary weights (misclassified → λ_up, rest → 1).
        from train import discover_jtt_weights
        weights, discovery_metrics = discover_jtt_weights(cfg, loaders, device)
        set_seed(cfg.training.seed)
        model = _make_model(cfg, loaders, "erm", device)
        train_jtt(cfg, model, loaders, device, run, weights, discovery_metrics)

    elif method == "group_dro":
        model = _make_model(cfg, loaders, "erm", device)  # single-head model
        train_group_dro(cfg, model, loaders, device, run)

    else:
        raise NotImplementedError(f"method '{method}' not yet implemented")

    run.finish()


if __name__ == "__main__":
    main()
