from __future__ import annotations

from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


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

    # TODO: call train(cfg, run) once train.py is implemented
    raise NotImplementedError("Wire up train() here once train.py is implemented.")

    run.finish()


if __name__ == "__main__":
    main()
