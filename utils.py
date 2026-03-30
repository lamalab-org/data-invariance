from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility.

    Covers Python, NumPy, and PyTorch (CPU + GPU). Does not set
    CUBLAS_WORKSPACE_CONFIG — add that env var if you need full
    determinism on CUDA at the cost of some performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_metrics(run: object, metrics: dict[str, float], step: int) -> None:
    """Log metrics to wandb and print to stdout.

    This is the single source of truth for metric names — training code must
    never call run.log() directly. Keeping names centralised here means wandb
    reports stay stable even as we add new methods.
    """
    run.log(metrics, step=step)
    parts = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"[epoch {step:03d}] {parts}")


def get_device() -> torch.device:
    """Return the best available device.

    Prefers MPS (Apple Silicon) over CPU so development on a MacBook
    gets hardware acceleration without requiring CUDA.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
