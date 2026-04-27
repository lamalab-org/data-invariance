"""Shared utilities for cross-sample fragility analyses.

All paper analysis scripts (``make_main_table.py``, ``make_pareto.py``, …)
import from here so the load and metric logic exists in one place.

Naming convention
-----------------
NPZ files saved by ``cross_sample_train.py`` follow:
  ``erm_train{seed}.npz``
  ``bagging_train{seed}_K{K}.npz``
  ``deep_ensemble_train{seed}_K{K}.npz``
  ``twin_indep_train{seed}_lam{lam}.npz``

Each contains either ``id_probs`` (single-model methods) or
``id_probs_avg`` (averaged across heads/ensemble), plus ``id_labels``,
``id_indices``, and the ood_* counterparts.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Distributional disagreement metrics
# ---------------------------------------------------------------------------

def sym_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row symmetric KL between two probability matrices (N, C)."""
    log_p = np.log(p + eps)
    log_q = np.log(q + eps)
    return 0.5 * ((p * (log_p - log_q)).sum(-1) + (q * (log_q - log_p)).sum(-1))


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def _seed_from_filename(path: Path) -> int:
    return int(path.stem.split("train")[1].split("_")[0])


def load_runs(dataset_dir: Path, glob: str) -> list[tuple[int, dict]]:
    """Load every NPZ matching `glob`. Returns [(train_seed, dict), …] sorted by seed."""
    files = sorted(dataset_dir.glob(glob), key=_seed_from_filename)
    return [(_seed_from_filename(f), dict(np.load(f, allow_pickle=True))) for f in files]


def get_probs(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (id_probs, ood_probs), preferring averaged heads when present."""
    id_probs = d["id_probs_avg"] if "id_probs_avg" in d else d["id_probs"]
    ood_probs = d["ood_probs_avg"] if "ood_probs_avg" in d else d["ood_probs"]
    return id_probs, ood_probs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_run_accuracies(runs: Sequence[tuple[int, dict]]) -> tuple[list[float], list[float]]:
    """For a list of runs, return (id_accs, ood_accs) — one per train_seed."""
    id_accs, ood_accs = [], []
    for _, d in runs:
        idp, odp = get_probs(d)
        id_accs.append(float((idp.argmax(1) == d["id_labels"]).mean()))
        ood_accs.append(float((odp.argmax(1) == d["ood_labels"]).mean()))
    return id_accs, ood_accs


def pairwise_metrics(
    runs: Sequence[tuple[int, dict]],
) -> tuple[dict[tuple[int, int], dict[str, float]], list[tuple[int, int]]]:
    """Return per-pair churn and sym-KL on id and ood, plus the list of pairs.

    Used by ``analysis.bootstrap_paired`` to compute paired CIs.
    """
    pairs = list(combinations([s for s, _ in runs], 2))
    runs_by_seed = dict(runs)
    out: dict[tuple[int, int], dict[str, float]] = {}
    for sa, sb in pairs:
        idA, odA = get_probs(runs_by_seed[sa])
        idB, odB = get_probs(runs_by_seed[sb])
        out[(sa, sb)] = {
            "id_churn":  float((idA.argmax(1) != idB.argmax(1)).mean()),
            "id_sym_kl": float(sym_kl(idA, idB).mean()),
            "ood_churn": float((odA.argmax(1) != odB.argmax(1)).mean()),
            "ood_sym_kl": float(sym_kl(odA, odB).mean()),
        }
    return out, pairs


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean. Returns (mean, low, high)."""
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return (float(arr.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def bootstrap_paired(
    deltas: Iterable[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on the mean of `deltas`. CI excludes 0 ⇒ significant."""
    return bootstrap_ci(deltas, n_boot=n_boot, alpha=alpha, rng=rng)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

GLOBS = {
    "erm":              "erm_train*.npz",
    "deep_ensemble_5":  "deep_ensemble_train*_K5.npz",
    "bagging_2":        "bagging_train*_K2.npz",
    "bagging_5":        "bagging_train*_K5.npz",
    "twin_indep":       "twin_indep_train*_lam{lam}.npz",
}


def fmt_ci(t: tuple[float, float, float], pct: bool = False) -> str:
    """Format (mean, lo, hi) as 'm [lo, hi]' or 'm% [lo%, hi%]'."""
    if pct:
        return f"{t[0]*100:.1f} [{t[1]*100:.1f},{t[2]*100:.1f}]"
    return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"
