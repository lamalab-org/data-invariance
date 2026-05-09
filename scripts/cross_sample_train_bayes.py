"""Bayesian-optimization driver for ``_train_twin_indep``.

Sibling entry point to ``scripts/cross_sample_train.py``.  Same
cross-sample data protocol, but the twin-indep consistency weight
``lambda`` is chosen by a Gaussian-process BO before the final
predictions are saved.

Selection-vs-reporting hygiene
------------------------------
``id_test`` and ``ood_test`` are never seen during optimisation.
The default ``--objective_split val`` carves a deterministic split
of the canonical training pool (seed = ``--canonical_data_seed``):

* ``--cv_folds 1`` (default): hold out ``--val_frac`` of the pool once;
  BO scores on this val.
* ``--cv_folds k`` (k>=2): split the pool into k contiguous folds; each
  BO trial trains k twins (one per fold) and averages val score across
  folds.  Final retraining uses the full pool.

``--objective_split {id_test, ood_test}`` are upper-bound oracle modes
for ablation only (not held-out).

BO objective and selection
--------------------------
``--bo_objective`` switches what the BO maximises per trial:

* ``acc`` (default): ensemble val accuracy.
* ``churn_constrained``: ``-val_churn`` with a steep penalty when val
  accuracy falls more than ``--prereg_tolerance`` below the unregularised
  baseline (twin@lambda=0).  ``val_churn`` is the inter-network argmax
  disagreement on val (free at every twin training).  The first trial
  is forced to lambda=0 to set the baseline.

``--selection_rule`` switches how the final lambda is picked from the
trial pool:

* ``best_score`` (default): trial with the highest BO objective.
* ``rule_largest_lam``: the paper's pre-registered rule applied
  per-dataset -- among trials whose val accuracy is within
  ``--prereg_tolerance`` of the smallest-lambda trial, pick the largest
  lambda.  Feasibility is checked against ``metrics["acc"]`` directly
  so it stays correct under any ``--bo_objective``.

Optimizer
---------
sklearn ``GaussianProcessRegressor`` with a Matern-2.5 kernel; searches
in log10(lambda) over ``[--lam_min, --lam_max]``; expected-improvement
acquisition.  The initial ``--bayes_init_trials`` positive-lambda trials
are random uniform-in-log10; subsequent trials use the GP.

Provenance
----------
Each saved NPZ has a manifest sidecar (``_provenance``) recording git
commit, env, full command, canonical_data_seed, train_seed, lam,
cv_folds, fold_val_hashes (one per fold), val_hash, val_size,
bo_objective, churn_penalty, selection_rule, prereg_tolerance,
n_train, n_train_full, lam_min, lam_max, best_score.  test_hash always
identifies the canonical id_test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._provenance import make_context as _make_provenance  # noqa: E402
from scripts.cross_sample_train import (  # noqa: E402
    _is_regression,
    _predict,
    _predict_reg,
    _train_twin_indep,
)
from scripts.run_experiment import HPARAMS, build_cfg  # noqa: E402
from train import make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float,
                          xi: float) -> np.ndarray:
    improvement = mu - best - xi
    z = np.divide(
        improvement,
        sigma,
        out=np.zeros_like(improvement),
        where=sigma > 1e-12,
    )
    return improvement * _normal_cdf(z) + sigma * _normal_pdf(z)


def _parse_lams(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(x) for x in raw.split(",") if x.strip()]


def _candidate_lam(trials: list[dict], args) -> float:
    """Pick the next lambda with GP expected improvement."""
    positive_trials = [t for t in trials if t["lam"] > 0.0]
    if len(positive_trials) < max(1, args.bayes_init_trials):
        rng = np.random.default_rng(args.bayes_seed + len(trials))
        return float(10.0 ** rng.uniform(
            math.log10(args.lam_min),
            math.log10(args.lam_max),
        ))

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cross_sample_train_bayes.py needs scikit-learn for the Gaussian-process "
            "Bayesian optimizer. Install project dependencies or run through the "
            "project environment, for example: uv run python scripts/cross_sample_train_bayes.py ..."
        ) from exc

    x = np.array([[math.log10(t["lam"])] for t in positive_trials], dtype=np.float64)
    y = np.array([t["score"] for t in positive_trials], dtype=np.float64)
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=args.bayes_seed,
        n_restarts_optimizer=3,
    )
    gp.fit(x, y)

    grid = np.linspace(
        math.log10(args.lam_min),
        math.log10(args.lam_max),
        args.bayes_grid_size,
    ).reshape(-1, 1)
    mu, sigma = gp.predict(grid, return_std=True)
    ei = _expected_improvement(mu, sigma, best=float(y.max()), xi=args.bayes_xi)

    tried = {round(math.log10(t["lam"]), 12) for t in positive_trials}
    order = np.argsort(ei)[::-1]
    for idx in order:
        log_lam = float(grid[idx, 0])
        if round(log_lam, 12) not in tried:
            return float(10.0 ** log_lam)
    return float(10.0 ** grid[int(order[0]), 0])


def _twin_predict(cfg, m_a, m_b, loader, device):
    """Return ``(ensemble, labels, p_a, p_b)`` on the loader.

    For classification, ``ensemble = (softmax_A + softmax_B) / 2``
    (probabilities, shape ``[N, C]``); for regression, scalar averages.
    ``p_a`` and ``p_b`` are returned for inter-network churn.
    """
    if _is_regression(cfg):
        p_a, y, _ = _predict_reg(m_a, loader, device)
        p_b, _, _ = _predict_reg(m_b, loader, device)
        return 0.5 * (p_a + p_b), y, p_a, p_b
    p_a, y, _ = _predict(m_a, loader, device)
    p_b, _, _ = _predict(m_b, loader, device)
    return 0.5 * (p_a + p_b), y, p_a, p_b


def _twin_metrics(cfg, m_a, m_b, loader, device) -> dict[str, float]:
    """Within-twin metrics on `loader`.

    Returns ``acc`` (ensemble accuracy or negative MAE) and ``churn``
    (inter-network argmax disagreement, or |A-B|.mean() for regression).
    Both come from one forward-pass pair (A and B).
    """
    avg, y, p_a, p_b = _twin_predict(cfg, m_a, m_b, loader, device)
    if _is_regression(cfg):
        return {
            "acc": -float(np.abs(avg - y).mean()),
            "churn": float(np.abs(p_a - p_b).mean()),
        }
    return {
        "acc": float((avg.argmax(1) == y).mean()),
        "churn": float((p_a.argmax(1) != p_b.argmax(1)).mean()),
    }


def _cross_sample_metrics(cfg, twin_canonical, twin_shadow, loader, device
                          ) -> dict[str, float]:
    """Cross-train_seed metrics on `loader`.

    Both pairs are twins trained at the same lambda but with
    *different* canonical train_seeds.  The reported metrics are:

    * ``acc``         -- canonical ensemble accuracy (the deployed model).
    * ``churn``       -- inter-network disagreement of the canonical pair
                         (kept as a within-twin diagnostic).
    * ``cross_churn`` -- argmax disagreement between the two ensembles
                         on the same loader.  This is the direct
                         val-side analogue of the cross-sample churn
                         the paper reports on test, and is the BO
                         objective in ``cross_sample_constrained`` mode.

    For regression, ``cross_churn`` is the mean absolute difference
    between the two ensemble predictions.
    """
    m_a, m_b = twin_canonical
    m_c, m_d = twin_shadow
    canon_avg, y, p_a, p_b = _twin_predict(cfg, m_a, m_b, loader, device)
    shadow_avg, _, _, _ = _twin_predict(cfg, m_c, m_d, loader, device)
    if _is_regression(cfg):
        return {
            "acc": -float(np.abs(canon_avg - y).mean()),
            "churn": float(np.abs(p_a - p_b).mean()),
            "cross_churn": float(np.abs(canon_avg - shadow_avg).mean()),
        }
    return {
        "acc": float((canon_avg.argmax(1) == y).mean()),
        "churn": float((p_a.argmax(1) != p_b.argmax(1)).mean()),
        "cross_churn": float((canon_avg.argmax(1) != shadow_avg.argmax(1)).mean()),
    }


def _train_twin_pair(cfg, canonical_loaders, device, seed: int,
                     epochs: int, lam: float, pool_idx, n_train: int):
    """Wrapper around ``_train_twin_indep`` with explicit free-after-use.

    Caller is responsible for ``del`` and ``empty_cache`` when done with
    the returned ``(m_a, m_b)`` pair.
    """
    return _train_twin_indep(
        cfg, canonical_loaders, device, seed, epochs, lam,
        pool_idx=pool_idx, n_train=n_train,
    )


def _free_twins(*twins) -> None:
    for pair in twins:
        if pair is None:
            continue
        for m in pair:
            del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _trial_metrics(cfg, canonical_loaders, device, train_seed: int,
                   epochs: int, lam: float, *, folds, objective_loader,
                   pool_idx, n_train: int,
                   shadow_seed_offset: int | None = None
                   ) -> dict[str, float]:
    """Run one BO trial at ``lam`` and return mean ``{acc, churn[, cross_churn]}``.

    Standard mode (``shadow_seed_offset=None``): one twin pair per
    fold (or one for hold-out), measured by ``_twin_metrics``.

    Cross-sample mode (``shadow_seed_offset`` is an int): two twin
    pairs per fold -- canonical at ``train_seed``, shadow at
    ``train_seed + shadow_seed_offset`` -- measured by
    ``_cross_sample_metrics``.  ``cross_churn`` is the val-side analogue
    of the test-time cross-sample churn the paper reports, and feeds
    the ``cross_sample_constrained`` BO objective.

    k-fold mode averages metrics across folds and records per-fold
    arrays under ``*_per_fold``.
    """
    use_shadow = shadow_seed_offset is not None

    def _measure(loader, train_idx, n: int) -> dict[str, float]:
        canonical = _train_twin_pair(
            cfg, canonical_loaders, device, train_seed, epochs, lam,
            pool_idx=train_idx, n_train=n,
        )
        if not use_shadow:
            metrics = _twin_metrics(cfg, *canonical, loader, device)
            _free_twins(canonical)
            return metrics
        shadow = _train_twin_pair(
            cfg, canonical_loaders, device, train_seed + shadow_seed_offset,
            epochs, lam, pool_idx=train_idx, n_train=n,
        )
        metrics = _cross_sample_metrics(cfg, canonical, shadow, loader, device)
        _free_twins(canonical, shadow)
        return metrics

    if folds is not None:
        per_fold: list[dict[str, float]] = [
            _measure(f_val_loader, f_train_idx, len(f_train_idx))
            for f_train_idx, _, f_val_loader in folds
        ]
        out: dict[str, float] = {}
        for key in ("acc", "churn", "cross_churn"):
            if key not in per_fold[0]:
                continue
            vals = [m[key] for m in per_fold]
            out[key] = float(np.mean(vals))
            out[f"{key}_per_fold"] = [round(v, 4) for v in vals]
        return out

    return _measure(objective_loader, pool_idx, n_train)


def _bo_score(metrics: dict[str, float], args, baseline_acc: float | None
              ) -> float:
    """Compose a scalar BO objective from a trial's metrics dict.

    Three modes, switched by ``--bo_objective``:

    * ``acc``                       -- maximise val accuracy (legacy).
    * ``churn_constrained``         -- maximise (-val_churn) subject to
      the accuracy constraint.  ``val_churn`` is inter-network
      disagreement (within-twin); a known lower bound on the
      cross-sample churn the paper reports.
    * ``cross_sample_constrained``  -- maximise (-val_cross_churn)
      subject to the same accuracy constraint.  ``val_cross_churn`` is
      argmax disagreement between two ensembles trained at different
      ``train_seed``s on the same fold-train pool -- the val-side
      direct analogue of the test-time cross-sample churn the paper
      reports.

    The constraint is enforced via a steep linear penalty proportional
    to the accuracy deficit below ``baseline_acc - prereg_tolerance``;
    ``--churn_penalty`` sets the slope.  The first trial (forced to
    lambda=0) sets ``baseline_acc``; before that, the score is just
    the negated churn (no penalty), giving BO a valid initial target.
    """
    if args.bo_objective == "acc":
        return metrics["acc"]
    churn_key = ("cross_churn" if args.bo_objective == "cross_sample_constrained"
                 else "churn")
    if churn_key not in metrics:
        raise KeyError(
            f"BO objective {args.bo_objective!r} expected '{churn_key}' in "
            f"metrics; got keys {sorted(metrics)}"
        )
    if baseline_acc is None:
        return -metrics[churn_key]
    deficit = max(0.0, (baseline_acc - args.prereg_tolerance) - metrics["acc"])
    return -metrics[churn_key] - args.churn_penalty * deficit


def _idx_hash(loader) -> str:
    ds = loader.dataset
    idxs = getattr(ds, "indices", None)
    if idxs is None:
        idxs = list(range(len(ds)))
    return hashlib.sha256(np.asarray(idxs, dtype=np.int64).tobytes()).hexdigest()[:16]


def _make_val_loader(base, val_idx: np.ndarray, train_loader) -> DataLoader:
    """DataLoader iterating ``base[val_idx]`` with the canonical kwargs.

    Indices are sorted before subsetting so the val loader's iteration
    order is stable and independent of the shuffle that produced
    ``val_idx`` (loader's own ``shuffle=False`` then locks the order).
    """
    val_idx_sorted = np.sort(val_idx)
    return DataLoader(
        Subset(base, val_idx_sorted.tolist()),
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        collate_fn=train_loader.collate_fn,
    )


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()
                          ).hexdigest()[:16]


def _shuffled_pool(canonical_loaders, canonical_data_seed: int,
                   subsample_size: int | None) -> np.ndarray:
    """Deterministic permutation of the canonical training-pool indices.

    Used as the source-of-truth for both hold-out and k-fold paths so
    they share the same shuffle and stay reproducible by
    ``canonical_data_seed`` alone.
    """
    n = len(canonical_loaders["train"].dataset)
    shuffled = np.random.default_rng(canonical_data_seed).permutation(n)
    if subsample_size is not None and subsample_size < n:
        shuffled = shuffled[:subsample_size]
    return shuffled


def _carve_val(canonical_loaders, canonical_data_seed: int, val_frac: float,
               subsample_size: int | None
               ) -> tuple[np.ndarray, np.ndarray, DataLoader]:
    """Hold-out: split the canonical training pool into (train_pool, val).

    The split is deterministic in ``canonical_data_seed`` and lives entirely
    inside the canonical training set: ``id_test`` and ``ood_test`` are
    untouched.  When ``subsample_size`` is set, the subsample is drawn
    *first* and val is then carved from that prefix, so the subsample-vs-no
    subsample comparison stays apples-to-apples on the val fraction.

    Returns ``(pool_idx, val_idx, val_loader)``.  ``val_idx`` is in
    shuffle order (provenance hash is computed against the unsorted form);
    the loader sorts before subsetting so iteration order is stable.
    """
    shuffled = _shuffled_pool(canonical_loaders, canonical_data_seed,
                              subsample_size)
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"--val_frac must be in (0, 1); got {val_frac}")
    n_val = max(1, int(round(val_frac * len(shuffled))))
    if n_val >= len(shuffled):
        raise ValueError(
            f"--val_frac={val_frac} leaves no training pool "
            f"(pool size {len(shuffled)}, n_val {n_val})"
        )
    val_idx = shuffled[:n_val]
    pool_idx = shuffled[n_val:]

    val_loader = _make_val_loader(canonical_loaders["train"].dataset,
                                  val_idx, canonical_loaders["train"])
    return pool_idx, val_idx, val_loader


def _carve_kfold(canonical_loaders, canonical_data_seed: int, k: int,
                 subsample_size: int | None
                 ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, DataLoader]]]:
    """k-fold CV: split the canonical training pool into k contiguous folds.

    Each fold yields a ``(train_pool_idx, val_idx, val_loader)`` triple.
    All k folds together cover the canonical training pool; ``id_test``
    and ``ood_test`` remain untouched.

    Returns ``(full_pool, folds)``.  ``full_pool`` is the complete
    permutation -- used for the *final retraining* at the BO-selected
    lambda (no held-out portion).
    """
    if k < 2:
        raise ValueError(f"--cv_folds must be >= 2 for the k-fold path; got {k}")
    full_pool = _shuffled_pool(canonical_loaders, canonical_data_seed,
                               subsample_size)
    n = len(full_pool)
    if n < k * 2:
        raise ValueError(
            f"training pool too small ({n}) for {k} folds with >=2 per fold"
        )
    base = canonical_loaders["train"].dataset
    train_loader = canonical_loaders["train"]

    fold_size = n // k
    folds: list[tuple[np.ndarray, np.ndarray, DataLoader]] = []
    for f in range(k):
        start = f * fold_size
        end = (f + 1) * fold_size if f < k - 1 else n
        val_idx = full_pool[start:end]
        train_idx = np.concatenate([full_pool[:start], full_pool[end:]])
        loader = _make_val_loader(base, val_idx, train_loader)
        folds.append((train_idx, val_idx, loader))
    return full_pool, folds


def _save_trials(path: Path, trials: list[dict], best: dict) -> None:
    path.write_text(json.dumps({"best": best, "trials": trials}, indent=2))


def _select_lambda(trials: list[dict], args) -> dict:
    """Pick the BO trial that becomes the final model.

    ``best_score`` returns the trial with the highest BO objective.

    ``rule_largest_lam`` mirrors the paper's pre-registered selection
    rule (Section~\\ref{sec:methods}): among trials whose val accuracy
    lies within ``--prereg_tolerance`` of the smallest-lambda trial's
    val accuracy, pick the largest lambda.  Feasibility is checked
    against ``metrics["acc"]`` directly so it stays correct under any
    ``--bo_objective`` (the BO score is a different scalar in
    ``churn_constrained`` mode).  Falls back to ``best_score`` when no
    trial passes the constraint.
    """
    if args.selection_rule == "rule_largest_lam":
        baseline_trial = min(trials, key=lambda t: t["lam"])
        baseline_acc = float(baseline_trial["metrics"]["acc"])
        tol = args.prereg_tolerance
        feasible = [t for t in trials
                    if float(t["metrics"]["acc"]) >= baseline_acc - tol]
        if feasible:
            return max(feasible, key=lambda t: t["lam"])
    return max(trials, key=lambda t: t["score"])


def _save_final(args, cfg, canonical_loaders, out_root: Path, device, m_a, m_b,
                best: dict, train_seed: int, epochs: int, n_train: int,
                n_train_full: int, data_hash: str, test_hash: str,
                val_hash: str | None = None, val_size: int | None = None,
                fold_val_hashes: list[str] | None = None,
                fold_val_sizes: list[int] | None = None) -> None:
    lam = best["lam"]
    if _is_regression(cfg):
        id_p_a, id_y, id_i = _predict_reg(m_a, canonical_loaders["id_test"], device)
        id_p_b, _, _ = _predict_reg(m_b, canonical_loaders["id_test"], device)
        ood_p_a, ood_y, ood_i = _predict_reg(m_a, canonical_loaders["ood_test"], device)
        ood_p_b, _, _ = _predict_reg(m_b, canonical_loaders["ood_test"], device)
        id_avg = 0.5 * (id_p_a + id_p_b)
        ood_avg = 0.5 * (ood_p_a + ood_p_b)
        payload = dict(
            id_preds_A=id_p_a, id_preds_B=id_p_b, id_preds_avg=id_avg,
            id_labels=id_y, id_indices=id_i,
            ood_preds_A=ood_p_a, ood_preds_B=ood_p_b, ood_preds_avg=ood_avg,
            ood_labels=ood_y, ood_indices=ood_i,
            canonical_data_seed=args.canonical_data_seed,
            train_seed=train_seed, lam=lam, mode="twin_indep_bayes",
            task="regression", bayes_best_score=best["score"],
        )
        extra = {"task": "regression"}
        print(f"  final id_mae={np.abs(id_avg-id_y).mean():.4f}  "
              f"ood_mae={np.abs(ood_avg-ood_y).mean():.4f}  lambda={lam:.6g}")
    else:
        id_p_a, id_y, id_i = _predict(m_a, canonical_loaders["id_test"], device)
        id_p_b, _, _ = _predict(m_b, canonical_loaders["id_test"], device)
        ood_p_a, ood_y, ood_i = _predict(m_a, canonical_loaders["ood_test"], device)
        ood_p_b, _, _ = _predict(m_b, canonical_loaders["ood_test"], device)
        id_avg = 0.5 * (id_p_a + id_p_b)
        ood_avg = 0.5 * (ood_p_a + ood_p_b)
        payload = dict(
            id_probs_A=id_p_a, id_probs_B=id_p_b, id_probs_avg=id_avg,
            id_labels=id_y, id_indices=id_i,
            ood_probs_A=ood_p_a, ood_probs_B=ood_p_b, ood_probs_avg=ood_avg,
            ood_labels=ood_y, ood_indices=ood_i,
            canonical_data_seed=args.canonical_data_seed,
            train_seed=train_seed, lam=lam, mode="twin_indep_bayes",
            bayes_best_score=best["score"],
        )
        extra = {}
        print(f"  final id_acc={(id_avg.argmax(1)==id_y).mean():.4f}  "
              f"ood_acc={(ood_avg.argmax(1)==ood_y).mean():.4f}  lambda={lam:.6g}")

    name = f"twin_indep_bayes_train{train_seed}_lam{lam:.6g}"
    npz_path = out_root / f"{name}.npz"
    np.savez_compressed(npz_path, **payload)
    config = {
        "dataset": args.dataset,
        "mode": "twin_indep_bayes",
        "canonical_data_seed": args.canonical_data_seed,
        "train_seed": int(train_seed),
        "epochs": epochs,
        "subsample_size": args.subsample_size,
        "n_train": int(n_train),
        "n_train_full": int(n_train_full),
        "bayes_trials": args.bayes_trials,
        "bayes_init_trials": args.bayes_init_trials,
        "bayes_seed": args.bayes_seed,
        "objective_split": args.objective_split,
        "cv_folds": int(args.cv_folds),
        "bo_objective": args.bo_objective,
        "churn_penalty": (args.churn_penalty
                          if args.bo_objective in ("churn_constrained",
                                                   "cross_sample_constrained")
                          else None),
        "shadow_seed_offset": (args.shadow_seed_offset
                               if args.bo_objective == "cross_sample_constrained"
                               else None),
        "selection_rule": args.selection_rule,
        "prereg_tolerance": (args.prereg_tolerance
                             if (args.selection_rule == "rule_largest_lam"
                                 or args.bo_objective in (
                                     "churn_constrained",
                                     "cross_sample_constrained"))
                             else None),
        "val_frac": (args.val_frac if args.objective_split == "val"
                     and args.cv_folds == 1 else None),
        "val_size": val_size,
        "val_hash": val_hash,
        "fold_val_sizes": fold_val_sizes,
        "fold_val_hashes": fold_val_hashes,
        "lam": lam,
        "lam_min": args.lam_min,
        "lam_max": args.lam_max,
        "best_score": best["score"],
        **extra,
    }
    ctx = _make_provenance(npz_path, config)
    ctx.data_hash = data_hash
    ctx.test_hash = test_hash
    ctx.write()


def run(args) -> None:
    if args.lam_min <= 0 or args.lam_max <= args.lam_min:
        raise ValueError("--lam_min must be positive and less than --lam_max")
    if args.bayes_trials < 1:
        raise ValueError("--bayes_trials must be at least 1")
    if args.cv_folds > 1 and args.objective_split != "val":
        raise ValueError(
            "--cv_folds > 1 only makes sense with --objective_split val "
            "(BO objective is the cross-validation mean over folds)"
        )
    if args.bo_objective == "churn_constrained" and args.lam_min > 0.001:
        # The first trial is forced to lambda=0; after that the GP samples
        # in log10([lam_min, lam_max]).  A high lam_min would fence BO out
        # of the lightly-regularised regime where acc is close to baseline.
        print(f"  [warn] --bo_objective churn_constrained with --lam_min="
              f"{args.lam_min:g}; consider lam_min<=1e-3 so BO can probe "
              f"near the baseline.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.dataset)
    cfg.training.seed = args.canonical_data_seed
    set_seed(args.canonical_data_seed)
    canonical_loaders = make_dataloaders(cfg)
    epochs = args.epochs if args.epochs is not None else HPARAMS[args.dataset]["epochs"]
    n_train_full = len(canonical_loaders["train"].dataset)

    # Three selection paths: hold-out val, k-fold CV, or test-set oracle.
    folds: list[tuple[np.ndarray, np.ndarray, DataLoader]] | None = None
    objective_loader = None
    val_hash: str | None = None
    val_size: int | None = None
    fold_val_hashes: list[str] | None = None
    fold_val_sizes: list[int] | None = None

    if args.objective_split == "val" and args.cv_folds > 1:
        full_pool, folds = _carve_kfold(
            canonical_loaders, args.canonical_data_seed,
            args.cv_folds, args.subsample_size,
        )
        # Final retraining at the BO-selected lambda uses the full pool.
        pool_idx = full_pool
        n_train = len(full_pool)
        data_hash = _hash_idx(full_pool)
        fold_val_hashes = [_hash_idx(v) for _, v, _ in folds]
        fold_val_sizes = [int(len(v)) for _, v, _ in folds]
        print(f"k={args.cv_folds}-fold CV: pool n={n_train} -> folds of size "
              f"{fold_val_sizes}; id_test/ood_test untouched.")
    elif args.objective_split == "val":
        pool_idx, val_idx, val_loader = _carve_val(
            canonical_loaders, args.canonical_data_seed,
            args.val_frac, args.subsample_size,
        )
        n_train = len(pool_idx)
        val_size = int(len(val_idx))
        val_hash = _hash_idx(val_idx)
        data_hash = _hash_idx(pool_idx)
        objective_loader = val_loader
        print(f"Held-out val: n={val_size} carved from canonical pool "
              f"(seed={args.canonical_data_seed}, frac={args.val_frac}); "
              f"bootstrap pool n={n_train}.")
    elif args.subsample_size is not None and args.subsample_size < n_train_full:
        pool_rng = np.random.default_rng(args.canonical_data_seed)
        pool_idx = pool_rng.permutation(n_train_full)[: args.subsample_size]
        n_train = args.subsample_size
        data_hash = _hash_idx(pool_idx)
        objective_loader = canonical_loaders[args.objective_split]
        print(f"Subsample: training pool capped at M={n_train} "
              f"(canonical set has {n_train_full} examples).")
    else:
        pool_idx = None
        n_train = n_train_full
        data_hash = _idx_hash(canonical_loaders["train"])
        objective_loader = canonical_loaders[args.objective_split]

    # Provenance: canonical id_test indices identify what we *report* on.
    test_hash = _idx_hash(canonical_loaders["id_test"])

    print(f"Canonical test set: id n={len(canonical_loaders['id_test'].dataset)}, "
          f"ood n={len(canonical_loaders['ood_test'].dataset)}")
    print(f"Bayesian objective: split={args.objective_split}, "
          f"cv_folds={args.cv_folds}, "
          f"lambda in [{args.lam_min:g}, {args.lam_max:g}]")

    initial_lams = _parse_lams(args.initial_lams)
    # In any *_constrained mode the first trial must be the unregularised
    # baseline so we have a reference accuracy for the constraint.
    # ``--include_zero_trial`` opts in to the same lambda=0-first trial
    # for the unconstrained ``acc`` objective.  In both cases, only
    # insert if lambda=0 isn't already the first initial trial.
    needs_baseline = args.bo_objective in ("churn_constrained",
                                           "cross_sample_constrained")
    want_zero_first = needs_baseline or args.include_zero_trial
    if want_zero_first and (not initial_lams or initial_lams[0] != 0.0):
        initial_lams.insert(0, 0.0)

    # Shadow-seed offset is only meaningful for cross_sample_constrained;
    # forwarded to ``_trial_metrics`` to enable the second-twin training.
    shadow_offset = (args.shadow_seed_offset
                     if args.bo_objective == "cross_sample_constrained"
                     else None)

    for train_seed in (int(s) for s in args.train_seeds.split(",")):
        print(f"\n=== {args.dataset}  canonical={args.canonical_data_seed}  "
              f"train_seed={train_seed}  mode=twin_indep_bayes ===")
        trials: list[dict] = []
        trial_path = out_root / f"twin_indep_bayes_train{train_seed}_trials.json"
        baseline_acc: float | None = None  # set after the lam=0 trial

        for trial_idx in range(args.bayes_trials):
            lam = (initial_lams[trial_idx] if trial_idx < len(initial_lams)
                   else _candidate_lam(trials, args))
            metrics = _trial_metrics(
                cfg, canonical_loaders, device, train_seed, epochs, lam,
                folds=folds, objective_loader=objective_loader,
                pool_idx=pool_idx, n_train=n_train,
                shadow_seed_offset=shadow_offset,
            )
            score = _bo_score(metrics, args, baseline_acc)
            if needs_baseline and baseline_acc is None and lam == 0.0:
                baseline_acc = metrics["acc"]

            trials.append({
                "trial": trial_idx,
                "lam": float(lam),
                "score": float(score),
                "metrics": metrics,
            })
            best = max(trials, key=lambda t: t["score"])
            _save_trials(trial_path, trials, best)
            extra = (f" cross_churn={metrics['cross_churn']:.4f}"
                     if "cross_churn" in metrics else "")
            print(f"  trial={trial_idx:02d} lambda={lam:.6g} "
                  f"acc={metrics['acc']:.4f} churn={metrics['churn']:.4f}"
                  f"{extra} score={score:.4f} best={best['lam']:.6g}")

        best = _select_lambda(trials, args)
        print(f"  selected lambda={best['lam']:.6g} "
              f"score={best['score']:.4f} (rule={args.selection_rule}); "
              f"retraining final model")
        m_a, m_b = _train_twin_indep(
            cfg, canonical_loaders, device, train_seed, epochs, best["lam"],
            pool_idx=pool_idx, n_train=n_train,
        )
        _save_final(
            args, cfg, canonical_loaders, out_root, device, m_a, m_b, best,
            train_seed, epochs, n_train, n_train_full, data_hash, test_hash,
            val_hash=val_hash, val_size=val_size,
            fold_val_hashes=fold_val_hashes, fold_val_sizes=fold_val_sizes,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--canonical_data_seed", type=int, default=99,
                    help="Seed for canonical (test-set-fixed) dataset construction.")
    ap.add_argument("--train_seeds", default="1,2,3,4,5",
                    help="Comma-separated list of train_seeds.")
    ap.add_argument("--bayes_trials", type=int, default=12,
                    help="Number of lambda trials per train seed.")
    ap.add_argument("--bayes_init_trials", type=int, default=4,
                    help="Random positive-lambda trials before GP acquisition.")
    ap.add_argument("--bayes_seed", type=int, default=0,
                    help="Seed for lambda proposal randomness.")
    ap.add_argument("--bayes_grid_size", type=int, default=256,
                    help="Candidate grid size for expected improvement.")
    ap.add_argument("--bayes_xi", type=float, default=0.01,
                    help="Expected-improvement exploration offset.")
    ap.add_argument("--lam_min", type=float, default=1e-3)
    ap.add_argument("--lam_max", type=float, default=1e2)
    ap.add_argument("--initial_lams", default=None,
                    help="Optional comma-separated lambda values to try first.")
    ap.add_argument("--include_zero_trial", action="store_true",
                    help="Try lambda=0 before positive-lambda Bayesian search.")
    ap.add_argument("--objective_split", choices=["val", "id_test", "ood_test"],
                    default="val",
                    help="Split used to score Bayesian-optimization trials. "
                         "'val' carves a held-out fraction of the canonical "
                         "training pool (rigorous default). 'id_test' / "
                         "'ood_test' deliberately use the test set as the BO "
                         "oracle (upper-bound comparison only).")
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of the canonical training pool held out as "
                         "the BO objective when --objective_split=val and "
                         "--cv_folds=1.  Ignored otherwise.")
    ap.add_argument("--bo_objective",
                    choices=["acc", "churn_constrained",
                             "cross_sample_constrained"],
                    default="acc",
                    help="What BO maximises per trial.  'acc' (default): "
                         "ensemble val accuracy.  'churn_constrained': "
                         "(-val_churn) -- inter-network disagreement, a "
                         "lower-bound proxy for cross-sample churn.  "
                         "'cross_sample_constrained': (-val_cross_churn) -- "
                         "argmax disagreement between two ensembles trained "
                         "at different train_seeds, a *direct* val-side "
                         "analogue of the cross-sample churn the paper "
                         "reports on test (~2x compute per trial).  Both "
                         "constrained modes force the first trial to "
                         "lambda=0 to set the accuracy baseline.")
    ap.add_argument("--churn_penalty", type=float, default=100.0,
                    help="Slope of the accuracy-deficit penalty in "
                         "--bo_objective {churn,cross_sample}_constrained.  "
                         "100 -> a 0.01 acc shortfall costs 1.0 of score, "
                         "dwarfing the feasible-region range [-1, 0] of "
                         "the negated churn proxy.")
    ap.add_argument("--shadow_seed_offset", type=int, default=1000,
                    help="Offset added to train_seed for the shadow twin in "
                         "--bo_objective cross_sample_constrained.  E.g. "
                         "with the default, train_seed=1 pairs with the "
                         "shadow seed 1001.  Ignored for other objectives.")
    ap.add_argument("--selection_rule",
                    choices=["best_score", "rule_largest_lam"],
                    default="best_score",
                    help="How to pick the final lambda from the trial pool. "
                         "'best_score' (default): trial with the highest BO "
                         "objective.  'rule_largest_lam': the paper's "
                         "pre-registered rule applied per-dataset -- among "
                         "trials whose val acc is within --prereg_tolerance "
                         "of the smallest-lambda trial's val acc, pick the "
                         "largest lambda.  Use with --include_zero_trial so "
                         "the smallest-lambda trial is the unregularised "
                         "baseline.")
    ap.add_argument("--prereg_tolerance", type=float, default=0.02,
                    help="Accuracy tolerance for --selection_rule "
                         "rule_largest_lam.  Matches the rule used in the "
                         "paper's main protocol.")
    ap.add_argument("--cv_folds", type=int, default=1,
                    help="If >=2, replace the hold-out val with k-fold CV "
                         "over the canonical training pool: each BO trial "
                         "trains k twin-network pairs and averages val score "
                         "across the k folds.  Final retraining at the "
                         "selected lambda uses the FULL pool (no held-out "
                         "portion).  Cost is k x the hold-out path, so prefer "
                         "k=3 unless you specifically need k=5.  Only valid "
                         "with --objective_split=val.")
    ap.add_argument("--subsample_size", type=int, default=None,
                    help="If set, cap the canonical training pool at this many examples.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from HPARAMS.")
    ap.add_argument("--output_dir", default="outputs/cross_sample_bayes_val")
    run(ap.parse_args())
