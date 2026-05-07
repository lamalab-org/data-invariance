"""Bayesian-optimization driver for ``_train_twin_indep``.

This is a sibling entry point to ``scripts/cross_sample_train.py``.  It keeps
the same cross-sample data protocol, but chooses the twin-indep consistency
weight ``lambda`` with a small Gaussian-process Bayesian optimizer before
saving the final predictions.

Selection-vs-reporting hygiene
------------------------------
The default ``--objective_split val`` carves a deterministic
``--val_frac`` fraction off the canonical training pool *before*
bootstrapping (seed = ``--canonical_data_seed``).  Bootstraps are drawn
from the remaining ``train`` indices only; ``id_test`` and ``ood_test``
are never seen during optimisation.  This is the rigorous setting for
the preprint and revision.

``--objective_split id_test`` and ``ood_test`` remain available for
comparison runs that deliberately use the test set as the BO oracle
(an upper bound on what tuning could achieve, not a fair held-out
result).

The optimizer depends only on scikit-learn, which is already a project
dependency.  It searches in log10(lambda) and maximizes validation score:

* classification: ensemble accuracy on the chosen objective split
* regression: negative ensemble MAE on the chosen objective split
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


def _score_twin(cfg, m_a, m_b, loader, device) -> tuple[float, dict]:
    if _is_regression(cfg):
        p_a, y, _ = _predict_reg(m_a, loader, device)
        p_b, _, _ = _predict_reg(m_b, loader, device)
        avg = 0.5 * (p_a + p_b)
        mae = float(np.abs(avg - y).mean())
        return -mae, {"mae": mae}

    p_a, y, _ = _predict(m_a, loader, device)
    p_b, _, _ = _predict(m_b, loader, device)
    avg = 0.5 * (p_a + p_b)
    acc = float((avg.argmax(1) == y).mean())
    return acc, {"acc": acc}


def _idx_hash(loader) -> str:
    ds = loader.dataset
    idxs = getattr(ds, "indices", None)
    if idxs is None:
        idxs = list(range(len(ds)))
    return hashlib.sha256(np.asarray(idxs, dtype=np.int64).tobytes()).hexdigest()[:16]


def _carve_val(canonical_loaders, canonical_data_seed: int, val_frac: float,
               subsample_size: int | None
               ) -> tuple[np.ndarray, np.ndarray, DataLoader]:
    """Split the canonical training pool into (train_pool, val).

    The split is deterministic in ``canonical_data_seed`` and lives entirely
    inside the canonical training set: ``id_test`` and ``ood_test`` are
    untouched.  When ``subsample_size`` is set, the subsample is drawn
    *first* and val is then carved from that prefix, so the subsample-vs-no
    subsample comparison stays apples-to-apples on the val fraction.

    Returns ``(pool_idx, val_idx, val_loader)``.  ``pool_idx`` is the new
    bootstrap pool that ``_train_twin_indep`` consumes; ``val_loader``
    iterates the held-out indices in input order (no shuffle, no aug
    differences from the canonical train transforms).
    """
    train_loader = canonical_loaders["train"]
    base = train_loader.dataset
    n_train_full = len(base)

    shuffled = np.random.default_rng(canonical_data_seed).permutation(n_train_full)
    if subsample_size is not None and subsample_size < n_train_full:
        shuffled = shuffled[:subsample_size]

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

    val_loader = DataLoader(
        Subset(base, val_idx.tolist()),
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        collate_fn=train_loader.collate_fn,
    )
    return pool_idx, val_idx, val_loader


def _save_trials(path: Path, trials: list[dict], best: dict) -> None:
    path.write_text(json.dumps({"best": best, "trials": trials}, indent=2))


def _save_final(args, cfg, canonical_loaders, out_root: Path, device, m_a, m_b,
                best: dict, train_seed: int, epochs: int, n_train: int,
                n_train_full: int, data_hash: str, test_hash: str,
                val_hash: str | None = None, val_size: int | None = None) -> None:
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
        "val_frac": args.val_frac if args.objective_split == "val" else None,
        "val_size": val_size,
        "val_hash": val_hash,
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

    val_loader = None
    val_hash: str | None = None
    val_size: int | None = None
    if args.objective_split == "val":
        pool_idx, val_idx, val_loader = _carve_val(
            canonical_loaders, args.canonical_data_seed,
            args.val_frac, args.subsample_size,
        )
        n_train = len(pool_idx)
        val_size = int(len(val_idx))
        val_hash = hashlib.sha256(
            np.asarray(val_idx, dtype=np.int64).tobytes()).hexdigest()[:16]
        data_hash = hashlib.sha256(
            np.asarray(pool_idx, dtype=np.int64).tobytes()).hexdigest()[:16]
        print(f"Held-out val: n={val_size} carved from canonical pool "
              f"(seed={args.canonical_data_seed}, frac={args.val_frac}); "
              f"bootstrap pool n={n_train}.")
    elif args.subsample_size is not None and args.subsample_size < n_train_full:
        pool_rng = np.random.default_rng(args.canonical_data_seed)
        pool_idx = pool_rng.permutation(n_train_full)[: args.subsample_size]
        n_train = args.subsample_size
        data_hash = hashlib.sha256(
            np.asarray(pool_idx, dtype=np.int64).tobytes()).hexdigest()[:16]
        print(f"Subsample: training pool capped at M={n_train} "
              f"(canonical set has {n_train_full} examples).")
    else:
        pool_idx = None
        n_train = n_train_full
        data_hash = _idx_hash(canonical_loaders["train"])

    if args.objective_split == "val":
        objective_loader = val_loader
        objective_hash = val_hash
    else:
        objective_loader = canonical_loaders[args.objective_split]
        objective_hash = _idx_hash(objective_loader)
    test_hash = objective_hash

    print(f"Canonical test set: id n={len(canonical_loaders['id_test'].dataset)}, "
          f"ood n={len(canonical_loaders['ood_test'].dataset)}")
    print(f"Bayesian objective: split={args.objective_split}, "
          f"lambda in [{args.lam_min:g}, {args.lam_max:g}]")

    initial_lams = _parse_lams(args.initial_lams)
    if args.include_zero_trial:
        initial_lams.insert(0, 0.0)

    for train_seed in (int(s) for s in args.train_seeds.split(",")):
        print(f"\n=== {args.dataset}  canonical={args.canonical_data_seed}  "
              f"train_seed={train_seed}  mode=twin_indep_bayes ===")
        trials: list[dict] = []
        trial_path = out_root / f"twin_indep_bayes_train{train_seed}_trials.json"

        for trial_idx in range(args.bayes_trials):
            if trial_idx < len(initial_lams):
                lam = initial_lams[trial_idx]
            else:
                lam = _candidate_lam(trials, args)

            m_a, m_b = _train_twin_indep(
                cfg, canonical_loaders, device, train_seed, epochs, lam,
                pool_idx=pool_idx, n_train=n_train,
            )
            score, metrics = _score_twin(
                cfg, m_a, m_b, objective_loader, device)
            record = {
                "trial": trial_idx,
                "lam": float(lam),
                "score": float(score),
                "metrics": metrics,
            }
            trials.append(record)
            best = max(trials, key=lambda t: t["score"])
            _save_trials(trial_path, trials, best)
            metric_text = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  trial={trial_idx:02d} lambda={lam:.6g} "
                  f"score={score:.4f} {metric_text} best={best['lam']:.6g}")
            del m_a, m_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best = max(trials, key=lambda t: t["score"])
        print(f"  selected lambda={best['lam']:.6g} "
              f"score={best['score']:.4f}; retraining final model")
        m_a, m_b = _train_twin_indep(
            cfg, canonical_loaders, device, train_seed, epochs, best["lam"],
            pool_idx=pool_idx, n_train=n_train,
        )
        _save_final(
            args, cfg, canonical_loaders, out_root, device, m_a, m_b, best,
            train_seed, epochs, n_train, n_train_full, data_hash, test_hash,
            val_hash=val_hash, val_size=val_size,
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
                         "the BO objective when --objective_split=val.  "
                         "Ignored otherwise.")
    ap.add_argument("--subsample_size", type=int, default=None,
                    help="If set, cap the canonical training pool at this many examples.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from HPARAMS.")
    ap.add_argument("--output_dir", default="outputs/cross_sample_bayes_val")
    run(ap.parse_args())
