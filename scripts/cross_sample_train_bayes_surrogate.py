"""Bayesian-optimization surrogate-stability driver.

This is the experiment Kevin suggested, and deliberately does *not* optimize
the twin consistency weight.  Instead, it runs repeated Bayesian-optimization
campaigns over the canonical training pool and compares which samples are
picked, and how quickly the campaign finds high-response samples, under two
surrogate choices:

``default_gpr``
    A standard Gaussian-process regressor with expected improvement.

``stabilized_gpr``
    A twin-bootstrap GPR surrogate.  Two GPRs are fit on independent bootstrap
    resamples of the observed samples, their predictions are averaged, and the
    acquisition is penalized when the twin surrogates disagree.  The
    ``--stabilization_lam`` value is fixed for the run; it is not chosen by
    Bayesian optimization.

For every campaign/method pair, the script:

1. runs BO over the canonical training pool to select a sample subset;
2. trains a twin-bootstrap model on that BO-selected subset with fixed
   ``--lam``;
3. saves cross-sample-compatible NPZ predictions on canonical ID/OOD tests;
4. writes a JSON trace with the selected samples, surrogate predictions, and
   final model metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _safe_scale(y: np.ndarray) -> float:
    scale = float(np.std(y))
    return scale if scale >= 1e-12 else 1.0


def _diversity_bonus(x: np.ndarray, observed: np.ndarray,
                     candidates: np.ndarray) -> np.ndarray:
    """Normalized distance-to-nearest-observed bonus for candidate coverage."""
    if len(observed) == 0 or len(candidates) == 0:
        return np.zeros(len(candidates), dtype=np.float64)
    diff = x[candidates, None, :] - x[observed][None, :, :]
    min_dist = np.sqrt(np.square(diff).sum(axis=2)).min(axis=1)
    scale = float(np.median(min_dist[min_dist > 0.0])) if np.any(min_dist > 0.0) else 1.0
    return min_dist / max(scale, 1e-12)


def _parse_methods(raw: str) -> list[str]:
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    allowed = {"default_gpr", "stabilized_gpr"}
    bad = sorted(set(methods) - allowed)
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {sorted(allowed)}")
    return methods


def _load_training_context(args):
    from scripts.run_experiment import HPARAMS, build_cfg
    from train import make_dataloaders
    from utils import set_seed

    cfg = build_cfg(args.dataset)
    cfg.training.seed = args.canonical_data_seed
    set_seed(args.canonical_data_seed)
    loaders = make_dataloaders(cfg)
    train_ds = loaders["train"].dataset
    epochs = args.epochs if args.epochs is not None else HPARAMS[args.dataset]["epochs"]

    if not hasattr(train_ds, "images") or not hasattr(train_ds, "labels"):
        raise ValueError(
            f"{args.dataset} does not expose tabular dataset.images/labels. "
            "Use a Morgan/descriptors dataset, not chemberta/gin/resnet."
        )

    x = train_ds.images.detach().cpu().numpy().astype(np.float64)
    y = train_ds.labels.detach().cpu().numpy().astype(np.float64)
    indices = np.arange(len(y), dtype=np.int64)

    if args.subsample_size is not None and args.subsample_size < len(y):
        rng = np.random.default_rng(args.canonical_data_seed)
        keep = rng.permutation(len(y))[:args.subsample_size]
        x, y, indices = x[keep], y[keep], indices[keep]

    data_hash = hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()[:16]
    return cfg, loaders, epochs, x, y, indices, data_hash


def _preprocess_features(x: np.ndarray, args) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This surrogate BO driver needs scikit-learn. Install project "
            "dependencies or run through the project environment."
        ) from exc

    if args.pca_dim and args.pca_dim > 0 and x.shape[1] > args.pca_dim:
        n_components = min(args.pca_dim, x.shape[0], x.shape[1])
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=args.bayes_seed),
        )
    else:
        pipe = make_pipeline(StandardScaler())
    return pipe.fit_transform(x).astype(np.float64)


def _make_gp(seed: int, args):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=args.gpr_noise, noise_level_bounds=(1e-9, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=args.gpr_restarts,
        alpha=args.gpr_alpha,
    )


def _default_acquisition(x: np.ndarray, y: np.ndarray, observed: np.ndarray,
                         candidates: np.ndarray, seed: int, args) -> dict[str, np.ndarray]:
    gp = _make_gp(seed, args)
    gp.fit(x[observed], y[observed])
    mu, sigma = gp.predict(x[candidates], return_std=True)
    ei = _expected_improvement(mu, sigma, best=float(y[observed].max()), xi=args.bayes_xi)
    diversity = _diversity_bonus(x, observed, candidates)
    acquisition = ei + args.diversity_lam * _safe_scale(y[observed]) * diversity
    return {
        "acquisition": acquisition,
        "ei": ei,
        "mu": mu,
        "sigma": sigma,
        "diversity": diversity,
    }


def _stabilized_acquisition(x: np.ndarray, y: np.ndarray, observed: np.ndarray,
                            candidates: np.ndarray, seed: int, args) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    boot_a = rng.choice(observed, size=len(observed), replace=True)
    boot_b = rng.choice(observed, size=len(observed), replace=True)

    gp_a = _make_gp(seed * 100 + 1, args)
    gp_b = _make_gp(seed * 100 + 2, args)
    gp_a.fit(x[boot_a], y[boot_a])
    gp_b.fit(x[boot_b], y[boot_b])

    mu_a, sigma_a = gp_a.predict(x[candidates], return_std=True)
    mu_b, sigma_b = gp_b.predict(x[candidates], return_std=True)
    mu = 0.5 * (mu_a + mu_b)
    sigma = 0.5 * (sigma_a + sigma_b)
    ei = _expected_improvement(mu, sigma, best=float(y[observed].max()), xi=args.bayes_xi)

    disagreement = np.abs(mu_a - mu_b)
    scale = _safe_scale(y[observed])
    diversity = _diversity_bonus(x, observed, candidates)
    acquisition = (
        ei
        - args.stabilization_lam * disagreement
        + args.diversity_lam * scale * diversity
    )
    return {
        "acquisition": acquisition,
        "ei": ei,
        "mu": mu,
        "sigma": sigma,
        "mu_a": mu_a,
        "sigma_a": sigma_a,
        "mu_b": mu_b,
        "sigma_b": sigma_b,
        "disagreement": disagreement,
        "disagreement_scaled": disagreement / scale,
        "diversity": diversity,
    }


def _selected_prediction(acq: dict[str, np.ndarray], pos: int) -> dict[str, float]:
    return {key: float(values[pos]) for key, values in acq.items()}


def _campaign(x: np.ndarray, y_score: np.ndarray, y_raw: np.ndarray,
              pool_indices: np.ndarray,
              method: str, campaign_seed: int, args) -> dict:
    rng = np.random.default_rng(campaign_seed)
    n = len(y_score)
    initial = rng.choice(n, size=args.initial_samples, replace=False)
    observed = list(int(i) for i in initial)
    selected = list(observed)
    acquisition_values: list[float | None] = [None] * len(selected)
    surrogate_predictions: list[dict[str, float] | None] = [None] * len(selected)

    for step in range(args.bo_steps):
        observed_arr = np.asarray(observed, dtype=np.int64)
        observed_mask = np.zeros(n, dtype=bool)
        observed_mask[observed_arr] = True
        candidates = np.flatnonzero(~observed_mask)
        if len(candidates) == 0:
            break

        if method == "default_gpr":
            acq = _default_acquisition(
                x, y_score, observed_arr, candidates, campaign_seed * 1000 + step, args)
        elif method == "stabilized_gpr":
            acq = _stabilized_acquisition(
                x, y_score, observed_arr, candidates, campaign_seed * 1000 + step, args)
        else:
            raise ValueError(f"unknown method: {method}")

        best_pos = int(np.argmax(acq["acquisition"]))
        next_idx = int(candidates[best_pos])
        observed.append(next_idx)
        selected.append(next_idx)
        acquisition_values.append(float(acq["acquisition"][best_pos]))
        surrogate_predictions.append(_selected_prediction(acq, best_pos))

    selected_arr = np.asarray(selected, dtype=np.int64)
    selected_y = y_raw[selected_arr]
    selected_score = y_score[selected_arr]
    best_score_so_far = np.maximum.accumulate(selected_score)
    if args.objective_direction == "max":
        best_so_far = np.maximum.accumulate(selected_y)
        global_best_raw = float(y_raw.max())
    else:
        best_so_far = np.minimum.accumulate(selected_y)
        global_best_raw = float(y_raw.min())
    global_best_score = float(y_score.max())
    hit = np.flatnonzero(selected_score >= global_best_score)
    hit_step = int(hit[0]) if len(hit) else None
    return {
        "method": method,
        "campaign_seed": int(campaign_seed),
        "initial_samples": int(args.initial_samples),
        "bo_steps": int(args.bo_steps),
        "selected_pool_positions": selected,
        "selected_indices": pool_indices[selected_arr].astype(int).tolist(),
        "selected_y": selected_y.astype(float).tolist(),
        "selected_score": selected_score.astype(float).tolist(),
        "best_so_far": best_so_far.astype(float).tolist(),
        "best_score_so_far": best_score_so_far.astype(float).tolist(),
        "acquisition_values": acquisition_values,
        "surrogate_predictions": surrogate_predictions,
        "final_best": float(best_so_far[-1]),
        "global_best": global_best_raw,
        "global_best_score": global_best_score,
        "hit_global_best_step": hit_step,
    }


def _summarize(records: Iterable[dict]) -> dict:
    by_method: dict[str, list[dict]] = {}
    for record in records:
        by_method.setdefault(record["method"], []).append(record)

    summary = {}
    for method, rows in by_method.items():
        curves = np.asarray([r["best_so_far"] for r in rows], dtype=np.float64)
        hit_steps = [r["hit_global_best_step"] for r in rows
                     if r["hit_global_best_step"] is not None]
        summary[method] = {
            "n_campaigns": len(rows),
            "mean_final_best": float(np.mean([r["final_best"] for r in rows])),
            "mean_best_so_far": curves.mean(axis=0).astype(float).tolist(),
            "hit_global_best_rate": float(len(hit_steps) / len(rows)),
            "mean_hit_global_best_step": (
                float(np.mean(hit_steps)) if hit_steps else None
            ),
        }
    return summary


def _sym_kl_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    log_p = np.log(p + eps)
    log_q = np.log(q + eps)
    return 0.5 * ((p * (log_p - log_q)).sum(-1) + (q * (log_q - log_p)).sum(-1))


def _model_summary(records: Iterable[dict]) -> dict:
    by_method: dict[str, list[tuple[int, dict]]] = {}
    for record in records:
        metrics = record.get("model_metrics") or {}
        path = metrics.get("npz_path")
        if not path:
            continue
        d = dict(np.load(path, allow_pickle=True))
        if "id_probs_avg" not in d:
            continue
        by_method.setdefault(record["method"], []).append((int(d["train_seed"]), d))

    out = {}
    for method, runs in by_method.items():
        runs = sorted(runs, key=lambda x: x[0])
        id_acc = []
        ood_acc = []
        for _, d in runs:
            id_acc.append(float((d["id_probs_avg"].argmax(1) == d["id_labels"]).mean()))
            ood_acc.append(float((d["ood_probs_avg"].argmax(1) == d["ood_labels"]).mean()))

        pair_rows = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                seed_a, da = runs[i]
                seed_b, db = runs[j]
                pair_rows.append({
                    "pair": [seed_a, seed_b],
                    "id_churn": float(
                        (da["id_probs_avg"].argmax(1) != db["id_probs_avg"].argmax(1)).mean()),
                    "ood_churn": float(
                        (da["ood_probs_avg"].argmax(1) != db["ood_probs_avg"].argmax(1)).mean()),
                    "id_sym_kl": float(_sym_kl_np(
                        da["id_probs_avg"], db["id_probs_avg"]).mean()),
                    "ood_sym_kl": float(_sym_kl_np(
                        da["ood_probs_avg"], db["ood_probs_avg"]).mean()),
                })

        out[method] = {
            "n_runs": len(runs),
            "n_pairs": len(pair_rows),
            "id_acc_mean": float(np.mean(id_acc)) if id_acc else None,
            "ood_acc_mean": float(np.mean(ood_acc)) if ood_acc else None,
            "id_churn_mean": float(np.mean([r["id_churn"] for r in pair_rows]))
            if pair_rows else None,
            "ood_churn_mean": float(np.mean([r["ood_churn"] for r in pair_rows]))
            if pair_rows else None,
            "id_sym_kl_mean": float(np.mean([r["id_sym_kl"] for r in pair_rows]))
            if pair_rows else None,
            "ood_sym_kl_mean": float(np.mean([r["ood_sym_kl"] for r in pair_rows]))
            if pair_rows else None,
            "pairs": pair_rows,
        }
    return out


def _idx_hash(loader) -> str:
    ds = loader.dataset
    idxs = getattr(ds, "indices", None)
    if idxs is None:
        idxs = list(range(len(ds)))
    return hashlib.sha256(np.asarray(idxs, dtype=np.int64).tobytes()).hexdigest()[:16]


def _model_seed_for_campaign(args, campaign_seed: int) -> int:
    if args.final_model_seed_mode == "fixed":
        return int(args.final_model_seed)
    if args.final_model_seed_mode == "offset":
        return int(args.final_model_seed + campaign_seed)
    return int(campaign_seed)


def _train_and_save_final(args, cfg, canonical_loaders, out_root: Path, device,
                          record: dict, method: str, train_seed: int,
                          model_seed: int,
                          epochs: int, n_train_full: int,
                          data_hash: str, test_hash: str) -> dict[str, float | str]:
    from scripts._provenance import make_context as _make_provenance
    from scripts.cross_sample_train import (
        _is_regression,
        _predict,
        _predict_reg,
        _train_twin_indep,
    )

    selected_indices = np.asarray(record["selected_indices"], dtype=np.int64)
    n_train = len(selected_indices)
    m_a, m_b = _train_twin_indep(
        cfg, canonical_loaders, device, model_seed, epochs, args.lam,
        pool_idx=selected_indices, n_train=n_train,
    )
    mode = f"twin_indep_bayes_surrogate_{method}"

    if _is_regression(cfg):
        id_p_a, id_y, id_i = _predict_reg(m_a, canonical_loaders["id_test"], device)
        id_p_b, _, _ = _predict_reg(m_b, canonical_loaders["id_test"], device)
        ood_p_a, ood_y, ood_i = _predict_reg(m_a, canonical_loaders["ood_test"], device)
        ood_p_b, _, _ = _predict_reg(m_b, canonical_loaders["ood_test"], device)
        id_avg = 0.5 * (id_p_a + id_p_b)
        ood_avg = 0.5 * (ood_p_a + ood_p_b)
        id_metric = float(np.abs(id_avg - id_y).mean())
        ood_metric = float(np.abs(ood_avg - ood_y).mean())
        payload = dict(
            id_preds_A=id_p_a, id_preds_B=id_p_b, id_preds_avg=id_avg,
            id_labels=id_y, id_indices=id_i,
            ood_preds_A=ood_p_a, ood_preds_B=ood_p_b, ood_preds_avg=ood_avg,
            ood_labels=ood_y, ood_indices=ood_i,
            canonical_data_seed=args.canonical_data_seed,
            train_seed=train_seed, model_seed=model_seed, lam=args.lam, mode=mode,
            task="regression", bayes_surrogate_method=method,
            bayes_surrogate_final_best=record["final_best"],
            bayes_surrogate_hit_step=-1 if record["hit_global_best_step"] is None
            else record["hit_global_best_step"],
            bayes_surrogate_selected_indices=selected_indices,
        )
        metrics: dict[str, float | str] = {
            "id_mae": id_metric,
            "ood_mae": ood_metric,
        }
        extra = {"task": "regression"}
        print(f"    final method={method} train_seed={train_seed} model_seed={model_seed} "
              f"id_mae={id_metric:.4f} ood_mae={ood_metric:.4f} lambda={args.lam:g}")
    else:
        id_p_a, id_y, id_i = _predict(m_a, canonical_loaders["id_test"], device)
        id_p_b, _, _ = _predict(m_b, canonical_loaders["id_test"], device)
        ood_p_a, ood_y, ood_i = _predict(m_a, canonical_loaders["ood_test"], device)
        ood_p_b, _, _ = _predict(m_b, canonical_loaders["ood_test"], device)
        id_avg = 0.5 * (id_p_a + id_p_b)
        ood_avg = 0.5 * (ood_p_a + ood_p_b)
        id_metric = float((id_avg.argmax(1) == id_y).mean())
        ood_metric = float((ood_avg.argmax(1) == ood_y).mean())
        payload = dict(
            id_probs_A=id_p_a, id_probs_B=id_p_b, id_probs_avg=id_avg,
            id_labels=id_y, id_indices=id_i,
            ood_probs_A=ood_p_a, ood_probs_B=ood_p_b, ood_probs_avg=ood_avg,
            ood_labels=ood_y, ood_indices=ood_i,
            canonical_data_seed=args.canonical_data_seed,
            train_seed=train_seed, model_seed=model_seed, lam=args.lam, mode=mode,
            bayes_surrogate_method=method,
            bayes_surrogate_final_best=record["final_best"],
            bayes_surrogate_hit_step=-1 if record["hit_global_best_step"] is None
            else record["hit_global_best_step"],
            bayes_surrogate_selected_indices=selected_indices,
        )
        metrics = {"id_acc": id_metric, "ood_acc": ood_metric}
        extra = {}
        print(f"    final method={method} train_seed={train_seed} model_seed={model_seed} "
              f"id_acc={id_metric:.4f} ood_acc={ood_metric:.4f} lambda={args.lam:g}")

    name = f"twin_indep_bayes_surrogate_{method}_train{train_seed}_lam{args.lam:g}"
    npz_path = out_root / f"{name}.npz"
    np.savez_compressed(npz_path, **payload)
    selected_hash = hashlib.sha256(selected_indices.astype(np.int64).tobytes()).hexdigest()[:16]
    config = {
        "dataset": args.dataset,
        "mode": mode,
        "canonical_data_seed": args.canonical_data_seed,
        "train_seed": int(train_seed),
        "model_seed": int(model_seed),
        "epochs": epochs,
        "subsample_size": args.subsample_size,
        "n_train": int(n_train),
        "n_train_full": int(n_train_full),
        "lam": args.lam,
        "bayes_seed": args.bayes_seed,
        "campaign_seed": record["campaign_seed"],
        "campaign_index": record["campaign_index"],
        "surrogate_method": method,
        "initial_samples": args.initial_samples,
        "bo_steps": args.bo_steps,
        "objective_direction": args.objective_direction,
        "stabilization_lam": args.stabilization_lam,
        "diversity_lam": args.diversity_lam,
        "selected_pool_hash": selected_hash,
        **extra,
    }
    ctx = _make_provenance(npz_path, config)
    ctx.data_hash = data_hash
    ctx.test_hash = test_hash
    ctx.write()
    metrics["npz_path"] = str(npz_path)
    metrics["selected_pool_hash"] = selected_hash
    return metrics


def run(args) -> None:
    import torch

    if args.initial_samples < 2:
        raise ValueError("--initial_samples must be at least 2 for GP fitting")
    if args.bo_steps < 1:
        raise ValueError("--bo_steps must be at least 1")
    if args.campaigns < 1:
        raise ValueError("--campaigns must be at least 1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    cfg, canonical_loaders, epochs, x_raw, y_raw, pool_indices, data_hash = (
        _load_training_context(args)
    )
    n_train_full = len(canonical_loaders["train"].dataset)
    test_hash = _idx_hash(canonical_loaders["id_test"])
    y_score = y_raw if args.objective_direction == "max" else -y_raw
    if args.initial_samples + args.bo_steps > len(y_raw):
        raise ValueError("initial_samples + bo_steps exceeds candidate-pool size")

    x = _preprocess_features(x_raw, args)
    methods = _parse_methods(args.methods)

    print(f"Candidate pool: dataset={args.dataset} n={len(y_raw)} d={x.shape[1]} "
          f"direction={args.objective_direction} data_hash={data_hash}")
    print(f"Methods: {', '.join(methods)}; initial={args.initial_samples} "
          f"bo_steps={args.bo_steps} campaigns={args.campaigns} lambda={args.lam:g}")

    records = []
    for campaign_idx in range(args.campaigns):
        campaign_seed = args.bayes_seed * max(args.campaigns, 1) + campaign_idx
        for method in methods:
            record = _campaign(
                x, y_score, y_raw, pool_indices, method, campaign_seed, args)
            record["campaign_index"] = campaign_idx
            records.append(record)
            print(f"  campaign={campaign_idx:03d} method={method} "
                  f"final_best={record['final_best']:.4g} "
                  f"hit_step={record['hit_global_best_step']}")
            train_seed = campaign_seed
            model_seed = _model_seed_for_campaign(args, campaign_seed)
            record["model_metrics"] = _train_and_save_final(
                args, cfg, canonical_loaders, out_root, device, record, method,
                train_seed, model_seed, epochs, n_train_full, data_hash, test_hash,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "config": {
            "dataset": args.dataset,
            "canonical_data_seed": args.canonical_data_seed,
            "bayes_seed": args.bayes_seed,
            "campaigns": args.campaigns,
            "initial_samples": args.initial_samples,
            "bo_steps": args.bo_steps,
            "methods": methods,
            "objective_direction": args.objective_direction,
            "bayes_xi": args.bayes_xi,
            "stabilization_lam": args.stabilization_lam,
            "diversity_lam": args.diversity_lam,
            "final_model_seed_mode": args.final_model_seed_mode,
            "final_model_seed": args.final_model_seed,
            "gpr_alpha": args.gpr_alpha,
            "gpr_noise": args.gpr_noise,
            "gpr_restarts": args.gpr_restarts,
            "pca_dim": args.pca_dim,
            "subsample_size": args.subsample_size,
            "epochs": epochs,
            "lam": args.lam,
            "data_hash": data_hash,
        },
        "summary": _summarize(records),
        "model_summary": _model_summary(records),
        "campaigns": records,
    }
    out_path = out_root / (
        f"surrogate_bo_seed{args.bayes_seed}_"
        f"campaigns{args.campaigns}_steps{args.bo_steps}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))

    from scripts._provenance import make_context as _make_provenance
    ctx = _make_provenance(out_path, payload["config"])
    ctx.data_hash = data_hash
    ctx.test_hash = test_hash
    ctx.write()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--canonical_data_seed", type=int, default=99,
                    help="Seed for canonical candidate-pool construction.")
    ap.add_argument("--bayes_seed", type=int, default=0,
                    help="Base seed for repeated BO campaigns.")
    ap.add_argument("--campaigns", type=int, default=20,
                    help="Number of repeated BO campaigns per method.")
    ap.add_argument("--initial_samples", type=int, default=5,
                    help="Random initial samples shared by both methods per campaign.")
    ap.add_argument("--bo_steps", type=int, default=25,
                    help="Sequential BO picks after the initial samples.")
    ap.add_argument("--lam", type=float, default=300.0,
                    help="Fixed twin-indep consistency weight for final model training.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from HPARAMS for final model training.")
    ap.add_argument("--methods", default="default_gpr,stabilized_gpr",
                    help="Comma-separated subset of default_gpr,stabilized_gpr.")
    ap.add_argument("--objective_direction", choices=["max", "min"], default="max",
                    help="Whether BO should maximize or minimize the dataset label/target.")
    ap.add_argument("--bayes_xi", type=float, default=0.01,
                    help="Expected-improvement exploration offset.")
    ap.add_argument("--stabilization_lam", type=float, default=0.1,
                    help="Fixed penalty on twin-surrogate disagreement.")
    ap.add_argument("--diversity_lam", type=float, default=0.0,
                    help=(
                        "Optional distance-to-observed coverage bonus in acquisition "
                        "units. Positive values discourage tiny, clustered BO subsets."
                    ))
    ap.add_argument("--final_model_seed_mode",
                    choices=["campaign", "fixed", "offset"], default="campaign",
                    help=(
                        "How to seed final neural/twin training. 'campaign' preserves "
                        "the original behavior; 'fixed' holds training randomness "
                        "constant across BO campaigns; 'offset' uses final_model_seed "
                        "+ campaign_seed."
                    ))
    ap.add_argument("--final_model_seed", type=int, default=0,
                    help="Seed used by --final_model_seed_mode fixed/offset.")
    ap.add_argument("--gpr_alpha", type=float, default=1e-8,
                    help="Numerical diagonal jitter passed to GaussianProcessRegressor.")
    ap.add_argument("--gpr_noise", type=float, default=1e-5,
                    help="Initial WhiteKernel noise level.")
    ap.add_argument("--gpr_restarts", type=int, default=2,
                    help="GPR hyperparameter optimizer restarts.")
    ap.add_argument("--pca_dim", type=int, default=32,
                    help="PCA dimension for high-dimensional tabular fingerprints; <=0 disables PCA.")
    ap.add_argument("--subsample_size", type=int, default=None,
                    help="If set, cap the canonical candidate pool at this many examples.")
    ap.add_argument("--output_dir", default="outputs/cross_sample_bayes_surrogate")
    run(ap.parse_args())
