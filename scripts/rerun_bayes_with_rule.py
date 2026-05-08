"""Re-pick lambda for each existing twin_indep_bayes trials.json using the
paper's rule (largest lambda within --tolerance of the smallest-lambda
trial's val accuracy), then retrain a single twin-indep model per
(dataset, train_seed) at the rule-selected lambda and write the final
NPZ + manifest into a separate output root.

Saves the 30-trial BO loop entirely -- the existing trials.json is the
trial pool we apply the rule to.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._provenance import make_context as _make_provenance  # noqa: E402
from scripts.cross_sample_train import (  # noqa: E402
    _is_regression,
    _predict,
    _predict_reg,
    _train_twin_indep,
)
from scripts.cross_sample_train_bayes import (  # noqa: E402
    _hash_idx,
    _idx_hash,
    _shuffled_pool,
)
from scripts.run_experiment import HPARAMS, build_cfg  # noqa: E402
from train import make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402


def _select_rule(trials: list[dict], tolerance: float) -> dict:
    baseline = min(trials, key=lambda t: t["lam"])
    feasible = [t for t in trials
                if float(t["score"]) >= float(baseline["score"]) - tolerance]
    return max(feasible, key=lambda t: t["lam"]) if feasible else max(
        trials, key=lambda t: t["score"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_root", required=True,
                    help="Existing root with trials.json files "
                         "(e.g. outputs/cross_sample_bayes_kfold3).")
    ap.add_argument("--output_root", required=True,
                    help="Where to write rule-selected NPZs.")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--canonical_data_seed", type=int, default=99)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--cv_folds", type=int, default=3,
                    help="Mirror the BO run's cv_folds so the final retrain "
                         "uses the same training pool.")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    for dataset in args.datasets:
        cfg = build_cfg(dataset)
        cfg.training.seed = args.canonical_data_seed
        set_seed(args.canonical_data_seed)
        canonical_loaders = make_dataloaders(cfg)
        epochs = (args.epochs if args.epochs is not None
                  else HPARAMS[dataset]["epochs"])
        n_train_full = len(canonical_loaders["train"].dataset)

        full_pool = _shuffled_pool(canonical_loaders, args.canonical_data_seed,
                                   subsample_size=None)
        n_train = len(full_pool)
        data_hash = _hash_idx(full_pool)
        # test_hash always identifies the canonical id_test split; the
        # selection-set hash is recorded separately in the trial JSON's
        # source_trials reference (which carries the cv_fold hashes).
        test_hash = _idx_hash(canonical_loaders["id_test"])

        out_dir = Path(args.output_root) / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        trial_files = sorted(
            (Path(args.source_root) / dataset).glob(
                "twin_indep_bayes_train*_trials.json"))
        print(f"\n=== {dataset}: {len(trial_files)} seeds in {args.source_root} ===")

        for tf in trial_files:
            train_seed = int(tf.name.split("train")[1].split("_")[0])
            d = json.loads(tf.read_text())
            trials = d["trials"]
            best = _select_rule(trials, args.tolerance)
            baseline = min(trials, key=lambda t: t["lam"])
            print(f"  seed={train_seed} baseline lam={baseline['lam']:.3g} "
                  f"score={baseline['score']:.4f}; "
                  f"rule selects lam={best['lam']:.6g} score={best['score']:.4f}")

            lam = float(best["lam"])
            m_a, m_b = _train_twin_indep(
                cfg, canonical_loaders, device, train_seed, epochs, lam,
                pool_idx=full_pool, n_train=n_train,
            )
            if _is_regression(cfg):
                id_p_a, id_y, id_i = _predict_reg(m_a, canonical_loaders["id_test"], device)
                id_p_b, _, _ = _predict_reg(m_b, canonical_loaders["id_test"], device)
                ood_p_a, ood_y, ood_i = _predict_reg(m_a, canonical_loaders["ood_test"], device)
                ood_p_b, _, _ = _predict_reg(m_b, canonical_loaders["ood_test"], device)
                payload = dict(
                    id_preds_A=id_p_a, id_preds_B=id_p_b,
                    id_preds_avg=0.5 * (id_p_a + id_p_b),
                    id_labels=id_y, id_indices=id_i,
                    ood_preds_A=ood_p_a, ood_preds_B=ood_p_b,
                    ood_preds_avg=0.5 * (ood_p_a + ood_p_b),
                    ood_labels=ood_y, ood_indices=ood_i,
                    canonical_data_seed=args.canonical_data_seed,
                    train_seed=train_seed, lam=lam, mode="twin_indep_bayes",
                    task="regression", bayes_best_score=best["score"],
                )
            else:
                id_p_a, id_y, id_i = _predict(m_a, canonical_loaders["id_test"], device)
                id_p_b, _, _ = _predict(m_b, canonical_loaders["id_test"], device)
                ood_p_a, ood_y, ood_i = _predict(m_a, canonical_loaders["ood_test"], device)
                ood_p_b, _, _ = _predict(m_b, canonical_loaders["ood_test"], device)
                payload = dict(
                    id_probs_A=id_p_a, id_probs_B=id_p_b,
                    id_probs_avg=0.5 * (id_p_a + id_p_b),
                    id_labels=id_y, id_indices=id_i,
                    ood_probs_A=ood_p_a, ood_probs_B=ood_p_b,
                    ood_probs_avg=0.5 * (ood_p_a + ood_p_b),
                    ood_labels=ood_y, ood_indices=ood_i,
                    canonical_data_seed=args.canonical_data_seed,
                    train_seed=train_seed, lam=lam, mode="twin_indep_bayes",
                    bayes_best_score=best["score"],
                )

            name = f"twin_indep_bayes_train{train_seed}_lam{lam:.6g}"
            npz_path = out_dir / f"{name}.npz"
            np.savez_compressed(npz_path, **payload)

            config = {
                "dataset": dataset,
                "mode": "twin_indep_bayes",
                "canonical_data_seed": args.canonical_data_seed,
                "train_seed": int(train_seed),
                "epochs": epochs,
                "n_train": int(n_train),
                "n_train_full": int(n_train_full),
                "selection_rule": "rule_largest_lam",
                "prereg_tolerance": args.tolerance,
                "cv_folds": args.cv_folds,
                "lam": lam,
                "best_score": best["score"],
                "baseline_lam": baseline["lam"],
                "baseline_score": baseline["score"],
                "source_trials": str(tf),
            }
            ctx = _make_provenance(npz_path, config)
            ctx.data_hash = data_hash
            ctx.test_hash = test_hash
            ctx.write()


if __name__ == "__main__":
    main()
