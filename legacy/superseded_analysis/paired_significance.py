"""Paired bootstrap significance tests for Twin_indep − Bagging vs ERM.

Per dataset, treats each (train_seed_A, train_seed_B) pair's churn delta as
an observation and bootstraps 95% CIs on the *difference* between methods.
Paired comparison: at each pair, all three methods see the same two
training samples (same canonical_data_seed=99 test set, same train_seeds).
This removes seed variance from the comparison.

Output: per-dataset table showing paired Δ with bootstrap 95% CI.
A method is significantly better than ERM if the CI excludes 0.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


def _bootstrap_ci_paired(deltas, n_boot=10_000, alpha=0.05):
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(deltas.mean()), float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


def _load_pairs(d, glob, key="id"):
    """Return dict: (seed_a, seed_b) → churn_value (argmax disagreement)."""
    files = sorted(d.glob(glob),
                   key=lambda p: int(p.stem.split("train")[1].split("_")[0]))
    if not files:
        return {}
    runs = {int(f.stem.split("train")[1].split("_")[0]):
            dict(np.load(f, allow_pickle=True)) for f in files}
    pairs = {}
    for sa, sb in combinations(sorted(runs), 2):
        a = runs[sa].get(f"{key}_probs_avg", runs[sa].get(f"{key}_probs"))
        b = runs[sb].get(f"{key}_probs_avg", runs[sb].get(f"{key}_probs"))
        pairs[(sa, sb)] = float((a.argmax(1) != b.argmax(1)).mean())
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_sample")
    args = ap.parse_args()
    root = Path(args.root)
    methods = [
        ("ERM", "erm_train*.npz"),
        ("Bagging K=2", "bagging_train*_K2.npz"),
        ("Bagging K=5", "bagging_train*_K5.npz"),
        ("Twin_indep λ=300", "twin_indep_train*_lam300.0.npz"),
    ]
    datasets = ["bbbp", "tadf", "mof_solvent", "mof_thermal"]

    print("=== Paired bootstrap CIs on (method − ERM) churn delta ===")
    print("Negative Δ means method has lower churn than ERM (= better stability).")
    print("CI excludes 0 ⇒ statistically significant difference.\n")

    for ds in datasets:
        for split in ["id", "ood"]:
            erm_pairs = _load_pairs(root / ds, "erm_train*.npz", key=split)
            print(f"-- {ds:12s} ({split.upper()}) --")
            for tag, glob in methods[1:]:
                m_pairs = _load_pairs(root / ds, glob, key=split)
                # Paired Δ
                common = set(erm_pairs) & set(m_pairs)
                deltas = [m_pairs[p] - erm_pairs[p] for p in common]
                mean, lo, hi = _bootstrap_ci_paired(deltas)
                sig = "  **" if (lo < 0 and hi < 0) else "    "
                print(f"   {tag:18s}  Δ_churn={mean:+.4f}  "
                      f"[{lo:+.4f}, {hi:+.4f}]{sig}  "
                      f"(n={len(deltas)} pairs)")
            print()


if __name__ == "__main__":
    main()
