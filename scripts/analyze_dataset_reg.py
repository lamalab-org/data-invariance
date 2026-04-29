"""Per-dataset paired-bootstrap analysis for regression.

Design decisions
----------------
1. **Fragility metric.**  For classification we report argmax churn
   (per-example argmax disagreement rate).  For regression there is no
   argmax, so we use the natural analogue: per-example absolute
   prediction difference between two retrainings,
   ``frag_reg(x) = E[|f_A(x) - f_B(x)|]``.  This is computable from the
   same NPZ files saved by ``cross_sample_train.py`` (predictions
   stored as ``id_preds`` for ERM / ``id_preds_avg`` for ensembles
   and twin-indep).

2. **Method comparison.**  Each method has its own glob pattern
   (see ``METHODS`` dict).  We deliberately include bagging-K=2 in
   the comparison so the matched-compute claim ($2\\times$-ERM
   compute, same as twin-indep) is computable from one script run.

3. **Paired bootstrap.**  All deltas are paired across the
   ``binom(10, 2) = 45`` seed-pair distribution that the canonical
   protocol generates.  ``bootstrap_paired`` is implemented in
   ``scripts/_analysis_lib.py``.

Usage: python scripts/analyze_dataset_reg.py <dataset>
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, bootstrap_paired, fmt_ci, load_runs

ROOT = Path("outputs/cross_sample")


def get_preds(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (id_preds, ood_preds), preferring averaged when present."""
    id_preds = d["id_preds_avg"] if "id_preds_avg" in d else d["id_preds"]
    ood_preds = d["ood_preds_avg"] if "ood_preds_avg" in d else d["ood_preds"]
    return id_preds, ood_preds


def per_run_mae(runs):
    id_maes, ood_maes = [], []
    for _, d in runs:
        idp, odp = get_preds(d)
        id_maes.append(float(np.abs(idp - d["id_labels"]).mean()))
        ood_maes.append(float(np.abs(odp - d["ood_labels"]).mean()))
    return id_maes, ood_maes


def pairwise_fragility(runs):
    """Per-pair cross-sample fragility (mean abs prediction diff) on id and ood.

    Returns: dict[(s_a, s_b)] -> {id_frag, ood_frag}, plus pair list.
    """
    pairs = list(combinations([s for s, _ in runs], 2))
    runs_by_seed = dict(runs)
    out = {}
    for sa, sb in pairs:
        idA, odA = get_preds(runs_by_seed[sa])
        idB, odB = get_preds(runs_by_seed[sb])
        out[(sa, sb)] = {
            "id_frag":  float(np.abs(idA - idB).mean()),
            "ood_frag": float(np.abs(odA - odB).mean()),
        }
    return out, pairs


METHODS = {
    "ERM":              "erm_train*.npz",
    "Bagging-K=2":      "bagging_train*_K2.npz",
    "Bagging-K=5":      "bagging_train*_K5.npz",
    "Twin-indep λ=1":   "twin_indep_train*_lam1.0.npz",
    "Twin-indep λ=3":   "twin_indep_train*_lam3.0.npz",
}


def main(dataset: str) -> None:
    data_dir = ROOT / dataset
    if not data_dir.exists():
        print(f"  no output dir: {data_dir}")
        sys.exit(1)
    runs = {name: load_runs(data_dir, glob) for name, glob in METHODS.items()}

    print(f"\n=== {dataset} ===")
    summary = {}
    for name, rs in runs.items():
        if len(rs) < 2:
            print(f"  {name:18s}  insufficient runs ({len(rs)})")
            continue
        id_maes, _ = per_run_mae(rs)
        pm, pairs = pairwise_fragility(rs)
        frag = [pm[p]["id_frag"] for p in pairs]
        summary[name] = {
            "id_mae": bootstrap_ci(id_maes),
            "id_frag": bootstrap_ci(frag),
            "frag_pairs": np.array(frag),
            "pairs": pairs,
        }
        print(f"  {name:18s}  id_mae {fmt_ci(summary[name]['id_mae'])}  "
              f"id_frag {fmt_ci(summary[name]['id_frag'])}")

    if "ERM" not in summary:
        return
    erm_pairs = summary["ERM"]["pairs"]
    erm_frag = summary["ERM"]["frag_pairs"]
    print("  Paired Δ vs ERM (same 45 seed-pairs):")
    for name in METHODS:
        if name == "ERM" or name not in summary:
            continue
        m_pm, _ = pairwise_fragility(runs[name])
        m_frag = np.array([m_pm[p]["id_frag"] for p in erm_pairs if p in m_pm])
        if len(m_frag) != len(erm_frag):
            print(f"    {name:18s}  pair mismatch")
            continue
        ci = bootstrap_paired(m_frag - erm_frag)
        rel = 100 * ci[0] / summary["ERM"]["id_frag"][0]
        print(f"    {name:18s}  Δ id_frag {fmt_ci(ci)} ({rel:+.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: analyze_dataset_reg.py <dataset>")
        sys.exit(1)
    main(sys.argv[1])
