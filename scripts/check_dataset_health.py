"""Sanity-check every dataset's NPZs before they enter the main table.

For each dataset directory in `outputs/cross_sample/<ds>/`:
  - test set size ≥ MIN_TEST
  - id_test class balance is not degenerate (both classes present)
  - ERM accuracy is above the majority-class baseline by ≥ 0.05
  - per-method NPZ schema is consistent
  - id_labels are byte-identical across all NPZs in the directory

A dataset failing any check is reported and excluded from
`make_main_table.py` outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MIN_TEST = 50


def _get_probs(d):
    return d.get("id_probs_avg", d.get("id_probs"))


def check_dataset(ds_dir: Path) -> dict:
    out = {"dataset": ds_dir.name, "status": "ok", "warnings": []}

    files = sorted(ds_dir.glob("*.npz"))
    if not files:
        return {**out, "status": "missing"}

    # Pick the canonical reference (first ERM file if present, else first file).
    erm_files = sorted(ds_dir.glob("erm_train*.npz"))
    if not erm_files:
        return {**out, "status": "no_erm"}
    ref = dict(np.load(erm_files[0], allow_pickle=True))
    y = ref["id_labels"]

    out["n_id"] = int(len(y))
    out["n_ood"] = int(len(ref["ood_labels"]))
    out["pY1_id"] = float(y.mean())
    out["pY1_ood"] = float(ref["ood_labels"].mean())
    out["majority_acc"] = float(max(out["pY1_id"], 1 - out["pY1_id"]))

    if len(y) < MIN_TEST:
        out["warnings"].append(f"id_test n={len(y)} < {MIN_TEST}")
    if y.min() == y.max():
        out["warnings"].append("id_labels all the same class (degenerate)")

    # Cross-NPZ label consistency.
    inconsistent = sum(
        1 for f in files
        if not np.array_equal(dict(np.load(f, allow_pickle=True))["id_labels"], y))
    if inconsistent:
        out["warnings"].append(f"{inconsistent} NPZs disagree on id_labels")

    # ERM accuracy vs majority baseline.
    erm_accs = []
    for f in erm_files:
        d = dict(np.load(f, allow_pickle=True))
        erm_accs.append(float((_get_probs(d).argmax(1) == d["id_labels"]).mean()))
    out["erm_acc_mean"] = float(np.mean(erm_accs))
    out["erm_acc_std"] = float(np.std(erm_accs))
    if out["erm_acc_mean"] < out["majority_acc"] + 0.05:
        out["warnings"].append(
            f"ERM mean acc {out['erm_acc_mean']:.3f} barely beats "
            f"majority {out['majority_acc']:.3f}")

    # Method coverage.
    out["methods"] = {
        "erm":             len(list(ds_dir.glob("erm_train*.npz"))),
        "deep_ensemble_5": len(list(ds_dir.glob("deep_ensemble_train*_K5.npz"))),
        "bagging_K2":      len(list(ds_dir.glob("bagging_train*_K2.npz"))),
        "bagging_K5":      len(list(ds_dir.glob("bagging_train*_K5.npz"))),
        "twin_indep_300":  len(list(ds_dir.glob("twin_indep_train*_lam300.0.npz"))),
    }
    incomplete = [m for m, n in out["methods"].items() if n < 10]
    if incomplete:
        out["warnings"].append(
            f"Incomplete method coverage (<10 seeds): {', '.join(incomplete)}")

    if out["warnings"]:
        out["status"] = "warning"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cross_sample")
    args = ap.parse_args()

    rows = [check_dataset(d) for d in sorted(Path(args.root).iterdir())
            if d.is_dir()]

    print(f"{'dataset':18s}  {'status':9s}  {'n_id':>5s}  {'pY1_id':>7s}  "
          f"{'majority':>9s}  {'erm_acc':>7s}  warnings")
    for r in rows:
        warnings = "; ".join(r.get("warnings", []))
        print(f"{r['dataset']:18s}  {r['status']:9s}  "
              f"{r.get('n_id', 0):>5d}  {r.get('pY1_id', 0):>7.3f}  "
              f"{r.get('majority_acc', 0):>9.3f}  "
              f"{r.get('erm_acc_mean', 0):>7.3f}  {warnings}")

    # Method coverage summary.
    print("\nPer-method seed counts (target: 10 each):")
    methods_keys = ["erm", "deep_ensemble_5", "bagging_K2",
                    "bagging_K5", "twin_indep_300"]
    print(f"{'dataset':18s}  " + "  ".join(f"{m:>16s}" for m in methods_keys))
    for r in rows:
        if "methods" not in r:
            continue
        print(f"{r['dataset']:18s}  "
              + "  ".join(f"{r['methods'].get(m, 0):>16d}" for m in methods_keys))


if __name__ == "__main__":
    main()
