"""Aggregate per-seed BO selections into a per-dataset lambda.

Reads the NPZ outputs of ``cross_sample_train_bayes.py`` (one NPZ per
(dataset, train_seed) at the seed-level BO-selected lambda) and writes
a CSV of the per-dataset median lambda.  This CSV is the bridge
between *stage 1* of the per-dataset BO protocol (per-seed BO inside
``cross_sample_train_bayes.py``) and *stage 2* (fixed-lambda retraining
across all seeds, scheduled by
``slurm/full_retraining/11_bo_perdataset_lam.sh``).

Usage::

    uv run python scripts/select_perdataset_lambda.py \\
        --source outputs/cross_sample_bayes_cross \\
        --csv    outputs/bo_perdataset_lambdas.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


# Lambda is encoded in the NPZ filename: twin_indep_bayes_train{seed}_lam{X}.npz
_NPZ_RE = re.compile(r"twin_indep_bayes_train\d+_lam(?P<lam>[0-9eE.+-]+)\.npz$")


def _per_seed_lambdas(dataset_dir: Path) -> list[float]:
    """Return one lambda per train_seed in ``dataset_dir``."""
    lams: list[float] = []
    for f in sorted(dataset_dir.glob("twin_indep_bayes_train*_lam*.npz")):
        m = _NPZ_RE.search(f.name)
        if m is None:
            raise ValueError(f"could not parse lambda from {f.name}")
        lams.append(float(m.group("lam")))
    return lams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="Source root (e.g. outputs/cross_sample_bayes_cross). "
                         "Each subdirectory is a dataset.")
    ap.add_argument("--csv", required=True,
                    help="Output CSV with columns: dataset, median_lambda, "
                         "n_seeds, min, max.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Optional whitelist of dataset names.  Default: "
                         "every subdirectory of --source with at least one NPZ.")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"--source {source} is not a directory")

    rows: list[dict[str, object]] = []
    datasets = args.datasets or sorted(
        d.name for d in source.iterdir() if d.is_dir())
    for ds in datasets:
        lams = _per_seed_lambdas(source / ds)
        if not lams:
            print(f"[skip] {ds}: no NPZs in {source / ds}")
            continue
        rows.append({
            "dataset": ds,
            "median_lambda": f"{statistics.median(lams):.6g}",
            "n_seeds": len(lams),
            "min": f"{min(lams):.6g}",
            "max": f"{max(lams):.6g}",
        })
        print(f"  {ds:>18s}: median={rows[-1]['median_lambda']:>10s} "
              f"(n={len(lams)}, range [{rows[-1]['min']}, {rows[-1]['max']}])")

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "median_lambda",
                                          "n_seeds", "min", "max"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
