"""Check the BACE λ-selection rule on every canonical seed.

The pre-registered rule: pick the largest λ in PARETO_LAMS such that
twin-bootstrap BACE id-accuracy is within `tolerance` of ERM id-accuracy.
We verify the rule selects the same λ on each replicated canonical seed
({99, 7, 42}) so the methods comparison rests on a stable choice.

Reads:  outputs/cross_sample{,_seed7,_seed42}/bace/{erm,twin_indep_lam*}.npz
Writes: outputs/pareto_rule_per_seed.csv
        stdout: per-seed picked λ
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from _analysis_lib import GLOBS, load_runs, per_run_accuracies
from paper_constants import FROZEN_LAM, PARETO_LAMS

OUT = Path("outputs")
TOL = 0.02
ROOTS = {
    99: OUT / "cross_sample",
    7:  OUT / "cross_sample_seed7",
    42: OUT / "cross_sample_seed42",
}


def _mean_id_acc(ds_dir: Path, glob: str) -> float | None:
    runs = load_runs(ds_dir, glob)
    if not runs:
        return None
    accs, _ = per_run_accuracies(runs)
    return float(np.mean(accs)) if accs else None


def main() -> None:
    rows = []
    print(f"{'seed':>5}  {'erm_acc':>10}  {'tolerance':>10}  "
          f"{'picked λ':>10}  {'  per-λ id_acc'}")
    for seed in sorted(ROOTS):
        root = ROOTS[seed] / "bace"
        if not root.exists():
            print(f"{seed:>5}  (missing {root})")
            continue
        erm_acc = _mean_id_acc(root, GLOBS["erm"])
        if erm_acc is None:
            print(f"{seed:>5}  (no ERM runs)")
            continue
        thresh = erm_acc - TOL
        per_lam = []
        for lam in sorted(PARETO_LAMS):
            twin_acc = _mean_id_acc(root, GLOBS["twin_indep"].format(lam=lam))
            per_lam.append((lam, twin_acc))
        # Largest λ within tolerance.
        feasible = [lam for lam, a in per_lam
                    if a is not None and a >= thresh]
        picked = max(feasible) if feasible else None
        rows.append({
            "canonical_seed": seed,
            "erm_id_acc": erm_acc,
            "tolerance_threshold": thresh,
            "picked_lambda": picked,
            **{f"twin_acc_lam{lam}": a for lam, a in per_lam},
        })
        per_lam_str = "  ".join(
            f"{lam:>4.0f}={(a if a is not None else float('nan')):.3f}"
            for lam, a in per_lam
        )
        print(f"{seed:>5}  {erm_acc:>10.3f}  {thresh:>10.3f}  "
              f"{(picked if picked is not None else '---'):>10}  "
              f"{per_lam_str}")

    # CSV dump.
    out_csv = OUT / "pareto_rule_per_seed.csv"
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\nWrote {out_csv}")

    picks = sorted({int(r["picked_lambda"]) for r in rows
                    if r["picked_lambda"] is not None})
    if len(picks) == 1 and picks[0] == int(FROZEN_LAM):
        print(f"\n[ok] Rule picks λ={picks[0]} on every canonical seed.")
    else:
        print(f"\n[warn] Rule picks vary across canonical seeds: {picks}")


if __name__ == "__main__":
    main()
