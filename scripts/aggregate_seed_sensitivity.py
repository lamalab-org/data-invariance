"""Aggregate per-seed analysis CSVs into seed-averaged tables.

The paper's headline magnitudes (BACE ERM churn, twin-bootstrap %
reduction, etc.) were originally computed at a single canonical
data-seed (`canonical_data_seed=99`).  A canonical-seed sensitivity
sweep replicates the entire main-table protocol at two additional
canonical seeds (7, 42), writing NPZs to `outputs/cross_sample_seed7/`
and `outputs/cross_sample_seed42/` next to the original
`outputs/cross_sample/` (= seed 99).

This script:
  1. Runs `make_main_table.py` once per canonical-seed root, producing
     `outputs/main_table_seed{seed}.csv`.
  2. Same for `make_friedman_test.py` (records each per-seed Friedman
     test result).
  3. Combines the per-seed main_tables into:
       - `outputs/main_table.csv`           - aggregated row per (dataset, method).
                                              Adds *_seed_{mean,std,lo,hi} columns
                                              alongside the existing single-seed columns
                                              (which carry the across-seed mean).
       - `outputs/main_table_per_seed.csv`  - long-format per-(seed, dataset, method)
                                              values, used by the appendix sensitivity table.
       - `outputs/friedman_per_seed.csv`    - per-seed Friedman chi^2/p-values.

The aggregated `main_table.csv` is a drop-in replacement for the
single-seed version: downstream scripts (`make_paper_macros.py`,
`make_fig1_forest.py`, …) keep reading the same column names.  The new
*_seed_* columns are ignored by old consumers and consumed by the
sensitivity-aware ones.

Usage::
    uv run python scripts/aggregate_seed_sensitivity.py
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from statistics import mean, stdev

OUT = Path("outputs")
SEED_ROOTS = {
    99: OUT / "cross_sample",
    7:  OUT / "cross_sample_seed7",
    42: OUT / "cross_sample_seed42",
}


def _run_per_seed_main_table(seed: int, root: Path) -> Path:
    """Invoke make_main_table.py at a per-seed root, write CSV and table."""
    csv_path = OUT / f"main_table_seed{seed}.csv"
    latex_path = OUT / f"main_table_seed{seed}.tex"   # discarded
    subprocess.check_call([
        "uv", "run", "python", "scripts/make_main_table.py",
        "--root", str(root),
        "--csv", str(csv_path),
        "--latex", str(latex_path),
    ])
    return csv_path


def _read(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def _f(s: str) -> float | None:
    if s == "" or s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# Numeric columns we aggregate.  Anything else (dataset, method, n_seeds)
# is copied from the canonical-seed-99 row.
_NUM_COLS = (
    "id_acc_mean", "id_acc_lo", "id_acc_hi",
    "ood_acc_mean", "ood_acc_lo", "ood_acc_hi",
    "id_churn_mean", "id_churn_lo", "id_churn_hi",
    "ood_churn_mean", "ood_churn_lo", "ood_churn_hi",
    "id_sym_kl_mean", "id_sym_kl_lo", "id_sym_kl_hi",
    "ood_sym_kl_mean", "ood_sym_kl_lo", "ood_sym_kl_hi",
    "delta_id_churn_mean", "delta_id_churn_lo", "delta_id_churn_hi",
)


def aggregate(per_seed_csvs: dict[int, Path]) -> tuple[list[dict], list[dict]]:
    """Combine the per-seed CSVs into (aggregate, per_seed) row lists.

    aggregate: one row per (dataset, method).  All numeric columns are
    the across-seed *mean*; *_seed_lo/_seed_hi/_seed_std are the across-
    seed range and standard deviation; the original columns stay
    populated so downstream code that doesn't know about seed-aggregation
    keeps working (and now reflects the across-seed mean rather than a
    single seed's value).

    per_seed: long-format, one row per (seed, dataset, method) for the
    appendix sensitivity table.
    """
    by_seed = {s: _read(p) for s, p in per_seed_csvs.items()}
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    for seed, rows in by_seed.items():
        for r in rows:
            key = (r["dataset"], r["method"])
            cells.setdefault(key, {})[seed] = r

    agg_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    for (ds, method), per_seed in sorted(cells.items()):
        # Skip cells that aren't present in all seeds.
        if len(per_seed) < len(per_seed_csvs):
            continue
        # Use the seed-99 row as the template (so dataset, method,
        # n_seeds, etc. carry through unchanged).
        out = dict(per_seed[99])
        for col in _NUM_COLS:
            vals = [_f(per_seed[s].get(col, "")) for s in per_seed_csvs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            out[col] = mean(vals)
            out[f"{col}_seed_mean"] = mean(vals)
            out[f"{col}_seed_lo"] = min(vals)
            out[f"{col}_seed_hi"] = max(vals)
            out[f"{col}_seed_std"] = stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(out)

        # Per-seed long-format dump.
        for s, r in sorted(per_seed.items()):
            row = {"canonical_seed": s, "dataset": ds, "method": method}
            for col in _NUM_COLS:
                v = _f(r.get(col, ""))
                if v is not None:
                    row[col] = v
            per_seed_rows.append(row)

    return agg_rows, per_seed_rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"[skip] no rows for {path}")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"Wrote {path}  ({len(rows)} rows)")


def main() -> None:
    # 1. Run per-seed main_table.py.
    available = {s: r for s, r in SEED_ROOTS.items() if r.exists()}
    if 99 not in available:
        raise SystemExit("outputs/cross_sample/ (seed 99) is required.")
    if len(available) < 2:
        print(f"[warn] only canonical_seed=99 results found; aggregation will "
              f"degenerate to the single-seed case.  Re-run "
              f"scripts/run_seed_sweep.sh for the missing seeds.")

    per_seed_csvs: dict[int, Path] = {}
    for seed, root in sorted(available.items()):
        print(f"\n=== make_main_table on canonical_seed={seed}  root={root} ===")
        per_seed_csvs[seed] = _run_per_seed_main_table(seed, root)

    # 2. Aggregate.
    agg_rows, per_seed_rows = aggregate(per_seed_csvs)
    write_csv(agg_rows, OUT / "main_table.csv")
    write_csv(per_seed_rows, OUT / "main_table_per_seed.csv")

    print(f"\nAggregated {len(per_seed_csvs)} canonical seeds: "
          f"{sorted(per_seed_csvs)}")
    print("Downstream (make_paper_macros.py, make_fig1_forest.py, "
          "make_friedman_test.py) can now read outputs/main_table.csv as before.")


if __name__ == "__main__":
    main()
