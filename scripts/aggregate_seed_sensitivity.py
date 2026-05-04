"""Aggregate per-seed analysis CSVs into seed-averaged tables.

The paper's headline magnitudes (BACE ERM churn, twin-bootstrap %
reduction, etc.) were originally computed at a single canonical
data-seed (`canonical_data_seed=99`).  A canonical-seed sensitivity
sweep replicates the entire main-table protocol at two additional
canonical seeds (7, 42), writing NPZs to `outputs/cross_sample_seed7/`
and `outputs/cross_sample_seed42/` next to the original
`outputs/cross_sample/` (= seed 99).

This script:
  1. For each analysis script that takes a `--root` and writes a CSV,
     runs it once per canonical-seed root, producing per-seed CSVs at
     `outputs/<basename>_seed{seed}.csv`.
  2. Aggregates each per-seed CSV family into `outputs/<basename>.csv`
     by averaging numeric columns across seeds (grouped on dataset /
     method / scope / key, depending on the schema), and adds
     `_seed_{lo,hi,std}` columns with the across-seed range and
     standard deviation.
  3. Writes a long-format `outputs/<basename>_per_seed.csv` for each
     family for the appendix sensitivity table.

Downstream consumers (`make_paper_macros.py`, `make_fig1_forest.py`)
keep reading the original CSV paths -- they now contain the cross-seed
mean rather than a single seed's value, plus optional `_seed_*` columns
for sensitivity-aware code.

Usage::
    uv run python scripts/aggregate_seed_sensitivity.py
"""
from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

OUT = Path("outputs")
SEED_ROOTS: dict[int, Path] = {
    99: OUT / "cross_sample",
    7:  OUT / "cross_sample_seed7",
    42: OUT / "cross_sample_seed42",
}


# ---------------------------------------------------------------------------
# Job table.  One per analysis script that consumes per-seed NPZs and
# emits a CSV.  `key_cols` are the natural per-row identifiers (string-
# valued, copied from the seed-99 row); every other column is treated
# as numeric and averaged across seeds.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    name: str             # used to name per-seed CSVs (outputs/<name>_seed{seed}.csv)
    script: str           # path to the analysis script
    csv_basename: str     # final aggregated CSV under outputs/
    key_cols: tuple[str, ...]


JOBS: list[Job] = [
    Job("main_table",            "scripts/make_main_table.py",
        "main_table.csv",            ("dataset", "method", "n_seeds")),
    Job("fragility_magnitudes",  "scripts/make_fragility_magnitudes_table.py",
        "fragility_magnitudes.csv",  ("dataset", "group", "n_seeds")),
    Job("distributional",        "scripts/make_distributional_table.py",
        "distributional.csv",        ("dataset", "method")),
    Job("per_class_churn",       "scripts/make_per_class_churn_table.py",
        "per_class_churn.csv",       ("dataset", "minority")),
    Job("additional_metrics",    "scripts/make_additional_metrics_table.py",
        "additional_metrics.csv",    ("dataset",)),
    Job("regression",            "scripts/make_regression_table.py",
        "regression.csv",            ("dataset", "method")),
    Job("convergence_recall",    "scripts/make_fig_convergence.py",
        "convergence_recall.csv",    ("dataset", "K")),
    Job("entropy_vs_fragility",  "scripts/make_entropy_vs_fragility.py",
        "entropy_vs_fragility.csv",  ("dataset",)),
    Job("friedman",              "scripts/make_friedman_test.py",
        "friedman.csv",              ("scope", "key")),
]


# ---------------------------------------------------------------------------

def _read(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def _f(s: str | None) -> float | None:
    if s in (None, ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _per_seed_csv(name: str, seed: int) -> Path:
    return OUT / f"{name}_seed{seed}.csv"


def _per_seed_latex(name: str, seed: int) -> Path:
    """A discardable per-seed latex output path (so the script's --latex
    arg has somewhere to write; downstream we only use the canonical
    paper/sections/tables/<>.tex regenerated from the aggregated CSV)."""
    return OUT / "_per_seed_latex" / f"{name}_seed{seed}.tex"


def run_per_seed(job: Job, seed: int, root: Path) -> Path | None:
    """Invoke the analysis script for a single canonical seed.

    Args common to most scripts: --root, --csv, --latex.  A few scripts
    (make_friedman_test.py, make_fig_convergence.py) use different flag
    names; we pass the right form per script.
    """
    csv_path = _per_seed_csv(job.name, seed)
    latex_path = _per_seed_latex(job.name, seed)
    latex_path.parent.mkdir(parents=True, exist_ok=True)

    if job.name == "friedman":
        # make_friedman_test.py writes outputs/friedman.csv directly,
        # has no --csv/--latex.  Run it with --root and then move the
        # output to the per-seed slot.
        subprocess.check_call([
            "uv", "run", "python", job.script, "--root", str(root),
        ])
        produced = OUT / "friedman.csv"
        if produced.exists():
            produced.replace(csv_path)
        else:
            return None
    elif job.name == "convergence_recall":
        # --out_csv / --out_pdf
        out_pdf = OUT / "_per_seed_latex" / f"convergence_seed{seed}.pdf"
        subprocess.check_call([
            "uv", "run", "python", job.script,
            "--root", str(root),
            "--out_csv", str(csv_path),
            "--out_pdf", str(out_pdf),
        ])
    elif job.name == "entropy_vs_fragility":
        subprocess.check_call([
            "uv", "run", "python", job.script,
            "--root", str(root),
            "--csv", str(csv_path),
            "--latex", str(latex_path),
        ])
    else:
        subprocess.check_call([
            "uv", "run", "python", job.script,
            "--root", str(root),
            "--csv", str(csv_path),
            "--latex", str(latex_path),
        ])
    return csv_path if csv_path.exists() else None


def aggregate_csv(name: str, key_cols: tuple[str, ...],
                  per_seed_csvs: dict[int, Path]) -> tuple[list[dict], list[dict]]:
    """Combine per-seed CSVs into (aggregated, long-form-per-seed) row lists.

    aggregated: one row per unique (key_cols ...) tuple.  Numeric columns
    carry the across-seed mean; new `<col>_seed_{mean,std,lo,hi}` columns
    record the across-seed range.

    long-form-per-seed: one row per (canonical_seed, key_cols ...) for the
    appendix sensitivity table.
    """
    # Read all rows, keyed by canonical seed.
    by_seed = {s: _read(p) for s, p in per_seed_csvs.items()}

    # Collect all key tuples.
    cells: dict[tuple, dict[int, dict]] = {}
    all_cols: set[str] = set()
    for s, rows in by_seed.items():
        for r in rows:
            key = tuple(r.get(k, "") for k in key_cols)
            cells.setdefault(key, {})[s] = r
            all_cols.update(r.keys())

    # Numeric columns are everything not in key_cols.
    num_cols = sorted(c for c in all_cols if c not in key_cols)

    agg_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    for key, by_s in sorted(cells.items()):
        # Drop cells absent from at least one seed.
        if len(by_s) < len(per_seed_csvs):
            continue
        # Use the seed-99 row (or first available) as the template.
        template = by_s[99] if 99 in by_s else next(iter(by_s.values()))
        out = {k: template.get(k, "") for k in key_cols}
        for col in num_cols:
            vals = [_f(by_s[s].get(col)) for s in by_s]
            vals = [v for v in vals if v is not None]
            if not vals:
                # Try to copy a string value from the template.
                if template.get(col, "") != "":
                    out[col] = template[col]
                continue
            out[col] = mean(vals)
            out[f"{col}_seed_mean"] = mean(vals)
            out[f"{col}_seed_lo"] = min(vals)
            out[f"{col}_seed_hi"] = max(vals)
            out[f"{col}_seed_std"] = stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(out)
        for s, r in sorted(by_s.items()):
            row = {"canonical_seed": s, **{k: r.get(k, "") for k in key_cols}}
            for col in num_cols:
                v = _f(r.get(col))
                if v is not None:
                    row[col] = v
                elif r.get(col, "") != "":
                    row[col] = r[col]
            per_seed_rows.append(row)
    return agg_rows, per_seed_rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  [skip] no rows for {path}")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {path}  ({len(rows)} rows)")


def main() -> None:
    available = {s: r for s, r in SEED_ROOTS.items() if r.exists()}
    if 99 not in available:
        raise SystemExit("outputs/cross_sample/ (canonical seed 99) is required.")
    print(f"Aggregating across canonical seeds: {sorted(available)}")
    if len(available) < 2:
        print("[warn] only canonical_seed=99 found; aggregation degenerates "
              "to single-seed.")

    for job in JOBS:
        print(f"\n--- {job.name} ---")
        per_seed_csvs: dict[int, Path] = {}
        for seed, root in sorted(available.items()):
            p = run_per_seed(job, seed, root)
            if p is None:
                print(f"  [warn] seed={seed}: no CSV produced for {job.name}")
                continue
            per_seed_csvs[seed] = p
        if not per_seed_csvs:
            print(f"  [skip] no per-seed CSVs for {job.name}")
            continue
        agg, per_seed = aggregate_csv(job.name, job.key_cols, per_seed_csvs)
        write_csv(agg, OUT / job.csv_basename)
        write_csv(per_seed,
                  OUT / job.csv_basename.replace(".csv", "_per_seed.csv"))


if __name__ == "__main__":
    main()
