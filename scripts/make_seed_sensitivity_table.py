"""Per-canonical-seed values for every (dataset, method) cell in the
main table.

Reads:  outputs/main_table_per_seed.csv  (long-format produced by
        scripts/aggregate_seed_sensitivity.py)
Writes: paper/sections/tables/seed_sensitivity.tex

Two columns per (dataset, method) cell: id-churn rate (%) and paired
Δ id-churn vs. ERM (pp), each repeated once per canonical seed.  The
table substantiates the across-seed range numbers that the appendix
prose quotes via macros.
"""
from __future__ import annotations

import csv
from pathlib import Path

from paper_constants import DEV_DATASET, FROZEN_LAM, HEADLINE_DATASETS, display


CSV = Path("outputs/main_table_per_seed.csv")
OUT = Path("paper/sections/tables/seed_sensitivity.tex")


METHOD_DISPLAY = {
    "ERM":              "ERM",
    "SWA":              "SWA",
    "MC dropout":       "MC dropout",
    "Deep Ensemble K=5": r"Deep Ens.\ $K{=}5$",
    "Bagging K=2":      "Bag $K{=}2$",
    "Bagging K=5":      "Bag $K{=}5$",
    f"Twin_indep λ={FROZEN_LAM} (frozen)":
                        r"Twin $\lambda{=}300$",
}
METHOD_ORDER = list(METHOD_DISPLAY.keys())


def _f(s: str) -> float | None:
    if s in (None, ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    if not CSV.exists():
        print(f"[skip] {CSV} not found.  Run "
              "scripts/aggregate_seed_sensitivity.py first.")
        return

    with CSV.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[skip] {CSV} is empty.")
        return

    seeds = sorted({int(r["canonical_seed"]) for r in rows})
    cells: dict[tuple[str, str], dict[int, dict[str, float | None]]] = {}
    for r in rows:
        s = int(r["canonical_seed"])
        cells.setdefault((r["dataset"], r["method"]), {})[s] = {
            "churn":  _f(r.get("id_churn_mean")),
            "delta":  _f(r.get("delta_id_churn_mean")),
        }

    datasets = [DEV_DATASET] + list(HEADLINE_DATASETS)

    # ----- Build LaTeX (longtable so the 60+ rows can break across pages) -----
    seed_header = (
        r"    & & \multicolumn{3}{c}{Canonical seed} \\"
        + "\n"
        + r"    \cmidrule(lr){3-5}"
        + "\n"
        + r"    Dataset & Method & "
        + " & ".join(str(s) for s in seeds)
        + r" \\"
    )
    lines = [
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{ll" + "rrr" * 1 + "}",
        r"  \caption{\textbf{Per-canonical-seed values for the main "
        r"table.}  Each method-dataset cell is reported on three "
        r"independent canonical splits ($99$, $7$, $42$).  Top number "
        r"per cell: id-churn rate (\%); bottom: paired $\Delta$ "
        r"id-churn vs.\ ERM (pp).  Both columns are aggregated over "
        r"the $\nSeedPairs$ paired-bootstrap seed pairs at fixed "
        r"canonical seed.  ERM rows have no paired-$\Delta$ entry.}"
        r"  \label{tab:seed_sensitivity} \\",
        r"    \toprule",
        seed_header,
        r"    \midrule",
        r"    \endfirsthead",
        r"    \multicolumn{5}{l}{\emph{Table~\ref{tab:seed_sensitivity} continued from previous page}} \\",
        r"    \toprule",
        seed_header,
        r"    \midrule",
        r"    \endhead",
        r"    \midrule",
        r"    \multicolumn{5}{r}{\emph{continued on next page}} \\",
        r"    \endfoot",
        r"    \bottomrule",
        r"    \endlastfoot",
    ]

    for ds in datasets:
        ds_label = display(ds) + (r"\,(dev)" if ds == DEV_DATASET else "")
        first = True
        for method in METHOD_ORDER:
            cell = cells.get((ds, method))
            if cell is None:
                continue
            disp = METHOD_DISPLAY.get(method, method)
            ds_col = ds_label if first else ""
            first = False
            churn_cells = []
            delta_cells = []
            for s in seeds:
                v = cell.get(s, {})
                ch = v.get("churn")
                dl = v.get("delta")
                churn_cells.append(f"{ch*100:.1f}" if ch is not None else "---")
                if method == "ERM":
                    # ERM is the reference: no paired-Δ entry.  Empty
                    # second-line keeps row height consistent without
                    # rendering three em-dashes as a visible rule.
                    delta_cells.append(r"\strut")
                else:
                    delta_cells.append(f"{dl*100:+.1f}" if dl is not None else "---")
            # Use \makecell for two-line stacked rows in a single \\.
            lines.append(
                f"    {ds_col} & {disp} & "
                + " & ".join(
                    rf"\makecell[r]{{{c}\\{d}}}"
                    for c, d in zip(churn_cells, delta_cells)
                )
                + r" \\"
            )
        # Horizontal rule between datasets (but not after the last one;
        # the longtable's \endlastfoot adds a \bottomrule which would
        # double-print otherwise).
        if ds is not datasets[-1]:
            lines.append(r"    \midrule")
    lines += [r"\end{longtable}", r"}", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
