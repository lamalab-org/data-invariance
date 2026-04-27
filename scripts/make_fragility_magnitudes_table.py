"""Generate the fragility-magnitudes table for paper §4.

For every dev + headline dataset, compute mean cross-bootstrap sym-KL and
mean argmax churn between ERM models trained on independent bootstraps,
with paired-bootstrap 95% CIs over the C(N_seeds, 2) seed pairs.

Reads:  outputs/cross_sample/{dataset}/erm_train*.npz  (10 seeds, sorted)
Writes: paper/sections/tables/fragility_magnitudes.tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, load_runs, pairwise_metrics, per_run_accuracies
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, N_TRAIN, display


def _row(dataset: str, root: Path) -> dict | None:
    runs = load_runs(root / dataset, "erm_train*.npz")
    if len(runs) < 2:
        return None
    id_accs, _ = per_run_accuracies(runs)
    pair_metrics, _ = pairwise_metrics(runs)
    churns = [m["id_churn"] for m in pair_metrics.values()]
    sym_kls = [m["id_sym_kl"] for m in pair_metrics.values()]
    return {
        "dataset": dataset,
        "n_train": N_TRAIN.get(dataset, len(runs[0][1]["id_indices"])),
        "id_acc_mean": float(np.mean(id_accs)),
        "churn": bootstrap_ci(churns),
        "sym_kl": bootstrap_ci(sym_kls),
        "n_seeds": len(runs),
    }


def _fmt_pct_ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]*100:.1f} [{t[1]*100:.1f}, {t[2]*100:.1f}]"


def _fmt_ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main() -> None:
    root = Path("outputs/cross_sample")
    out_path = Path("paper/sections/tables/fragility_magnitudes.tex")
    datasets = [DEV_DATASET] + HEADLINE_DATASETS
    rows = [r for r in (_row(ds, root) for ds in datasets) if r is not None]
    if not rows:
        print("No ERM runs found under outputs/cross_sample/.")
        return

    rows.sort(key=lambda r: r["n_train"])
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Cross-bootstrap fragility magnitudes on the canonical "
        r"id-test of each dataset. Argmax churn is the per-example "
        r"disagreement rate between two ERM models trained on independent "
        r"bootstraps; sym-KL is the corresponding distributional gap. "
        r"Both are means over $\binom{n_{\text{seeds}}}{2}$ seed pairs with "
        r"$95\%$ paired-bootstrap CIs ($10{,}000$ resamples).}",
        r"  \label{tab:fragility-magnitudes}",
        r"  \small",
        r"  \begin{tabular}{lrrll}",
        r"    \toprule",
        r"    Dataset & $N$ & ERM id-acc & Argmax churn (\%) & Sym-KL (nats) \\",
        r"    \midrule",
    ]
    for r in rows:
        lines.append(
            f"    {display(r['dataset'])} & {r['n_train']} & "
            f"{r['id_acc_mean']:.3f} & {_fmt_pct_ci(r['churn'])} & "
            f"{_fmt_ci(r['sym_kl'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    for r in rows:
        print(f"  {r['dataset']:16s}  N={r['n_train']:>5d}  "
              f"acc={r['id_acc_mean']:.3f}  "
              f"churn={_fmt_pct_ci(r['churn'])}  "
              f"sym-kl={_fmt_ci(r['sym_kl'])}")


if __name__ == "__main__":
    main()
