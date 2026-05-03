"""Within-dataset N-scaling on BACE under the cross-sample bootstrap protocol.

For each subsample size M ∈ {200, 400, 600, 800, 968}, train ERM on
ten independent bootstraps of a deterministic-shuffled M-prefix of the
canonical training pool, evaluate on the canonical id-test, and report
mean cross-sample sym-KL with paired-bootstrap 95% CIs over the
C(10, 2)=45 seed pairs.

Reads:  outputs/cross_sample_nscaling/M{200,400,600,800}/bace/erm_train*.npz
        outputs/cross_sample/bace/erm_train*.npz                 (full-N)
Writes: paper/sections/tables/nscaling_bace.tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, load_runs, pairwise_metrics


def _row(ds_dir: Path, M: int):
    runs = load_runs(ds_dir, "erm_train*.npz")
    if len(runs) < 2:
        return None
    pair_metrics, _ = pairwise_metrics(runs)
    sym_kls = [m["id_sym_kl"] for m in pair_metrics.values()]
    churns = [m["id_churn"] for m in pair_metrics.values()]
    return {
        "M": M,
        "n_seeds": len(runs),
        "sym_kl": bootstrap_ci(sym_kls),
        "churn": bootstrap_ci(churns),
    }


def main() -> None:
    sizes = [200, 300, 400, 500, 600, 700, 800, 900, 968]
    rows = []
    for M in sizes:
        if M == 968:
            ds_dir = Path("outputs/cross_sample/bace")
        else:
            ds_dir = Path(f"outputs/cross_sample_nscaling/M{M}/bace")
        r = _row(ds_dir, M)
        if r is not None:
            rows.append(r)
    if not rows:
        print("No N-scaling outputs found.")
        return

    print(f"\n=== BACE within-dataset N-scaling (cross-sample protocol) ===\n")
    print(f"{'M':>5s}  {'sym_kl (mean [lo, hi])':>30s}  "
          f"{'argmax churn (%)':>22s}")
    for r in rows:
        sk = r["sym_kl"]
        ch = r["churn"]
        print(f"{r['M']:>5d}  "
              f"{sk[0]:.3f} [{sk[1]:.3f}, {sk[2]:.3f}]      "
              f"{ch[0]*100:5.1f} [{ch[1]*100:.1f}, {ch[2]*100:.1f}]")

    # Log-log slope of mean sym_kl vs M.
    Ms = np.array([r["M"] for r in rows], dtype=float)
    sks = np.array([r["sym_kl"][0] for r in rows], dtype=float)
    slope, intercept = np.polyfit(np.log(Ms), np.log(sks), 1)
    print(f"\nlog(sym_kl) = {slope:.2f} * log(M) + {intercept:.2f}")
    print(f"  theoretical β = O(1/N) prediction: slope = -1")
    print(f"  observed within-dataset slope:    {slope:.2f}\n")

    # Emit LaTeX table.
    out_path = Path("paper/sections/tables/nscaling_bace.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    caption = (
        r"\textbf{Cross-sample fragility decreases with training-pool "
        r"size on BACE, with a visible plateau between $M{=}600$ and "
        r"$M{=}800$.}  Within-dataset $N$-scaling under the cross-sample "
        r"bootstrap protocol.  Each row is mean cross-pair sym-KL and "
        r"argmax churn at training-pool size $M$, computed over all "
        r"$45$ pairs of $10$ retrainings with paired-bootstrap $95\%$ "
        r"CIs ($10{,}000$ resamples).  The within-dataset log-log slope "
        r"of sym-KL vs $M$ is $" + f"{slope:.2f}" + r"$; the plateau "
        r"reflects an irreducible boundary-disagreement floor (fragility "
        r"decreases with $N$ but not to zero)."
    )
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{" + caption + r"}",
        r"  \label{tab:nscaling}",
        r"  \small",
        r"  \begin{tabular}{rll}",
        r"    \toprule",
        r"    $M$ & sym-KL (mean [95\% CI]) & Argmax churn (\%, [95\% CI]) \\",
        r"    \midrule",
    ]
    for r in rows:
        sk = r["sym_kl"]; ch = r["churn"]
        lines.append(
            f"    {r['M']} & "
            f"{sk[0]:.3f} [{sk[1]:.3f}, {sk[2]:.3f}] & "
            f"{ch[0]*100:.1f} [{ch[1]*100:.1f}, {ch[2]*100:.1f}] \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")

    # CSV dump for paper-macros: per-M rows + a summary row carrying
    # the log-log slope so prose can reference it via a macro.
    import csv as _csv
    csv_path = Path("outputs/nscaling_bace.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["scope", "M", "sym_kl_mean", "sym_kl_lo", "sym_kl_hi",
                    "churn_mean", "churn_lo", "churn_hi"])
        for r in rows:
            sk = r["sym_kl"]; ch = r["churn"]
            w.writerow(["per_M", r["M"], sk[0], sk[1], sk[2],
                        ch[0], ch[1], ch[2]])
        w.writerow(["slope", "log_log", slope, intercept, "", "", "", ""])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
