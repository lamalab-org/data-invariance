"""Summarise ``cross_sample_train_bayes.py`` outputs.

Reads one or more output roots containing files named
``<root>/<dataset>/twin_indep_bayes_train{seed}_lam{lambda}.npz`` and writes
a long CSV plus a compact LaTeX table.

Examples
--------
Fixed-lambda Bayesian-driver runs saved into separate roots::

    uv run python scripts/make_bayes_table.py \\
      --roots bo=outputs/cross_sample_bayes \\
              lam0=outputs/cross_sample_bayes_lam0 \\
              lam300=outputs/cross_sample_bayes_lam300 \\
      --compare-run bo \\
      --baseline-runs lam0 lam300

The comparison table also adds ``BO vs. ERM`` by default, using
``outputs/cross_sample`` for ``erm_train*.npz`` files.

Default Bayesian-search outputs::

    uv run python scripts/make_bayes_table.py --roots bo=outputs/cross_sample
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _analysis_lib import (
    GLOBS,
    bootstrap_ci,
    bootstrap_paired,
    fmt_ci,
    get_probs,
    load_runs,
    pairwise_metrics,
    per_run_accuracies,
)
from paper_constants import DEV_DATASET, HEADLINE_DATASETS, N_TRAIN, display


GLOB = "twin_indep_bayes_train*_lam*.npz"
RUN_GLOBS = {
    "erm": GLOBS["erm"],
}


def _label_root(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label, Path(path)
    path = Path(raw)
    return path.name, path


def _scalar(d: dict, key: str) -> float | None:
    if key not in d:
        return None
    arr = d[key]
    try:
        return float(arr.item())
    except Exception:
        return float(arr)


def _run_lambdas(runs: list[tuple[int, dict]]) -> list[float]:
    vals = []
    for _, d in runs:
        lam = _scalar(d, "lam")
        if lam is not None:
            vals.append(lam)
    return vals


def metrics_with_cis(dataset_dir: Path, glob: str = GLOB) -> dict | None:
    runs = load_runs(dataset_dir, glob)
    if not runs:
        return None

    id_accs, ood_accs = per_run_accuracies(runs)
    pair_metrics, _ = pairwise_metrics(runs)
    pair_rows = list(pair_metrics.values())
    lams = _run_lambdas(runs)

    out = {
        "n_seeds": len(runs),
        "id_acc": bootstrap_ci(id_accs),
        "ood_acc": bootstrap_ci(ood_accs),
        "id_churn": bootstrap_ci(r["id_churn"] for r in pair_rows),
        "ood_churn": bootstrap_ci(r["ood_churn"] for r in pair_rows),
        "id_sym_kl": bootstrap_ci(r["id_sym_kl"] for r in pair_rows),
        "ood_sym_kl": bootstrap_ci(r["ood_sym_kl"] for r in pair_rows),
        "lam_mean": float(np.mean(lams)) if lams else float("nan"),
        "lam_median": float(np.median(lams)) if lams else float("nan"),
        "lam_min": float(np.min(lams)) if lams else float("nan"),
        "lam_max": float(np.max(lams)) if lams else float("nan"),
    }
    return out


def _acc_by_seed(runs: list[tuple[int, dict]]) -> dict[int, dict[str, float]]:
    out = {}
    for seed, d in runs:
        idp, odp = get_probs(d)
        out[seed] = {
            "id_acc": float((idp.argmax(1) == d["id_labels"]).mean()),
            "ood_acc": float((odp.argmax(1) == d["ood_labels"]).mean()),
        }
    return out


def deltas_vs_baseline(root: Path, baseline_root: Path, dataset: str,
                       glob: str = GLOB, baseline_glob: str = GLOB,
                       ) -> dict[str, object] | None:
    """Paired deltas for ``root - baseline_root`` on one dataset.

    Accuracy deltas are matched by train seed.  Churn/sym-KL deltas are
    matched by seed pair.  Positive accuracy deltas are better; negative churn
    and sym-KL deltas are better.
    """
    runs = load_runs(root / dataset, glob)
    base_runs = load_runs(baseline_root / dataset, baseline_glob)
    if not runs or not base_runs:
        return None

    acc = _acc_by_seed(runs)
    base_acc = _acc_by_seed(base_runs)
    common_seeds = sorted(set(acc).intersection(base_acc))
    if not common_seeds:
        return None

    out: dict[str, object] = {
        "delta_n_seeds": len(common_seeds),
    }
    for key in ("id_acc", "ood_acc"):
        vals = [acc[s][key] - base_acc[s][key] for s in common_seeds]
        out[f"delta_{key}"] = bootstrap_paired(vals)

    pair_metrics, _ = pairwise_metrics(runs)
    base_pair_metrics, _ = pairwise_metrics(base_runs)
    common_pairs = sorted(set(pair_metrics).intersection(base_pair_metrics))
    out["delta_n_pairs"] = len(common_pairs)
    for key in ("id_churn", "ood_churn", "id_sym_kl", "ood_sym_kl"):
        if common_pairs:
            vals = [pair_metrics[p][key] - base_pair_metrics[p][key]
                    for p in common_pairs]
            out[f"delta_{key}"] = bootstrap_paired(vals)
        else:
            out[f"delta_{key}"] = (float("nan"), float("nan"), float("nan"))
    return out


def _flatten_row(label: str, dataset: str, m: dict) -> dict:
    row = {
        "run": label,
        "dataset": dataset,
        "n_train": N_TRAIN.get(dataset, ""),
        "n_seeds": m["n_seeds"],
        "lam_mean": m["lam_mean"],
        "lam_median": m["lam_median"],
        "lam_min": m["lam_min"],
        "lam_max": m["lam_max"],
    }
    for key in ("id_acc", "ood_acc", "id_churn", "ood_churn",
                "id_sym_kl", "ood_sym_kl"):
        row[f"{key}_mean"], row[f"{key}_lo"], row[f"{key}_hi"] = m[key]
    return row


def _add_delta_fields(row: dict, deltas: dict[str, object] | None) -> None:
    if deltas is None:
        return
    row["delta_n_seeds"] = deltas["delta_n_seeds"]
    row["delta_n_pairs"] = deltas["delta_n_pairs"]
    for key in ("id_acc", "ood_acc", "id_churn", "ood_churn",
                "id_sym_kl", "ood_sym_kl"):
        mean, lo, hi = deltas[f"delta_{key}"]
        row[f"delta_{key}_mean"] = mean
        row[f"delta_{key}_lo"] = lo
        row[f"delta_{key}_hi"] = hi


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run", "dataset", "n_train", "n_seeds",
        "lam_mean", "lam_median", "lam_min", "lam_max",
        "id_acc_mean", "id_acc_lo", "id_acc_hi",
        "ood_acc_mean", "ood_acc_lo", "ood_acc_hi",
        "id_churn_mean", "id_churn_lo", "id_churn_hi",
        "ood_churn_mean", "ood_churn_lo", "ood_churn_hi",
        "id_sym_kl_mean", "id_sym_kl_lo", "id_sym_kl_hi",
        "ood_sym_kl_mean", "ood_sym_kl_lo", "ood_sym_kl_hi",
        "delta_n_seeds", "delta_n_pairs",
        "delta_id_acc_mean", "delta_id_acc_lo", "delta_id_acc_hi",
        "delta_ood_acc_mean", "delta_ood_acc_lo", "delta_ood_acc_hi",
        "delta_id_churn_mean", "delta_id_churn_lo", "delta_id_churn_hi",
        "delta_ood_churn_mean", "delta_ood_churn_lo", "delta_ood_churn_hi",
        "delta_id_sym_kl_mean", "delta_id_sym_kl_lo", "delta_id_sym_kl_hi",
        "delta_ood_sym_kl_mean", "delta_ood_sym_kl_lo", "delta_ood_sym_kl_hi",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def _fmt_lam(row: dict) -> str:
    lo, hi = row["lam_min"], row["lam_max"]
    med = row["lam_median"]
    if not np.isfinite(med):
        return "---"
    if abs(lo - hi) <= max(1e-12, 1e-9 * max(abs(lo), abs(hi), 1.0)):
        return f"{med:.4g}"
    return f"{med:.4g} [{lo:.4g},{hi:.4g}]"


def _fmt_lam_mean(row: dict) -> str:
    mean = row["lam_mean"]
    if not np.isfinite(mean):
        return "---"
    return f"{mean:.4g}"


def _fmt_delta(t: tuple[float, float, float], pct: bool = True) -> str:
    if not np.isfinite(t[0]):
        return "---"
    scale = 100.0 if pct else 1.0
    return f"{t[0]*scale:+.1f} [{t[1]*scale:+.1f},{t[2]*scale:+.1f}]"


def _fmt_signed_cell(t: tuple[float, float, float], *, good: str) -> str:
    cell = _fmt_delta(t)
    if cell == "---":
        return cell
    mean = t[0]
    if (good == "positive" and mean > 0) or (good == "negative" and mean < 0):
        return r"\textbf{" + cell + r"}"
    return cell


def _delta_tuple(row: dict, key: str) -> tuple[float, float, float]:
    return (
        row.get(f"delta_{key}_mean", float("nan")),
        row.get(f"delta_{key}_lo", float("nan")),
        row.get(f"delta_{key}_hi", float("nan")),
    )


def _glob_for(label: str) -> str:
    return RUN_GLOBS.get(label, GLOB)


def _comparison_rows(root_by_label: dict[str, Path], compare_run: str,
                     baseline_runs: list[str], datasets: list[str]) -> list[dict]:
    compare_root = root_by_label.get(compare_run)
    if compare_root is None:
        raise SystemExit(f"compare run {compare_run!r} was not supplied in --roots")

    rows = []
    for baseline_run in baseline_runs:
        n_before = len(rows)
        baseline_root = root_by_label.get(baseline_run)
        if baseline_root is None:
            print(f"[warn] baseline run {baseline_run!r} was not supplied; skipping.")
            continue
        for ds in datasets:
            m = metrics_with_cis(compare_root / ds, _glob_for(compare_run))
            if m is None:
                continue
            deltas = deltas_vs_baseline(
                compare_root, baseline_root, ds,
                _glob_for(compare_run), _glob_for(baseline_run))
            if deltas is None:
                continue
            row = _flatten_row(compare_run, ds, m)
            row["baseline_run"] = baseline_run
            _add_delta_fields(row, deltas)
            rows.append(row)
        if len(rows) == n_before:
            print(f"[warn] no comparison rows built for "
                  f"{compare_run!r} vs {baseline_run!r}; expected "
                  f"{compare_root}/<dataset>/{_glob_for(compare_run)} and "
                  f"{baseline_root}/<dataset>/{_glob_for(baseline_run)}.")
    return rows


def _pairwise_comparison_rows(root_by_label: dict[str, Path],
                              comparisons: list[tuple[str, str]],
                              datasets: list[str]) -> list[dict]:
    rows = []
    for compare_run, baseline_run in comparisons:
        n_before = len(rows)
        compare_root = root_by_label.get(compare_run)
        baseline_root = root_by_label.get(baseline_run)
        if compare_root is None or baseline_root is None:
            print(f"[warn] comparison {compare_run} vs {baseline_run} skipped; "
                  "one or both roots were not supplied.")
            continue
        for ds in datasets:
            m = metrics_with_cis(compare_root / ds, _glob_for(compare_run))
            if m is None:
                continue
            deltas = deltas_vs_baseline(
                compare_root, baseline_root, ds,
                _glob_for(compare_run), _glob_for(baseline_run))
            if deltas is None:
                continue
            row = _flatten_row(compare_run, ds, m)
            row["baseline_run"] = baseline_run
            _add_delta_fields(row, deltas)
            rows.append(row)
        if len(rows) == n_before:
            print(f"[warn] no comparison rows built for "
                  f"{compare_run!r} vs {baseline_run!r}; expected "
                  f"{compare_root}/<dataset>/{_glob_for(compare_run)} and "
                  f"{baseline_root}/<dataset>/{_glob_for(baseline_run)}.")
    return rows


def write_latex(rows: list[dict], path: Path, baseline_run: str,
                compare_run: str | None = None) -> None:
    if compare_run is not None and rows and "baseline_run" in rows[0]:
        return write_comparison_latex(rows, path, compare_run)

    by_run_ds = {(r["run"], r["dataset"]): r for r in rows}
    runs = [r for r in dict.fromkeys(r["run"] for r in rows)
            if r != baseline_run]
    datasets = [DEV_DATASET] + list(HEADLINE_DATASETS)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Bayesian-driver twin-bootstrap deltas versus "
        r"$\lambda{=}0$.} "
        r"Cells report method minus the $\lambda{=}0$ baseline with paired "
        r"bootstrap $95\%$ confidence intervals. Positive accuracy deltas "
        r"are better; negative churn deltas are better. "
        r"The $\lambda$ column reports median selected $\lambda$ with "
        r"range across seeds; fixed-$\lambda$ runs therefore show a single "
        r"value.}",
        r"  \label{tab:bayes_twin}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}llrrrrr@{}}",
        r"    \toprule",
        r"    Run & Dataset & $\lambda$ & ID acc & OOD acc & ID churn & OOD churn \\",
        r"    \midrule",
    ]
    for run in runs:
        first_run = True
        for ds in datasets:
            r = by_run_ds.get((run, ds))
            if r is None:
                continue
            run_cell = run if first_run else ""
            first_run = False
            ds_label = display(ds) + (r"\,(dev)" if ds == DEV_DATASET else "")
            lines.append(
                f"    {run_cell} & {ds_label} & {_fmt_lam(r)} & "
                f"{_fmt_delta(_delta_tuple(r, 'id_acc'))} & "
                f"{_fmt_delta(_delta_tuple(r, 'ood_acc'))} & "
                f"{_fmt_delta(_delta_tuple(r, 'id_churn'))} & "
                f"{_fmt_delta(_delta_tuple(r, 'ood_churn'))}"
                + r" \\"
            )
        lines.append(r"    \addlinespace[2pt]")
    lines += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def write_comparison_latex(rows: list[dict], path: Path, compare_run: str) -> None:
    datasets = [DEV_DATASET] + list(HEADLINE_DATASETS)
    blocks = list(dict.fromkeys((r["run"], r["baseline_run"]) for r in rows))
    by_block_ds = {(r["run"], r["baseline_run"], r["dataset"]): r for r in rows}
    has_erm_block = any(baseline == "erm" for _, baseline in blocks)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Bayesian optimization discovers dataset-specific operating",
        r" points on the accuracy--stability frontier.} Cells report the first method",
        r" named in each block minus the second with paired bootstrap $95\%$ confidence",
        r" intervals. Positive accuracy deltas indicate improved performance, while",
        r" negative churn deltas indicate improved stability. Bayesian optimization (BO)",
        r" generally identifies intermediate $\lambda$ values that improve ID accuracy and",
        r" often improve OOD accuracy relative to $\lambda{=}0$, while trading off some",
        r" stability relative to strongly regularized models ($\lambda{=}300$).",
    ]
    if has_erm_block:
        lines += [
            r" The BO-vs.-ERM block checks whether optimized $\lambda$ values retain a churn",
            r" reduction relative to ordinary retraining.",
        ]
    lines += [
        r" The $\lambda$",
        r" column reports the mean optimized $\lambda$ for the first method in each comparison",
        r" block. Bolded entries indicate cases where the point estimate favors the first",
        r" method.}",
        r"  \label{tab:bayes_twin}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2pt}",
    ]
    for i, (run, baseline) in enumerate(blocks):
        if i > 0:
            lines.append(r"  \par\vspace{0.75em}")
        pretty_run = _pretty_run_label(run)
        pretty_baseline = _pretty_run_label(baseline)
        lines += [
            rf"  \noindent\makebox[\textwidth][c]{{\textbf{{{pretty_run} vs.\ {pretty_baseline}}}}}",
            r"  \vspace{-0.35em}",
            r"  \resizebox{\textwidth}{!}{%",
            r"  \begin{tabular}{@{}lrlllll@{}}",
            r"    \toprule",
            r"    Dataset & $N$ & Mean $\lambda$ & ID acc & OOD acc & ID churn & OOD churn \\",
            r"    \midrule",
        ]
        for ds in datasets:
            r = by_block_ds.get((run, baseline, ds))
            if r is None:
                continue
            ds_label = display(ds) + (r"\,(dev)" if ds == DEV_DATASET else "")
            lines.append(
                f"    {ds_label} & {N_TRAIN.get(ds, '---')} & {_fmt_lam_mean(r)} & "
                f"{_fmt_signed_cell(_delta_tuple(r, 'id_acc'), good='positive')} & "
                f"{_fmt_signed_cell(_delta_tuple(r, 'ood_acc'), good='positive')} & "
                f"{_fmt_signed_cell(_delta_tuple(r, 'id_churn'), good='negative')} & "
                f"{_fmt_signed_cell(_delta_tuple(r, 'ood_churn'), good='negative')}"
                + r" \\"
            )
        lines += [r"    \bottomrule", r"  \end{tabular}%", r"  }"]
    lines += [r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def _pretty_run_label(run: str) -> str:
    if run == "bo":
        return "BO"
    if run == "erm":
        return "ERM"
    if run.startswith("lam"):
        return run.replace("lam", r"$\lambda{=}") + "$"
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["bo=outputs/cross_sample"],
                    help="Output roots as PATH or LABEL=PATH.")
    ap.add_argument("--datasets", nargs="+",
                    default=[DEV_DATASET] + HEADLINE_DATASETS)
    ap.add_argument("--csv", default="outputs/bayes_table.csv")
    ap.add_argument("--latex", default="paper/sections/tables/bayes_twin.tex")
    ap.add_argument("--baseline-run", default="lam0",
                    help="Run label to use as the lambda=0 baseline.")
    ap.add_argument("--compare-run", default=None,
                    help="If set, LaTeX reports this run against --baseline-runs.")
    ap.add_argument("--baseline-runs", nargs="+", default=None,
                    help="Baselines for --compare-run comparisons.")
    ap.add_argument("--extra-comparisons", nargs="*", default=[],
                    help="Extra LaTeX comparisons as RUN:BASELINE, e.g. lam0:lam300.")
    ap.add_argument("--erm-root", default="outputs/cross_sample",
                    help="Root containing ERM runs for the BO-vs-ERM churn check.")
    ap.add_argument("--skip-erm-comparison", action="store_true",
                    help="Do not add the BO-vs-ERM comparison block.")
    args = ap.parse_args()

    root_specs = [_label_root(r) for r in args.roots]
    root_by_label = dict(root_specs)
    if not args.skip_erm_comparison and args.erm_root:
        root_by_label.setdefault("erm", Path(args.erm_root))
    baseline_root = root_by_label.get(args.baseline_run)
    if baseline_root is None:
        print(f"[warn] baseline run {args.baseline_run!r} was not supplied; "
              "delta columns will be empty.")

    rows = []
    print("== Bayesian twin-bootstrap outputs ==\n")
    print(f"{'run':14s} {'dataset':18s} {'n':>3s} {'lambda':>18s} "
          f"{'id_acc':>18s} {'ood_acc':>18s} {'id_churn':>18s} "
          f"{'ood_churn':>18s} {'Δid_acc':>18s} {'Δood_acc':>18s} "
          f"{'Δid_churn':>18s} {'Δood_churn':>18s}")
    for label, root in root_specs:
        for ds in args.datasets:
            m = metrics_with_cis(root / ds)
            if m is None:
                continue
            row = _flatten_row(label, ds, m)
            if baseline_root is not None:
                _add_delta_fields(row, deltas_vs_baseline(root, baseline_root, ds))
            rows.append(row)
            d_id = _delta_tuple(row, "id_acc")
            d_ood = _delta_tuple(row, "ood_acc")
            d_ch = _delta_tuple(row, "id_churn")
            d_ood_ch = _delta_tuple(row, "ood_churn")
            print(f"{label:14s} {ds:18s} {m['n_seeds']:3d} "
                  f"{_fmt_lam(row):>18s} "
                  f"{fmt_ci(m['id_acc'], pct=True):>18s} "
                  f"{fmt_ci(m['ood_acc'], pct=True):>18s} "
                  f"{fmt_ci(m['id_churn'], pct=True):>18s} "
                  f"{fmt_ci(m['ood_churn'], pct=True):>18s} "
                  f"{_fmt_delta(d_id):>18s} "
                  f"{_fmt_delta(d_ood):>18s} "
                  f"{_fmt_delta(d_ch):>18s} "
                  f"{_fmt_delta(d_ood_ch):>18s}")
        print()

    if not rows:
        raise SystemExit("No Bayesian NPZ files found.")

    write_csv(rows, Path(args.csv))
    if args.compare_run is not None:
        comparison_baselines = args.baseline_runs or [args.baseline_run]
        if not args.skip_erm_comparison and "erm" in root_by_label:
            comparison_baselines = list(comparison_baselines)
            if "erm" not in comparison_baselines:
                comparison_baselines.append("erm")
        comp_rows = _comparison_rows(
            root_by_label, args.compare_run, comparison_baselines, args.datasets)
        extra_pairs = []
        for raw in args.extra_comparisons:
            if ":" not in raw:
                raise SystemExit(f"Bad --extra-comparisons item {raw!r}; expected RUN:BASELINE")
            run, base = raw.split(":", 1)
            extra_pairs.append((run, base))
        comp_rows.extend(_pairwise_comparison_rows(
            root_by_label, extra_pairs, args.datasets))
        if not comp_rows:
            raise SystemExit("No comparison rows could be built.")
        write_latex(comp_rows, Path(args.latex), args.baseline_run,
                    compare_run=args.compare_run)
    else:
        write_latex(rows, Path(args.latex), args.baseline_run)


if __name__ == "__main__":
    main()
