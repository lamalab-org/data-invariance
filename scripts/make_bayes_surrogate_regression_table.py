"""Summarise Bayesian-surrogate regression outputs.

Reads outputs from ``scripts/cross_sample_train_bayes_surrogate.py``:

``<root>/<dataset>/twin_indep_bayes_surrogate_{method}_train{seed}_lam{lam}.npz``

and writes:

* a long CSV with per-method MAE/fragility metrics and paired deltas;
* a compact LaTeX table comparing ``stabilized_gpr`` against ``default_gpr``.

Regression fragility is the per-example absolute prediction difference between
two retrainings, averaged over the canonical test split.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from _analysis_lib import bootstrap_ci, bootstrap_paired, load_runs


DATASETS = [
    ("esol_reg", "ESOL", 1128),
    ("freesolv_reg", "FreeSolv", 642),
    ("lipo_reg", "Lipo", 4200),
]

METHOD_LABELS = {
    "default_gpr": "Default GPR",
    "stabilized_gpr": "Stabilized GPR",
}


def _glob(method: str, lam: str) -> str:
    return f"twin_indep_bayes_surrogate_{method}_train*_lam{lam}.npz"


def _scalar(d: dict, key: str) -> float | None:
    if key not in d:
        return None
    try:
        return float(d[key].item())
    except Exception:
        return float(d[key])


def _mean_or_nan(vals: list[float | None]) -> float:
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _preds(d: dict, split: str) -> np.ndarray:
    avg_key = f"{split}_preds_avg"
    single_key = f"{split}_preds"
    return d[avg_key] if avg_key in d else d[single_key]


def _maes(runs: list[tuple[int, dict]], split: str) -> list[float]:
    return [
        float(np.abs(_preds(d, split) - d[f"{split}_labels"]).mean())
        for _, d in runs
    ]


def _pairwise_fragility(runs: list[tuple[int, dict]]) -> dict[tuple[int, int], dict[str, float]]:
    pairs = list(combinations([s for s, _ in runs], 2))
    by_seed = dict(runs)
    out = {}
    for sa, sb in pairs:
        da = by_seed[sa]
        db = by_seed[sb]
        out[(sa, sb)] = {
            "id_frag": float(np.abs(_preds(da, "id") - _preds(db, "id")).mean()),
            "ood_frag": float(np.abs(_preds(da, "ood") - _preds(db, "ood")).mean()),
        }
    return out


def _campaign_json(root: Path, dataset: str) -> Path | None:
    candidates = sorted((root / dataset).glob("surrogate_bo_seed*_campaigns*_steps*.json"))
    return candidates[-1] if candidates else None


def _bo_summary(root: Path, dataset: str, method: str) -> dict[str, float | None]:
    path = _campaign_json(root, dataset)
    if path is None:
        return {
            "bo_final_best_mean": None,
            "bo_hit_global_best_rate": None,
            "bo_mean_hit_global_best_step": None,
        }
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {}).get(method, {})
    return {
        "bo_final_best_mean": summary.get("mean_final_best"),
        "bo_hit_global_best_rate": summary.get("hit_global_best_rate"),
        "bo_mean_hit_global_best_step": summary.get("mean_hit_global_best_step"),
    }


def metrics_with_cis(root: Path, dataset: str, method: str, lam: str) -> dict | None:
    runs = load_runs(root / dataset, _glob(method, lam))
    if not runs:
        return None

    pair_rows = list(_pairwise_fragility(runs).values())
    lams = [_scalar(d, "lam") for _, d in runs]
    selected_sizes = []
    for _, d in runs:
        selected = d.get("bayes_surrogate_selected_indices")
        if selected is not None:
            selected_sizes.append(float(len(selected)))

    out = {
        "method": method,
        "n_seeds": len(runs),
        "n_pairs": len(pair_rows),
        "id_mae": bootstrap_ci(_maes(runs, "id")),
        "ood_mae": bootstrap_ci(_maes(runs, "ood")),
        "id_frag": bootstrap_ci(r["id_frag"] for r in pair_rows),
        "ood_frag": bootstrap_ci(r["ood_frag"] for r in pair_rows),
        "lam_mean": _mean_or_nan(lams),
        "selected_n_mean": float(np.mean(selected_sizes)) if selected_sizes else float("nan"),
    }
    out.update(_bo_summary(root, dataset, method))
    return out


def _mae_by_seed(runs: list[tuple[int, dict]]) -> dict[int, dict[str, float]]:
    out = {}
    for seed, d in runs:
        out[seed] = {
            "id_mae": float(np.abs(_preds(d, "id") - d["id_labels"]).mean()),
            "ood_mae": float(np.abs(_preds(d, "ood") - d["ood_labels"]).mean()),
        }
    return out


def deltas_vs_default(root: Path, dataset: str, method: str, lam: str) -> dict | None:
    if method == "default_gpr":
        return None
    runs = load_runs(root / dataset, _glob(method, lam))
    base_runs = load_runs(root / dataset, _glob("default_gpr", lam))
    if not runs or not base_runs:
        return None

    mae = _mae_by_seed(runs)
    base_mae = _mae_by_seed(base_runs)
    common_seeds = sorted(set(mae).intersection(base_mae))
    if not common_seeds:
        return None

    out: dict[str, object] = {
        "delta_n_seeds": len(common_seeds),
    }
    for key in ("id_mae", "ood_mae"):
        vals = [mae[s][key] - base_mae[s][key] for s in common_seeds]
        out[f"delta_{key}"] = bootstrap_paired(vals)

    pair_metrics = _pairwise_fragility(runs)
    base_pair_metrics = _pairwise_fragility(base_runs)
    common_pairs = sorted(set(pair_metrics).intersection(base_pair_metrics))
    out["delta_n_pairs"] = len(common_pairs)
    for key in ("id_frag", "ood_frag"):
        if common_pairs:
            vals = [pair_metrics[p][key] - base_pair_metrics[p][key]
                    for p in common_pairs]
            out[f"delta_{key}"] = bootstrap_paired(vals)
        else:
            out[f"delta_{key}"] = (float("nan"), float("nan"), float("nan"))
    return out


def _flatten_row(dataset: str, label: str, n_train: int, method: str,
                 m: dict, deltas: dict | None) -> dict:
    row = {
        "dataset": dataset,
        "dataset_label": label,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "n_train": n_train,
        "n_seeds": m["n_seeds"],
        "n_pairs": m["n_pairs"],
        "lam_mean": m["lam_mean"],
        "selected_n_mean": m["selected_n_mean"],
        "bo_final_best_mean": m["bo_final_best_mean"],
        "bo_hit_global_best_rate": m["bo_hit_global_best_rate"],
        "bo_mean_hit_global_best_step": m["bo_mean_hit_global_best_step"],
    }
    for key in ("id_mae", "ood_mae", "id_frag", "ood_frag"):
        row[f"{key}_mean"], row[f"{key}_lo"], row[f"{key}_hi"] = m[key]
    if deltas:
        row["delta_n_seeds"] = deltas["delta_n_seeds"]
        row["delta_n_pairs"] = deltas["delta_n_pairs"]
        for key in ("id_mae", "ood_mae", "id_frag", "ood_frag"):
            mean, lo, hi = deltas[f"delta_{key}"]
            row[f"delta_{key}_mean"] = mean
            row[f"delta_{key}_lo"] = lo
            row[f"delta_{key}_hi"] = hi
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "dataset_label", "method", "method_label", "n_train",
        "n_seeds", "n_pairs", "lam_mean", "selected_n_mean",
        "bo_final_best_mean", "bo_hit_global_best_rate", "bo_mean_hit_global_best_step",
        "id_mae_mean", "id_mae_lo", "id_mae_hi",
        "ood_mae_mean", "ood_mae_lo", "ood_mae_hi",
        "id_frag_mean", "id_frag_lo", "id_frag_hi",
        "ood_frag_mean", "ood_frag_lo", "ood_frag_hi",
        "delta_n_seeds", "delta_n_pairs",
        "delta_id_mae_mean", "delta_id_mae_lo", "delta_id_mae_hi",
        "delta_ood_mae_mean", "delta_ood_mae_lo", "delta_ood_mae_hi",
        "delta_id_frag_mean", "delta_id_frag_lo", "delta_id_frag_hi",
        "delta_ood_frag_mean", "delta_ood_frag_lo", "delta_ood_frag_hi",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _fmt_ci(row: dict, key: str) -> str:
    mean = row.get(f"{key}_mean")
    lo = row.get(f"{key}_lo")
    hi = row.get(f"{key}_hi")
    if mean is None or not np.isfinite(mean):
        return "---"
    return f"{mean:.2f} [{lo:.2f},{hi:.2f}]"


def _fmt_ci_best(row: dict, key: str, best: float | None) -> str:
    cell = _fmt_ci(row, key)
    mean = row.get(f"{key}_mean")
    if best is not None and mean is not None and np.isfinite(mean) and np.isclose(mean, best):
        return r"\textbf{" + cell + r"}"
    return cell


def _fmt_delta(row: dict, key: str) -> str:
    mean = row.get(f"delta_{key}_mean", float("nan"))
    lo = row.get(f"delta_{key}_lo", float("nan"))
    hi = row.get(f"delta_{key}_hi", float("nan"))
    if not np.isfinite(mean):
        return "---"
    cell = f"{mean:+.2f} [{lo:+.2f},{hi:+.2f}]"
    if mean < 0:
        return r"\textbf{" + cell + r"}"
    return cell


def write_latex(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Effect of GPR stabilization on regression benchmark "
        r"performance.} Metrics are computed from final twin-bootstrap model "
        r"predictions. MAE and fragility are reported on in-distribution (ID) "
        r"and out-of-distribution (OOD) test splits. Bold highlights the better "
        r"value between the default and stabilized GPR surrogate.} ",
        r"  \label{tab:bayes-surrogate-regression}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{ID} & \multicolumn{2}{c}{OOD} \\",
        r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"    Dataset & Metric & Default & Stabilized & Default & Stabilized \\",
        r"    \midrule",
    ]
    by_dataset = {short: [] for short, _, _ in DATASETS}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset, dataset_rows in by_dataset.items():
        ordered = sorted(dataset_rows, key=lambda r: 0 if r["method"] == "default_gpr" else 1)
        best = {}
        for split in ("id", "ood"):
            for metric in ("mae", "frag"):
                key = f"{split}_{metric}"
                vals = [
                    row.get(f"{key}_mean")
                    for row in ordered
                    if row.get(f"{key}_mean") is not None
                    and np.isfinite(row.get(f"{key}_mean"))
                ]
                best[key] = min(vals) if vals else None
        rows_by_method = {row["method"]: row for row in ordered}
        default = rows_by_method.get("default_gpr")
        stabilized = rows_by_method.get("stabilized_gpr")
        if default is None or stabilized is None:
            continue
        for i, (metric, label) in enumerate((("mae", "MAE"), ("frag", "Frag."))):
            dataset_label = default["dataset_label"] if i == 0 else ""
            lines.append(
                f"    {dataset_label} & {label} & "
                f"{_fmt_ci_best(default, f'id_{metric}', best[f'id_{metric}'])} & "
                f"{_fmt_ci_best(stabilized, f'id_{metric}', best[f'id_{metric}'])} & "
                f"{_fmt_ci_best(default, f'ood_{metric}', best[f'ood_{metric}'])} & "
                f"{_fmt_ci_best(stabilized, f'ood_{metric}', best[f'ood_{metric}'])} \\\\"
            )
        if ordered:
            lines.append(r"    \addlinespace")
    if lines[-1] == r"    \addlinespace":
        lines.pop()
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path("outputs/cross_sample_bayes_surrogate_regression"))
    ap.add_argument("--datasets", nargs="+", default=[d[0] for d in DATASETS],
                    help="Datasets to include.")
    ap.add_argument("--methods", nargs="+", default=["default_gpr", "stabilized_gpr"])
    ap.add_argument("--lam", default="3",
                    help="Lambda string used in filenames, e.g. 3 or 3.0.")
    ap.add_argument("--csv", type=Path,
                    default=Path("outputs/bayes_surrogate_regression_table.csv"))
    ap.add_argument("--tex", type=Path,
                    default=Path("paper/sections/tables/bayes_surrogate_regression.tex"))
    args = ap.parse_args()

    meta = {short: (label, n_train) for short, label, n_train in DATASETS}
    rows = []
    for dataset in args.datasets:
        label, n_train = meta.get(dataset, (dataset, ""))
        for method in args.methods:
            metrics = metrics_with_cis(args.root, dataset, method, args.lam)
            if metrics is None:
                print(f"[warn] no runs for dataset={dataset} method={method} lam={args.lam}")
                continue
            deltas = deltas_vs_default(args.root, dataset, method, args.lam)
            rows.append(_flatten_row(dataset, label, n_train, method, metrics, deltas))

    if not rows:
        raise SystemExit("No surrogate regression NPZ files found.")

    write_csv(rows, args.csv)
    write_latex(rows, args.tex)

    print("\n== Bayesian surrogate regression outputs ==")
    print(f"{'dataset':12s} {'method':16s} {'n':>3s} {'id_mae':>9s} "
          f"{'id_frag':>9s} {'d_id_mae':>10s} {'d_id_frag':>11s}")
    for row in rows:
        dmae = row.get("delta_id_mae_mean")
        dfrag = row.get("delta_id_frag_mean")
        dmae_s = f"{dmae:10.3f}" if dmae is not None else f"{'---':>10s}"
        dfrag_s = f"{dfrag:11.3f}" if dfrag is not None else f"{'---':>11s}"
        print(f"{row['dataset']:12s} {row['method']:16s} {row['n_seeds']:3d} "
              f"{row['id_mae_mean']:9.3f} {row['id_frag_mean']:9.3f} "
              f"{dmae_s} {dfrag_s}")


if __name__ == "__main__":
    main()
