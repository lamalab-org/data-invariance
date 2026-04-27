"""Parse v2 instrumented run_experiment.py logs into one tidy CSV.

Per-(dataset, seed) the logs contain:
  - discovery line:  signal_ratio, reliability, lambda, optional assign_corr
  - one line per method (erm | jtt | lff | ours): WGA-sel, SWA-groups,
    Free-sel, SWA-free, plus epochs
  - a `candidates: vrex=x jtt_fallback=y erm_fallback=z [vrex_gated_off]` line
    emitted just before 'ours' in runs with the post-2026-04-17 instrumentation

Output: one CSV with one row per (dataset, seed, method) plus the
discovery / candidate columns. CSV path is printed to stdout.

The parser is deliberately conservative: it ignores transient warning lines
and non-numeric "candidates: ..." mid-training prints.
"""
import argparse
import csv
import re
from pathlib import Path


LINE_DISC = re.compile(
    r"discovery:\s+signal_ratio=([\d.]+)\s+reliability=([\d.]+)\s+"
    r"lambda=([\d.]+)(?:\s+assign_corr=([\d.]+))?")

# Example: "  erm  : WGA-sel=0.7944(ep6)  SWA-groups=0.7336  "
#          "Free-sel=0.6838(ep10)  SWA-free=0.7523  [192s]"
LINE_METHOD = re.compile(
    r"^\s{0,4}(erm|jtt|lff|ours)\s*:\s*"
    r"WGA-sel=([\d.]+)\(ep(-?\d+)\)\s+"
    r"SWA-groups=([\d.]+)\s+"
    r"Free-sel=([\d.]+)\(ep(-?\d+)\)\s+"
    r"SWA-free=([\d.]+)"
    r"(?:\s+\[(\d+)s\])?"
    r"(?:\s+\[([a-z_]+)\])?")

LINE_CAND = re.compile(
    r"candidates:\s+vrex=([\d.]+)\s+jtt_fallback=([\d.]+)"
    r"(?:\s+erm_fallback=([\d.]+))?(?:\s+\[(vrex_gated_off)\])?")

LINE_SEED = re.compile(r"---\s+SEED\s+(\d+)\s+---")


def parse_log(path: Path, dataset: str):
    rows = []
    seed = None
    disc = {"signal_ratio": None, "reliability": None,
            "lambda": None, "assign_corr": None}
    cand = {"vrex_cand": None, "jtt_cand": None,
            "erm_cand": None, "vrex_gated": False}

    with path.open() as fh:
        for line in fh:
            m = LINE_SEED.search(line)
            if m:
                seed = int(m.group(1))
                # reset per-seed state
                disc = dict.fromkeys(disc, None)
                cand = {"vrex_cand": None, "jtt_cand": None,
                        "erm_cand": None, "vrex_gated": False}
                continue

            m = LINE_DISC.search(line)
            if m:
                disc = {
                    "signal_ratio": float(m.group(1)),
                    "reliability": float(m.group(2)),
                    "lambda": float(m.group(3)),
                    "assign_corr": (float(m.group(4))
                                    if m.group(4) is not None else None),
                }
                continue

            m = LINE_CAND.search(line)
            if m:
                cand = {
                    "vrex_cand": float(m.group(1)),
                    "jtt_cand":  float(m.group(2)),
                    "erm_cand":  (float(m.group(3))
                                  if m.group(3) is not None else None),
                    "vrex_gated": bool(m.group(4)),
                }
                continue

            m = LINE_METHOD.match(line)
            if m:
                method = m.group(1)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "wga_sel":   float(m.group(2)),
                    "wga_epoch": int(m.group(3)),
                    "swa_groups": float(m.group(4)),
                    "free_sel":   float(m.group(5)),
                    "free_epoch": int(m.group(6)),
                    "swa_free":   float(m.group(7)),
                    "seconds":    int(m.group(8)) if m.group(8) else None,
                    "picked":     m.group(9) or "",
                    **disc,
                    **(cand if method == "ours" else
                       {"vrex_cand": None, "jtt_cand": None,
                        "erm_cand": None, "vrex_gated": False}),
                }
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs/v2",
                    help="Directory containing v2_<dataset>.log files.")
    ap.add_argument("--out", default="outputs/v2_results.csv")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for log in sorted(logs_dir.glob("v2_*.log")):
        dataset = log.stem.replace("v2_", "")
        rows = parse_log(log, dataset)
        all_rows.extend(rows)
        print(f"{log.name}: {len(rows)} rows")

    if not all_rows:
        print("No rows parsed.")
        return

    fields = ["dataset", "seed", "method",
              "wga_sel", "wga_epoch", "swa_groups",
              "free_sel", "free_epoch", "swa_free",
              "seconds", "picked",
              "signal_ratio", "reliability", "lambda", "assign_corr",
              "vrex_cand", "jtt_cand", "erm_cand", "vrex_gated"]

    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
