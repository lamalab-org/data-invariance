# SLURM dispatch

This directory holds slurm submit scripts for running the paper's
cross-sample churn experiments on the Draco cluster (or any slurm
cluster — adjust `--partition` and `--gres` as needed).

## Layout

```
slurm/
├── full_retraining/    # canonical full-paper retraining sweep
│   ├── 01_headline_cls.sh         # 9 datasets × 6 methods (CPU)
│   ├── 02_pareto_bace.sh          # 6 lambdas (CPU)
│   ├── 03_chemberta.sh            # 6 datasets × 3 methods (GPU)
│   ├── 04_gin_bace.sh             # 4 methods (GPU)
│   ├── 05_waterbirds.sh           # 3 methods (GPU)
│   ├── 06_regression.sh           # 3 datasets × 4 methods (CPU)
│   ├── 07_nscaling_bace.sh        # 5 sizes (CPU)
│   ├── 08_borderline.sh           # 3 datasets, ERM-only (CPU)
│   ├── 09_excluded.sh             # 5 datasets, ERM-only (CPU)
│   └── submit_all.sh              # dispatch every block
├── setup.sh            # one-time: install uv, sync deps on the cluster
├── legacy/             # archived one-off scripts from earlier sweeps
└── README.md
```

## One-time setup on the cluster

```bash
git clone <repo-url> data-invariance
cd data-invariance
bash slurm/setup.sh
```

## Full paper retraining

From the repo root on the cluster:

```bash
bash slurm/full_retraining/submit_all.sh
```

This submits every block as a separate slurm array.  Job ids are
written to `logs/full_retraining_jobs.txt`; monitor with:

```bash
squeue --jobs $(paste -sd, logs/full_retraining_jobs.txt) -u $USER
```

Each array task runs all 10 train-seeds for a single
(dataset, method, K, λ) tuple in one Python process.  NPZs land under
`outputs/cross_sample/<dataset>/`; each save also writes a
`<basename>.manifest.json` sidecar (git commit, command, env, data
hashes, wallclock) and appends one line to
`outputs/cross_sample/RUN_LEDGER.jsonl`.

## After the sweep

Pull NPZs back to a workstation and regenerate every artefact:

```bash
make analysis    # rebuild every CSV from saved NPZs
make tables      # rebuild every paper/sections/tables/*.tex
make figures     # rebuild every paper/figures/*.pdf
make macros      # rebuild paper/sections/macros.tex from CSVs
```

Then build the paper:

```bash
cd paper && latexmk -pdf main.tex
```

## Canonical-seed sensitivity sweep on pretrained backbones

The CPU-side seed sweep (`scripts/run_seed_sweep.sh 7` /
`scripts/run_seed_sweep.sh 42`) covers the headline chemistry MLP +
regression sweeps.  The pretrained-backbone analyses
(ChemBERTa, GIN, Waterbirds) need GPU and dispatch via slurm:

```bash
bash slurm/full_retraining/submit_pretrained_seed_sweep.sh
```

This calls `CANON=7 sbatch …` and `CANON=42 sbatch …` for blocks 03
(ChemBERTa, 6 datasets x 3 methods x 10 seeds), 04 (GIN, 4 methods x
10 seeds), and 05 (Waterbirds, 3 methods x 10 seeds), writing NPZs to
`outputs/cross_sample_seed${CANON}/<dataset>/`.  The canonical
`outputs/cross_sample/` (seed 99) tree is untouched.

Each individual block also accepts `CANON` via env if dispatched on
its own:

```bash
CANON=7  sbatch slurm/full_retraining/03_chemberta.sh
CANON=42 sbatch slurm/full_retraining/05_waterbirds.sh
```

After the dispatch finishes (rsync NPZs back to the workstation
first), re-aggregate:

```bash
uv run python scripts/aggregate_seed_sensitivity.py
make tables
```

The aggregator degenerates gracefully -- if the dispatch hasn't
finished, the per-seed step prints `[warn] seed=X: no CSV produced
for <job>` and the aggregate falls back to canonical-seed-99 alone.

## Cluster-specific tweaks

`--partition`, `--gres`, `--mem`, `--time` are set for Draco's GPU
queue with A100s.  Adjust at the top of each `*.sh` for other
clusters.
