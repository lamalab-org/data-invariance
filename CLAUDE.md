# Cross-sample prediction churn — agent instructions

## Project

NeurIPS 2026 submission *Reducing cross-sample prediction churn in
scientific machine learning*.  Two classifiers trained on independent
bootstraps of the same chemistry training set assign different classes
to 8–22% of test molecules; aggregate accuracy moves only 1–4 pp.  The
paper measures this gap (cross-sample churn), shows it dominates the
parameter-variance axis deep ensembles / MC dropout / SWA capture, and
reduces it with bagging (free lunch: 43–52% cut at no accuracy cost)
and twin-bootstrap (two networks with sym-KL consistency on
independent bootstraps; further median 41% beyond bagging-K=2 at
matched 2x-ERM compute).

Paper source: `paper/main.tex` + `paper/sections/*.tex`.  Body fits
the NeurIPS 9-page budget.

## Active code paths

```
scripts/cross_sample_train.py    training driver — writes NPZ + manifest
                                 sidecar to outputs/cross_sample{,_seed7,_seed42}/
                                 dataset/method_train{seed}*.npz
scripts/run_experiment.py        HPARAMS table + build_cfg (sourced by
                                 cross_sample_train; not an entry point)
train.py                         make_dataloaders + _build_model
scripts/_analysis_lib.py         load_runs, pairwise_metrics, bootstrap_paired
scripts/_provenance.py           manifest sidecar writer
scripts/paper_constants.py       dataset partitioning, frozen lambda, method order
scripts/make_*_table.py          per-table generators (CSV + .tex)
scripts/analyze_*.py             per-experiment analyses (chemberta, gin,
                                 waterbirds, regression)
scripts/make_paper_macros.py     emits paper/sections/macros.tex from CSVs
scripts/aggregate_seed_sensitivity.py  combines per-canonical-seed CSVs into
                                       cross-seed averages
scripts/run_cpu_blocks_local.sh  full sweep at canonical_seed=99
scripts/run_seed_sweep.sh <s>    sweep at one additional canonical seed
```

## Source-of-truth chain

```
NPZ (cross_sample_train.py)
  → CSV (analyze_*.py / make_*_table.py)
    → paper/sections/tables/*.tex                  (table fragments)
    → paper/sections/macros.tex                    (every quoted prose number)
```

Every prose number in `paper/sections/*.tex` references a
`\newcommand` defined in `macros.tex` — no literal numbers in body
prose.  124 macros, all used.

## Reproducibility

```bash
make analysis    # CSVs from NPZs
make tables      # .tex fragments + macros
make figures     # PDFs
cd paper && latexmk -pdf main.tex
```

`outputs/` is git-ignored; CI does not run training, only the analysis
pipeline.  Bit-for-bit reproducibility is verified after any refactor
of the training code by re-running BACE ERM train_seed=1 and
diffing the resulting NPZ against `outputs/cross_sample/bace/erm_train1.npz`.

## How to work on this project

### Scientist stays at the wheel
- Explain every non-obvious design choice before implementing it.
- When there are real alternatives, name them and state the tradeoff —
  don't silently pick one.
- Flag when a result is surprising or contradicts the prior — don't
  just report numbers.

### Planning
- Enter plan mode for tasks with 3+ steps or an architectural choice.
- If something goes sideways, stop and re-plan; don't keep pushing.

### Verification
- Never call a step done without showing it works.
- For training code: ERM train_seed=1 on BACE should match the
  existing manifest bit-for-bit.  Use `np.allclose` to check.
- For paper changes: rebuild the PDF, check the log for overfull
  boxes / undefined references, check the macro audit prints
  `0 unused`.

### Code quality
- Simplicity first; minimal impact on surrounding code; no speculative
  abstractions.
- Find root causes; no temporary fixes.
- Comments explain *why* or *what the design choice is* — never just
  restate the variable name.
- Don't write top-of-file docstrings that re-narrate what's obvious
  from the code.

### Subagents
- Use subagents to offload literature lookups, code exploration, and
  parallel analyses.  One focused task per subagent.

### Git commits
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`).
- Commit at natural milestones, not per file.
- Co-Authored-By trailer with the model name.
