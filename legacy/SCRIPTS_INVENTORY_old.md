# Scripts inventory (post-cleanup, 2026-04-27)

The cross-sample paper depends only on the seven scripts under "**Paper
pipeline**" below. Everything else either belongs to a sibling project
(the predecessor pipeline paper) or has been archived to ``legacy/``.

## Paper pipeline

A clean dependency graph: training → analysis. Each analysis script
loads NPZs from ``outputs/cross_sample/`` and prints a table; nothing
shells out to anything else.

| Script | Purpose |
|---|---|
| ``scripts/_analysis_lib.py``           | Shared utilities (NPZ load, sym-KL, bootstrap CI, paired CI). Imported by every ``make_*`` script. |
| ``scripts/cross_sample_train.py``      | Train one of {ERM, deep_ensemble, bagging, twin_indep} on a canonical (fixed test set) protocol. |
| ``scripts/make_main_table.py``         | **Paper Table 1.** Held-out comparison ERM ↔ ensembles ↔ Twin_indep with bootstrap CIs and paired Δ. |
| ``scripts/make_pareto.py``             | **Paper Figure: development Pareto.** λ vs (acc, churn) on the dev dataset, identifies the frozen λ. |
| ``scripts/make_fragility_table.py``    | **Paper Table: fragility magnitude.** Cross-dataset 1/N scaling; loads ``outputs/fragility/``. |
| ``scripts/make_nscaling_table.py``     | **Paper Figure: within-dataset N-scaling.** Subsampled BACE; loads ``outputs/fragility_nscaling/``. |
| ``scripts/make_churn_table.py``        | **Paper Table: churn predictivity.** Top-decile fragility predicts argmax flips. |

## Shell drivers

These wrap multi-dataset, multi-seed sweeps. Each one is just a loop over
dataset/lam/seed values calling ``cross_sample_train.py``. Keep when running
end-to-end; can also be replaced by ``Makefile.paper``.

- ``scripts/run_lambda_pareto.sh``
- ``scripts/run_cross_sample_sweep.sh``
- ``scripts/run_ensemble_baselines.sh``
- ``scripts/run_fragility_sweep.sh``
- ``scripts/run_n_scaling.sh``

## Pre-existing pipeline-paper scripts (separate project)

These belong to the predecessor research direction (group-free OOD
robustness pipeline). They are not required for the cross-sample paper
and should not be touched without a discussion of the larger codebase.

- ``run_experiment.py``, ``correlation_sweep.py``, ``discovery_quality.py``,
  ``ingredient_ablation.py``, ``k_detection.py``, ``make_figures.py``,
  ``resampling_stability_test.py``, ``swa_analysis.py``

## Archived

- ``legacy/superseded_analysis/`` — earlier analysis scripts replaced by
  ``make_main_table.py`` and ``make_pareto.py``: cross_sample_summary,
  pareto_analysis, strict_dev_test, method_comparison, paired_significance.
- ``legacy/predecessor_methods/`` — earlier twin trainers
  (partition_pair_train, twin_fragility_train, twin_analysis).
- ``legacy/broken_protocol/`` — scripts that ran against the old
  varying-test-set protocol (parse_v2, aggregate_v2). Results invalidated.
- ``legacy/failed_attempts/`` — null results documented for provenance
  (meta_reptile, weight-decay sweep, joint_selective, aggregation_solution,
  attribution_eval, stability_eval).
- ``legacy/alpha_investigation/`` — original "partition-sensitivity as
  uncertainty" investigation that motivated the pivot.

## Outputs layout

```
outputs/
  cross_sample/<dataset>/{erm,deep_ensemble,bagging,twin_indep}_train{seed}*.npz
  fragility/<dataset>/{erm,ensemble,partition}_seed{seed}_k{k}.npz
  fragility_nscaling/M{size}/<dataset>/{erm,partition}_seed{seed}_k{k}.npz
  main_table.csv          ← from make_main_table.py
  pareto_table.csv        ← from make_pareto.py
```
