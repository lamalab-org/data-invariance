# Refactor Plan

Written 2026-04-09 after the story converged on **loss-discovery + V-REx + SWA**
as the final method (with upweighted ERM as the imbalanced-val fallback).

## Why refactor now

The codebase has accumulated ~9 months of exploration:

- **train.py is 2034 lines.** It contains 10 `train_*` functions, 5 of which
  implement methods (`adversarial_split`, `adversarial_split_multi`,
  `random_split`, `oracle_split`, `resampling`) that **the final method does not
  use** and that we have empirically shown do not work.
- **`discover_environments` is 425 lines** because it supports 6 scoring
  criteria (`loss`, `entropy`, `counterfactual`, `confident_wrong`,
  `cartography`, `permutation`, `activation`) plus 4 splitting strategies. The
  ablations are done; only `loss` + median-split is on the critical path.
- **models.py has SplitMLP and MultiHeadMLP** which are *only* used by
  the deleted adversarial methods.
- **tests/test_train.py is 39 KB** and ~60 % of its tests cover obsolete code.
- **scripts/** has 11 files; 6 are stability-score validation work that has been
  deprioritised, and 1 is a one-off (`dro_k2_waterbirds.py`) superseded by
  `dro_discovered.py`.
- **configs/method/** has 9 method YAMLs; 4 are for obsolete methods.
- The main `configs/config.yaml` has ~50 hyperparameter fields, ~20 of which
  control adversarial-split internals that no longer exist anywhere on the
  critical path.

The cost of leaving this is high:

1. **Reviewer cognitive load.** The first impression of the repo for the paper
   reviewer / open-source user is "this looks like a research scratch directory."
2. **Maintenance burden.** Every `from train import …` line risks pulling in dead
   code; every config field is one more thing the user has to ignore.
3. **Bug surface.** The auto-switch between V-REx and DRO inside
   `train_discovered_split` is dead code (we use `aggregation_ablation.py` for
   head-to-head). It can mislead future-us into thinking it's load-bearing.
4. **Test fragility.** The 40+ tests for adversarial-split internals will break
   the moment we touch SplitMLP, even though SplitMLP isn't used.

## The principle

> **Keep exactly what is required to reproduce the paper's tables and figures.
> Move everything else to `legacy/` (or delete) so that anyone who reads the
> repo sees only the final method.**

We do not delete history — git keeps it. We move dead code to a marked location
so the *current* surface is clean.

## Code map (current → target)

### Load-bearing for the final method (KEEP, possibly simplify)

| File | Purpose | Action |
|---|---|---|
| `data.py` | Dataset classes (Waterbirds, CelebA, CMNIST, Multi-CMNIST, TADF) | **Keep as-is.** All datasets are used. |
| `models.py` → `MLP`, `make_resnet_backbone`, `_make_mlp_backbone` | The single-head model used everywhere | **Keep, delete `SplitMLP` and `MultiHeadMLP`.** |
| `train.py` → `make_dataloaders`, `evaluate`, `_ModelSelector`, `_val_score`, `_eval_metrics`, `train_erm`, `discover_environments` (pruned), `train_discovered_split` (simplified), `train_jtt`, `train_group_dro`, `train_dfr`, `discover_jtt_weights` | The training pipelines and the discovery pipeline | **Keep but prune.** Target: ~700–900 lines. |
| `evaluate.py` | Currently a kitchen-sink of analysis helpers | **Move stability-score functions to legacy.** Keep nothing or only what `train.py` actually imports. |
| `utils.py` | Tiny — `set_seed`, `get_device`, `log_metrics` | **Keep.** |
| `configs/config.yaml` | Hydra root config | **Prune ~20 obsolete fields.** |
| `configs/dataset/{cmnist,waterbirds,celeba,multi_cmnist,tadf,continuous_cmnist}.yaml` | Per-dataset configs | **Keep.** |
| `configs/method/{erm,discovered_split,jtt,group_dro,dfr}.yaml` | Per-method configs | **Keep.** |
| `scripts/dro_discovered.py` | Standalone DRO + SWA training (used as DRO baseline in aggregation ablation) | **Keep.** Acts as the reproducible entry point for the DRO baseline. |
| `scripts/aggregation_ablation.py` | The head-to-head ERM-vs-V-REx-vs-DRO table | **Keep.** Produces the paper's main ablation table. |
| `scripts/k_detection.py` | Loss-histogram K detection diagnostic | **Keep.** Produces the K-detection figure. |

### Dead or obsolete (MOVE to `legacy/` or DELETE)

| File / function | Why obsolete | Action |
|---|---|---|
| `models.py:SplitMLP`, `models.py:MultiHeadMLP` | Only used by adversarial-split methods which we have shown do not work | **Delete** (or move to `legacy/models_split.py`) |
| `train.py:symmetric_kl` | Used only by adversarial / random / oracle / resampling splits | **Delete** |
| `train.py:train_random_split`, `train_oracle_split`, `train_resampling`, `train_adversarial_split`, `train_adversarial_split_multi` | All require SplitMLP/MultiHeadMLP, all use the soft-assignment learnable-partition idea that didn't work | **Delete** (history is in git) |
| `train.py:discover_environments` branches: `permutation`, `activation`, `counterfactual`, `cartography`, `confident_wrong`, plus the `reweight`, `extremes`, and `cartography natural grouping` branches in the assignment block | All ablated, all worse than `loss` median-split. Only `loss` + median-split survives in the final method. | **Delete branches**, keep the loss + median-split path. ~250 → ~100 lines. |
| `train.py:train_discovered_split` — `balanced_sampling`, `env_mixup`, `training_noise`, the `env_balance < 0.5` auto-switch to DRO | Either ablated or moved into separate scripts (DRO is in `dro_discovered.py`) | **Delete the branches**, keep V-REx-only training. |
| `evaluate.py:compute_assignment_correlation`, `compute_assignment_correlation_multi` | Used only by `train_adversarial_split[_multi]`. Once those are gone, no callers remain. | **Delete** |
| `evaluate.py:compute_stability_scores`, `disagreement_stability_scores`, `adaptive_stability_scores`, `evaluate_stability_discrimination` | Stability-score work was deprioritised after we found ERM uncertainty already tracks difficulty on natural data | **Move to `legacy/stability.py`** in case we want to revisit it for an appendix table |
| `scripts/dro_k2_waterbirds.py` | Superseded by `dro_discovered.py --dataset waterbirds` | **Delete** |
| `scripts/evaluate_stability.py`, `resampling_stability.py`, `resampling_stability_id.py`, `stability_chemistry.py`, `stability_waterbirds.py`, `validate_adaptive_scores.py` | All stability-score validation, deprioritised | **Move to `legacy/scripts/`** |
| `scripts/multiseed_waterbirds.sh` | Bash sweep over methods, the methods it sweeps include adversarial_split | **Delete or rewrite** to use the new entry points |
| `configs/method/adversarial_split.yaml`, `oracle_split.yaml`, `random_split.yaml`, `resampling.yaml` | Configs for deleted methods | **Delete** |
| `configs/config.yaml` fields: `adv_init`, `adv_init_scale`, `head_noise`, `adv_warmup_epochs`, `adv_steps_per_model_step`, `adv_entropy_bonus`, `adv_mode`, `lambda_threshold`, `lambda_ramp_range`, `lambda_warmup_epochs`, `discovery_quantile`, `discovery_reweight`, `discovery_rounds`, `freeze_backbone`, `balanced_sampling`, `env_mixup`, `training_noise`, `discovery_criterion` (or fix it to `"loss"`) | All control branches that we are deleting | **Delete** from config |
| `tests/test_train.py` — tests for `train_adversarial_split*`, `train_oracle_split`, `train_random_split`, `train_resampling`, `SplitMLP`, `MultiHeadMLP`, `compute_assignment_correlation*` | ~40 functions out of ~60. Test obsolete code. | **Delete** these test functions; **add** new tests for `train_discovered_split` (the simplified version) and `dro_discovered.run_seed` |

### What we GAIN by doing this

- **train.py: 2034 → ~900 lines.** Easier to navigate, easier to read, the main
  method's training loop is no longer hidden behind 6 unrelated training loops.
- **models.py: 202 → ~100 lines.** One model class, two backbone factories.
- **discover_environments: 425 → ~120 lines.** A single criterion, a single
  split rule, the permutation test, the diagnostics. Nothing else.
- **tests/test_train.py: ~39 KB → ~12 KB.** Faster, less brittle.
- **One config field per actual hyperparameter,** not 50 with 30 dead.
- **scripts/** becomes 4 files (not 11), each tied to a paper figure.

## Phased plan

Each phase is independent — you can stop after any phase and the repo is still
in a working state. The order minimises risk to the running CelebA experiment
(which uses `aggregation_ablation.py` → `dro_discovered.py` → `train.py`'s
`make_dataloaders`, `discover_environments`, `evaluate`, `_ModelSelector`,
`_val_score` — none of which we touch in phases 0–2).

### Phase 0 — `legacy/` directory (zero risk, do this first)

1. `mkdir -p legacy/scripts`
2. `mkdir -p legacy/configs_method`
3. Add a `legacy/README.md` explaining what's in there and why
4. Move (`git mv`) the deprioritised script files into `legacy/scripts/`:
   `evaluate_stability.py`, `resampling_stability.py`, `resampling_stability_id.py`,
   `stability_chemistry.py`, `stability_waterbirds.py`, `validate_adaptive_scores.py`
5. Move the obsolete method config files into `legacy/configs_method/`:
   `adversarial_split.yaml`, `oracle_split.yaml`, `random_split.yaml`, `resampling.yaml`
6. Delete `scripts/dro_k2_waterbirds.py` (superseded, no value to keep)
7. Verify the running CelebA experiment still has all its imports — none of the
   above is touched by it.

**Test:** `uv run python -c "from scripts.aggregation_ablation import *"` and
`uv run python scripts/dro_discovered.py --dataset cmnist --seeds 42 --device cpu`
both work.

### Phase 1 — Prune `evaluate.py`

1. Move the four stability-score functions (`compute_stability_scores`,
   `disagreement_stability_scores`, `adaptive_stability_scores`,
   `evaluate_stability_discrimination`) into `legacy/stability.py`
2. Delete `compute_assignment_correlation` and `compute_assignment_correlation_multi`
3. If `evaluate.py` becomes empty, delete it and remove its import from `train.py`

**Test:** `pytest tests/` — at this point only the assignment-correlation tests
should fail. We'll fix them in phase 4.

### Phase 2 — Prune `configs/config.yaml`

1. Delete the 18 obsolete fields listed above
2. Keep `discovery_criterion` only if you want to leave the door open for
   re-running entropy/counterfactual ablations; otherwise delete
3. Update inline comments to reflect the simplified config

**Test:** `uv run python run.py method=erm dataset=cmnist training.epochs=1` and
`uv run python run.py method=discovered_split dataset=cmnist training.epochs=1`.

### Phase 3 — Wait for CelebA to finish, THEN refactor `train.py` and `models.py`

This is the big one. Do not start until the CelebA run completes.

1. Delete `train_random_split`, `train_oracle_split`, `train_resampling`,
   `train_adversarial_split`, `train_adversarial_split_multi`, and `symmetric_kl`
   from `train.py`
2. Delete `SplitMLP` and `MultiHeadMLP` from `models.py`
3. Prune `discover_environments`:
   - Keep only the `loss` criterion
   - Keep only the median-split assignment path (`q == 0.5`, `upweight > 0`)
   - Keep the discovery diagnostics (correlation, env sizes, permutation test)
   - Remove the `criterion` config dependency from the function signature
   - Result: ~100 lines of focused logic
4. Simplify `train_discovered_split`:
   - Delete `balanced_sampling`, `env_mixup`, `training_noise`,
     `env_balance` auto-switch, the DRO group-weight tracking
   - Pure V-REx training with adaptive λ from `discovery_metrics["adaptive/reliability"]`
   - Add SWA model selection (window=5 anchored at best-by-val epoch)
5. Update `run.py` to remove the dispatch for the deleted methods
6. Verify with a 1-epoch end-to-end run on each kept dataset

**Test:** `uv run python run.py method=discovered_split dataset=waterbirds training.epochs=2`
should produce a final WGA in the right ballpark (>70 %).

### Phase 4 — Tests

1. Delete the ~40 obsolete test functions from `tests/test_train.py`
2. Add new tests:
   - `test_discover_environments_loss_split` — verifies env A and env B differ
     in correlation on a tiny CMNIST
   - `test_train_discovered_split_with_swa` — runs 2 epochs, checks the SWA
     model has lower training loss than a single-epoch checkpoint
   - `test_train_discovered_split_returns_metrics` — keys are present, all finite
   - `test_aggregation_ablation_smoke` — runs aggregation_ablation.py for 1 seed
     1 epoch, checks the result dict structure
3. Run `pytest tests/ -q` and ensure 100 % pass

### Phase 5 — Reproducibility scaffolding (the user explicitly cares about this)

Per `CLAUDE.md`: *one command per paper figure / table.*

Add a `Makefile` (or a `scripts/run_paper.sh` if you prefer) with targets:

```make
.PHONY: table_main table_aggregation table_discovery_ablation \
        table_k_detection figure_swa_window all

table_main:        ## Headline table: ours vs baselines on 4 datasets, 5 seeds
	uv run python scripts/run_main_table.py --seeds 42,123,789,2024,7

table_aggregation: ## Aggregator head-to-head (ERM/V-REx/DRO) on Waterbirds + CelebA
	uv run python scripts/aggregation_ablation.py --dataset waterbirds --seeds 42,123,789
	uv run python scripts/aggregation_ablation.py --dataset celeba     --seeds 42,123,789

table_discovery_ablation: ## Loss-averaging vs single-epoch, with vs without SWA, etc.
	uv run python scripts/ingredient_ablation.py --dataset waterbirds --seeds 42,123,789

table_k_detection:
	uv run python scripts/k_detection.py --dataset cmnist
	uv run python scripts/k_detection.py --dataset multi_cmnist
	uv run python scripts/k_detection.py --dataset tadf
	uv run python scripts/k_detection.py --dataset waterbirds

all: table_main table_aggregation table_discovery_ablation table_k_detection
```

The two new scripts (`run_main_table.py`, `ingredient_ablation.py`) need to be
written in this phase. The rest already exist.

### Phase 6 — README and documentation

1. Rewrite `README.md` to point at the final method, not the historical idea.
2. Add a section "How to reproduce paper tables" that lists `make table_main`
   etc.
3. Add a `CITATION.bib` placeholder.
4. Move the historical CLAUDE.md content (the original adversarial-split
   research idea) into a section "Project history" or just leave it — it's
   useful context for future-us, even if the final method differs.

## What NOT to do

- **Do not refactor while the CelebA run is in flight.** The risk of import
  paths breaking the running process is low (Python caches the source) but the
  risk of *us getting confused* about what's the current state is high.
- **Do not delete tests without writing replacements.** If we delete the
  adversarial-split tests in phase 4, we should add the new ones in the same
  commit so test coverage doesn't temporarily drop to nothing.
- **Do not use this as an excuse to add features.** The goal is to make what's
  already working easier to understand. Save K-detection integration into the
  training pipeline, the new auto-step-size formula, and the WILDS extension
  for *after* this refactor lands.
- **Do not move files without `git mv`.** Preserve history.
- **Do not delete the legacy folder.** It's the safety net if a reviewer asks
  about an experiment we didn't include in the paper.

## Risk assessment

| Phase | Risk | Mitigation |
|---|---|---|
| 0 (legacy/) | Zero — pure file moves, no code touched | Verify imports after each move |
| 1 (evaluate.py) | Low — only obsolete tests will break | Skip the failing tests in phase 1, fix in phase 4 |
| 2 (config.yaml) | Low — runs may break if a deleted field is accessed | Run a smoke test after the prune |
| 3 (train.py + models.py) | **Medium** — this is the bulk of the work | Do it in a separate branch; merge only after phase-3 smoke tests pass on every dataset |
| 4 (tests) | Low | Run `pytest -q` after every deletion |
| 5 (Makefile) | Low — additive, no deletion | — |
| 6 (README) | Zero | — |

## Effort estimate

Roughly half a working day if done in one sitting. The actual code-moving is
mechanical; most of the time goes to running smoke tests after each phase to
make sure nothing broke. The CelebA-running constraint adds a wait of ~10 hours
between phase 2 and phase 3.

## Success criteria

After the refactor:

1. `wc -l train.py` ≤ 1000
2. `wc -l models.py` ≤ 120
3. `pytest tests/ -q` passes 100 %
4. `make table_main` reproduces the headline numbers
5. A new reader can find the final method's training loop within 30 seconds of
   opening `train.py`
6. The `legacy/` folder has a README explaining what's in it and which paper
   experiments those files contributed to (with logbook section references)
