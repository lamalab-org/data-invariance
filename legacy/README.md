# `legacy/` — code from earlier experiments that did not make it into the paper

This directory holds work from the project's exploration phase that was either
**ablated and shown to underperform**, **superseded by simpler alternatives**, or
**deprioritised** after the method converged.

It exists for two reasons:

1. **Reviewer questions.** If a reviewer asks "did you try X?", the code is
   here, runnable, and the relevant experiments are documented in
   `experiments/LOGBOOK.md`.
2. **Future revisits.** Some of this work (especially the stability scores)
   might still be salvageable for an appendix or follow-up paper.

Nothing in this directory is on the critical path for reproducing the paper's
main tables. The active code is in `train.py`, `models.py`, `data.py`,
`evaluate.py`, `utils.py`, and `scripts/{dro_discovered, aggregation_ablation, k_detection}.py`.

---

## Contents

### `scripts/` — exploration scripts deprioritised after the stability-score work failed

| File | What it did | Why deprioritised |
|---|---|---|
| `evaluate_stability.py` | Computed multiple per-example stability scores (confidence, entropy, MC dropout, disagreement) and ROC for OOD-flip prediction | On natural data (Waterbirds, TADF) ERM's plain confidence already tracks difficulty better than our scores. Only beats ERM under strong synthetic confounding (CMNIST corr=0.9). |
| `resampling_stability.py` / `resampling_stability_id.py` | Bootstrapped subsets to measure prediction sensitivity to training data composition | Computationally expensive and the signal collapsed on Waterbirds because ERM's predictions are stable on the held-in distribution |
| `stability_chemistry.py` / `stability_waterbirds.py` | Per-dataset stability score evaluation pipelines | Same root cause |
| `validate_adaptive_scores.py` | Tested the "blend ERM and ours by reliability" idea | Worked on CMNIST, didn't help on natural data |

See `experiments/LOGBOOK.md` section "Stability Scores" (and the "What doesn't
work" subsection) for the full ablation history.

### `configs_method/` — Hydra method configs for methods that didn't work

| File | Method | Why removed |
|---|---|---|
| `adversarial_split.yaml` | The original learnable-partition adversarial split (the project's seed idea) | The adversary never reliably found the colour-correlated partition. Best CMNIST OOD was ~55 % vs the discovered-split method's ~70 %. The learned soft assignments did not converge to the spurious feature even with various initialisation, warmup, and entropy-bonus schemes. |
| `oracle_split.yaml` | Two-head model with the *ground-truth* spurious attribute as the partition | Used as the upper bound for the adversarial split. Now obsolete because the discovered-split method directly produces a per-example scalar weight, not a binary partition. |
| `random_split.yaml` | Two-head model with a random partition (V-REx-style baseline) | Same — superseded by single-model V-REx on loss-discovered environments. |
| `resampling.yaml` | Per-step random environments with V-REx | Variant of `random_split` with fresh batches; same outcome. |

The corresponding training functions (`train_adversarial_split`,
`train_adversarial_split_multi`, `train_oracle_split`, `train_random_split`,
`train_resampling`) are removed from `train.py` in phase 3 of the refactor.
They live in git history (`git log --diff-filter=D --summary -- train.py`).

The two model classes that supported them (`SplitMLP`, `MultiHeadMLP`) are
also removed in phase 3.

---

## Updates from the cross-sample-paper iteration (2026-04-23 → 04-26)

After we pivoted from the group-free pipeline paper to the cross-sample-fragility
method paper, additional code accumulated and was archived here.

### `predecessor_methods/`
Earlier iterations of the twin-network method, superseded by the unified
`scripts/cross_sample_train.py`.

- `partition_pair_train.py` — original partition-pair trainer.
- `twin_fragility_train.py` — second-iteration twin trainer with multiple
  partition modes (fixed/per_epoch/bootstrap/adversarial).
- `twin_analysis.py` — analysis script for the above.

### `broken_protocol/`
Scripts that ran against the old cross-sample protocol where `data_seed`
varied the test set itself (giving spurious ~50% churn). Results invalidated.

- `parse_v2.py`
- `aggregate_v2.py`

### `failed_attempts/`
Approaches tested rigorously but not yielding positive paper results.

- `meta_reptile_train.py` — meta-learning over hard-subset partitions.
  Marginal cross-sample stability gain at meaningful accuracy cost;
  superseded by `twin_indep` consistency training.
- `wd_analysis.py` — weight-decay sweep showing weight decay alone does
  *not* reduce cross-sample fragility.
- `joint_selective.py` — fragility-aware selective prediction; mild
  positive only.
- `aggregation_solution.py` — deployment-time bagging evaluation under
  the (later-fixed) broken protocol.
- `attribution_eval.py` — label-noise attribution via partition-pair
  disagreement (α direction). Negative on CMNIST.
- `stability_eval.py` — original α uncertainty evaluation.

### `alpha_investigation/`
The "partition-sensitivity as epistemic uncertainty" investigation (α).
Failed at small-benchmark scale; pivoted to method paper. See its
README for details.
