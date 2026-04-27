# α-investigation: partition-sensitivity as epistemic uncertainty (null result)

**Status:** null result, archived 2026-04-24.

## Hypothesis

Disagreement between models trained on *disjoint* partitions of the training
data is a novel epistemic-uncertainty signal — "the model's prediction
depends on which training data it saw" — and a tractable proxy for
algorithmic stability (Bousquet & Elisseeff, 2002). We tested whether it
beats softmax entropy and deep-ensemble disagreement on (a) test-time
uncertainty estimation and (b) label-noise attribution.

## What we ran

- `partition_pair_train.py` — trains K models on either disjoint partitions,
  full-data ensembles, or a single ERM. Saves per-example softmax probs on
  id_test, ood_test, and optionally the full training set.
- `stability_eval.py` — computes partition / ensemble / entropy signals and
  scores them on selective prediction AUC, OOD-detection AUROC, and
  per-group selective AUC.
- `attribution_eval.py` — tests whether partition-pair disagreement on the
  training set identifies mislabeled examples. Compares against loss_in,
  entropy_in, conf_gap.

## Results (negative)

| Task | Dataset | partition_shift | best baseline |
|---|---|---|---|
| Selective AUC on OOD (Waterbirds, single seed) | Waterbirds | 0.0197 | 0.0135 (ensemble) |
| OOD-detection AUROC (Waterbirds) | Waterbirds | 0.515 | 0.503 (ensemble) |
| Label-noise AUROC (CMNIST, corr=0.9, noise=0.25) | CMNIST | 0.487 | 0.792 (loss_in) |
| Label-noise AUROC (CMNIST, corr=0.5, noise=0.20) | CMNIST | 0.515 | 0.987 (loss_in) |

r(partition, ensemble) on Waterbirds OOD = 0.27 — partition is a *different*
signal, but it is not a *better* signal on any task we tested.

## Why it fails

- **Test-time uncertainty:** on tasks where deep ensembles do disagree
  (Waterbirds, ResNet50), partition disagreement is decorrelated from
  ensemble disagreement but uniformly worse at ranking prediction
  confidence. On tasks where models collapse to identical solutions (CMNIST
  with strong spurious feature), all disagreement signals are ~0.
- **Label-noise attribution:** modern training (SGD + weight decay + early
  stopping) prevents memorization of noisy examples, so in-model and
  out-model predictions agree even for mislabeled examples. Classical
  per-example loss is near-perfect (AUROC ≥ 0.99) because the in-model
  predicts the *true* label against the flipped training label, which is a
  direct signal that partition-shift cannot exploit.

## What this means for the paper

**Updated 2026-04-24 — direction revived at smaller scale.**

The partition-sensitivity idea fails on the benchmarks we tested, BUT those
benchmarks are exactly where algorithmic-stability theory predicts the
effect should vanish:
  - Waterbirds (N=4.8K + ImageNet-pretrained ResNet50): effective sample
    complexity is tiny, β = O(1/N) ~ noise floor.
  - CMNIST (N=60K, simple MLP): models collapse to identical shortcut
    solutions; no room for meaningful between-partition variance.

Algorithmic stability scales as ~1/N. On small-data scientific ML
(BACE 968, BBBP 1.3K, TADF 1K, MOF 862–1.3K), β should be ~100× larger.
The phenomenon we failed to detect on Waterbirds should be clearly
visible there.

The scripts in this directory (`partition_pair_train.py`,
`stability_eval.py`, `attribution_eval.py`) have been **moved back to
`scripts/`** and will be used for the new target: a measurement /
benchmark paper on *sample fragility* in small-data scientific ML. This
README stays as context for the negative-result pivot on large benchmarks.
