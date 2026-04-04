# Experiment Logbook

## Method: Discovered Environment Split + V-REx

### Core pipeline
1. Train throw-away ERM for K epochs (learns the spurious shortcut)
2. Score each training example by cross-entropy loss (high loss = minority group)
3. Median split into env A (low loss, spurious-aligned) and env B (high loss, spurious-misaligned)
4. Upweight high-loss examples: `w_i = 1 + factor * (loss_i / max_loss)`
5. Train new model with V-REx penalty: `(loss_A - loss_B)^2` on weighted environment losses
6. Early stop on risk variance (patience-based, with warmup)

### Key design choices and why
- **Single model, not two heads:** SplitMLP caused domain blindness (each head only saw its own env). Single MLP with V-REx forces the one model to find features that work in both environments.
- **Loss scoring, not entropy:** At high spurious correlation, the ERM is confidently *wrong* on minority examples (low entropy, high loss). Entropy misses them; loss catches them.
- **Upweighting, not exclusion:** Excluding the middle 70% of examples (quantile split) loses shape signal. Upweighting keeps all data while amplifying the minority group in the V-REx penalty.
- **Dataset-agnostic:** No assumption about where the spurious feature lives. Only uses the ERM's own mistakes as signal.

---

## CMNIST Experiments

### Dataset
- Binary classification: digits 0-4 vs 5-9
- Spurious feature: colour (red=0, green=1), correlated with label
- Configurable correlation: train_correlation controls spurious strength
- OOD test: flipped correlation (0.1 when train is 0.9)
- 60K train, 10K test, label_noise=0.25

### Baselines (all at default config unless noted)

| Method | train_corr | ID acc | OOD acc | Notes |
|--------|-----------|--------|---------|-------|
| ERM | 0.7 | ~72% | 55.6% | Learns colour, degrades OOD |
| ERM | 0.9 | ~90% | ~22% | Heavily colour-dependent |

### Discovered split results at corr=0.7

| Config | OOD peak | OOD final (ep29) | Notes |
|--------|----------|-----------------|-------|
| lambda=1, no anneal | 64.2% (ep6) | 50.2% | Works but overfits |
| lambda=10, no anneal | **71.5%** (ep0) | 53.7% | Best absolute OOD |
| lambda=100, no anneal | 71.4% (ep0) | 54.3% | Too strong, unstable |
| lambda=10, anneal 10x | 72.7% (ep3) | 55.4% | Annealing helps final |

**Key finding:** +16pp over ERM at peak (71.5% vs 55.6%).

### Discovered split results at corr=0.9

#### Splitting strategy comparison (lambda=10, anneal 10x)

| Strategy | corr_A | corr_B | OOD peak | Problem |
|----------|--------|--------|----------|---------|
| Entropy, median (q=0.5) | 0.945 | 0.654 | 15% | corr_B too high |
| Loss, median (q=0.5) | 1.0 | 0.60 | 13% | corr_B too high |
| Loss, q=0.15 extremes | 1.0 | -0.33 | 74% | Learns anti-colour |
| Loss, q=0.20 extremes | 1.0 | 0.00 | 55% | Colour-invariant but random |
| Reweight=10, median | 1.0 | 0.60 | 13% | Corr_B too high for any reweight |
| Reweight=50, median | 1.0 | 0.60 | 15% | Same |

**Key finding:** At corr=0.9, median splits can't create different-enough environments. Extreme quantiles find great environments but the model can't learn shape from only extreme examples.

#### Loss-based upweighting (the breakthrough)

| Upweight | Discovery | OOD peak | ID at peak | Notes |
|----------|-----------|----------|------------|-------|
| 0 | loss, median | ~13% | ~89% | V-REx has no signal |
| 20 | loss, median | 61.5% (ep11) | 57% | First time beating ERM |
| 50 | loss, median | **74.3%** (ep7) | 53% | Best result |

**Key finding:** Upweighting makes the minority group dominate the *weighted* env B loss, even though the *unweighted* env B has corr=0.60. The effective contrast is in the loss contributions, not the environment labels. +52pp over ERM (74.3% vs 22%).

**Limitation:** OOD peaks early then degrades. Early stopping on risk variance (patience=5, warmup=5 epochs) auto-selects epoch 5 with 70.4% OOD.

---

## Waterbirds Experiments

### Dataset
- Binary classification: waterbird (1) vs landbird (0)
- Spurious feature: background (water=1, land=0), 95% correlated with label
- Standard splits: 4,795 train / 1,199 val / 5,794 test
- Training group composition:
  - landbird + land: 3,498 (73%)
  - waterbird + water: 1,057 (22%)
  - landbird + water: 184 (4%)
  - waterbird + land: 56 (1%) -- hardest group
- Val/test sets balanced across groups
- Key metric: **worst-group accuracy** (accuracy on the hardest group)
- Model: ResNet-50 pretrained on ImageNet, fine-tuned

### Results

| Method | Disc. epochs | Upweight | Best worst-group | At epoch | Test acc | Notes |
|--------|-------------|----------|-----------------|----------|----------|-------|
| ERM | -- | -- | 80.7% | 5 | 90.7% | Baseline |
| Discovered V-REx | 3 | 20 | 81.6% | 6 | 90.1% | Marginal improvement |
| Discovered V-REx | 5 | 50 | **85.4%** | 1 | 88.7% | Best worst-group, early peak |
| Discovered V-REx | 10 | 50 | 84.1% | 8 | 90.9% | More stable, still strong |

### Discovery diagnostics

| Disc. epochs | assignment_color_abs_corr | corr_A | corr_B | Notes |
|-------------|--------------------------|--------|--------|-------|
| 3 | 0.51 | 0.947 | 0.810 | Weak contrast |
| 5 | 0.37 | 0.986 | 0.801 | Sharper ERM, better scoring |
| 10 | 0.30 | 0.987 | 0.796 | Diminishing returns on contrast |

**Key finding:** Method generalises from synthetic (CMNIST) to real images (Waterbirds). +4.7pp worst-group accuracy over ERM (85.4% vs 80.7%) without any knowledge of the spurious feature.

**Note:** Results are noisy -- worst-group oscillates heavily across epochs. Early stopping is essential. Multiple seeds needed for reliable comparisons.

---

## Open Questions and TODO

### For ablations (NeurIPS)
- [ ] Run all experiments with 5+ random seeds and report mean +/- std
- [ ] Ablate discovery_epochs systematically: {1, 3, 5, 10, 20}
- [ ] Ablate upweight systematically: {0, 5, 10, 20, 50, 100}
- [ ] Ablate lambda_disagree: {1, 5, 10, 50, 100}
- [ ] Ablate early_stop_patience: {3, 5, 10, none}
- [ ] Compare to published baselines: Group DRO, JTT, GEORGE, CnC, DFR
- [ ] CelebA dataset (hair colour with gender as spurious feature)
- [ ] Chemistry datasets (scaffold splits, property-based splits)

### Methodological questions
- [ ] Why does upweighting work better than quantile exclusion? Formal analysis of effective environment correlation under upweighting
- [ ] Can we use the discovery ERM's *features* (not just loss) for better environment construction? e.g., cluster in feature space
- [ ] Is there an optimal upweight that can be estimated from the data? (e.g., based on the loss distribution's shape)
- [ ] Connection to JTT (Liu et al. 2021): our method is JTT + V-REx. What does V-REx add over simple upweighting?

### For the paper
- [ ] Frame as "environment discovery via ERM residuals + V-REx"
- [ ] Position relative to JTT, GEORGE, CnC, DFR (all post-hoc correction methods)
- [ ] Our contribution: V-REx on discovered environments with loss-based upweighting
- [ ] Theoretical angle: connection to algorithmic stability (Bousquet & Elisseeff 2002)
- [ ] Stability scores as a byproduct: KL(head_A || head_B) as epistemic uncertainty

---

## Architecture and Hyperparameter Notes

### CMNIST
- Model: 2-layer MLP, hidden_dim=256
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Batch size: 256
- Gradient clipping: max_norm=1.0

### Waterbirds
- Model: ResNet-50 (ImageNet pretrained), fine-tuned
- Discovery model: ResNet-50 with frozen backbone (only linear head trained)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Batch size: 64
- Transforms: Resize(256), RandomResizedCrop(224), RandomHorizontalFlip, ImageNet normalise
- Gradient clipping: max_norm=1.0
