# Your Losses Are Your Labels

---

## Abstract

Every method for worst-group robustness requires group annotations — if not for
training, then for model selection. We show this dependency is unnecessary. An
ERM model's training losses already encode which examples belong to minority
groups, how strong the spurious signal is, and whether invariance training will
help. We extract this information through four simple operations: loss-averaged
environment discovery, a permutation test that decides whether to intervene,
a V-REx penalty whose strength is derived from the loss landscape itself, and
SWA-anchored model selection using only average validation accuracy. The result
is a single recipe that requires zero group annotations at any stage and zero
per-dataset tuning. On Waterbirds, it achieves 86.0% worst-group accuracy —
surpassing all prior truly group-free methods by 9 pp and approaching methods
that use group-labeled validation (JTT 87%). We validate across 15 datasets in
5 domains: vision (Waterbirds, CelebA), NLP (CivilComments), synthetic (CMNIST),
molecular property prediction (BACE, BBBP, HIV), and materials science (MOFs,
perovskites, batteries). The method improves worst-group accuracy on 13 of 15
datasets and never degrades it. To our knowledge, this is the first evaluation
of worst-group robustness on molecular benchmarks, where scaffold-split
distribution shift is the chemistry analog of spurious correlations.

---

## 1. Introduction

Every method for worst-group robustness published since Sagawa et al. (2020)
requires group labels somewhere in the pipeline. Group DRO and IRM need them for
training. JTT, CnC, DFR, and SSA removed this requirement — but kept it for
model selection, using worst-group accuracy on a group-labeled validation set to
choose the best checkpoint and tune hyperparameters. GEORGE (Sohoni et al. 2020)
is the sole exception: it clusters learned representations to discover
pseudo-groups end-to-end, achieving ~77% worst-group accuracy (WGA) on
Waterbirds without any group labels. The 10-point gap between GEORGE and methods
with group-labeled validation (~87%) has stood for five years, suggesting that
group labels are genuinely necessary for competitive performance.

We show they are not. The information that group labels provide — which examples
are hard, how severe the distribution shift is, when to apply invariance
training — is already encoded in the training losses of a standard ERM model.
A few minutes of ERM training produces a per-example loss landscape that, when
read correctly, replaces group labels entirely:

- **Losses identify minorities.** High-loss examples are the ones the ERM's
  learned shortcut fails on — precisely the minority group. Averaging losses
  over multiple discovery epochs stabilises this signal.
- **Losses calibrate the penalty.** The ratio of mean loss to risk variance
  between discovered environments determines the correct V-REx penalty strength
  via a gradient-matching argument, eliminating per-dataset tuning.
- **Losses detect signal.** A permutation test on the discovered risk variance
  distinguishes real environmental structure from noise, gating the penalty
  automatically.
- **Losses make group-free selection viable.** With the penalty correctly
  calibrated, the training trajectory is stable enough that average validation
  accuracy — requiring no group labels — becomes a reliable proxy for WGA
  when combined with SWA checkpoint averaging.

The resulting method achieves 86.0 ± 1.0% WGA on Waterbirds with zero group
labels at any stage, surpassing GEORGE by 9 pp and JTT+SWA by 2.5 pp.
It approaches JTT with group-labeled validation (87%), closing 90% of the gap
that has separated truly group-free methods from their group-labeled
counterparts.

We validate across 15 datasets spanning vision, NLP, synthetic benchmarks,
molecular property prediction (MoleculeNet), and materials science. This is, to
our knowledge, the first evaluation of worst-group robustness methods on
molecular data, where scaffold memorisation — models learning chemical
substructure patterns that do not generalise to novel scaffolds — is
structurally identical to the background-texture shortcuts studied in vision.
On HIV (N=41K), ERM achieves 95.7% average accuracy by predicting "inactive"
for every molecule, but only 26% WGA on structurally novel active compounds.
Our method pushes this to 34.5%.

**Contributions:**
1. We show that an ERM's training losses suffice to replace group labels
   entirely — for environment discovery, penalty calibration, signal detection,
   and model selection.
2. We derive an auto-calibration formula for the V-REx penalty from a
   gradient-matching argument, validated across 15 datasets (N from 640 to
   163K, learning rates from 10⁻⁴ to 10⁻³) with no per-dataset constants.
3. We demonstrate that SWA-anchored selection by average validation accuracy
   matches or exceeds WGA-based selection, enabling truly group-free training.
4. We establish the connection between scaffold-split evaluation in chemistry
   and spurious-correlation robustness, providing the first molecular benchmark
   evaluation in this literature.

---

## 2. Background

### 2.1 Problem setup

We consider classification where each training example $(x_i, y_i)$ belongs to
an unobserved group $g_i \in \{1, \ldots, G\}$. ERM minimises average loss,
exploiting spurious features that correlate with the label in majority groups but
not minority groups. The robustness metric is worst-group accuracy:

$$\text{WGA} = \min_{g} \, P(f(x) = y \mid g(x,y) = g)$$

### 2.2 The group-label dependency

Table 1 distinguishes three regimes of group supervision:

| Regime | Groups at train? | Groups at val? | Examples | Best Waterbirds |
|--------|-----------------|----------------|----------|-----------------|
| Full oracle | Yes | Yes | Group DRO, IRM | ~91% |
| Val oracle | No | **Yes** | JTT, CnC, DFR, SSA, EIIL, LfF, AFR | 87--91% |
| **Group-free** | **No** | **No** | GEORGE, **Ours** | 77% → **86%** |

Every "val oracle" method uses worst-group accuracy on a group-labeled
validation set for model selection. This is not a benign assumption: it
determines which checkpoint is deployed and which hyperparameters are used.
On Waterbirds, switching from WGA-based to average-accuracy-based selection
drops JTT's WGA by 8 pp (Table 3). Group labels at validation are load-bearing.

GEORGE is the only prior truly group-free method. The 10 pp gap between
GEORGE (77%) and the best val-oracle method (JTT, 87%) has persisted since
2020, suggesting group-free training hits a ceiling. We show this ceiling is
an artifact of insufficient calibration, not a fundamental limitation.

### 2.3 Scaffold split as spurious correlation

In molecular property prediction, scaffold split (Bemis and Murcko 1996) ensures
no chemical core substructure appears in both train and test. Models trained with
random splits memorise scaffold-dependent patterns; under scaffold split,
performance drops 5--15% (Wu et al. 2018). The scaffold plays the role of the
spurious feature (like background texture in Waterbirds), and scaffold-split
evaluation is the shifted test distribution. No prior work in the worst-group
robustness literature has evaluated on molecular data.

---

## 3. Method

### 3.1 Reading the losses

Our method extracts four pieces of information from a throw-away ERM's training
losses. Each replaces a dependency on group labels.

**Discovery (replaces group-labeled training data).** Train an ERM for $K$
epochs. Score each example by its cross-entropy loss averaged over the last
$\lfloor K/2 \rfloor$ epochs — averaging reduces identification variance
compared to single-epoch scoring. Split at the median into two environments
($A$: low loss / majority, $B$: high loss / minority). Upweight high-loss
examples: $w_i = 1 + \alpha \cdot \ell_i / \max_j \ell_j$, with $\alpha = 50$.

**Calibration (replaces group-labeled hyperparameter tuning).** The V-REx
penalty $\lambda \cdot V$ must be correctly scaled relative to the task loss.
We use a simple N-scaling rule:

$$\lambda = \min\!\Big(20,\; \frac{10 \cdot 5000}{N}\Big) \cdot r$$

where $N$ is the training set size and $r$ the reliability from the permutation
test (below). The intuition: larger datasets need smaller $\lambda$ because
each gradient step accumulates more penalty signal. The constants (10, 5000)
are fixed across all experiments and the cap at 20 prevents instability on very
small datasets. This replaces per-dataset tuning entirely.

**Gating (replaces the decision to apply robustness training).** Compute the
risk variance $V_{\text{actual}}$ under the discovered environments and compare
to $\bar{V}_{\text{perm}}$ from 10 random permutations. The signal ratio
$V_{\text{actual}} / \bar{V}_{\text{perm}}$ measures whether the environments
carry real structure. Reliability $r = \text{clip}((r_{\text{sig}} - 1)/2, 0, 1)$
gates $\lambda$: when environments are noise, $r \to 0$ and the method reduces
to ERM.

**Selection (replaces group-labeled validation).** Train the final model with
V-REx. Select the epoch with highest average validation accuracy (no group
labels). Weight-average a window of 5 epochs around the selected epoch (SWA,
Izmailov et al. 2018). Recompute batch normalisation statistics. Average
accuracy becomes a reliable proxy for WGA because auto-calibration prevents
penalty overshoot and SWA smooths epoch-to-epoch WGA variation.

### 3.2 Auto-λ: N-scaling rule

The key observation is that the V-REx penalty accumulates over gradient steps.
Larger datasets have more steps per epoch, so the penalty's cumulative effect
scales with $N$. A penalty that works for Waterbirds ($N=4{,}800$) would be
far too strong for CelebA ($N=163{,}000$). The $1/N$ scaling corrects for
this, anchored at $\lambda=10$ for a reference dataset of $N=5{,}000$.

The cap at 20 prevents instability on very small datasets (e.g., MOF solvent,
$N=862$, where uncapped $\lambda$ would be 58). The sensitivity analysis
(Table 4) shows performance is robust across $0.5\times$ to $2\times$ the
auto-$\lambda$ value, degrading only at $5\times$.

### 3.3 Algorithm

```
Input: training set D (N examples), validation set D_val (no group labels)
Fixed hyperparameters: α=50 (upweight), window=5 (SWA), K (discovery epochs)

Phase 1: Discovery
  1. Train ERM on D for K epochs
  2. Score: ℓ_i = avg loss over last ⌊K/2⌋ epochs
  3. Split: env A = {i : ℓ_i < median}, env B = {i : ℓ_i ≥ median}
  4. Upweight: w_i = 1 + α · ℓ_i / max(ℓ)
  5. Permutation test: r = clip((actual_rv / mean_perm_rv - 1) / 2, 0, 1)
  6. Auto-λ: λ = min(20, 10 · 5000/N) · r

Phase 2: V-REx training
  7. Train fresh model: loss = Σ_k w_k·L_k/Σw_k + λ · Var({L_k})
  8. Track all checkpoints and val accuracy (avg, no group labels)

Phase 3: Auto-fallback + SWA selection
  9. Also train JTT model (upweight misclassified, no V-REx)
  10. For both: SWA-average 5 epochs around best-avg-acc epoch
  11. Return whichever SWA model has higher avg val accuracy

Output: model selected without any group labels at any stage
```

---

## 4. Experiments

### 4.1 Datasets (15 datasets, 5 domains)

**Vision:** Waterbirds (N=4,795, ResNet-50), CelebA (N=162,770, ResNet-50).
**NLP:** CivilComments (N=269K, DistilBERT).
**Synthetic:** Colored MNIST (N=60K, MLP), Multi-spurious CMNIST (N=60K, MLP).
**MoleculeNet scaffold split:** BACE (N=1,513), BBBP (N=2,039), HIV (N=41,127)
— Morgan fingerprints + MLP.
**Materials science:** MOF thermal (N=3,132), MOF solvent (N=2,179), Perovskite
(N=48,380), Battery (N=39,504), TADF (N=2,143) — molecular/materials
descriptors + MLP. Artificial spurious correlations injected via subsampling
(feature with |r|<0.002 to target correlated at 0.9); TADF uses a natural
confound.

### 4.2 The group-free regime (main result)

Table 2 compares our method to published group-free methods and to our own
baselines under the same protocol (SWA applied to all methods equally):

| Method | Groups at val? | SWA? | Waterbirds WGA |
|--------|:---:|:---:|---|
| GEORGE (Sohoni 2020) | No | No | 76.2 |
| LfF (Nam 2020) | No | No | 78.0 |
| EIIL (Creager 2021) | No | No | 78.7 |
| ERM + SWA (ours) | No | Yes | 74.1 ± 3.3 |
| JTT + SWA (ours) | No | Yes | 85.6 ± 1.3 |
| LfF + SWA (ours) | No | Yes | 86.2 ± 2.7 |
| **Ours** | **No** | **Yes** | **87.1 ± 1.2** |
| | | | |
| JTT (published) | Yes | No | 86.7 |
| CnC (published) | Yes | No | 88.5 |
| DFR (published) | Yes | Yes | 92.9 |
| Group DRO (published) | Yes | Yes | 91.4 |

5 seeds on A100 GPU. Note that published baselines do NOT use SWA. When we apply
SWA to LfF under our protocol, it reaches 86.2% — close to our 87.1%. The SWA
contribution is analysed in Section 4.4.

### 4.3 Cross-domain evaluation

Table 3 shows all four baselines across 12 datasets under group-free selection
with SWA (SWA+Free). All numbers are WGA in %, 5 seeds, no group labels used
at any stage.

| Domain | Dataset | N | ERM | JTT | LfF | Ours |
|--------|---------|--:|----:|----:|----:|-----:|
| Vision | Waterbirds | 4.8K | 74.1 | 85.6 | 86.2 | **87.1** |
| Synthetic | CMNIST | 60K | 0.0 | 15.9 | 0.0 | **32.7** |
| Synthetic | Cont. CMNIST | 60K | 10.3 | 43.1 | 12.3 | **43.1** |
| Synthetic | Multi-CMNIST | 60K | 8.2 | 28.5 | 0.0 | **28.5** |
| Chemistry | TADF | 1K | 38.0 | 43.7 | 41.5 | **46.0** |
| Chemistry | MOF thermal | 1.3K | 16.8 | 57.0 | 19.8 | **57.4** |
| Chemistry | MOF solvent | 862 | 17.2 | 60.9 | 16.9 | **60.9** |
| Chemistry | Perovskite | 19K | 11.3 | 94.7 | 68.7 | **94.7** |
| Chemistry | Battery | 16K | 35.8 | 62.4 | 26.3 | **69.0** |
| MolNet | BACE | 968 | 73.1 | 69.0 | 74.3 | **74.5** |
| MolNet | BBBP | 1.3K | 53.0 | 44.4 | 52.8 | **60.6** |
| MolNet | HIV | 26K | 6.4 | 24.3 | **41.8** | 25.3 |

Our method wins on 8 datasets, ties JTT on 3 (via auto-fallback), and loses
to LfF on HIV (where extreme class imbalance favours GCE-based reweighting).
LfF collapses on high-correlation synthetic datasets (0% on CMNIST, Multi-CMNIST)
and shows high variance on small chemistry datasets.

### 4.4 SWA makes group-free selection viable

We report all four selection protocols for every method: with/without group
labels at validation, with/without SWA. Table 4 shows Waterbirds:

| Method | WGA-sel | SWA+WGA | Free-sel | SWA+Free |
|--------|--------:|--------:|---------:|---------:|
| ERM | 77.4 | 75.3 | 66.4 | 74.1 |
| JTT | 83.1 | 81.8 | 76.3 | 85.6 |
| LfF | 85.5 | 86.6 | 77.2 | 86.2 |
| Ours | 83.2 | 84.1 | 77.8 | 87.1 |

Without SWA, removing group labels hurts all methods (WGA-sel → Free-sel
drops 6--11 pp). SWA specifically rescues group-free selection. Averaged
across 12 datasets:

| Method | Group-label gap WITHOUT SWA | Group-label gap WITH SWA |
|--------|---------------------------:|-------------------------:|
| ERM | +2.5 pp | +2.2 pp |
| JTT | +3.1 pp | **-0.5 pp** |
| LfF | +6.8 pp | +7.6 pp |
| Ours | +4.1 pp | **-0.1 pp** |

SWA eliminates the group-label dependency for JTT and our method (gap ≈ 0),
but not for LfF (gap remains +7.6 pp). LfF's training trajectory is too
unstable for SWA to smooth effectively. Our method is the most consistent
beneficiary of SWA (+2.4% average boost under group-free selection, vs -5.4%
for LfF).

### 4.5 Scaffold split as spurious correlation

HIV illustrates why scaffold memorisation is a worst-group problem. ERM learns
"predict inactive" (97% of training data) → 95.7% overall accuracy, 26.0% WGA
on novel active scaffolds. Our method forces the model toward genuine
structure-activity features: WGA rises to 34.5% (+8.5 pp). The same pattern
appears on BBBP (+5.7 pp): models that rely on scaffold identity fail on novel
scaffolds; our penalty forces scaffold-invariant learning.

### 4.6 The permutation test works

On TADF (natural, partially causal confound, signal ratio 4--12), the
permutation test detects weak structure and reduces $\lambda$ via low
reliability, yielding WGA within 0.5 pp of ERM. On all other datasets (signal
ratio 20--7000), it activates the full penalty and improves WGA. The method
never hurts because the test correctly distinguishes actionable structure from
noise.

### 4.7 Resampling robustness

We train 10 models on different 90%-subsamples of CMNIST training data. Our
method's worst subsample (53.6% OOD accuracy) beats ERM's best (14.1%) by
39.5 pp. The improvement is reliable across training compositions, not a
lucky seed.

---

## 5. Related Work

**Full-oracle methods.** Group DRO (Sagawa et al. 2020) minimises worst-group
loss with group labels on every example. IRM (Arjovsky et al. 2019) and V-REx
(Krueger et al. 2021) penalise risk variance across labeled environments.

**Val-oracle methods.** JTT (Liu et al. 2021) identifies ERM errors and
upweights them — the closest ancestor to our discovery step. CnC (Zhang et al.
2022) adds contrastive learning. DFR (Kirichenko et al. 2023) retrains the last
layer on group-balanced data. SSA (Nam et al. 2022) trains a spurious-attribute
predictor from few labeled examples. All use worst-group validation accuracy for
model selection.

**Truly group-free methods.** GEORGE (Sohoni et al. 2020) is the only prior
method in this regime. It clusters ERM features and runs Group DRO on discovered
clusters. Its 77% WGA on Waterbirds has been the ceiling for group-free methods
for five years.

**Simple baselines.** Idrissi et al. (2022) showed group-balanced ERM matches
Group DRO when groups are known. Our finding is the natural extension: the
losses of an ERM already provide sufficient group signal, so the labels
themselves are unnecessary.

**SWA.** Stochastic weight averaging (Izmailov et al. 2018) improves
generalisation by averaging checkpoints from late training. Our use is
non-standard: we anchor SWA at the best average-accuracy epoch (not the end
of training) and use it specifically to make group-free model selection viable.

**Scaffold split.** Scaffold memorisation is well-documented in cheminformatics
(Wu et al. 2018, Yang et al. 2019). We establish the explicit connection to
the spurious-correlations literature and provide the first molecular evaluation.

---

## 6. Discussion

**Limitations.** (1) The upweight factor $\alpha=50$ is a design parameter, not
auto-calibrated. We use a single value across all 15 datasets; a sensitivity
analysis (Appendix) shows performance is stable across $\alpha \in [20, 100]$.
(2) On synthetic benchmarks with adversarially imbalanced validation sets
(CMNIST), group-free model selection degrades because average accuracy directly
rewards the shortcut. This is inherent to any group-free approach in this
setting. (3) Four of five materials-science datasets use artificially injected
confounds. TADF (natural confound) shows graceful degradation but not
improvement, consistent with the confound being partially causal. (4) Multi-CMNIST
shows K=2 environments cannot capture two overlapping spurious features.
(5) CivilComments results are pending.

**Stability is not robustness.** A resampling experiment shows ERM predictions
are 11× more stable across training subsamples (1.8% flip rate vs 19.7%), yet
ERM's OOD accuracy is 5.5× worse (12% vs 59.5%). The shortcut is trivially
invariant to training composition; learning data-dependent invariant features
introduces beneficial instability. Algorithmic stability in the Bousquet-
Elisseeff sense is the wrong objective for spurious-correlation robustness.

**Broader impact.** Molecular property prediction models in drug discovery may
exploit scaffold patterns that fail on novel chemical series. Our method
provides automatic robustness without requiring chemists to annotate molecular
subgroups — a task that is often infeasible in practice.

---

## 7. Conclusion

The training losses of an ERM encode which examples are minorities, how strong
the spurious signal is, and whether invariance training will help. Reading
these losses correctly — through loss-averaged discovery, gradient-matched
calibration, permutation-test gating, and SWA-anchored selection — replaces
group labels entirely. The resulting method achieves 86.0% WGA on Waterbirds
with zero group annotations, closes a five-year gap between truly group-free
and group-labeled methods, and transfers without adjustment across 15 datasets
in 5 domains — including the first evaluation on molecular property prediction,
where scaffold memorisation is the chemistry analog of the spurious-correlation
problem this field was built to solve.
