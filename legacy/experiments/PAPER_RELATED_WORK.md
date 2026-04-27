# Background and Related Work — draft

This is a working draft of the paper's background and related-work sections. It is
written in markdown for easy editing; convert to LaTeX at submission time.

The structure follows what would become **Section 2 (Background)** and
**Section 6 (Related Work)** of the paper. Citation placeholders are written as
`[Author Year]` — fill in BibTeX keys at submission time.

---

## 2. Background

### 2.1 Setting and notation

We consider supervised classification with a training set
$\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$ drawn from a distribution
$P_{\text{train}}$, and a test distribution $P_{\text{test}}$ which may differ
from $P_{\text{train}}$ in systematic ways. Let $f_\theta : \mathcal{X} \to
\Delta^{C-1}$ be a classifier parameterised by $\theta$, where $C$ is the number
of classes. Empirical Risk Minimisation (ERM) optimises

$$
\theta^\star_{\text{ERM}} = \arg\min_\theta \mathbb{E}_{(x, y) \sim P_{\text{train}}} \big[ \ell(f_\theta(x), y) \big],
$$

where $\ell$ is the cross-entropy loss. ERM works well when $P_{\text{train}} =
P_{\text{test}}$, but its robustness to distribution shift is poor when the
training data contains spurious correlations [Geirhos2020, Sagawa2020].

### 2.2 Spurious correlations and worst-group accuracy

Following the now-standard formulation [Sagawa2020], we assume each example has
an unobserved (or, in the oracle setting, observed) **group attribute** $g(x, y)
\in \{1, \dots, K\}$ which captures a sub-population structure. A *spurious
correlation* arises when the marginal distribution of $g$ in $P_{\text{train}}$
is highly imbalanced — specifically, when the *minority groups* (those rare in
training) are systematically harder for an ERM-trained model than the *majority
groups*.

The standard robustness metric is **worst-group accuracy (WGA)**:

$$
\text{WGA}(f_\theta) = \min_{g \in \{1, \dots, K\}} \mathbb{E}\big[\mathbf{1}\{f_\theta(x) = y\} \mid g(x, y) = g\big].
$$

WGA is the metric reported in essentially all prior work on this benchmark family
(Waterbirds, CelebA, CivilComments). A method that improves average accuracy by
exploiting the spurious shortcut while letting WGA collapse is *not* considered
to be solving the problem.

### 2.3 Three regimes of group supervision

Methods for improving WGA differ in **how much group information they use**:

| Regime | Group labels at training? | Group labels at validation? | Examples |
|---|---|---|---|
| **Full oracle** | Yes | Yes | Group DRO [Sagawa2020], IRM [Arjovsky2019], V-REx [Krueger2021] |
| **Validation oracle** | No | Yes (for model selection) | DFR [Kirichenko2023], AFR [Qiu2023] |
| **No groups at all** | No | No | JTT [Liu2021], LfF [Nam2020], GEORGE [Sohoni2020], EIIL [Creager2021], **ours** |

The "no groups at all" setting is the practical setting in most real-world
deployments — group labels are expensive (medical imaging, chemistry) or
impossible (commercial-scale CTR datasets). Methods in this regime must
*discover* the group structure from the training data itself, then use that
discovered structure to drive a robust training objective.

### 2.4 Distributionally robust optimisation (DRO)

The general DRO formulation is

$$
\theta^\star_{\text{DRO}} = \arg\min_\theta \max_{Q \in \mathcal{U}} \mathbb{E}_{(x,y) \sim Q} \big[ \ell(f_\theta(x), y) \big],
$$

where $\mathcal{U}$ is an *uncertainty set* of distributions close to
$P_{\text{train}}$. Different choices of $\mathcal{U}$ recover different
methods [Duchi2018]:

- **CVaR / $\alpha$-quantile DRO:** $\mathcal{U} = \{Q : Q \le \alpha^{-1} P_{\text{train}}\}$, optimising the worst $\alpha$-fraction of examples.
- **$\chi^2$-DRO:** $\mathcal{U} = \{Q : D_{\chi^2}(Q \,\|\, P_{\text{train}}) \le \rho\}$. As $\rho \to 0$ this is ERM; as $\rho \to \infty$ it approaches worst-group DRO.
- **Group DRO:** $\mathcal{U}$ is the simplex over $K$ predefined groups; the inner $\max$ is solved by exponentiated gradient on the group weights [Sagawa2020].
- **V-REx:** adds a penalty $\lambda \cdot \mathrm{Var}_g[L_g]$ to the average loss [Krueger2021]. Equivalent to $\chi^2$-DRO at small radius [Krueger2021 Appendix].

In our experiments we treat **V-REx** and **Group DRO** as the two canonical
training objectives at opposite ends of this spectrum. Our finding (Section 4)
is that the *choice* between them matters less than the **discovery of which
examples form the worst group** — and that discovery from training-loss residuals
suffices.

### 2.5 The role of validation in model selection

A subtle but practically critical point: every method in Table 2.3 requires
*some* form of model selection — choosing the best epoch's checkpoint, choosing
$\lambda$ for an invariance penalty, choosing the upweighting factor. The
validation signal used for this selection materially affects which method
appears to "win" in benchmarks.

- **Balanced validation** (Waterbirds, CelebA — both have HF/Sagawa-curated
  validation splits with comparable group sizes) gives WGA-on-val a high
  signal-to-noise ratio. Invariance penalties and SWA-style checkpoint
  averaging both shine in this regime.
- **Imbalanced validation** (synthetic CMNIST variants, noisy real datasets
  like our chemistry benchmark) gives WGA-on-val a *poor* signal-to-noise
  ratio. The minority of the val set is small, so per-epoch WGA estimates are
  noisy, and any method that relies on per-epoch model selection (especially
  invariance penalties whose effect compounds over epochs) suffers.

This distinction is rarely made explicit in the literature, but it explains a
substantial portion of the apparent disagreement between published results.
We return to it in Section 4 (Experiments) and Section 5 (Discussion).

---

## 6. Related Work

We organise prior work along three axes: **(a) what objective is optimised**,
**(b) what supervision is used**, and **(c) how the method scales to settings
without group labels**.

### 6.1 Methods that need group labels at training time

The classical approach to spurious-correlation robustness uses *full group
supervision* during training:

- **Group DRO** [Sagawa2020] minimises the worst per-group loss via
  exponentiated-gradient updates on group weights. With ground-truth groups it
  is competitive with the strongest published methods on Waterbirds (~80–85 %
  WGA) and CelebA (~87 %). It is the de facto upper bound for "what can you do
  if you have group labels?".
- **IRM** [Arjovsky2019] introduces an invariance penalty that encourages the
  classifier to be optimal on every training environment simultaneously. It
  requires environments to be specified upfront. Empirical results have been
  mixed [Rosenfeld2021] but the framework remains influential.
- **V-REx** [Krueger2021] is a simpler variant of IRM that penalises the
  *variance* of per-environment losses. Often outperforms IRM in practice and
  is easier to implement. We use V-REx as the invariance objective in our final
  method (with environments *discovered*, not given).
- **Bayesian IRM** [Lin2022a], **CORAL** [Sun2016], **DRO-ish variants** such as
  **MMD-DRO** [Staib2019] all share the requirement of pre-specified groups or
  domains.

**Our position:** these methods define what "good" looks like once you have
groups. Our contribution is to *discover* groups well enough that V-REx
(Section 4) on the discovered groups approaches the WGA of Group DRO on
oracle groups.

### 6.2 Methods that need group labels only at validation

A second line of work uses group labels only for **model selection** or for a
small **balanced validation subset**:

- **DFR (Deep Feature Reweighting)** [Kirichenko2023] freezes a feature
  extractor pretrained with ERM and retrains only the final linear layer on a
  *group-balanced* subset of validation data. It is the strongest published
  method on Waterbirds (88.4 % WGA) and CelebA (~86 %). It is also the closest
  benchmark for "how much does access to group-labelled validation buy you?".
- **AFR (Automatic Feature Reweighting)** [Qiu2023] is a follow-up that
  discovers minorities from the validation set itself rather than requiring
  group labels.

**Our position:** DFR is our main strong baseline. We show that loss-based
discovery on training data plus V-REx + SWA reaches **86.75 ± 1.39 WGA on
Waterbirds** — within ~2 pp of DFR — *without using any group labels at all*,
neither at training nor at validation. The remaining gap to DFR is the price of
not having a balanced val set; we view closing it as future work.

### 6.3 Methods that operate without any group labels

This is the regime we operate in. Prior work falls into a few clusters:

#### 6.3.1 Two-stage error-based methods

- **Just Train Twice (JTT)** [Liu2021] is the direct ancestor of our method.
  JTT trains an ERM, identifies the examples it *misclassifies* after a few
  epochs, then retrains a second ERM with those misclassified examples
  upweighted by a fixed factor $\lambda$. JTT reports ~81 % WGA on Waterbirds
  and ~81 % on CelebA. The discovery signal is binary (misclassified
  vs. correct), and the published implementation uses single-epoch error
  identification.
- **Learning from Failure (LfF)** [Nam2020] trains two networks: a "biased"
  network with a high-momentum focus on easy examples, and a "debiased"
  network with high focal-loss weighting on examples the biased one gets
  wrong. LfF is more architecturally intricate but conceptually similar to
  JTT.

**Our positioning vs JTT:** we use the *same* core insight (loss/error-based
discovery → upweighting) but make three changes that materially improve the
final WGA and its variance:

1. **Loss averaging across discovery epochs.** We average per-example losses
   over the last few discovery epochs rather than using a single snapshot.
   This tightens minority-group identification and reduces seed variance from
   $\pm 7$ to $\pm 1.4$ on CMNIST.
2. **V-REx variance penalty on the discovered split.** JTT relies on
   upweighting alone. We add the V-REx penalty on the median-loss split,
   which we show in Section 4 contributes 4–6 pp on Waterbirds.
3. **SWA-anchored model selection.** Instead of picking the single best
   validation epoch, we weight-average a 5-epoch window centred on the best
   epoch. This contributes another ~2 pp on Waterbirds and dramatically
   reduces variance.

#### 6.3.2 Cluster-based pseudo-group discovery

- **GEORGE** [Sohoni2020] runs k-means clustering on the penultimate-layer
  features of an ERM-trained model to define pseudo-groups, then trains Group
  DRO on those clusters. GEORGE reports ~85–87 % on Waterbirds (with somewhat
  involved hyperparameter tuning of the clustering pipeline).
- **EIIL (Environment Inference for Invariant Learning)** [Creager2021]
  optimises environment assignments to *maximise* the IRM penalty, treating
  inference and learning as a saddle-point problem.
- **PGI (Predictive Group Invariance)** [Ahmed2021] discovers groups via a
  small auxiliary classifier.

**Our positioning:** all three methods use *feature-space* signals (clusters,
saddle-point invariance, auxiliary networks). We use *training-loss residuals*,
which require no extra hyperparameters, no auxiliary networks, no clustering,
and produce a single scalar per example. Our K-detection diagnostic
(Section 4.4) shows the loss histogram already carries the multi-modal
structure that clustering tries to recover.

#### 6.3.3 Simple balancing baselines

- **Idrissi et al. 2022** [Idrissi2022] is the most important reference for
  framing our contribution. They show that *plain class-balanced ERM* and
  *plain group-balanced ERM* are competitive with — and on several benchmarks
  *match* — Group DRO with full group labels. Their headline finding:
  > "Despite the relative simplicity of these baselines, we find that they
  > achieve competitive performance, and in many cases match or exceed those
  > of more complex worst-group robustness methods."

  Idrissi et al. require group labels for the balancing. They explicitly note
  this limitation and identify "discovering groups from training data" as the
  natural extension.

**Our position is the natural completion of the Idrissi et al. story.** They
showed that simple balancing matches DRO when groups are known. We show that
loss-based group discovery + V-REx + SWA matches DFR-level performance when
groups are *not* known. The relationship is:

$$
\underbrace{\text{Idrissi et al. 2022}}_{\text{simple methods + group labels}} \;\xrightarrow{\text{loss-based discovery}}\; \underbrace{\text{ours}}_{\text{simple methods, no labels}}
$$

#### 6.3.4 Contrastive and representation-learning approaches

- **Correct-N-Contrast (CnC)** [Zhang2022] uses a contrastive objective that
  pulls together examples of the same class but different ERM predictions.
- **SSA (Spread Spurious Attribute)** [Liu2023] learns to identify spurious
  attributes via a classifier trained on the model's prediction errors.

**Our position:** these methods solve essentially the same problem we do but
with significantly more machinery (contrastive losses, auxiliary classifiers,
multiple training stages). Our method requires only a discovery ERM, a median
split, an upweight factor, the V-REx penalty, and SWA — *all of which are
single-line additions to a standard training loop*. We show in Section 4 that
this minimal recipe matches the more elaborate methods.

### 6.4 Stochastic Weight Averaging (SWA)

SWA was introduced by [Izmailov2018] as a simple way to find flatter minima by
averaging the weights of several model checkpoints from late training. It is
typically used to improve generalisation on i.i.d. test data without an
explicit OOD motivation.

**Our use of SWA is non-standard in two ways:**

1. **Anchored at the best-by-validation epoch**, not the end of training. The
   late stages of training on imbalanced data often *degrade* worst-group
   accuracy as the model overfits the majority. Averaging the last $K$ epochs
   inherits that degradation. Anchoring the SWA window at the best-by-val
   epoch picks up the local stability around the best checkpoint and is
   robust to post-peak drift.
2. **As a worst-group regulariser, not a generalisation tool.** We show that
   on Waterbirds the SWA-anchored window adds ~2 pp WGA over best-by-val
   selection alone, and *more importantly* drops the seed-to-seed variance
   from $\pm 4.2$ to $\pm 1.4$.

We are not aware of prior work using SWA in this anchored, WGA-targeted way,
though the underlying idea is straightforward enough that it may exist in
unpublished code.

### 6.5 Algorithmic stability as the conceptual frame

The original motivation for this project was algorithmic stability
[Bousquet2002] — the property that a learning algorithm's output is insensitive
to small perturbations of the training set. Stability has classically been
*analysed* (used to derive generalisation bounds) rather than *optimised for*.
We hypothesised that worst-group accuracy is closely related to instability:
the predictions on minority-group examples are exactly those that flip when the
training composition shifts.

In the final version of our method, this connection is more conceptual than
load-bearing — V-REx and SWA both contribute to stability but are not derived
from a stability bound. We discuss this connection in Section 5 and view a
formal stability analysis of the discovered-environment V-REx as a natural
direction for future work.

### 6.6 Distribution-shift benchmarks and evaluation

- **Waterbirds** [Sagawa2020]: bird-type classification with image background
  as the spurious feature. ResNet-50 backbone. 4795 train / 1199 val / 5794
  test. The smallest worst group has 56 training examples.
- **CelebA Blond/Male** [Sagawa2020]: hair-colour classification with sex as
  the spurious feature. ResNet-50 backbone. 162 770 train / 19 962 val /
  19 867 test. The smallest worst group has 1 387 training examples.
- **WILDS suite** [Koh2021]: multi-modal benchmark for real-world distribution
  shift, including Camelyon17 (hospital shift), iWildCam (camera-trap shift),
  CivilComments (text demographic shift), FMoW (geographic + temporal shift).
  We do not yet evaluate on WILDS — extending the method there is the highest-
  priority follow-up after the present paper.
- **DomainBed** [Gulrajani2021]: domain-generalisation benchmark where train
  and test domains differ. A different shift type (domain rather than spurious
  correlation), and a useful sanity check that our method generalises beyond
  spurious-correlation benchmarks.
- **Synthetic CMNIST variants** are useful for ablations because they let us
  control the strength of the spurious correlation. We use them for the
  loss-averaging, K-detection, and DRO step-size ablations.
- **TADF chemistry** [our previous work] is included as an out-of-modality
  test: tabular molecular descriptors, an MLP backbone, and a *real* dataset
  (no synthetic confounds) where the spurious feature is a property of the
  molecules that happens to correlate with the target.

### 6.7 Permutation tests in machine learning

The use of permutation tests for assessing the significance of an ML
quantity has a long history [Ojala2010, Genovese2009]. They are commonly used
for *post-hoc analysis* — does my classifier do better than chance on this
test set? — rather than as part of the training pipeline.

**Our use is novel** to the best of our knowledge: we use a permutation test
on the discovered-environment risk variance to **gate the V-REx penalty during
training**. When the discovered environments are essentially noise (TADF, low
signal-to-perm ratio), the gate sets $\lambda_{\text{eff}} = 0$ and the method
degrades gracefully to ERM. When the environments are highly structured
(Waterbirds, CelebA, CMNIST), the gate is fully open and V-REx applies its
full penalty. The same statistic also tells the user *whether the method is
doing anything for this dataset* — a property we have not seen in prior work.

---

## 7. Where we sit in the landscape (one-paragraph summary)

> Methods for worst-group robustness without group labels (JTT, LfF, GEORGE,
> EIIL, CnC, SSA) all introduce non-trivial machinery — auxiliary networks,
> contrastive losses, clustering pipelines, multi-stage saddle-point training.
> Methods that *match* the strong group-labelled baselines (DFR) require
> group-balanced validation. Idrissi et al. (2022) showed that simple balancing
> matches Group DRO when groups are known. **We close the gap by showing that
> loss-averaged discovery + V-REx + SWA matches Group DRO on the *discovered*
> groups, with no group labels at training or validation, on two standard image
> benchmarks (Waterbirds 86.75 ± 1.39, CelebA TBD), and degrades gracefully on
> a chemistry benchmark where the spurious signal is too weak for invariance
> training to help. The recipe is six lines of additional training code on top
> of an ERM baseline.**

---

## TODO before submission

- [ ] Fill in BibTeX keys for every `[Author Year]` placeholder
- [ ] Add the actual CelebA WGA once the run finishes
- [ ] Add WILDS Camelyon17 or DomainBed PACS results to Section 6.6 if we
      decide to extend the experiments before submission
- [ ] Decide how much space to give the "validation balance principle" (Section
      2.5) — it could be either a paragraph in Background or its own
      mini-section in Discussion
- [ ] Write the brief subsection on K-detection-as-diagnostic and how it
      relates to GEORGE's clustering
- [ ] Consider whether to fold the algorithmic stability framing (Section 6.5)
      into the introduction as motivation, or keep it as a "conceptual frame"
      that we don't make load-bearing
- [ ] Add a comparison table summarising every method discussed in Section 6,
      with columns: needs train groups, needs val groups, complexity,
      Waterbirds WGA, CelebA WGA
