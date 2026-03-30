# Adversarial Data Splitting for Training Stability

## Research Idea

### Problem
Machine learning models are sensitive to the composition of their training data. Small changes in which examples are included can meaningfully change predictions. Existing work on out-of-distribution generalization (IRM, V-REx) requires pre-defined environments. Ensemble methods (bagging) capture disagreement but never feed it back into training. Consistency regularization penalizes sensitivity to input perturbations, not to training data composition.

### Core Contribution: Adversarial Data Splitting
Instead of randomly partitioning training data and penalizing disagreement, **learn the worst-case partition** — the split of your training set that maximizes prediction disagreement — and then regularize against that.

This is a minimax game:
- **Inner loop (adversary):** find the data partition that most destabilizes the model's predictions
- **Outer loop (model):** train to be robust to that worst-case split

### Why This Is New
- **IRM/V-REx:** require known environments. We discover them.
- **Bagging/ensembles:** average away disagreement. We optimize against it.
- **Consistency regularization:** penalizes input perturbations. We penalize training data composition sensitivity.
- **DRO:** robustifies against test distribution shift. We robustify against training data composition.

### Key Byproducts
1. **Instance-level stability scores:** For each test prediction, measure how much the worst-case data partition can destabilize it. This is a new form of epistemic uncertainty — not "the model is uncertain" but "the model's confidence is an artifact of which data it trained on."
2. **Data attribution maps:** The adversarial assignment weights reveal which training examples are "contentious" — their inclusion/exclusion flips predictions elsewhere.

### Paper Positioning
"Algorithmic stability has been studied as a property to analyze. We turn it into a training objective." Connects to uniform stability bounds (Bousquet & Elisseeff, 2002) but as something you minimize during training.

---

## Prior Work to Reference

| Method | What It Does | Key Limitation |
|--------|-------------|----------------|
| IRM (Arjovsky et al., 2019) | Learns invariant representations across known environments | Requires pre-defined environment labels |
| V-REx (Krueger et al., 2021) | Penalizes variance of risk across environments | Also requires known environments |
| Bayesian IRM (Lin et al., 2022) | Bayesian treatment of IRM | Same environment requirement |
| ZIN (Lin et al., 2022) | Jointly learns environment partition and invariant representation | Requires auxiliary information |
| Deep Ensembles (Lakshminarayanan et al., 2017) | Multiple models for uncertainty | Disagreement not fed back into training |
| Stability Training (Zheng et al., 2016) | Penalizes sensitivity to input perturbations | Wrong axis — input noise, not data composition |
| Co-teaching (Han et al., 2018) | Two networks, different data, for noise robustness | Different goal (label noise, not stability) |
| StARS (Liu et al., 2010) | Uses subsample stability for model selection | Post-hoc selection, not a training objective |

---

## Experimental Plan

### Ablation Conditions (This Is Critical)

| Condition | Description | Purpose |
|-----------|-------------|---------|
| **ERM** | Standard training, no splitting | Baseline |
| **Random Split + Penalty** | Random K=2 partition, KL penalty between heads | Essentially V-REx; shows random splits help a little |
| **Adversarial Split + Penalty** | Learned worst-case partition, KL penalty | **Our method** |
| **Oracle Split** | Split by known spurious feature (e.g., color in CMNIST) | Upper bound; verifies the right split helps |

The **key comparison** is Random vs Adversarial. If adversarial is substantially better, the paper works.

### Datasets (in order of complexity)

1. **Colored MNIST** (primary, for development and validation)
   - We control the spurious feature (color), so we can verify the method discovers it
   - Binary classification: digits 0-4 vs 5-9
   - Configurable color-label correlation
2. **Synthetic Gaussian data** (optional, for clean theoretical illustrations)
3. **Waterbirds or CelebA** (stretch goal, standard OOD benchmarks)

### Metrics

- **In-distribution accuracy** (should remain competitive with ERM)
- **Out-of-distribution accuracy** (should beat ERM and Random Split)
- **Stability score quality:** ROC curve for "does stability score predict which examples will flip under distribution shift?"
- **Attribution quality:** Mutual information between learned assignment s_i and the true spurious attribute (color)
- **Calibration:** ECE in-distribution and out-of-distribution

### Experiments to Validate Stability Scores

1. Train model with adversarial splitting on CMNIST (correlation=0.9)
2. At test time, compute stability score for each example: `KL(head_A(x) || head_B(x))`
3. Evaluate the same examples under shifted test set (correlation=0.1)
4. Check: do low-stability examples flip predictions more often? Make ROC curve.
5. Compare stability score discrimination against:
   - Softmax entropy (from single ERM model)
   - MC Dropout variance
   - Deep ensemble disagreement (2 independently trained models)

### Experiments to Validate Attribution / Learned Splits

1. After training with adversarial splitting on CMNIST:
   - Extract learned assignment weights s_i for each training example
   - Compute correlation between s_i and the color label
   - If high correlation → method discovered color as the spurious feature automatically
2. Visualize: scatter plot of training examples colored by learned s_i, check if it separates by color

---

## Architecture

### K=2 Adversarial Split Setup

```
                    ┌──────────┐
                    │  Shared  │
        x ────────>│ Backbone │────────┐
                    │  (MLP)   │        │
                    └──────────┘        │
                         │              │
                    shared features     │
                    ┌────┴────┐         │
                    │         │         │
               ┌────▼───┐ ┌──▼─────┐   │
               │ Head A │ │ Head B │   │
               └────┬───┘ └──┬─────┘   │
                    │        │         │
                 pred_A   pred_B       │
                    │        │         │
                    ▼        ▼         │
              KL(pred_A || pred_B)     │
              = disagreement loss      │
                                       │
    ┌──────────────────────────────────┘
    │
    │  s_i ∈ [0,1] per training example
    │  (adversary's learnable weights)
    │
    │  Head A loss = Σ s_i * CE(pred_A_i, y_i)
    │  Head B loss = Σ (1-s_i) * CE(pred_B_i, y_i)
    │
    │  Model minimizes: L_A + L_B + λ * KL(A||B)
    │  Adversary maximizes: KL(A||B) by optimizing s_i
```

### Training Loop (Pseudocode)

```
for each batch (x, y, idx):
    # Forward through shared backbone + both heads
    features = backbone(x)
    pred_A = head_A(features)
    pred_B = head_B(features)

    # Weighted losses using soft assignments
    s = sigmoid(assignment_logits[idx])
    loss_A = (s * cross_entropy(pred_A, y)).mean()
    loss_B = ((1-s) * cross_entropy(pred_B, y)).mean()

    # Disagreement
    disagreement = kl_div(pred_A, pred_B) + kl_div(pred_B, pred_A)

    # Model step: minimize task loss + minimize disagreement
    model_loss = loss_A + loss_B + lambda * disagreement
    model_loss.backward()
    model_optimizer.step()

    # Adversary step: maximize disagreement via assignment weights
    adv_loss = -disagreement  # negate because we maximize
    adv_loss.backward()
    assignment_optimizer.step()
```

---

## Step-by-Step Build Plan

Everything in PyTorch. Simple, clean, minimal boilerplate. Config-driven so experiments are easy to swap.

### Step 1: Colored MNIST Dataset

**File:** `data.py`

Implement `ColoredMNIST(Dataset)`:
- **`__init__` params:**
  - `env_correlation: float` — color-label correlation (0.9 for train, 0.1 for test)
  - `label_noise: float = 0.25` — probability of flipping the binary label
  - `split: str = "train"` — which MNIST split to use
  - `data_dir: str = "./data"` — where to download MNIST
- **Construction (all in `__init__`, deterministic once built):**
  1. Load MNIST via `torchvision.datasets.MNIST`
  2. Binarize labels: y = (digit >= 5).long()
  3. Flip labels with probability `label_noise` using `torch.bernoulli`
  4. Assign color: with probability `env_correlation`, color = label; else color = 1 - label
  5. Build 3-channel images: put grayscale values in red channel if color=0, green channel if color=1, zeros elsewhere
  6. Store: images (N, 3, 28, 28), labels (N,), colors (N,), as tensors
- **`__getitem__` returns:** `{"image": tensor, "label": int, "color": int, "index": int}`
- **`__len__`:** number of examples

**Test it:** Create train (corr=0.9) and test (corr=0.1). Print shapes, check label distribution, visualize a few examples.

### Step 2: Baseline ERM Training

**File:** `models.py`

Simple MLP:
```python
class MLP(nn.Module):
    # backbone: Linear(3*28*28, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU
    # head: Linear(hidden, 2)
    # hidden_dim = 256 as default
```

**File:** `train.py`

Clean training loop with:
- Config via a simple dataclass or dictionary (not argparse yet, keep it simple)
- Training function that takes model, dataloaders, config
- Evaluation function that reports accuracy on both in-distribution and OOD test sets
- Basic logging (print or simple list accumulation, no wandb yet)

**Config fields needed:**
```python
@dataclass
class Config:
    # Data
    train_correlation: float = 0.9
    test_correlation: float = 0.1
    label_noise: float = 0.25
    # Model
    hidden_dim: int = 256
    # Training
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 10
    seed: int = 42
    # Method: "erm", "random_split", "adversarial_split", "oracle_split"
    method: str = "erm"
    # Disagreement penalty
    lambda_disagree: float = 1.0
    # For adversarial split
    adv_lr: float = 1e-2
```

**Target:** Train ERM, report train acc, ID test acc (corr=0.9), OOD test acc (corr=0.1). ERM should do well ID and poorly OOD.

### Step 3: Random Split + Disagreement Penalty

**File:** Extend `models.py`

```python
class SplitMLP(nn.Module):
    # shared backbone: same as MLP but without the final head
    # head_a: Linear(hidden, 2)
    # head_b: Linear(hidden, 2)
    # forward returns: (pred_a, pred_b, features)
```

**File:** Extend `train.py`

- At dataset construction, randomly assign each example to subset 0 or 1
- Training loop computes weighted CE for each head + KL penalty
- Same evaluation as step 2

### Step 4: Adversarial Split (The Novel Method)

**File:** Extend `train.py`

- Add learnable `assignment_logits` parameter (one scalar per training example)
- Alternating optimization: model step then adversary step
- Key detail: use `sigmoid` on logits to get soft assignments in [0,1]
- Key detail: may need gradient clipping or learning rate tuning on the adversary

### Step 5: Stability Scores

**File:** `evaluate.py`

- After training, compute stability score for each test example
- `stability(x) = 0.5 * (KL(pA || pB) + KL(pB || pA))` (symmetric KL)
- Evaluate discrimination: bin examples by stability score, check OOD accuracy per bin
- ROC curve: stability score predicts "will this example's prediction change under OOD shift"
- Compare against softmax entropy baseline

### Step 6: Attribution Scores

**File:** Extend `evaluate.py`

- Extract learned `sigmoid(assignment_logits)` values
- Correlate with ground-truth color labels
- Visualization: histogram of assignment values, colored by true color attribute
- Optional (uses `torch.func`): per-test-example attribution via gradient of disagreement w.r.t. assignment weights

---

## Code Style and Conventions

- **Pure PyTorch**, no Lightning — the adversarial training loop is the novel contribution; hiding it behind Lightning's `manual_optimization` path would obscure exactly the part that matters most
- **Minimal abstractions:** functions over classes where possible
- **Config via Hydra** (`configs/` directory, YAML + method config groups); every run's resolved config is saved automatically to `outputs/` for provenance
- **Reproducibility:** set seeds everywhere (torch, numpy, random)
- **Device handling:** MPS → CUDA → CPU priority (see `utils.get_device()`)
- **File structure:**
  ```
  project/
  ├── configs/
  │   ├── config.yaml          # root defaults
  │   └── method/              # one yaml per method (erm, random_split, …)
  ├── data.py                  # ColoredMNIST dataset
  ├── models.py                # MLP, SplitMLP
  ├── train.py                 # Training loops for all methods
  ├── evaluate.py              # Stability scores, attribution, metrics
  ├── utils.py                 # Seed setting, device helpers
  └── run.py                   # Entry point: @hydra.main → train()
  ```
- **No premature optimization:** get it correct first, fast later
- **All tensor operations in comments** explaining shapes

---

## Key Design Decisions Still Open

1. **Symmetric vs asymmetric KL:** Should disagreement be `KL(A||B)`, `KL(B||A)`, or `0.5*(KL(A||B)+KL(B||A))`? Start with symmetric, ablate later.
2. **How often to update adversary vs model:** Every step? Every N steps? Start with alternating every step.
3. **Assignment initialization:** All zeros (sigmoid=0.5, uniform) or random? Start with zeros.
4. **Temperature on assignments:** Should we anneal the sigmoid toward hard assignments? Maybe later.
5. **Regularization on assignments:** Should we penalize assignments that are too uniform or too extreme? Try entropy penalty on s if needed.

---

## How to Run Experiments

Once all steps are built, the ablation is:

```bash
# ERM baseline
python run.py method=erm

# Random split
python run.py method=random_split

# Adversarial split (our method)
python run.py method=adversarial_split

# Oracle split
python run.py method=oracle_split

# Sweep lambda (Hydra multirun)
python run.py -m method=adversarial_split training.lambda_disagree=0.1,1.0,10.0
```

---

## Wandb Conventions

All methods log through a shared helper in `utils.py` — metric names must never be inlined ad-hoc in training code. This is the only way wandb reports stay comparable across runs.

**Run naming:** `{method}_{MMDD-HHMM}`.
**Tags:** `dev` by default; override on CLI: `python run.py "wandb.tags=[dev,final]"`.

---

## Success Criteria

The project works if:
1. **ERM fails OOD** (relies on color) — this is the known baseline
2. **Random split helps a little** OOD — confirms existing V-REx-like findings
3. **Adversarial split helps substantially more** OOD — this is the key result
4. **Oracle split is the upper bound** — adversarial should approach this
5. **Learned assignments s_i correlate with color** — method discovers spurious feature
6. **Stability scores predict OOD failures** better than softmax entropy / ensembles

---

## How Claude Should Work on This Project

### Scientist stays at the wheel
- Explain every non-obvious design choice before implementing it
- When there are real alternatives, name them and state the tradeoff — don't silently pick one
- Flag when an experiment result is surprising or contradicts the hypothesis; don't just report numbers

### Planning
- Enter plan mode for any task with 3+ steps or an architectural decision
- If something goes sideways, stop and re-plan — don't keep pushing through a bad approach
- Write a clear spec before writing code; ambiguity caught early is cheap

### Verification
- Never call a step done without showing it works (run the code, check the output)
- For training code: show loss curves or accuracy numbers, not just "it ran"
- Ask: "Would a staff researcher trust this result?" before moving on

### Code quality
- Simplicity first — minimal impact on surrounding code, no speculative abstractions
- Find root causes; no temporary fixes or workarounds
- Before finishing any non-trivial implementation: pause and ask whether a simpler or more elegant approach exists
- Comments must explain *why* or *what the design choice is* — never just restate the variable name

### Subagents
- Use subagents to keep the main context clean for research reasoning
- Offload literature lookups, code exploration, and parallel analyses to subagents
- One focused task per subagent

### Self-correction
- After any correction: record the pattern in `tasks/lessons.md` so the same mistake doesn't recur
- Review `tasks/lessons.md` at the start of each session

### Git commits
Use [Conventional Commits](https://www.conventionalcommits.org/). Commit at natural milestones — end of each completed step, not per file.

| Prefix | When to use |
|--------|-------------|
| `feat:` | new capability (new model, new training method, new evaluation) |
| `test:` | adding or fixing tests |
| `fix:` | bug fix in existing code |
| `refactor:` | restructuring without behaviour change |
| `docs:` | README, comments, docstrings only |
| `chore:` | config, deps, tooling |

Always include `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` trailer.
