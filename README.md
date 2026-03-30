# Adversarial Data Splitting for Training Stability

## The problem

Machine learning models are sensitive to the composition of their training data. Small changes in which examples are included can meaningfully shift predictions. Existing approaches to this problem either require pre-defined environment labels (IRM, V-REx) or capture disagreement without feeding it back into training (ensembles).

## The idea

Instead of randomly partitioning training data and penalising disagreement, we **learn the worst-case partition** — the split of the training set that maximises prediction disagreement between two model heads — and then regularise against it.

This is a minimax game:

- **Inner loop (adversary):** find the data partition that most destabilises the model's predictions
- **Outer loop (model):** train to be robust to that worst-case split

Each training example gets a soft assignment weight $s_i \in [0, 1]$ learned by the adversary. Head A is trained on examples weighted by $s_i$, head B on examples weighted by $1 - s_i$. The adversary maximises the KL divergence between the two heads' predictions; the model minimises it.

## Why this is different

| Method | Key limitation |
|--------|----------------|
| IRM, V-REx | Require pre-defined environment labels |
| Deep ensembles | Disagreement is not fed back into training |
| Consistency regularisation | Penalises input perturbations, not training data composition |
| DRO | Robustifies against test distribution shift, not training data composition |

## Byproducts

1. **Instance-level stability scores** — for each test prediction, how much can the worst-case data partition destabilise it? A new form of epistemic uncertainty: not "the model is uncertain" but "the model's confidence is an artefact of which data it trained on."
2. **Data attribution** — the learned assignment weights $s_i$ reveal which training examples are "contentious" — their inclusion or exclusion flips predictions elsewhere.

## Experiments

We use **Colored MNIST** as the primary benchmark. Digits 0–4 vs 5–9 (binary), with color assigned as a spurious cue. Training correlation 0.9 (color strongly predicts label), test correlation 0.1 (color is mostly misleading). A model that relies on color will fail badly out-of-distribution.

Four conditions:

| Condition | Description |
|-----------|-------------|
| ERM | Standard training, single head — the baseline that fails OOD |
| Random split | Random K=2 partition + KL penalty — a V-REx-like baseline |
| **Adversarial split** | Learned worst-case partition + KL penalty — **our method** |
| Oracle split | Partition by ground-truth color label — upper bound |

The key comparison is random vs adversarial. If adversarial splitting is substantially better, the paper works.

## Setup

```bash
# Install dependencies
make install

# Run ERM baseline
python run.py method=erm

# Run adversarial split (our method)
python run.py method=adversarial_split

# Sweep lambda
python run.py -m method=adversarial_split training.lambda_disagree=0.1,1.0,10.0
```

Requires Python 3.11+. Experiments are logged to [Weights & Biases](https://wandb.ai) under the `data-invariance` project.
