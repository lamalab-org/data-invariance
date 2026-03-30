from __future__ import annotations

import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import ColoredMNIST
from utils import log_metrics


def make_dataloaders(cfg: DictConfig) -> dict[str, DataLoader]:
    """Build train, ID-test, and OOD-test dataloaders.

    All three datasets share the same label_noise and seed so only the
    color-label correlation differs between them. This isolates the
    distribution shift to exactly one variable.

    The datasets are constructed on CPU and kept there; DataLoader moves
    batches to device per-step. This avoids pinning 60k images in GPU memory.

    Returns:
        {"train": ..., "id_test": ..., "ood_test": ...}
    """
    seed = cfg.training.seed
    noise = cfg.data.label_noise
    data_dir = cfg.data.data_dir

    train_ds = ColoredMNIST(cfg.data.train_correlation, label_noise=noise, split="train", data_dir=data_dir, seed=seed)
    # ID test: same correlation as train — measures in-distribution performance
    id_test_ds = ColoredMNIST(cfg.data.train_correlation, label_noise=noise, split="test", data_dir=data_dir, seed=seed)
    # OOD test: flipped correlation — the spurious color cue now hurts; this is the key metric
    ood_test_ds = ColoredMNIST(cfg.data.test_correlation, label_noise=noise, split="test", data_dir=data_dir, seed=seed)

    kwargs = dict(batch_size=cfg.training.batch_size, num_workers=0, pin_memory=False)
    return {
        "train": DataLoader(train_ds, shuffle=True, **kwargs),
        "id_test": DataLoader(id_test_ds, shuffle=False, **kwargs),
        "ood_test": DataLoader(ood_test_ds, shuffle=False, **kwargs),
    }


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two sets of logits, averaged over the batch.

    KL is computed over the softmax probability distributions derived from the logits.
    We use log_softmax for numerical stability: F.kl_div expects log-probabilities as
    input and regular probabilities as target.

    Symmetric KL = 0.5 * (KL(P||Q) + KL(Q||P)) — avoids the asymmetry of plain KL,
    where KL(P||Q) → ∞ when Q assigns zero mass to regions where P has mass.

    Args:
        logits_a: (B, C) raw logits from head A
        logits_b: (B, C) raw logits from head B
    Returns:
        scalar — mean symmetric KL over the batch
    """
    log_pa = logits_a.log_softmax(dim=1)   # (B, C) log-probabilities for head A
    log_pb = logits_b.log_softmax(dim=1)   # (B, C) log-probabilities for head B
    pa = log_pa.exp()                      # (B, C) probabilities for head A
    pb = log_pb.exp()                      # (B, C) probabilities for head B

    # batchmean: sums over classes, averages over batch — the correct reduction for KL
    kl_ab = F.kl_div(log_pa, pb, reduction="batchmean")
    kl_ba = F.kl_div(log_pb, pa, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate model on a dataloader, returning accuracy, loss, precision, recall, and AUROC.

    Calls model.predict(x) which returns class probabilities (B, 2). Both MLP and
    SplitMLP implement predict(), so this function is model-agnostic.

    Precision/recall are included now so we notice if the model collapses to predicting
    one class — accuracy alone can hide that on balanced datasets and will miss it entirely
    if we move to an imbalanced setting later.

    AUROC is threshold-free and robust to class imbalance, making it a better summary
    statistic than accuracy when comparing across conditions.
    """
    model.eval()

    # task="binary" because we have exactly 2 classes (digits 0-4 vs 5-9)
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    auroc = torchmetrics.AUROC(task="binary").to(device)

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)   # (B, 3, 28, 28)
            y = batch["label"].to(device)   # (B,) int64

            probs = model.predict(x)        # (B, 2) — probabilities, model-agnostic

            # Cross-entropy from probabilities: use log + nll_loss
            # sum reduction so we can compute a properly weighted mean at the end
            total_loss += F.nll_loss(probs.log(), y, reduction="sum").item()
            total_n += len(y)

            pos_probs = probs[:, 1]         # (B,) probability of positive class
            preds = probs.argmax(dim=1)     # (B,) predicted class

            accuracy.update(preds, y)
            precision.update(preds, y)
            recall.update(preds, y)
            auroc.update(pos_probs, y)

    return {
        "acc": accuracy.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "auroc": auroc.compute().item(),
        "loss": total_loss / total_n,
    }


def train_erm(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """Standard ERM training loop.

    One optimizer step per batch, per-epoch evaluation on both test sets.
    AdamW is used over Adam because Adam's weight decay is mathematically
    incorrect (it decays the adapted parameters rather than the weights
    directly). AdamW fixes this, giving cleaner regularisation at no extra cost.

    Returns the final epoch's metrics dict.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    metrics: dict[str, float] = {}
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0

        for batch in loaders["train"]:
            x = batch["image"].to(device)   # (B, 3, 28, 28)
            y = batch["label"].to(device)   # (B,) int64

            logits = model(x)               # (B, 2)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Multiply by batch size to undo the mean reduction, so we can
            # accumulate a properly weighted sum across variable-size batches.
            epoch_loss += loss.item() * len(y)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_n += len(y)

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            "eval/id_acc": id_metrics["acc"],
            "eval/id_auroc": id_metrics["auroc"],
            "eval/id_precision": id_metrics["precision"],
            "eval/id_recall": id_metrics["recall"],
            "eval/ood_acc": ood_metrics["acc"],
            "eval/ood_auroc": ood_metrics["auroc"],
            "eval/ood_precision": ood_metrics["precision"],
            "eval/ood_recall": ood_metrics["recall"],
        }
        log_metrics(run, metrics, step=epoch)

    return metrics


def train_random_split(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """Random split + KL disagreement penalty training loop.

    Each training example is assigned to head A (s=1) or head B (s=0) once
    before training begins. The assignment is fixed for all epochs — this is
    analogous to V-REx with two fixed environments, and makes a clean comparison
    to the adversarial split (which has a *learned* rather than random assignment).

    Loss = loss_A + loss_B + lambda * symmetric_KL(head_A, head_B)

    where loss_A weights each example by s_i (1 for assigned examples, 0 otherwise),
    and loss_B weights by (1 - s_i). This ensures every example contributes to
    exactly one head's task loss, while the KL term is computed on all examples.

    Returns the final epoch's metrics dict.
    """
    train_ds = loaders["train"].dataset
    N = len(train_ds)

    # Fixed random assignment seeded for reproducibility.
    # Using a separate generator so this doesn't interact with model init randomness.
    g = torch.Generator().manual_seed(cfg.training.seed)
    # assignment[i] = 0 → example i trains head A (s=1)
    # assignment[i] = 1 → example i trains head B (s=0)
    assignment = torch.randint(0, 2, (N,), generator=g).to(device)  # (N,) ∈ {0, 1}

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    metrics: dict[str, float] = {}
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        epoch_disagree = 0.0

        for batch in loaders["train"]:
            x = batch["image"].to(device)    # (B, 3, 28, 28)
            y = batch["label"].to(device)    # (B,) int64
            idx = batch["index"].to(device)  # (B,) — global example indices

            logits_a, logits_b = model(x)    # each (B, 2)

            # s=1 → assigned to head A, s=0 → assigned to head B
            s = (assignment[idx] == 0).float()   # (B,) ∈ {0.0, 1.0}

            # reduction="none" gives per-example losses (B,) so we can weight them
            ce_a = F.cross_entropy(logits_a, y, reduction="none")  # (B,)
            ce_b = F.cross_entropy(logits_b, y, reduction="none")  # (B,)

            # Each example contributes to exactly one head; mean over batch
            loss_a = (s * ce_a).mean()
            loss_b = ((1.0 - s) * ce_b).mean()

            # Disagreement on all examples — not just the assigned ones.
            # We want the heads to agree everywhere, not just on their own subset.
            disagree = symmetric_kl(logits_a, logits_b)

            loss = loss_a + loss_b + cfg.training.lambda_disagree * disagree

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            # Use averaged head predictions for training accuracy tracking
            avg_probs = (logits_a.softmax(1) + logits_b.softmax(1)) / 2
            epoch_correct += (avg_probs.argmax(1) == y).sum().item()
            epoch_n += len(y)
            epoch_disagree += disagree.item() * len(y)

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            "train/disagreement": epoch_disagree / epoch_n,
            "eval/id_acc": id_metrics["acc"],
            "eval/id_auroc": id_metrics["auroc"],
            "eval/id_precision": id_metrics["precision"],
            "eval/id_recall": id_metrics["recall"],
            "eval/ood_acc": ood_metrics["acc"],
            "eval/ood_auroc": ood_metrics["auroc"],
            "eval/ood_precision": ood_metrics["precision"],
            "eval/ood_recall": ood_metrics["recall"],
        }
        log_metrics(run, metrics, step=epoch)

    return metrics
