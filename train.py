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


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate model on a dataloader, returning accuracy, loss, precision, recall, and AUROC.

    Precision/recall are included now so we notice if the model collapses to predicting
    one class — accuracy alone can hide that on balanced datasets and will miss it entirely
    if we move to an imbalanced setting later.

    AUROC is threshold-free and robust to class imbalance, making it a better summary
    statistic than accuracy when comparing across conditions.

    All metrics are accumulated with torchmetrics, which handles edge cases (e.g. zero
    division when a class is never predicted) correctly.
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

            logits = model(x)               # (B, 2)

            # sum reduction so we can compute a properly weighted mean at the end
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            total_n += len(y)

            # torchmetrics expects probabilities for AUROC; softmax over the 2 logits,
            # then take the positive-class (index 1) probability
            probs = logits.softmax(dim=1)[:, 1]   # (B,)
            preds = logits.argmax(dim=1)           # (B,)

            accuracy.update(preds, y)
            precision.update(preds, y)
            recall.update(preds, y)
            auroc.update(probs, y)

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
