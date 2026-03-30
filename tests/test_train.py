from __future__ import annotations

import math

import torch
import wandb
from omegaconf import OmegaConf

from models import MLP
from train import evaluate, make_dataloaders, train_erm


def make_cfg():
    return OmegaConf.create({
        "data": {"train_correlation": 0.9, "test_correlation": 0.1, "label_noise": 0.25, "data_dir": "./data"},
        "model": {"hidden_dim": 64},
        "training": {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "epochs": 1, "seed": 0},
        "method": {"name": "erm"},
        "wandb": {"enabled": False},
    })


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def test_mlp_output_shape():
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64)
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    assert out.shape == (4, 2), f"unexpected shape: {out.shape}"


def test_mlp_output_is_logits():
    """Output should be raw logits, not probabilities — no softmax in forward.

    Returning logits is preferred because F.cross_entropy applies log-softmax
    internally in a numerically stable way. Applying softmax before cross_entropy
    would lose that stability.
    """
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64)
    x = torch.randn(16, 3, 28, 28)
    out = model(x)
    # If output were probabilities, all values would be in [0,1] and rows would sum to 1.
    # Raw logits can be any real number.
    row_sums = out.softmax(dim=1).sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(16)), "rows should sum to 1 after softmax"
    # And at least some logits should be negative (probabilities never are)
    assert out.min().item() < 0, "expected negative logits from an untrained model"


# ---------------------------------------------------------------------------
# make_dataloaders
# ---------------------------------------------------------------------------

def test_dataloaders_keys():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    assert set(loaders.keys()) == {"train", "id_test", "ood_test"}


def test_dataloaders_nonempty():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    for name, loader in loaders.items():
        assert len(loader) > 0, f"{name} loader is empty"


def test_dataloader_batch_shape():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[1:] == (3, 28, 28)
    assert batch["label"].shape == (cfg.training.batch_size,)


def test_id_and_ood_correlation_differ():
    """ID and OOD test sets must have different color-label correlations.

    This is the entire point of the benchmark — if these are identical,
    we are not measuring OOD generalisation at all.
    """
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    id_ds = loaders["id_test"].dataset
    ood_ds = loaders["ood_test"].dataset
    id_corr = (id_ds.labels == id_ds.colors).float().mean().item()
    ood_corr = (ood_ds.labels == ood_ds.colors).float().mean().item()
    assert abs(id_corr - ood_corr) > 0.5, (
        f"ID corr {id_corr:.2f} and OOD corr {ood_corr:.2f} are too similar — "
        "distribution shift is not working"
    )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_metric_range():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)
    metrics = evaluate(model, loaders["id_test"], device)

    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["loss"] >= 0.0


def test_evaluate_metric_keys():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)
    metrics = evaluate(model, loaders["id_test"], device)
    assert set(metrics.keys()) == {"acc", "precision", "recall", "auroc", "loss"}


# ---------------------------------------------------------------------------
# train_erm
# ---------------------------------------------------------------------------

def test_train_erm_returns_correct_metric_keys():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_erm(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall",
    }
    assert set(metrics.keys()) == expected


def test_train_erm_loss_below_random():
    """After one epoch, train loss should beat the random-model baseline.

    For balanced binary classification, a model predicting uniformly at random
    achieves cross-entropy = log(2) ≈ 0.693. A model that has learned anything
    should do strictly better. If it does not, something is wrong with the
    optimisation (wrong loss, wrong labels, gradient not flowing).
    """
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_erm(cfg, model, loaders, device, run)
    run.finish()

    random_baseline = math.log(2)  # ≈ 0.693
    assert metrics["train/loss"] < random_baseline, (
        f"train loss {metrics['train/loss']:.4f} not below random baseline {random_baseline:.4f} — "
        "check that gradients are flowing and the loss is computed correctly"
    )
