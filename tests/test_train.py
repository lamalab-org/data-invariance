"""Tests for the training pipeline.

Tests the 3 methods (ERM, JTT, V-REx) and the discovery pipeline
against the refactored train.py. Uses CMNIST with small configs for speed.
"""
from __future__ import annotations

import torch
import wandb
from omegaconf import OmegaConf

from models import MLP
from train import (
    _build_model,
    _val_score,
    auto_lambda,
    discover_environments,
    discover_jtt_weights,
    evaluate,
    make_dataloaders,
    train_erm,
    train_jtt,
    train_vrex,
)


def make_cfg():
    return OmegaConf.create({
        "dataset": {"name": "cmnist", "arch": "mlp", "train_correlation": 0.9,
                    "test_correlation": 0.1, "label_noise": 0.25, "data_dir": "./data"},
        "model": {"hidden_dim": 64},
        "training": {
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "epochs": 1,
            "seed": 0, "discovery_epochs": 2, "discovery_upweight": 50.0,
            "num_discovery_envs": 2, "early_stop_patience": 0,
            "lambda_disagree": 10.0,
        },
        "method": {"name": "erm"},
        "wandb": {"enabled": False},
    })


# --- Model tests ---

def test_mlp_output_shape():
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64)
    x = torch.randn(4, 3, 28, 28)
    out = model(x)
    assert out.shape == (4, 2)


def test_mlp_predict_is_probabilities():
    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64)
    x = torch.randn(4, 3, 28, 28)
    probs = model.predict(x)
    assert probs.shape == (4, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (probs >= 0).all()


# --- Dataloader tests ---

def test_dataloaders_keys():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    assert set(loaders.keys()) == {"train", "id_test", "ood_test"}


def test_dataloader_batch_shape():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[1:] == (3, 28, 28)
    assert "label" in batch
    assert "index" in batch


# --- Evaluate tests ---

def test_evaluate_returns_metrics():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    model = _build_model(cfg, loaders, torch.device("cpu"))
    metrics = evaluate(model, loaders["id_test"], torch.device("cpu"))
    assert "acc" in metrics
    assert "loss" in metrics
    assert "auroc" in metrics
    assert 0 <= metrics["acc"] <= 1


def test_evaluate_computes_wga():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    model = _build_model(cfg, loaders, torch.device("cpu"))
    metrics = evaluate(model, loaders["id_test"], torch.device("cpu"))
    assert "worst_group_acc" in metrics
    assert 0 <= metrics["worst_group_acc"] <= 1


# --- Val score tests ---

def test_val_score_group_free():
    metrics = {"acc": 0.9, "worst_group_acc": 0.5}
    assert _val_score(metrics, group_free=True) == 0.9
    assert _val_score(metrics, group_free=False) == 0.5


def test_val_score_no_wga():
    metrics = {"acc": 0.9}
    assert _val_score(metrics, group_free=False) == 0.9
    assert _val_score(metrics, group_free=True) == 0.9


# --- Discovery tests ---

def test_discover_environments():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    assignment, weights, diag = discover_environments(cfg, loaders, torch.device("cpu"))
    N = len(loaders["train"].dataset)
    assert assignment.shape == (N,)
    assert weights.shape == (N,)
    assert assignment.min() >= 0
    assert assignment.max() <= 1
    assert weights.min() >= 1.0
    assert "adaptive/signal_ratio" in diag
    assert "adaptive/reliability" in diag
    assert 0 <= diag["adaptive/reliability"] <= 1


def test_auto_lambda():
    diag = {
        "discovery/n_env_A": 3000.0,
        "discovery/n_env_B": 3000.0,
        "adaptive/reliability": 1.0,
        "adaptive/actual_risk_var": 0.5,
    }
    cfg = make_cfg()
    lam = auto_lambda(diag, cfg)
    assert lam > 0
    # N=6000, so lambda = min(20, 10 * 5000/6000) * 1.0 = 8.33
    assert 5 < lam < 20


def test_auto_lambda_caps():
    diag = {
        "discovery/n_env_A": 200.0,
        "discovery/n_env_B": 200.0,
        "adaptive/reliability": 1.0,
        "adaptive/actual_risk_var": 0.5,
    }
    cfg = make_cfg()
    lam = auto_lambda(diag, cfg)
    # N=400, uncapped would be 10*5000/400=125, capped at 20
    assert lam == 20.0


# --- Training tests ---

def test_train_erm():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    model = _build_model(cfg, loaders, torch.device("cpu"))
    run = wandb.init(mode="disabled")
    result = train_erm(cfg, model, loaders, torch.device("cpu"), run, group_free=True)
    run.finish()
    assert "train/loss" in result
    assert "eval/ood_acc" in result


def test_train_jtt():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    weights, diag = discover_jtt_weights(cfg, loaders, torch.device("cpu"))
    assert weights.shape == (len(loaders["train"].dataset),)
    assert "jtt/n_misclassified" in diag

    model = _build_model(cfg, loaders, torch.device("cpu"))
    run = wandb.init(mode="disabled")
    result = train_jtt(cfg, model, loaders, torch.device("cpu"), run,
                       weights, diag, group_free=True)
    run.finish()
    assert "train/loss" in result


def test_train_vrex():
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    assignment, weights, diag = discover_environments(cfg, loaders, torch.device("cpu"))

    model = _build_model(cfg, loaders, torch.device("cpu"))
    run = wandb.init(mode="disabled")
    result = train_vrex(cfg, model, loaders, torch.device("cpu"), run,
                        assignment, weights, diag, group_free=True)
    run.finish()
    assert "train/loss" in result
    assert "train/risk_variance" in result
    assert "train/lambda" in result


def test_train_vrex_group_free_vs_labeled():
    """The group_free flag should only affect model selection, not training."""
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    assignment, weights, diag = discover_environments(cfg, loaders, torch.device("cpu"))

    # Both should produce valid results
    for gf in [True, False]:
        model = _build_model(cfg, loaders, torch.device("cpu"))
        run = wandb.init(mode="disabled")
        result = train_vrex(cfg, model, loaders, torch.device("cpu"), run,
                            assignment, weights, diag, group_free=gf)
        run.finish()
        assert "train/loss" in result
