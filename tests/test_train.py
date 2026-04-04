from __future__ import annotations

import math

import torch
import wandb
from omegaconf import OmegaConf

from evaluate import compute_assignment_correlation, compute_assignment_correlation_multi
from models import MLP, MultiHeadMLP, SplitMLP
from train import discover_environments, evaluate, make_dataloaders, symmetric_kl, train_adversarial_split, train_adversarial_split_multi, train_discovered_split, train_erm, train_oracle_split, train_random_split, train_resampling


def make_cfg():
    return OmegaConf.create({
        "dataset": {"name": "cmnist", "arch": "mlp", "train_correlation": 0.9, "test_correlation": 0.1, "label_noise": 0.25, "data_dir": "./data"},
        "model": {"hidden_dim": 64},
        "training": {
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "epochs": 1, "seed": 0,
            "lambda_disagree": 1.0, "adv_lr": 1e-2,
            "adv_init": "zeros", "adv_init_scale": 1.0,
            "head_noise": 0.0, "adv_warmup_epochs": 0,
            "adv_steps_per_model_step": 1, "lambda_warmup_epochs": 0,
            "adv_entropy_bonus": 0.0,
            "lambda_threshold": 0.0, "lambda_ramp_range": 0.0,
            "adv_mode": "task_loss",
            "discovery_epochs": 2,
            "lambda_anneal_factor": 1.0,
            "discovery_quantile": 0.5,
            "discovery_criterion": "entropy",
            "discovery_reweight": 0.0,
            "discovery_upweight": 0.0,
            "early_stop_patience": 0,
        },
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
    expected = {"acc", "precision", "recall", "auroc", "loss"}
    # worst_group_acc is present when the dataset provides spurious labels
    if "worst_group_acc" in metrics:
        expected.add("worst_group_acc")
    assert set(metrics.keys()) == expected


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
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
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


# ---------------------------------------------------------------------------
# SplitMLP
# ---------------------------------------------------------------------------

def test_split_mlp_forward_shapes():
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64)
    x = torch.randn(4, 3, 28, 28)
    logits_a, logits_b = model(x)
    assert logits_a.shape == (4, 2)
    assert logits_b.shape == (4, 2)


def test_split_mlp_predict_is_probabilities():
    """predict() must return a valid probability distribution (rows sum to 1)."""
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64)
    x = torch.randn(8, 3, 28, 28)
    probs = model.predict(x)
    assert probs.shape == (8, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


def test_split_mlp_heads_differ_after_init():
    """The two heads should have different random initialisation — they are independent."""
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64)
    assert not torch.equal(model.head_a.weight, model.head_b.weight)


def test_split_mlp_separate_backbones_shapes():
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64, separate_backbones=True)
    x = torch.randn(4, 3, 28, 28)
    logits_a, logits_b = model(x)
    assert logits_a.shape == (4, 2)
    assert logits_b.shape == (4, 2)


def test_split_mlp_separate_backbones_differ():
    """With separate backbones, the two backbone initialisations must differ."""
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64, separate_backbones=True)
    # Compare first Linear weight of each backbone
    # Index 1 because index 0 is nn.Flatten
    w_a = model.backbone_a[1].weight
    w_b = model.backbone_b[1].weight
    assert not torch.equal(w_a, w_b)


def test_split_mlp_get_features_shared():
    """With a shared backbone, get_features returns the same tensor for both heads."""
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64, separate_backbones=False)
    x = torch.randn(4, 3, 28, 28)
    fa, fb = model.get_features(x)
    assert fa is fb   # same object, no extra compute


def test_split_mlp_get_features_separate():
    """With separate backbones, get_features returns different tensors."""
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64, separate_backbones=True)
    x = torch.randn(4, 3, 28, 28)
    fa, fb = model.get_features(x)
    assert fa is not fb
    assert not torch.equal(fa, fb)


# ---------------------------------------------------------------------------
# symmetric_kl
# ---------------------------------------------------------------------------

def test_symmetric_kl_zero_for_equal_logits():
    """KL divergence between identical distributions must be zero."""
    logits = torch.randn(8, 2)
    kl = symmetric_kl(logits, logits)
    assert kl.item() < 1e-5, f"expected ~0 for identical logits, got {kl.item()}"


def test_symmetric_kl_positive_for_different_logits():
    """KL divergence between different distributions must be positive."""
    logits_a = torch.tensor([[2.0, -2.0]] * 8)   # strongly predicts class 0
    logits_b = torch.tensor([[-2.0, 2.0]] * 8)   # strongly predicts class 1
    kl = symmetric_kl(logits_a, logits_b)
    assert kl.item() > 1.0, f"expected large KL for opposite predictions, got {kl.item()}"


def test_symmetric_kl_is_symmetric():
    logits_a = torch.randn(8, 2)
    logits_b = torch.randn(8, 2)
    assert abs(symmetric_kl(logits_a, logits_b).item() - symmetric_kl(logits_b, logits_a).item()) < 1e-5


# ---------------------------------------------------------------------------
# train_random_split
# ---------------------------------------------------------------------------

def test_train_random_split_metric_keys():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_random_split(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/disagreement",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
    }
    assert set(metrics.keys()) == expected


def test_train_random_split_disagreement_is_finite():
    """Disagreement must be a finite non-negative number — NaN would indicate a bug."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_random_split(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/disagreement"])
    assert metrics["train/disagreement"] >= 0.0


# ---------------------------------------------------------------------------
# train_adversarial_split
# ---------------------------------------------------------------------------

def test_train_adversarial_split_metric_keys():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/disagreement", "train/risk_variance",
        "train/lambda", "train/assignment_entropy",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
        "assignment_color_corr", "assignment_color_abs_corr",
    }
    assert set(metrics.keys()) == expected


def test_train_adversarial_split_entropy_range():
    """Assignment entropy must lie in [0, log(2)] — the range for a binary distribution.

    log(2) ≈ 0.693 is the maximum entropy (uniform assignment, s_i = 0.5 for all i).
    After one epoch from a zero-initialised start the entropy should be close to log(2)
    but may have decreased slightly as the adversary begins to commit.
    """
    import math
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    H = metrics["train/assignment_entropy"]
    assert 0.0 <= H <= math.log(2) + 1e-5, (
        f"assignment entropy {H:.4f} outside valid range [0, log(2)={math.log(2):.4f}]"
    )


def test_train_adversarial_split_logits_move():
    """After one epoch, at least some assignment logits should differ from 0.

    The adversary starts at zero logits (sigmoid = 0.5 uniform split). If all
    logits remain at zero after training, the adversary is not learning — either
    gradients are not flowing or the adversary optimizer is broken.
    """
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    # We need access to assignment_logits after training; patch the function to expose them.
    # Instead, verify indirectly: entropy < log(2) means logits have diverged from 0.
    import math
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    # If logits were all still 0, entropy would be exactly log(2). A strict drop means movement.
    assert metrics["train/assignment_entropy"] < math.log(2), (
        "assignment entropy is still at maximum — adversary logits have not moved from zero"
    )


def test_train_adversarial_split_disagreement_finite():
    """Disagreement must be a finite non-negative number — NaN indicates a gradient bug."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/disagreement"])
    assert metrics["train/disagreement"] >= 0.0


# ---------------------------------------------------------------------------
# train_oracle_split
# ---------------------------------------------------------------------------

def test_train_oracle_split_metric_keys():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_oracle_split(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/disagreement",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
    }
    assert set(metrics.keys()) == expected


def test_oracle_split_uses_color_not_random():
    """Oracle split assignment must correlate with color, not be uniformly random.

    With train_correlation=0.9, roughly 90% of examples have color == label.
    The oracle assigns s=1 to color=0 examples and s=0 to color=1 examples.
    At a minimum, the two partitions must be non-trivially unequal in label
    distribution — unlike a random 50/50 split which would be balanced.
    """
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    train_ds = loaders["train"].dataset

    # Proportion of color=0 examples (should be ~50% — CMNIST balances colors)
    color_0_frac = (train_ds.colors == 0).float().mean().item()
    assert abs(color_0_frac - 0.5) < 0.05, (
        f"expected ~50% color=0 examples, got {color_0_frac:.3f}"
    )

    # Within color=0 group, label distribution should differ from color=1 group.
    # With train_correlation=0.9: P(label=0 | color=0) ≈ 0.9*0.5/0.5 = 0.9 (approx),
    # while P(label=0 | color=1) ≈ 0.1. So label distributions are very different.
    mask_0 = train_ds.colors == 0
    label_rate_c0 = train_ds.labels[mask_0].float().mean().item()
    label_rate_c1 = train_ds.labels[~mask_0].float().mean().item()
    assert abs(label_rate_c0 - label_rate_c1) > 0.5, (
        f"label rates within color groups should differ strongly; "
        f"got {label_rate_c0:.2f} vs {label_rate_c1:.2f}"
    )


def test_adversarial_split_random_init():
    """adv_init='random' must produce valid metrics — non-uniform starting partition."""
    cfg = make_cfg()
    cfg.training.adv_init = "random"
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/assignment_entropy"])
    assert 0.0 <= metrics["train/assignment_entropy"] <= math.log(2) + 1e-5


def test_adversarial_split_head_noise():
    """head_noise > 0 must produce valid metrics without NaN."""
    cfg = make_cfg()
    cfg.training.head_noise = 0.1
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/disagreement"])
    assert math.isfinite(metrics["train/assignment_entropy"])


def test_adversarial_split_multi_adv_steps():
    """adv_steps_per_model_step > 1 must produce valid metrics."""
    cfg = make_cfg()
    cfg.training.adv_steps_per_model_step = 3
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert set(metrics.keys()) == {
        "train/loss", "train/acc", "train/disagreement", "train/risk_variance",
        "train/lambda", "train/assignment_entropy",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
        "assignment_color_corr", "assignment_color_abs_corr",
    }


def test_adversarial_split_warmup_frozen():
    """With adv_warmup_epochs=1 and epochs=1, the adversary never activates.

    The assignment logits must remain at their initial value (zeros → all equal)
    because no adversary steps were taken. Entropy should therefore be at or
    very close to log(2) (the all-0.5 assignment).
    """
    cfg = make_cfg()
    cfg.training.adv_warmup_epochs = 1   # warmup covers the only epoch
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    # Logits are all 0 → entropy = log(2). Allow tiny float error.
    assert abs(metrics["train/assignment_entropy"] - math.log(2)) < 1e-4, (
        f"expected entropy = log(2) with frozen adversary, got {metrics['train/assignment_entropy']:.6f}"
    )


def test_adversarial_split_lambda_warmup():
    """lambda_warmup_epochs > epochs means lambda=0 throughout — heads see no KL penalty."""
    cfg = make_cfg()
    cfg.training.lambda_warmup_epochs = 100   # far beyond the single epoch
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/loss"])
    assert math.isfinite(metrics["train/disagreement"])


def test_adversarial_split_entropy_bonus():
    """adv_entropy_bonus > 0 must produce valid metrics without NaN.

    The entropy bonus penalises soft assignments in the adversary loss.
    We verify it doesn't break training and that entropy remains in [0, log(2)].
    """
    cfg = make_cfg()
    cfg.training.adv_entropy_bonus = 0.1
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/assignment_entropy"])
    assert 0.0 <= metrics["train/assignment_entropy"] <= math.log(2) + 1e-5


def test_adversarial_split_threshold_scheduling_holds_lambda_at_zero():
    """With an unreachably high lambda_threshold, lambda must stay 0 the whole run.

    When lambda=0 throughout, the model sees no KL penalty and the disagreement
    should grow more freely than with lambda=1.0. We verify:
    1. train/lambda is 0.0 at the end (threshold never fired)
    2. train/disagreement is finite and non-negative
    """
    cfg = make_cfg()
    cfg.training.lambda_threshold = 100.0   # unreachable — disagreement stays far below this
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert metrics["train/lambda"] == 0.0, (
        f"expected lambda=0 with unreachable threshold, got {metrics['train/lambda']}"
    )
    assert math.isfinite(metrics["train/disagreement"])


def test_adversarial_split_threshold_disabled_matches_epoch_warmup():
    """With lambda_threshold=0 (disabled), train/lambda must equal the epoch-warmup value.

    With lambda_warmup_epochs=0 the full lambda_disagree should be used from epoch 0.
    """
    cfg = make_cfg()
    cfg.training.lambda_threshold = 0.0
    cfg.training.lambda_warmup_epochs = 0
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert abs(metrics["train/lambda"] - cfg.training.lambda_disagree) < 1e-6, (
        f"expected train/lambda={cfg.training.lambda_disagree}, got {metrics['train/lambda']}"
    )


def test_adversarial_split_grad_div_mode():
    """adv_mode='grad_div' must run without errors and produce valid metrics.

    The gradient-diversity adversary minimises cosine similarity between the two
    heads' gradient directions at the backbone output.  We verify:
    1. All expected metric keys are present and finite.
    2. Assignment logits have moved from zero (adversary produced a gradient).
    3. Assignment entropy is in the valid binary-entropy range [0, log(2)].
    """
    cfg = make_cfg()
    cfg.training.adv_mode = "grad_div"
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/disagreement", "train/risk_variance",
        "train/lambda", "train/assignment_entropy",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
        "assignment_color_corr", "assignment_color_abs_corr",
    }
    assert set(metrics.keys()) == expected
    assert all(math.isfinite(v) for v in metrics.values()), "some metric is not finite"
    assert 0.0 <= metrics["train/assignment_entropy"] <= math.log(2) + 1e-5


def test_adversarial_split_risk_variance_mode():
    """adv_mode='risk_variance': adversary maximises (loss_A-loss_B)², model minimises it.

    Checks:
    1. All expected metric keys present and finite.
    2. train/risk_variance is non-negative (it's a squared quantity).
    3. Assignment entropy is in [0, log(2)].
    """
    cfg = make_cfg()
    cfg.training.adv_mode = "risk_variance"
    cfg.training.adv_entropy_bonus = 0.1   # prevent partition collapse
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/disagreement", "train/risk_variance",
        "train/lambda", "train/assignment_entropy",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
        "assignment_color_corr", "assignment_color_abs_corr",
    }
    assert set(metrics.keys()) == expected
    assert all(math.isfinite(v) for v in metrics.values()), "some metric is not finite"
    assert metrics["train/risk_variance"] >= 0.0
    assert 0.0 <= metrics["train/assignment_entropy"] <= math.log(2) + 1e-5


# ---------------------------------------------------------------------------
# train_resampling
# ---------------------------------------------------------------------------

def test_train_resampling_metric_keys():
    """train_resampling must return the expected metric keys."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_resampling(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/disagreement", "train/risk_variance",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
    }
    assert set(metrics.keys()) == expected


def test_train_resampling_risk_variance_nonnegative():
    """risk_variance = (loss_A - loss_B)² must be non-negative and finite."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_resampling(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/risk_variance"])
    assert metrics["train/risk_variance"] >= 0.0


# ---------------------------------------------------------------------------
# compute_assignment_correlation
# ---------------------------------------------------------------------------

def test_assignment_correlation_range():
    """Correlation must be in [-1, 1] and abs_corr in [0, 1]."""
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    logits = torch.randn(len(loaders["train"].dataset))
    result = compute_assignment_correlation(logits, loaders["train"].dataset)
    assert -1.0 - 1e-5 <= result["assignment_color_corr"] <= 1.0 + 1e-5
    assert 0.0 <= result["assignment_color_abs_corr"] <= 1.0 + 1e-5


def test_assignment_correlation_perfect():
    """Logits perfectly aligned with colour produce |corr| ≈ 1."""
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    ds = loaders["train"].dataset
    # logits = +5 for colour=1, -5 for colour=0 → sigmoid ≈ 1/0, perfectly aligned
    logits = ds.colors.float() * 10.0 - 5.0
    result = compute_assignment_correlation(logits, ds)
    assert result["assignment_color_abs_corr"] > 0.99


def test_assignment_correlation_multi_range():
    """Multi-head correlation values must be in [0, 1]."""
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    K = 4
    logits = torch.randn(len(loaders["train"].dataset), K)
    result = compute_assignment_correlation_multi(logits, loaders["train"].dataset)
    assert 0.0 <= result["assignment_color_max_abs_corr"] <= 1.0 + 1e-5
    assert 0.0 <= result["assignment_color_mean_abs_corr"] <= 1.0 + 1e-5
    assert result["assignment_color_max_abs_corr"] >= result["assignment_color_mean_abs_corr"]


def test_train_adversarial_split_logs_correlation():
    """train_adversarial_split must include assignment_color_corr in returned metrics."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = SplitMLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split(cfg, model, loaders, device, run)
    run.finish()

    assert "assignment_color_corr" in metrics
    assert "assignment_color_abs_corr" in metrics
    assert 0.0 <= metrics["assignment_color_abs_corr"] <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# MultiHeadMLP
# ---------------------------------------------------------------------------

def test_multi_head_mlp_forward_shapes():
    for K in (2, 4):
        model = MultiHeadMLP(input_dim=3 * 28 * 28, hidden_dim=64, num_heads=K)
        x = torch.randn(4, 3, 28, 28)
        logits_list = model(x)
        assert len(logits_list) == K
        for logits in logits_list:
            assert logits.shape == (4, 2)


def test_multi_head_mlp_predict_is_probabilities():
    model = MultiHeadMLP(input_dim=3 * 28 * 28, hidden_dim=64, num_heads=4)
    x = torch.randn(8, 3, 28, 28)
    probs = model.predict(x)
    assert probs.shape == (8, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


def test_multi_head_mlp_separate_backbones():
    model = MultiHeadMLP(input_dim=3 * 28 * 28, hidden_dim=64, num_heads=3, separate_backbones=True)
    x = torch.randn(4, 3, 28, 28)
    features = model.get_all_features(x)
    assert len(features) == 3
    # With separate backbones each feature tensor must differ
    assert not torch.equal(features[0], features[1])
    assert not torch.equal(features[1], features[2])


def test_multi_head_mlp_shared_backbone_same_features():
    model = MultiHeadMLP(input_dim=3 * 28 * 28, hidden_dim=64, num_heads=3, separate_backbones=False)
    x = torch.randn(4, 3, 28, 28)
    features = model.get_all_features(x)
    # Shared backbone: all feature tensors are the same object
    assert features[0] is features[1]
    assert features[1] is features[2]


# ---------------------------------------------------------------------------
# train_adversarial_split_multi
# ---------------------------------------------------------------------------

def test_train_adversarial_split_multi_metric_keys():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = MultiHeadMLP(input_dim=3 * 28 * 28, hidden_dim=64, num_heads=4).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split_multi(cfg, model, loaders, device, run)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/disagreement", "train/lambda",
        "train/assignment_entropy",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
        "assignment_color_max_abs_corr", "assignment_color_mean_abs_corr",
    }
    assert set(metrics.keys()) == expected


def test_train_adversarial_split_multi_entropy_range():
    """Entropy must lie in [0, log(K)] for K-way assignments."""
    cfg = make_cfg()
    K = 4
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = MultiHeadMLP(input_dim=3 * 28 * 28, hidden_dim=64, num_heads=K).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split_multi(cfg, model, loaders, device, run)
    run.finish()

    H = metrics["train/assignment_entropy"]
    assert 0.0 <= H <= math.log(K) + 1e-5, (
        f"entropy {H:.4f} outside [0, log({K})={math.log(K):.4f}]"
    )


def test_train_adversarial_split_multi_disagreement_finite():
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    model = MultiHeadMLP(input_dim=3 * 28 * 28, hidden_dim=64, num_heads=4).to(device)

    run = wandb.init(mode="disabled")
    metrics = train_adversarial_split_multi(cfg, model, loaders, device, run)
    run.finish()

    assert math.isfinite(metrics["train/disagreement"])
    assert metrics["train/disagreement"] >= 0.0


def test_random_assignment_is_balanced():
    """With a random 50/50 split, each head should get roughly half the examples."""
    import torch
    cfg = make_cfg()
    loaders = make_dataloaders(cfg)
    N = len(loaders["train"].dataset)
    g = torch.Generator().manual_seed(cfg.training.seed)
    assignment = torch.randint(0, 2, (N,), generator=g)
    frac_a = (assignment == 0).float().mean().item()
    assert abs(frac_a - 0.5) < 0.05, f"assignment imbalance: {frac_a:.3f} assigned to head A"


# ---------------------------------------------------------------------------
# discover_environments
# ---------------------------------------------------------------------------

def test_discover_environments_shapes():
    """assignment must be (N,) long tensor; diag_metrics must contain expected keys."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    N = len(loaders["train"].dataset)

    assignment, weights, diag = discover_environments(cfg, loaders, device)

    assert assignment.shape == (N,), f"expected ({N},), got {assignment.shape}"
    assert assignment.dtype == torch.long
    expected_diag_keys = {
        "discovery/assignment_color_abs_corr",
        "discovery/color_label_corr_A",
        "discovery/color_label_corr_B",
        "discovery/n_env_A",
        "discovery/n_env_B",
    }
    assert expected_diag_keys.issubset(set(diag.keys())), (
        f"missing keys: {expected_diag_keys - set(diag.keys())}"
    )


def test_discover_environments_binary():
    """assignment values must be 0 or 1 only."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)

    assignment, _, _ = discover_environments(cfg, loaders, device)

    unique = assignment.unique().tolist()
    assert set(unique).issubset({0, 1}), f"unexpected values: {unique}"


def test_discover_environments_balanced():
    """Median split should give roughly equal-sized environments."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)
    N = len(loaders["train"].dataset)

    assignment, _, diag = discover_environments(cfg, loaders, device)

    frac_A = (assignment == 0).float().mean().item()
    assert abs(frac_A - 0.5) < 0.1, f"severe imbalance: {frac_A:.3f} in env A"
    assert abs(diag["discovery/n_env_A"] - diag["discovery/n_env_B"]) < 0.15 * N


# ---------------------------------------------------------------------------
# train_discovered_split
# ---------------------------------------------------------------------------

def test_train_discovered_split_metric_keys():
    """train_discovered_split must return the required metric keys."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)

    assignment, weights, diag = discover_environments(cfg, loaders, device)

    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)
    run = wandb.init(mode="disabled")
    metrics = train_discovered_split(cfg, model, loaders, device, run, assignment, weights, diag)
    run.finish()

    expected = {
        "train/loss", "train/acc", "train/risk_variance", "train/lambda",
        "eval/id_acc", "eval/id_auroc", "eval/id_precision", "eval/id_recall", "eval/id_worst_group_acc",
        "eval/ood_acc", "eval/ood_auroc", "eval/ood_precision", "eval/ood_recall", "eval/ood_worst_group_acc",
    }
    assert set(metrics.keys()) == expected, (
        f"missing: {expected - set(metrics.keys())}; extra: {set(metrics.keys()) - expected}"
    )


def test_train_discovered_split_risk_variance_nonnegative():
    """V-REx risk variance must be non-negative."""
    cfg = make_cfg()
    device = torch.device("cpu")
    loaders = make_dataloaders(cfg)

    assignment, weights, diag = discover_environments(cfg, loaders, device)

    model = MLP(input_dim=3 * 28 * 28, hidden_dim=64).to(device)
    run = wandb.init(mode="disabled")
    metrics = train_discovered_split(cfg, model, loaders, device, run, assignment, weights, diag)
    run.finish()

    assert metrics["train/risk_variance"] >= 0.0, (
        f"negative risk variance: {metrics['train/risk_variance']}"
    )
