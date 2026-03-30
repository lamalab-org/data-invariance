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


def train_oracle_split(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """Oracle split training loop — upper bound for learned partition methods.

    Assignment is fixed to the ground-truth color label: color=0 examples train
    head A (s=1), color=1 examples train head B (s=0). Within each head's data,
    color is constant and therefore non-predictive, forcing both heads to learn
    the true digit feature rather than the spurious color shortcut.

    This is the best partition we could ever give the adversary. If adversarial
    split can approach this OOD accuracy, the learned partition is doing its job.
    If it can't, the adversary hasn't found the color-correlated split.

    The training loop is identical to train_random_split — only the assignment
    source differs (color label vs random coin flip).

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
        epoch_disagree = 0.0

        for batch in loaders["train"]:
            x = batch["image"].to(device)    # (B, 3, 28, 28)
            y = batch["label"].to(device)    # (B,) int64
            # color=0 → red → head A (s=1); color=1 → green → head B (s=0)
            s = (1 - batch["color"]).float().to(device)   # (B,) ∈ {0.0, 1.0}

            logits_a, logits_b = model(x)    # each (B, 2)

            ce_a = F.cross_entropy(logits_a, y, reduction="none")  # (B,)
            ce_b = F.cross_entropy(logits_b, y, reduction="none")  # (B,)

            loss_a = (s * ce_a).mean()
            loss_b = ((1.0 - s) * ce_b).mean()
            disagree = symmetric_kl(logits_a, logits_b)

            loss = loss_a + loss_b + cfg.training.lambda_disagree * disagree

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y)
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


def train_adversarial_split(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """Adversarial split + KL disagreement penalty training loop.

    The adversary maintains one learnable scalar logit per training example.
    sigmoid(logit_i) = s_i ∈ (0,1): s_i ≈ 1 routes example i to head A,
    s_i ≈ 0 routes it to head B.

    Each batch alternates two gradient steps:

    1. **Model step** — minimise weighted task loss + lambda * KL disagreement.
       s is detached so model gradients don't flow back through assignment logits.

    2. **Adversary step(s)** — maximise the weighted task loss. KL has zero
       gradient w.r.t. s_i (depends only on model weights), so the adversary
       maximises -(s*CE_A + (1-s)*CE_B), pushing each example toward its harder
       head, which causes specialisation and drives disagreement indirectly.
       ce_a/ce_b are detached from the model graph so model weights are fixed.
       Running adv_steps_per_model_step > 1 lets the adversary commit to a
       partition before the model can re-equalise the heads.

    Symmetry-breaking options (all off by default):
    - adv_init="random"        — non-uniform starting partition → CE_A ≠ CE_B
                                  from step 1, giving the adversary an immediate
                                  gradient signal.
    - head_noise > 0           — independent Gaussian noise added to backbone
                                  features before each head, preventing both heads
                                  collapsing to identical linear maps.
    - adv_warmup_epochs > 0    — assignment logits are frozen for the first N
                                  epochs (combined with adv_init="random": heads
                                  pre-differentiate on a fixed random partition
                                  before the adversary starts optimising it).
    - lambda_warmup_epochs > 0 — lambda is ramped linearly from 0 to
                                  lambda_disagree over N epochs, letting heads
                                  diverge before the disagreement penalty kicks in.

    Logged metrics:
    - train/disagreement       — KL between heads (same diagnostic as random split)
    - train/assignment_entropy — H(s_i) averaged over all training examples.
                                  Max = log(2) ≈ 0.693 (uniform); min = 0 (hard).
                                  Should decrease as the adversary commits.

    Returns the final epoch's metrics dict.
    """
    train_ds = loaders["train"].dataset
    N = len(train_ds)

    # --- Assignment logit initialisation ---
    # "zeros"  → sigmoid(0) = 0.5 everywhere; the adversary starts blind.
    # "random" → N(0, adv_init_scale²); immediately non-uniform so CE_A ≠ CE_B
    #            and the adversary has a real gradient from the very first step.
    #            Seed offset (+1) so this draw is independent of data/model seeds.
    if cfg.training.adv_init == "zeros":
        init_vals = torch.zeros(N)
    else:
        g = torch.Generator().manual_seed(cfg.training.seed + 1)
        init_vals = torch.randn(N, generator=g) * cfg.training.adv_init_scale
    assignment_logits = init_vals.to(device).requires_grad_(True)

    # Plain Adam for the adversary: AdamW's weight decay pulls logits toward 0
    # (sigmoid → 0.5, uniform split), directly fighting the adversary objective.
    adv_optimizer = torch.optim.Adam([assignment_logits], lr=cfg.training.adv_lr)
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    metrics: dict[str, float] = {}
    prev_disagree = 0.0   # previous epoch's mean disagreement — used for threshold gating
    for epoch in range(cfg.training.epochs):
        # --- Effective lambda for this epoch ---
        # Two independent schedules; the effective lambda is their minimum so that
        # BOTH conditions must be satisfied before the full penalty is applied.
        #
        # 1. Epoch-based warmup: ramp linearly from 0 to lambda_disagree over N epochs.
        warmup_e = cfg.training.lambda_warmup_epochs
        lam_epoch = cfg.training.lambda_disagree * (min(1.0, epoch / warmup_e) if warmup_e > 0 else 1.0)
        #
        # 2. Disagreement-threshold gating: hold lambda=0 until the heads have
        #    diverged enough, then ramp over lambda_ramp_range KL units.
        #    Uses prev_disagree (end of previous epoch) so it is lag-free within
        #    the current epoch and doesn't require an extra forward pass.
        thr = cfg.training.lambda_threshold
        if thr > 0.0:
            if prev_disagree < thr:
                lam_disagree = 0.0
            elif cfg.training.lambda_ramp_range > 0.0:
                progress = (prev_disagree - thr) / cfg.training.lambda_ramp_range
                lam_disagree = cfg.training.lambda_disagree * min(1.0, progress)
            else:
                lam_disagree = cfg.training.lambda_disagree   # hard step
        else:
            lam_disagree = cfg.training.lambda_disagree       # threshold disabled
        #
        lam = min(lam_epoch, lam_disagree)

        # Warmup phase: adversary is frozen. Assignment logits stay fixed so the
        # model pre-differentiates the heads on the current (possibly random) partition.
        adversary_active = epoch >= cfg.training.adv_warmup_epochs

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        epoch_disagree = 0.0

        for batch in loaders["train"]:
            x = batch["image"].to(device)    # (B, 3, 28, 28)
            y = batch["label"].to(device)    # (B,) int64
            idx = batch["index"].to(device)  # (B,) global example indices

            # --- Forward pass ---
            # get_features() works for both shared and separate-backbone models.
            # Noise is injected here (training only) — evaluate() uses predict()
            # which goes through the clean forward path without noise.
            features_a, features_b = model.get_features(x)   # (B, hidden_dim) each

            if cfg.training.head_noise > 0.0:
                # Independent draws per head — prevents collapse to identical maps
                # even when the backbone is shared (features_a is features_b).
                features_a = features_a + torch.randn_like(features_a) * cfg.training.head_noise
                features_b = features_b + torch.randn_like(features_b) * cfg.training.head_noise

            logits_a = model.head_a(features_a)   # (B, 2)
            logits_b = model.head_b(features_b)   # (B, 2)

            ce_a = F.cross_entropy(logits_a, y, reduction="none")   # (B,)
            ce_b = F.cross_entropy(logits_b, y, reduction="none")   # (B,)

            # --- Model step ---
            s = assignment_logits[idx].sigmoid().detach()   # (B,) — no grad through logits
            loss_a = (s * ce_a).mean()
            loss_b = ((1.0 - s) * ce_b).mean()
            disagree = symmetric_kl(logits_a, logits_b)

            model_loss = loss_a + loss_b + lam * disagree
            model_optimizer.zero_grad()
            model_loss.backward()
            model_optimizer.step()

            # --- Adversary step(s) ---
            # ce_a / ce_b are detached from the current model graph; model is fixed.
            # Multiple steps exploit the same per-example losses — this is equivalent
            # to a larger effective adversary learning rate with near-zero extra compute.
            #
            # Optional entropy bonus: -β * H(s_i) is subtracted from adv_loss,
            # penalising soft (uncertain) assignments. This pushes the adversary
            # toward hard 0/1 splits faster than the task-loss gradient alone.
            # β = 0 recovers the original formulation.
            if adversary_active:
                for _ in range(cfg.training.adv_steps_per_model_step):
                    s_adv = assignment_logits[idx].sigmoid()   # (B,) — live grad
                    task_term = -(s_adv * ce_a.detach() + (1.0 - s_adv) * ce_b.detach()).mean()
                    if cfg.training.adv_entropy_bonus > 0.0:
                        s_c = s_adv.clamp(1e-7, 1.0 - 1e-7)
                        batch_entropy = (-s_c * s_c.log() - (1.0 - s_c) * (1.0 - s_c).log()).mean()
                        adv_loss = task_term - cfg.training.adv_entropy_bonus * batch_entropy
                    else:
                        adv_loss = task_term
                    adv_optimizer.zero_grad()
                    adv_loss.backward()
                    adv_optimizer.step()
                    with torch.no_grad():
                        assignment_logits.clamp_(-5.0, 5.0)

            epoch_loss += model_loss.item() * len(y)
            avg_probs = (logits_a.softmax(1) + logits_b.softmax(1)) / 2
            epoch_correct += (avg_probs.argmax(1) == y).sum().item()
            epoch_n += len(y)
            epoch_disagree += disagree.item() * len(y)

        # Assignment entropy: H(s_i) averaged over all training examples.
        # Clamp guards log(0) even though logit clamping already bounds s to [0.007, 0.993].
        with torch.no_grad():
            s_all = assignment_logits.sigmoid().clamp(1e-7, 1.0 - 1e-7)
            entropy = (-s_all * s_all.log() - (1.0 - s_all) * (1.0 - s_all).log()).mean()

        prev_disagree = epoch_disagree / epoch_n   # for next epoch's threshold check

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            "train/disagreement": epoch_disagree / epoch_n,
            "train/lambda": lam,             # shows exactly when threshold fires in wandb
            "train/assignment_entropy": entropy.item(),
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
