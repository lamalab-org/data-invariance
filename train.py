from __future__ import annotations

import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import ColoredMNIST, ContinuousCMNIST, MultiSpuriousCMNIST, TADFDataset, WaterbirdsDataset
from evaluate import compute_assignment_correlation, compute_assignment_correlation_multi
from models import MLP, make_resnet_backbone
from utils import log_metrics


def make_dataloaders(cfg: DictConfig) -> dict[str, DataLoader]:
    """Build train, ID-test, and OOD-test dataloaders.

    Dispatches on cfg.dataset.name to construct the right dataset class.
    Returns {"train": ..., "id_test": ..., "ood_test": ...}.
    """
    if cfg.dataset.name == "cmnist":
        seed = cfg.training.seed
        noise = cfg.dataset.label_noise
        data_dir = cfg.dataset.data_dir

        train_ds = ColoredMNIST(cfg.dataset.train_correlation, label_noise=noise, split="train", data_dir=data_dir, seed=seed)
        id_test_ds = ColoredMNIST(cfg.dataset.train_correlation, label_noise=noise, split="test", data_dir=data_dir, seed=seed)
        ood_test_ds = ColoredMNIST(cfg.dataset.test_correlation, label_noise=noise, split="test", data_dir=data_dir, seed=seed)

    elif cfg.dataset.name == "continuous_cmnist":
        seed = cfg.training.seed
        noise = cfg.dataset.label_noise
        data_dir = cfg.dataset.data_dir
        beta_c = cfg.dataset.beta_concentration

        train_ds = ContinuousCMNIST(cfg.dataset.env_correlation, label_noise=noise, split="train", data_dir=data_dir, seed=seed, beta_concentration=beta_c)
        id_test_ds = ContinuousCMNIST(cfg.dataset.env_correlation, label_noise=noise, split="test", data_dir=data_dir, seed=seed, beta_concentration=beta_c)
        ood_test_ds = ContinuousCMNIST(cfg.dataset.test_correlation, label_noise=noise, split="test", data_dir=data_dir, seed=seed, beta_concentration=beta_c)

    elif cfg.dataset.name == "multi_cmnist":
        seed = cfg.training.seed
        noise = cfg.dataset.label_noise
        data_dir = cfg.dataset.data_dir

        train_ds = MultiSpuriousCMNIST(
            color_correlation=cfg.dataset.color_correlation,
            brightness_correlation=cfg.dataset.brightness_correlation,
            label_noise=noise, split="train", data_dir=data_dir, seed=seed,
        )
        id_test_ds = MultiSpuriousCMNIST(
            color_correlation=cfg.dataset.color_correlation,
            brightness_correlation=cfg.dataset.brightness_correlation,
            label_noise=noise, split="test", data_dir=data_dir, seed=seed,
        )
        # OOD: both correlations flipped
        ood_test_ds = MultiSpuriousCMNIST(
            color_correlation=cfg.dataset.test_color_correlation,
            brightness_correlation=cfg.dataset.test_brightness_correlation,
            label_noise=noise, split="test", data_dir=data_dir, seed=seed,
        )

    elif cfg.dataset.name == "tadf":
        seed = cfg.training.seed
        ppath = cfg.dataset.parquet_path
        spur_prop = getattr(cfg.dataset, "spurious_property", None)
        spur_corr = getattr(cfg.dataset, "spurious_correlation", 0.9)

        train_ds = TADFDataset(
            parquet_path=ppath, split="train", seed=seed,
            spurious_property=spur_prop, spurious_correlation=spur_corr,
        )
        # ID test: full test set (model selection)
        id_test_ds = TADFDataset(
            parquet_path=ppath, split="test", seed=seed,
        )
        # OOD test: only misaligned test examples (spurious ≠ label)
        # These are the counterexamples the model hasn't seen during biased training.
        ood_test_ds = TADFDataset(
            parquet_path=ppath, split="test_misaligned", seed=seed,
            spurious_property=spur_prop,
        )

    elif cfg.dataset.name == "waterbirds":
        data_dir = cfg.dataset.data_dir
        train_ds = WaterbirdsDataset(split="train", data_dir=data_dir)
        id_test_ds = WaterbirdsDataset(split="val", data_dir=data_dir)
        ood_test_ds = WaterbirdsDataset(split="test", data_dir=data_dir)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

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
    # Clamp probabilities (not log-probs) so that log(p) stays finite.
    # F.kl_div is avoided because it recomputes log(target) internally — on MPS,
    # target values near 0 underflow to exactly 0.0, giving 0 * log(0) = NaN.
    # Explicit formula with clamped probs is safe on every device.
    # eps=1e-7: log(1e-7) ≈ -16, so symmetric KL is bounded at ≈ 16 nats max —
    # much tighter than the previous -100 clamp and prevents gradient explosion.
    eps = 1e-7
    pa = logits_a.softmax(dim=1).clamp(min=eps)   # (B, C)
    pb = logits_b.softmax(dim=1).clamp(min=eps)   # (B, C)
    log_pa = pa.log()
    log_pb = pb.log()
    kl_ab = (pa * (log_pa - log_pb)).sum(dim=1).mean()   # KL(A || B)
    kl_ba = (pb * (log_pb - log_pa)).sum(dim=1).mean()   # KL(B || A)
    return 0.5 * (kl_ab + kl_ba)


def _val_score(id_metrics: dict[str, float]) -> float:
    """Validation score for model selection (higher = better).

    Uses worst-group accuracy when available (Waterbirds, multi-spurious);
    falls back to negative loss for standard CMNIST.
    """
    if "worst_group_acc" in id_metrics:
        return id_metrics["worst_group_acc"]
    return -id_metrics["loss"]


class _ModelSelector:
    """Track the best model checkpoint by validation score."""

    def __init__(self):
        import copy as _copy
        self._copy = _copy
        self.best_score = float("-inf")
        self.best_state = None
        self.best_metrics: dict[str, float] = {}

    def update(self, val_score: float, model: torch.nn.Module, metrics: dict[str, float]) -> None:
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_state = self._copy.deepcopy(model.state_dict())
            self.best_metrics = dict(metrics)

    def restore(self, model: torch.nn.Module) -> dict[str, float]:
        """Restore best checkpoint and return its metrics."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return self.best_metrics if self.best_metrics else {}


def _eval_metrics(prefix: str, m: dict[str, float]) -> dict[str, float]:
    """Build prefixed metric dict from evaluate() output, including worst_group_acc if present."""
    d = {
        f"{prefix}_acc": m["acc"],
        f"{prefix}_auroc": m["auroc"],
        f"{prefix}_precision": m["precision"],
        f"{prefix}_recall": m["recall"],
    }
    if "worst_group_acc" in m:
        d[f"{prefix}_worst_group_acc"] = m["worst_group_acc"]
    return d


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate model on a dataloader.

    Returns accuracy, precision, recall, AUROC, loss, and worst-group accuracy.
    Worst-group accuracy is the minimum accuracy across the 4 (label, spurious)
    groups — the standard metric for Waterbirds and other group-robustness
    benchmarks.  It is computed whenever the batch contains a "spurious" key.
    """
    model.eval()

    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    auroc = torchmetrics.AUROC(task="binary").to(device)

    total_loss = 0.0
    total_n = 0

    # Collect per-example predictions and group labels for worst-group accuracy.
    all_preds = []
    all_labels = []
    all_spurious = []
    has_spurious = False

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            probs = model.predict(x)          # (B, 2)
            preds = probs.argmax(dim=1)       # (B,)

            total_loss += F.nll_loss(probs.clamp(min=1e-7).log(), y, reduction="sum").item()
            total_n += len(y)

            accuracy.update(preds, y)
            precision.update(preds, y)
            recall.update(preds, y)
            auroc.update(probs[:, 1], y)

            if "spurious" in batch:
                has_spurious = True
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
                s = batch["spurious"]
                if isinstance(s, torch.Tensor):
                    all_spurious.append(s.clone())
                else:
                    all_spurious.append(torch.tensor(s if not isinstance(s, int) else [s]))

    result = {
        "acc": accuracy.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "auroc": auroc.compute().item(),
        "loss": total_loss / total_n,
    }

    # Worst-group accuracy: min accuracy over (label, spurious) groups.
    if has_spurious:
        preds_t = torch.cat(all_preds)
        labels_t = torch.cat(all_labels)
        spurious_t = torch.cat(all_spurious)
        # Group index: label * 2 + spurious → {0, 1, 2, 3}
        groups = labels_t * 2 + spurious_t
        group_accs = {}
        for g in range(4):
            mask = groups == g
            if mask.any():
                group_accs[g] = (preds_t[mask] == labels_t[mask]).float().mean().item()
        if group_accs:
            result["worst_group_acc"] = min(group_accs.values())

    return result


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

    Returns metrics of the best validation checkpoint.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0

        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_n += len(y)

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_metrics), model, metrics)

    return selector.restore(model)


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
    selector = _ModelSelector()
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
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_metrics), model, metrics)

    return selector.restore(model)


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
    selector = _ModelSelector()
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
            s = (1 - batch["spurious"]).float().to(device)   # (B,) ∈ {0.0, 1.0}

            logits_a, logits_b = model(x)    # each (B, 2)

            ce_a = F.cross_entropy(logits_a, y, reduction="none")  # (B,)
            ce_b = F.cross_entropy(logits_b, y, reduction="none")  # (B,)

            loss_a = (s * ce_a).mean()
            loss_b = ((1.0 - s) * ce_b).mean()
            # Oracle split: task loss only — no KL penalty.
            # The oracle partition is the upper bound for partition quality; adding
            # a KL penalty would fight the very specialisation we're trying to measure.
            # We still compute disagree for logging so the curve is comparable to
            # other methods, but it does not enter the gradient.
            loss = loss_a + loss_b
            with torch.no_grad():
                disagree = symmetric_kl(logits_a, logits_b)

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
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_metrics), model, metrics)

    return selector.restore(model)


def train_resampling(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """V-REx training with per-step random environments (resampling).

    At every gradient step the current mini-batch is split randomly into two
    equal halves. Head A is evaluated on the first half, head B on the second.
    A fresh random split is drawn every step, so there are no global per-example
    assignment parameters and no second optimiser.

    The model penalty is the **risk variance** between the two heads:

        model_loss = loss_A + loss_B + lambda * (loss_A − loss_B)²

    The squared difference term is the V-REx penalty (Krueger et al., 2021).
    Minimising it forces the model to perform equally on all data compositions.
    If a feature (e.g. colour) is useful only in a particular data composition,
    it will inflate (loss_A − loss_B)² and be penalised.

    KL(head_A || head_B) is logged as `train/disagreement` for comparability
    with the other split methods but does **not** enter the gradient.

    Returns the final epoch's metrics dict.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    # Separate generator so the per-step splits are independent of data/model seeds.
    split_rng = torch.Generator(device=device)
    split_rng.manual_seed(cfg.training.seed + 99)

    metrics: dict[str, float] = {}
    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        epoch_disagree = 0.0
        epoch_risk_var = 0.0

        for batch in loaders["train"]:
            x = batch["image"].to(device)    # (B, 3, 28, 28)
            y = batch["label"].to(device)    # (B,) int64
            B = len(y)

            logits_a, logits_b = model(x)    # (B, 2) each

            ce_a = F.cross_entropy(logits_a, y, reduction="none")  # (B,)
            ce_b = F.cross_entropy(logits_b, y, reduction="none")  # (B,)

            # Fresh random 50/50 split of this mini-batch.
            # idx_A → head A's environment this step; idx_B → head B's.
            perm = torch.randperm(B, device=device, generator=split_rng)
            half = B // 2
            idx_A = perm[:half]
            idx_B = perm[half:]

            loss_A = ce_a[idx_A].mean()   # head A task loss on its half
            loss_B = ce_b[idx_B].mean()   # head B task loss on its half

            # V-REx risk-variance penalty: (loss_A − loss_B)²
            # Gradient: 2*(loss_A−loss_B) * (∂loss_A/∂θ − ∂loss_B/∂θ)
            # This pushes the model toward equal performance on all splits.
            risk_var = (loss_A - loss_B) ** 2

            # KL logged for comparability; not in the gradient.
            with torch.no_grad():
                disagree = symmetric_kl(logits_a, logits_b)

            model_loss = loss_A + loss_B + cfg.training.lambda_disagree * risk_var

            optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += model_loss.item() * B
            avg_probs = (logits_a.softmax(1) + logits_b.softmax(1)) / 2
            epoch_correct += (avg_probs.argmax(1) == y).sum().item()
            epoch_n += B
            epoch_disagree += disagree.item() * B
            epoch_risk_var += risk_var.item() * B

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            "train/disagreement": epoch_disagree / epoch_n,
            "train/risk_variance": epoch_risk_var / epoch_n,
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_metrics), model, metrics)

    return selector.restore(model)


def train_adversarial_split(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """Adversarial split + disagreement penalty training loop.

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
    elif cfg.training.adv_init == "oracle":
        # Initialize using ground-truth color labels: color=0 → logit=+5 (s≈1, assign to head A),
        # color=1 → logit=-5 (s≈0, assign to head B). This gives the adversary the correct
        # partition from the start — a sanity check to verify the mechanism works when discovery
        # is not required. If OOD improves over ERM here, the mechanism is sound; the only
        # remaining challenge is the bootstrap/discovery problem.
        colors = train_ds.spurious  # (N,) tensor of 0/1 ground-truth spurious labels
        init_vals = torch.where(colors == 0, torch.tensor(5.0), torch.tensor(-5.0))
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
    selector = _ModelSelector()
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
        epoch_risk_var = 0.0

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

            # Two penalty modes for the model:
            #   "risk_variance" → (loss_a − loss_b)²: V-REx penalty, forces equal per-
            #                     environment risk. Adversary (below) maximises this.
            #   anything else   → lambda * KL: prediction-level disagreement penalty.
            risk_var = (loss_a - loss_b) ** 2
            if cfg.training.adv_mode == "risk_variance":
                model_loss = loss_a + loss_b + lam * risk_var
            else:
                model_loss = loss_a + loss_b + lam * disagree

            model_optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()

            # --- Adversary step(s) ---
            # Model is fixed; only assignment_logits are updated here.
            # Three modes controlled by cfg.training.adv_mode:
            #
            # "task_loss" (default): maximise -(s·CE_A + (1-s)·CE_B).
            #   Pushes examples toward harder heads → easy/hard partition.
            #
            # "risk_variance": maximise (loss_a − loss_b)².
            #   Paired with the model's V-REx penalty (minimise same term).
            #   Finds the split with maximum risk difference; model is forced to
            #   perform equally on that worst-case split → invariant predictions.
            #   adv_entropy_bonus regularises the partition to prevent collapse.
            #
            # "grad_div": minimise cosine similarity of gradient directions (experimental).
            if adversary_active:
                for _ in range(cfg.training.adv_steps_per_model_step):
                    s_adv = assignment_logits[idx].sigmoid()   # (B,) — live grad

                    if cfg.training.adv_mode == "risk_variance":
                        # Adversary maximises the risk variance to find worst-case split.
                        # ce_a/ce_b are detached — model weights fixed during adv step.
                        loss_a_adv = (s_adv * ce_a.detach()).mean()
                        loss_b_adv = ((1.0 - s_adv) * ce_b.detach()).mean()
                        adv_base = -(loss_a_adv - loss_b_adv) ** 2
                        if cfg.training.adv_entropy_bonus > 0.0:
                            s_c = s_adv.clamp(1e-7, 1.0 - 1e-7)
                            batch_entropy = (-s_c * s_c.log() - (1.0 - s_c) * (1.0 - s_c).log()).mean()
                            adv_loss = adv_base - cfg.training.adv_entropy_bonus * batch_entropy
                        else:
                            adv_loss = adv_base

                    elif cfg.training.adv_mode == "grad_div":
                        with torch.no_grad():
                            err_A = logits_a.softmax(1) - F.one_hot(y, num_classes=2).float()
                            err_B = logits_b.softmax(1) - F.one_hot(y, num_classes=2).float()
                            delta_A = err_A @ model.head_a.weight   # (B, hidden_dim)
                            delta_B = err_B @ model.head_b.weight   # (B, hidden_dim)
                        g_A = (s_adv.unsqueeze(1) * delta_A).sum(dim=0)
                        g_B = ((1.0 - s_adv).unsqueeze(1) * delta_B).sum(dim=0)
                        adv_loss = F.cosine_similarity(
                            g_A.unsqueeze(0), g_B.unsqueeze(0), dim=1, eps=1e-8
                        ).squeeze()

                    else:
                        # "task_loss" mode (default)
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
            epoch_risk_var += risk_var.item() * len(y)

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
            "train/risk_variance": epoch_risk_var / epoch_n,
            "train/lambda": lam,
            "train/assignment_entropy": entropy.item(),
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        # After the final epoch, compute how well the assignment tracks colour.
        # Logged only once so it appears as a scalar summary in wandb rather than
        # a curve — it's a property of the converged partition, not a trajectory.
        if epoch == cfg.training.epochs - 1:
            corr_metrics = compute_assignment_correlation(assignment_logits, loaders["train"].dataset)
            metrics.update(corr_metrics)

        log_metrics(run, metrics, step=epoch)

    return metrics

def train_adversarial_split_multi(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """Adversarial split for K>2 heads using softmax (K-simplex) assignments.

    Generalises train_adversarial_split from binary (sigmoid) to K-way (softmax).

    Assignment: (N, K) logits; s_ik = softmax(logits_i)[k] with sum_k s_ik = 1.
    Adversary pushes each example toward its hardest head — gradient of
    -(sum_k s_ik * CE_k(i)) w.r.t. logit_ij is -s_ij*(CE_j - avg_CE), negative
    when head j has above-average CE, so s_ij increases.

    Disagreement: average symmetric KL over all K(K-1)/2 head pairs.

    Assignment entropy: H(s_i) = -sum_k s_ik log s_ik, averaged over examples.
    Max = log(K) (uniform); min = 0 (hard single-head assignment).

    Lambda scheduling (threshold + warmup) identical to binary case.
    Correlation diagnostic uses compute_assignment_correlation_multi.

    Returns the final epoch's metrics dict.
    """
    train_ds = loaders["train"].dataset
    N = len(train_ds)
    K = model.num_heads
    n_pairs = K * (K - 1) // 2

    if cfg.training.adv_init == "zeros":
        init_vals = torch.zeros(N, K)
    else:
        g = torch.Generator().manual_seed(cfg.training.seed + 1)
        init_vals = torch.randn(N, K, generator=g) * cfg.training.adv_init_scale
    assignment_logits = init_vals.to(device).requires_grad_(True)

    adv_optimizer = torch.optim.Adam([assignment_logits], lr=cfg.training.adv_lr)
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    metrics: dict[str, float] = {}
    prev_disagree = 0.0
    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        # Lambda schedules — identical logic to binary case
        warmup_e = cfg.training.lambda_warmup_epochs
        lam_epoch = cfg.training.lambda_disagree * (min(1.0, epoch / warmup_e) if warmup_e > 0 else 1.0)

        thr = cfg.training.lambda_threshold
        if thr > 0.0:
            if prev_disagree < thr:
                lam_disagree = 0.0
            elif cfg.training.lambda_ramp_range > 0.0:
                lam_disagree = cfg.training.lambda_disagree * min(
                    1.0, (prev_disagree - thr) / cfg.training.lambda_ramp_range
                )
            else:
                lam_disagree = cfg.training.lambda_disagree
        else:
            lam_disagree = cfg.training.lambda_disagree

        lam = min(lam_epoch, lam_disagree)
        adversary_active = epoch >= cfg.training.adv_warmup_epochs

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        epoch_disagree = 0.0

        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            features_list = model.get_all_features(x)   # K × (B, hidden_dim)
            if cfg.training.head_noise > 0.0:
                features_list = [
                    f + torch.randn_like(f) * cfg.training.head_noise
                    for f in features_list
                ]
            logits_list = [model.heads[k](features_list[k]) for k in range(K)]   # K × (B,2)
            ce_list = [F.cross_entropy(logits_list[k], y, reduction="none") for k in range(K)]

            # Average pairwise symmetric KL
            disagree = sum(
                symmetric_kl(logits_list[j], logits_list[k])
                for j in range(K) for k in range(j + 1, K)
            ) / n_pairs

            # Model step
            s = assignment_logits[idx].softmax(dim=1).detach()   # (B, K)
            task_loss = sum((s[:, k] * ce_list[k]).mean() for k in range(K))
            model_loss = task_loss + lam * disagree
            model_optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()

            # Adversary step(s)
            if adversary_active:
                for _ in range(cfg.training.adv_steps_per_model_step):
                    s_adv = assignment_logits[idx].softmax(dim=1)   # (B, K)
                    task_term = -sum(
                        (s_adv[:, k] * ce_list[k].detach()).mean() for k in range(K)
                    )
                    if cfg.training.adv_entropy_bonus > 0.0:
                        s_c = s_adv.clamp(1e-7, 1.0 - 1e-7)
                        batch_entropy = -(s_c * s_c.log()).sum(dim=1).mean()
                        adv_loss = task_term - cfg.training.adv_entropy_bonus * batch_entropy
                    else:
                        adv_loss = task_term
                    adv_optimizer.zero_grad()
                    adv_loss.backward()
                    adv_optimizer.step()
                    with torch.no_grad():
                        assignment_logits.clamp_(-5.0, 5.0)

            epoch_loss += model_loss.item() * len(y)
            avg_probs = sum(logits_list[k].softmax(1) for k in range(K)) / K
            epoch_correct += (avg_probs.argmax(1) == y).sum().item()
            epoch_n += len(y)
            epoch_disagree += disagree.item() * len(y)

        # H(s_i) = -sum_k s_ik log s_ik, max = log(K)
        with torch.no_grad():
            s_all = assignment_logits.softmax(dim=1).clamp(1e-7, 1.0 - 1e-7)
            entropy = -(s_all * s_all.log()).sum(dim=1).mean()

        prev_disagree = epoch_disagree / epoch_n

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            "train/disagreement": epoch_disagree / epoch_n,
            "train/lambda": lam,
            "train/assignment_entropy": entropy.item(),
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }

        if epoch == cfg.training.epochs - 1:
            corr_metrics = compute_assignment_correlation_multi(
                assignment_logits, loaders["train"].dataset
            )
            metrics.update(corr_metrics)

        log_metrics(run, metrics, step=epoch)

    return metrics


# ---------------------------------------------------------------------------
# Two-phase environment discovery
# ---------------------------------------------------------------------------

def discover_environments(
    cfg: DictConfig,
    loaders: dict[str, DataLoader],
    device: torch.device,
    existing_model: torch.nn.Module | None = None,
    return_model: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]] | tuple[torch.Tensor, torch.Tensor, dict[str, float], torch.nn.Module]:
    """Score training examples and split into environments.

    When ``existing_model`` is None (default), trains a throw-away ERM for
    scoring.  When provided (iterative refinement), uses that model directly.

    Pipeline:
        1. Score each example (loss, entropy, or counterfactual sensitivity).
        2. Split into K environments by score rank.
        3. Optionally upweight high-score examples.

    Returns:
        assignment  - (N,) long: environment index per example.
        weights     - (N,) float: per-example importance weights.
        diag_metrics - discovery diagnostics.
    """
    train_ds = loaders["train"].dataset
    N = len(train_ds)

    freeze_bb = getattr(cfg.training, "freeze_backbone", False)

    if existing_model is not None:
        # Iterative refinement: reuse the model from the previous round.
        disc = existing_model
    else:
        # Train a throw-away ERM for scoring.
        # When freeze_backbone is requested, the discovery ERM must fine-tune
        # fully so its backbone learns dataset-specific features.  These
        # features are then frozen for V-REx training.
        if cfg.dataset.arch == "resnet":
            # Cartography needs the ERM to make mistakes → fine-tune fully.
            # Other criteria work well with frozen backbone (faster).
            freeze_disc = cfg.training.discovery_criterion != "cartography"
            backbone, out_dim = make_resnet_backbone(freeze=freeze_disc)
            disc = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
        else:
            disc = MLP(input_dim=train_ds.input_dim, hidden_dim=cfg.model.hidden_dim).to(device)
        opt = torch.optim.AdamW(disc.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

        # Track per-example losses across the last few epochs for stable scoring.
        n_epochs_disc = cfg.training.discovery_epochs
        avg_window = max(1, n_epochs_disc // 2)  # average over last half of training
        accumulated_losses = torch.zeros(N)
        accumulation_count = 0

        for ep in range(n_epochs_disc):
            disc.train()
            for batch in loaders["train"]:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                loss = F.cross_entropy(disc(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Accumulate per-example losses in the averaging window.
            if ep >= n_epochs_disc - avg_window:
                disc.eval()
                with torch.no_grad():
                    for batch in loaders["train"]:
                        x_b = batch["image"].to(device)
                        y_b = batch["label"].to(device)
                        idx_b = batch["index"]
                        ce = F.cross_entropy(disc(x_b), y_b, reduction="none").cpu()
                        accumulated_losses[idx_b] += ce
                accumulation_count += 1

        # Pre-computed average losses for loss-based scoring criteria.
        avg_losses = accumulated_losses / max(accumulation_count, 1)

    # Score every training example.
    criterion = cfg.training.discovery_criterion
    K = cfg.training.num_discovery_envs
    disc.eval()
    scores = torch.zeros(N)

    if criterion == "permutation":
        # Permutation-based discovery: find which features the ERM relies on
        # most (global importance), then score each example by how much it
        # depends on those features (per-example sensitivity).
        #
        # Step 1: Global feature importance — which features matter most?
        # Permute each feature, measure average loss change.
        # For efficiency, sample a random subset of features.
        input_dim = train_ds.input_dim
        n_features_to_test = min(200, input_dim)
        g_feat = torch.Generator().manual_seed(cfg.training.seed + 555)
        test_features = torch.randperm(input_dim, generator=g_feat)[:n_features_to_test]

        # Compute baseline losses.
        baseline_losses = torch.zeros(N)
        all_x = []
        all_y = []
        all_idx = []
        with torch.no_grad():
            for batch in loaders["train"]:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                idx = batch["index"]
                baseline_losses[idx] = F.cross_entropy(disc(x), y, reduction="none").cpu()
                all_x.append(x.cpu())
                all_y.append(y.cpu())
                all_idx.append(idx)

        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)
        # For tabular data, all_x is (N, D). For images, (N, C, H, W).
        is_tabular = all_x.dim() == 2

        if is_tabular:
            # Step 1: Global importance per feature.
            feat_importance = torch.zeros(n_features_to_test)
            with torch.no_grad():
                for fi, feat_idx in enumerate(test_features):
                    x_perm = all_x.clone()
                    perm = torch.randperm(N, generator=g_feat)
                    x_perm[:, feat_idx] = x_perm[perm, feat_idx]
                    perm_loss = F.cross_entropy(
                        disc(x_perm.to(device)), all_y.to(device), reduction="none"
                    ).cpu()
                    feat_importance[fi] = (perm_loss - baseline_losses).mean()

            # Top-K most important features = likely shortcuts.
            K_top = min(10, n_features_to_test)
            top_feat_indices = test_features[feat_importance.argsort(descending=True)[:K_top]]

            # Step 2: Per-example sensitivity to the top features.
            # For each example, permute the top features and measure loss change.
            per_example_sensitivity = torch.zeros(N)
            with torch.no_grad():
                for feat_idx in top_feat_indices:
                    x_perm = all_x.clone()
                    perm = torch.randperm(N, generator=g_feat)
                    x_perm[:, feat_idx] = x_perm[perm, feat_idx]
                    perm_loss = F.cross_entropy(
                        disc(x_perm.to(device)), all_y.to(device), reduction="none"
                    ).cpu()
                    per_example_sensitivity += (perm_loss - baseline_losses).abs()

            scores = per_example_sensitivity
        else:
            # For images, fall back to loss-based scoring.
            scores = baseline_losses

    elif criterion == "activation":
        # Activation-based discovery (similar to GEORGE): cluster examples
        # by their ERM penultimate-layer features.  Examples with similar
        # activations rely on similar features — clustering separates
        # shortcut-users from genuine-feature-users.
        #
        # Unlike other criteria that produce scores for median-splitting,
        # this directly assigns environments via K-means clustering.
        # We set scores to cluster distances so the upweighting still works
        # (examples far from centroids = boundary cases = higher weight).
        all_features = []
        all_idx = []
        with torch.no_grad():
            for batch in loaders["train"]:
                x = batch["image"].to(device)
                idx = batch["index"]
                feat = disc.backbone(x)  # (B, hidden_dim)
                all_features.append(feat.cpu())
                all_idx.append(idx)

        features = torch.cat(all_features)  # (N, D)
        idx_order = torch.cat(all_idx)
        ordered_features = torch.zeros_like(features)
        ordered_features[idx_order] = features

        # K-means clustering directly gives environment assignments.
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=cfg.training.seed, n_init=10)
        cluster_labels = km.fit_predict(ordered_features.numpy())

        # Use cluster label as the score: 0.0 for cluster 0, 1.0 for cluster 1.
        # The median split will then produce the same assignment as K-means.
        # For K>2, use the cluster label directly (rank-based assignment handles it).
        scores = torch.tensor(cluster_labels, dtype=torch.float32)

    elif criterion == "counterfactual":
        # Counterfactual scoring: for each example, measure how much the
        # prediction changes under random input perturbations.
        #
        # score_i = Var_k[ model(x_i + noise_k) ]  (variance of predictions)
        #
        # High variance → model relies on fragile features (likely spurious).
        # Low variance → prediction is robust to perturbation (invariant features).
        #
        # Unlike loss (which concentrates near 0 for confident models), this
        # score is well-distributed — every example gets a meaningful sensitivity
        # measure.  This matters for continuous spurious features where loss-based
        # splitting fails.
        n_perturbations = 10
        noise_std = 0.1
        g_noise = torch.Generator(device=device).manual_seed(cfg.training.seed + 99)

        with torch.no_grad():
            for batch in loaders["train"]:
                x = batch["image"].to(device)   # (B, C, H, W)
                idx = batch["index"]

                # Collect predictions under perturbation.
                pred_samples = []
                for _ in range(n_perturbations):
                    noise = torch.randn_like(x, generator=g_noise) * noise_std
                    perturbed_logits = disc(x + noise)
                    pred_samples.append(perturbed_logits.softmax(1)[:, 1])  # P(class=1)

                # Variance of P(class=1) across perturbations.
                preds_stack = torch.stack(pred_samples, dim=0)  # (K, B)
                var_per_example = preds_stack.var(dim=0)        # (B,)
                scores[idx] = var_per_example.cpu()
    else:
        with torch.no_grad():
            for batch in loaders["train"]:
                x   = batch["image"].to(device)
                y   = batch["label"].to(device)
                idx = batch["index"]
                logits = disc(x)
                if criterion == "loss":
                    # Use the averaged losses from the discovery window
                    # for more stable ranking across seeds.
                    scores[idx] = avg_losses[idx]
                    continue
                elif criterion == "confident_wrong":
                    # Data cartography-inspired: probability assigned to the
                    # WRONG class.  High = the model confidently predicts
                    # incorrectly = minority group (shortcut fails).
                    probs = logits.softmax(1)
                    correct_class_prob = probs.gather(1, y.unsqueeze(1)).squeeze(1)
                    s = 1.0 - correct_class_prob  # P(wrong class)
                elif criterion == "cartography":
                    # Full data cartography: 4 environments based on
                    # correctness × confidence.  Assigns each example a
                    # score that naturally creates the 4 cartography regions
                    # when split into K=4 environments:
                    #   env 0: correct + high confidence (easy, majority)
                    #   env 1: correct + low confidence (boundary)
                    #   env 2: wrong + low confidence (hard)
                    #   env 3: wrong + high confidence (minority)
                    # Score = -margin = -(P(correct) - P(wrong)).
                    # Ranges from -1 (maximally correct) to +1 (maximally wrong).
                    probs = logits.softmax(1)
                    correct_class_prob = probs.gather(1, y.unsqueeze(1)).squeeze(1)
                    margin = 2.0 * correct_class_prob - 1.0  # -1 to +1
                    s = -margin  # flip so high = wrong
                else:
                    # entropy
                    p = logits.softmax(1).clamp(min=1e-7)
                    s = -(p * p.log()).sum(1)
                scores[idx] = s.cpu()

    # --- Build assignment and per-example weights ---
    upweight = cfg.training.discovery_upweight
    reweight = cfg.training.discovery_reweight
    q = cfg.training.discovery_quantile

    K = cfg.training.num_discovery_envs

    # Cartography always uses natural grouping (correct/wrong × confident/uncertain).
    if criterion == "cartography":
        wrong = scores > 0
        correct = ~wrong
        assignment = torch.zeros(N, dtype=torch.long)
        if correct.any():
            correct_median = scores[correct].median()
            assignment[correct & (scores <= correct_median)] = 0  # easy
            assignment[correct & (scores > correct_median)] = 1   # boundary
        if wrong.any():
            wrong_median = scores[wrong].median()
            assignment[wrong & (scores <= wrong_median)] = 2      # hard
            assignment[wrong & (scores > wrong_median)] = 3       # minority
        # Upweight based on score magnitude.
        score_max = scores.abs().max()
        if upweight > 0 and score_max > 1e-8:
            weights = 1.0 + upweight * (scores.abs() / score_max)
        else:
            weights = torch.ones(N)
    elif upweight > 0:
        if q < 0.5:
            # Asymmetric split: bottom q% = env A, top q% = env B
            low_thresh = torch.quantile(scores, q)
            high_thresh = torch.quantile(scores, 1.0 - q)
            assignment = torch.full((N,), -1, dtype=torch.long)
            assignment[scores <= low_thresh] = 0
            assignment[scores >= high_thresh] = 1
        else:
            # Standard rank-based K-way split
            ranks = scores.argsort().argsort()
            assignment = (ranks.float() / N * K).long().clamp(max=K - 1)

        score_max = scores.max()
        if score_max > 1e-8:
            weights = 1.0 + upweight * (scores / score_max)
        else:
            weights = torch.ones(N)
    elif reweight > 0:
        # Balanced reweighting: ALL examples assigned via median split;
        # extremes get higher weight in the V-REx loss.
        #
        # weight_i = 1 + reweight * (2 * |percentile_rank_i - 0.5|)
        #   percentile 0 or 1 (most extreme):  w = 1 + reweight
        #   percentile 0.5 (median):            w = 1
        #
        # This keeps all examples in training (shape signal preserved)
        # while amplifying V-REx's sensitivity to colour-correlation
        # differences at the extremes.
        threshold = scores.median()
        assignment = (scores >= threshold).long()

        # Percentile rank ∈ [0, 1] for each example.
        ranks = scores.argsort().argsort().float() / (N - 1)
        extremeness = (2.0 * (ranks - 0.5).abs())  # 0 at median, 1 at tails
        weights = 1.0 + reweight * extremeness
    elif q == 0.5:
        # Balanced median split: bottom 50% -> env A, top 50% -> env B.
        threshold = scores.median()
        assignment = (scores >= threshold).long()
        weights = torch.ones(N)
    else:
        # Extremes split: bottom q% -> env A (0), top q% -> env B (1),
        # middle examples get -1 and are excluded from both environments.
        low_thresh  = torch.quantile(scores, q)
        high_thresh = torch.quantile(scores, 1.0 - q)
        assignment  = torch.full((N,), -1, dtype=torch.long)  # -1 = excluded
        assignment[scores <= low_thresh]  = 0   # env A: lowest score
        assignment[scores >= high_thresh] = 1   # env B: highest score
        weights = torch.ones(N)

    # --- Diagnostics ---
    colors = train_ds.spurious.float()
    labels = train_ds.labels.float()

    def _corr(mask: torch.Tensor) -> float:
        c, l = colors[mask], labels[mask]
        if len(c) < 2 or c.std() < 1e-8 or l.std() < 1e-8:
            return 0.0
        return torch.corrcoef(torch.stack([c, l]))[0, 1].item()

    mask_A = assignment == 0
    mask_B = assignment == 1
    mask_used = mask_A | mask_B   # excludes -1 examples

    # Correlation computed only over assigned examples (A and B).
    s_used = assignment[mask_used].float()
    c_used = colors[mask_used]
    disc_color_abs_corr = (
        abs(torch.corrcoef(torch.stack([s_used, c_used]))[0, 1].item())
        if len(s_used) > 1 and s_used.std() > 1e-8 and c_used.std() > 1e-8
        else 0.0
    )

    # --- Permutation test using the discovery ERM ---
    # Compare risk variance under real assignment vs random permutations.
    # Uses the DISCOVERY ERM (which has learned the shortcut), not the
    # untrained V-REx model.  This gives a meaningful signal — the ERM's
    # loss landscape has structure that the permutation test can detect.
    def _rv_for_assignment(assign_t):
        """Compute risk variance using pre-averaged losses (vectorised)."""
        env_sums = torch.zeros(K)
        env_counts = torch.zeros(K)
        w_cpu = weights.cpu() if weights.is_cuda or (hasattr(weights, 'device') and str(weights.device) != 'cpu') else weights
        for kk in range(K):
            mask = assign_t == kk
            if mask.any():
                env_sums[kk] = (w_cpu[mask] * avg_losses[mask]).sum()
                env_counts[kk] = w_cpu[mask].sum()
        env_l = env_sums / env_counts.clamp(min=1)
        return ((env_l - env_l.mean()) ** 2).sum().item()

    actual_rv = _rv_for_assignment(assignment)
    n_perms = 10
    g_perm = torch.Generator().manual_seed(cfg.training.seed + 777)
    perm_rvs = []
    for _ in range(n_perms):
        perm_assign = assignment[torch.randperm(N, generator=g_perm)]
        perm_rvs.append(_rv_for_assignment(perm_assign))
    mean_perm_rv = sum(perm_rvs) / n_perms

    if mean_perm_rv > 1e-12:
        signal_ratio = actual_rv / mean_perm_rv
    else:
        signal_ratio = 10.0 if actual_rv > 1e-12 else 1.0

    reliability = min(1.0, max(0.0, (signal_ratio - 1.0) / 2.0))

    diag_metrics: dict[str, float] = {
        "discovery/assignment_color_abs_corr": disc_color_abs_corr,
        "discovery/color_label_corr_A":        _corr(mask_A),
        "discovery/color_label_corr_B":        _corr(mask_B),
        "discovery/n_env_A":                   float(mask_A.sum().item()),
        "discovery/n_env_B":                   float(mask_B.sum().item()),
        "discovery/n_excluded":                float((assignment == -1).sum().item()),
        "discovery/reweight_max":              float(weights.max().item()),
        "adaptive/actual_risk_var":            actual_rv,
        "adaptive/mean_perm_risk_var":         mean_perm_rv,
        "adaptive/signal_ratio":               signal_ratio,
        "adaptive/reliability":                reliability,
    }
    if return_model:
        return assignment, weights, diag_metrics, disc
    return assignment, weights, diag_metrics


def train_discovered_split(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
    assignment: torch.Tensor,
    weights: torch.Tensor,
    discovery_metrics: dict[str, float],
) -> dict[str, float]:
    """Train a single MLP with V-REx penalty on discovered environments.

    The V-REx penalty forces equal risk across environments.  Because env A
    has higher spurious-label correlation than env B, minimising the risk
    difference pushes the model toward invariant features.

    Model selection uses the validation set (id_test).  Early stopping:
    if validation metric doesn't improve for ``patience`` epochs, stop.

    Args:
        assignment:        (N,) long tensor; 0 = env A, 1 = env B, -1 = excluded.
        weights:           (N,) float tensor of per-example importance weights.
        discovery_metrics: diagnostics logged at step 0 in wandb.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    assignment = assignment.to(device)
    weights = weights.to(device)

    # Build environment-balanced sampler: each batch has equal representation
    # from each environment.  This replaces loss-based upweighting with
    # structural balance — no tuning of upweight factor needed.
    K = cfg.training.num_discovery_envs
    env_indices = {k: (assignment.cpu() == k).nonzero(as_tuple=True)[0] for k in range(K)}
    balanced = getattr(cfg.training, "balanced_sampling", False)
    if balanced:
        from torch.utils.data import WeightedRandomSampler
        # Inverse-frequency weights: examples in smaller environments get higher sampling probability.
        sample_weights = torch.zeros(len(assignment))
        for k in range(K):
            mask = assignment.cpu() == k
            n_k = mask.sum().item()
            if n_k > 0:
                sample_weights[mask] = 1.0 / n_k
        sample_weights = sample_weights / sample_weights.sum()
        balanced_sampler = WeightedRandomSampler(sample_weights, num_samples=len(assignment), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            loaders["train"].dataset,
            batch_size=cfg.training.batch_size,
            sampler=balanced_sampler,
            num_workers=0,
        )
    else:
        train_loader = loaders["train"]

    log_metrics(run, discovery_metrics, step=0)

    # Adaptive λ: use the reliability from the discovery permutation test.
    reliability = discovery_metrics.get("adaptive/reliability", 1.0)

    # Adaptive penalty: blend V-REx and DRO based on environment balance.
    # V-REx (variance penalty) works best with balanced environments.
    # DRO (worst-group upweighting) works best with imbalanced environments.
    # Balance = min_env_size / max_env_size (0 = very imbalanced, 1 = balanced).
    env_sizes = [(assignment.cpu() == k).sum().item() for k in range(K)]
    non_empty = [s for s in env_sizes if s > 0]
    env_balance = min(non_empty) / max(non_empty) if len(non_empty) >= 2 else 1.0
    # DRO group weights (maintained across batches).
    dro_group_weights = torch.ones(K, device=device) / K
    dro_step_size = 0.01

    anneal_factor = cfg.training.lambda_anneal_factor
    patience = cfg.training.early_stop_patience

    selector = _ModelSelector()
    epochs_without_improvement = 0
    for epoch in range(cfg.training.epochs):
        # Lambda: base value × anneal factor × reliability.
        if cfg.training.epochs > 1 and anneal_factor != 1.0:
            progress = epoch / (cfg.training.epochs - 1)
            lam = cfg.training.lambda_disagree * (1.0 + (anneal_factor - 1.0) * progress)
        else:
            lam = cfg.training.lambda_disagree
        lam = lam * reliability  # adaptive scaling

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        epoch_risk_var = 0.0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            # Training noise: diversifies oversampled minority examples.
            noise_std = getattr(cfg.training, "training_noise", 0.0)
            if noise_std > 0 and model.training:
                x = x + torch.randn_like(x) * noise_std

            logits = model(x)                                       # (B, 2)
            ce = F.cross_entropy(logits, y, reduction="none")       # (B,)

            batch_assign = assignment[idx]
            w = weights[idx]

            # Compute weighted per-environment losses.
            env_losses = []
            for k in range(K):
                mask = batch_assign == k
                if mask.any():
                    wk = w[mask]
                    env_losses.append((wk * ce[mask]).sum() / wk.sum())

            if len(env_losses) >= 2:
                env_losses_t = torch.stack(env_losses)
                mean_loss = env_losses_t.mean()

                risk_var = ((env_losses_t - mean_loss) ** 2).sum()  # tracked for logging

                # Use Group DRO (upweight worst discovered group) when envs
                # are imbalanced; V-REx (variance penalty) when balanced.
                # Binary switch at balance threshold 0.5.
                if env_balance < 0.5:
                    # Imbalanced envs (e.g. cartography K=4): pure DRO
                    dro_w = dro_group_weights[:len(env_losses)].detach()
                    model_loss = (dro_w * env_losses_t).sum()
                else:
                    # Balanced envs (e.g. K=2 median): V-REx
                    model_loss = mean_loss + lam * risk_var
            elif len(env_losses) == 1:
                model_loss = env_losses[0]
                risk_var = torch.tensor(0.0)
            else:
                continue

            # Environment-aware mixup: create synthetic counterexamples
            # by interpolating between env A and env B features with the
            # same label.  Blends out spurious signal, preserves invariant.
            mixup_weight = getattr(cfg.training, "env_mixup", 0.0)
            if mixup_weight > 0 and len(env_losses) >= 2:
                mixup_loss = torch.tensor(0.0, device=device)
                for label_val in y.unique():
                    # Find examples from different envs with the same label.
                    for k1 in range(K):
                        for k2 in range(k1 + 1, K):
                            mask1 = (batch_assign == k1) & (y == label_val)
                            mask2 = (batch_assign == k2) & (y == label_val)
                            n1, n2 = mask1.sum(), mask2.sum()
                            if n1 > 0 and n2 > 0:
                                # Sample pairs and interpolate.
                                n_mix = min(n1, n2).item()
                                x1 = x[mask1][:n_mix]
                                x2 = x[mask2][:n_mix]
                                # Shape lambda to broadcast: (n_mix, 1) for tabular,
                                # (n_mix, 1, 1, 1) for images.
                                lam_shape = (n_mix,) + (1,) * (x1.dim() - 1)
                                lam_mix = torch.rand(lam_shape, device=device)
                                x_mixed = lam_mix * x1 + (1 - lam_mix) * x2
                                logits_mix = model(x_mixed)
                                y_mix = torch.full((n_mix,), label_val, device=device, dtype=torch.long)
                                mixup_loss = mixup_loss + F.cross_entropy(logits_mix, y_mix)
                model_loss = model_loss + mixup_weight * mixup_loss

            optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update DRO group weights AFTER backward (avoids in-place conflict).
            if len(env_losses) >= 2:
                with torch.no_grad():
                    dro_group_weights[:len(env_losses)] *= torch.exp(
                        dro_step_size * env_losses_t.detach()
                    )
                    dro_group_weights /= dro_group_weights.sum()

            epoch_loss += model_loss.item() * len(y)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_n += len(y)
            epoch_risk_var += risk_var.item() * len(y)

        avg_risk_var = epoch_risk_var / epoch_n

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss":          epoch_loss / epoch_n,
            "train/acc":           epoch_correct / epoch_n,
            "train/risk_variance": avg_risk_var,
            "train/lambda":        lam,
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)

        # Track best validation checkpoint.
        prev_best = selector.best_score
        selector.update(_val_score(id_metrics), model, metrics)
        improved = selector.best_score > prev_best

        # Early stopping: after warmup, stop if no improvement for `patience` epochs.
        min_epochs = min(cfg.training.epochs // 2, 5)
        if patience > 0 and epoch >= min_epochs:
            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    return selector.restore(model)


def discover_jtt_weights(
    cfg: DictConfig,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """JTT identification phase — faithful to Liu et al. 2021.

    Train a short ERM, then check which examples it *misclassifies*.
    Those get a fixed upweight factor; everything else gets weight 1.

    This is different from our continuous loss-based weighting:
    - JTT: binary (misclassified → upweight, correct → 1)
    - Ours: continuous (weight proportional to loss)

    Binary identification focuses weight precisely on the minority group.
    Continuous weighting dilutes across all high-loss examples including
    noisy majority-group examples.

    Returns:
        weights      - (N,) float: 1.0 for correct, upweight_factor for misclassified.
        diag_metrics - diagnostics (how many misclassified, overlap with spurious).
    """
    train_ds = loaders["train"].dataset
    N = len(train_ds)
    upweight_factor = cfg.training.discovery_upweight

    # Train throw-away ERM (same as discover_environments phase 1).
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone(freeze=True)
        disc = MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    else:
        disc = MLP(input_dim=train_ds.input_dim, hidden_dim=cfg.model.hidden_dim).to(device)
    opt = torch.optim.AdamW(disc.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    for _ in range(cfg.training.discovery_epochs):
        disc.train()
        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(disc(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Identify misclassified examples.
    disc.eval()
    misclassified = torch.zeros(N, dtype=torch.bool)
    with torch.no_grad():
        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"]
            preds = disc(x).argmax(1)
            misclassified[idx] = (preds != y).cpu()

    # Binary weights: misclassified → upweight, correct → 1.
    weights = torch.ones(N)
    weights[misclassified] = upweight_factor

    # Diagnostics.
    n_error = misclassified.sum().item()
    spurious = train_ds.spurious
    labels = train_ds.labels
    # What fraction of misclassified examples are from the minority group?
    minority_mask = spurious != labels  # spurious feature doesn't match label
    n_error_minority = (misclassified & minority_mask).sum().item()

    diag_metrics = {
        "jtt/n_misclassified": float(n_error),
        "jtt/frac_misclassified": float(n_error) / N,
        "jtt/frac_errors_from_minority": float(n_error_minority) / max(n_error, 1),
        "jtt/upweight_factor": float(upweight_factor),
    }
    return weights, diag_metrics


def train_jtt(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
    weights: torch.Tensor,
    discovery_metrics: dict[str, float],
) -> dict[str, float]:
    """JTT (Just Train Twice) — Liu et al. 2021.

    Phase 1 (discover_environments) has already run: we have per-example
    loss-based weights.  Phase 2 trains a fresh model with upweighted loss
    on the examples the discovery ERM got wrong.

    This is the ablation that isolates what V-REx adds: JTT uses the same
    discovery + upweighting but trains with plain weighted ERM (no environment
    split, no risk-variance penalty).

    Args:
        weights: (N,) float tensor from discover_environments.
        discovery_metrics: logged at step 0.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    weights = weights.to(device)

    log_metrics(run, discovery_metrics, step=0)

    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0

        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none")

            w = weights[idx]
            loss = (w * ce).sum() / w.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_n += len(y)

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_metrics), model, metrics)

    return selector.restore(model)


def train_group_dro(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """Group DRO — Sagawa et al. 2020.

    At each step, compute per-group loss, then upweight the group with the
    highest loss.  This directly minimises worst-group risk.

    Requires ground-truth group labels (label x spurious) on every training
    example — this is the oracle upper bound for methods that don't have
    group annotations.

    Groups are defined as (label, spurious) pairs:
        group 0: label=0, spurious=0
        group 1: label=0, spurious=1
        group 2: label=1, spurious=0
        group 3: label=1, spurious=1

    The group weights q are maintained on a simplex and updated via
    exponentiated gradient ascent:
        q_g <- q_g * exp(eta * loss_g)
        q <- q / sum(q)

    eta (step_size) controls how aggressively we upweight the worst group.
    Sagawa et al. use eta=0.01 as default.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    n_groups = 4
    # Group weights on the simplex — start uniform.
    group_weights = torch.ones(n_groups, device=device) / n_groups
    dro_step_size = 0.01  # eta from Sagawa et al.

    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0

        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            s = batch["spurious"]
            if isinstance(s, torch.Tensor):
                s = s.to(device)
            else:
                s = torch.tensor(s, device=device)

            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none")  # (B,)

            # Group index: label * 2 + spurious
            groups = y * 2 + s  # (B,) ∈ {0, 1, 2, 3}

            # Per-group mean loss.
            group_losses = torch.zeros(n_groups, device=device)
            for g in range(n_groups):
                mask = groups == g
                if mask.any():
                    group_losses[g] = ce[mask].mean()

            # DRO loss: weighted combination of per-group losses.
            loss = (group_weights * group_losses).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update group weights via exponentiated gradient ascent.
            with torch.no_grad():
                group_weights = group_weights * torch.exp(dro_step_size * group_losses)
                group_weights = group_weights / group_weights.sum()

            epoch_loss += loss.item() * len(y)
            epoch_correct += (logits.argmax(1) == y).sum().item()
            epoch_n += len(y)

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)

        metrics = {
            "train/loss": epoch_loss / epoch_n,
            "train/acc": epoch_correct / epoch_n,
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_metrics), model, metrics)

    return selector.restore(model)


def train_dfr(
    cfg: DictConfig,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run: object,
) -> dict[str, float]:
    """DFR (Deep Feature Reweighting) — Kirichenko et al. 2023.

    Phase 1: train ERM (full model, standard training).
    Phase 2: freeze the backbone, retrain only the last layer (head) on
    group-balanced data from the validation set.

    The insight: ERM learns good features but misweights them in the final
    layer toward spurious features.  A balanced last-layer refit corrects this.

    Requires group labels (spurious attribute) on the validation set for
    balanced sampling.  This is a stronger assumption than our method
    (which needs no group labels at all).
    """
    # Phase 1: train ERM normally.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        id_metrics = evaluate(model, loaders["id_test"], device)
        ood_metrics = evaluate(model, loaders["ood_test"], device)
        metrics = {
            "train/loss": loss.item(),
            "train/acc": 0.0,  # not tracked per-epoch for simplicity
            **_eval_metrics("eval/id", id_metrics),
            **_eval_metrics("eval/ood", ood_metrics),
        }
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_metrics), model, metrics)

    # Restore best ERM checkpoint.
    selector.restore(model)

    # Phase 2: freeze backbone, retrain head on group-balanced validation data.
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Reset the head weights.
    for param in model.head.parameters():
        torch.nn.init.zeros_(param)

    # Group-balanced sampling from validation set.
    val_ds = loaders["id_test"].dataset
    val_labels = val_ds.labels if hasattr(val_ds, "labels") else torch.tensor([val_ds[i]["label"] for i in range(len(val_ds))])
    val_spurious = val_ds.spurious if hasattr(val_ds, "spurious") else torch.zeros(len(val_ds))
    groups = val_labels * 2 + val_spurious
    group_weights = torch.zeros(len(val_ds))
    for g in groups.unique():
        mask = groups == g
        group_weights[mask] = 1.0 / mask.sum().float()
    group_weights = group_weights / group_weights.sum()

    from torch.utils.data import WeightedRandomSampler
    balanced_sampler = WeightedRandomSampler(group_weights, num_samples=len(val_ds), replacement=True)
    balanced_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.training.batch_size, sampler=balanced_sampler, num_workers=0,
    )

    # Train head with higher lr (it's a linear probe now).
    head_optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=cfg.training.lr * 10, weight_decay=1e-2,
    )

    dfr_epochs = 10
    for epoch in range(dfr_epochs):
        model.train()
        for batch in balanced_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(model(x), y)
            head_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step()

    # Final evaluation.
    id_metrics = evaluate(model, loaders["id_test"], device)
    ood_metrics = evaluate(model, loaders["ood_test"], device)
    metrics = {
        "train/loss": 0.0,
        "train/acc": 0.0,
        **_eval_metrics("eval/id", id_metrics),
        **_eval_metrics("eval/ood", ood_metrics),
    }
    log_metrics(run, metrics, step=cfg.training.epochs + dfr_epochs)

    return metrics
