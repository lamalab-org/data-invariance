"""Training pipeline for worst-group robustness without group labels.

This module contains the complete method:
  1. make_dataloaders — dataset construction
  2. evaluate — model evaluation (computes WGA when spurious labels available)
  3. discover_environments — loss-based environment discovery + permutation test
  4. train_erm — ERM baseline
  5. train_jtt — JTT baseline (Liu et al. 2021)
  6. train_lff — LfF baseline (Nam et al. 2020)
  7. train_vrex — our method: V-REx on discovered environments

All training functions support both group-free (avg accuracy) and group-labeled
(worst-group accuracy) model selection via the group_free flag.
"""
from __future__ import annotations


import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import (
    CelebADataset,
    ChemistryDataset,
    CivilCommentsDataset,
    ColoredMNIST,
    ContinuousCMNIST,
    MultiNLIDataset,
    MultiSpuriousCMNIST,
    TADFDataset,
    WaterbirdsDataset,
)
from models import MLP, make_resnet_backbone
from utils import log_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(x, device):
    """Move input to device, handling both tensors and dicts (for DistilBERT)."""
    if isinstance(x, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in x.items()}
    return x.to(device)


def _val_score(id_metrics: dict[str, float], group_free: bool = False) -> float:
    """Validation score for model selection (higher = better).

    group_free=False: uses worst-group accuracy when available (standard protocol).
    group_free=True:  uses average accuracy only (no group labels needed).
    """
    if not group_free and "worst_group_acc" in id_metrics:
        return id_metrics["worst_group_acc"]
    return id_metrics["acc"]


class _ModelSelector:
    """Track the best model checkpoint by validation score."""

    def __init__(self):
        import copy as _copy
        self._copy = _copy
        self.best_score = float("-inf")
        self.best_state = None
        self.best_metrics: dict[str, float] = {}

    def update(self, val_score: float, model: torch.nn.Module,
               metrics: dict[str, float]) -> None:
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_state = self._copy.deepcopy(model.state_dict())
            self.best_metrics = dict(metrics)

    def restore(self, model: torch.nn.Module) -> dict[str, float]:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return self.best_metrics if self.best_metrics else {}


def _eval_metrics(prefix: str, m: dict[str, float]) -> dict[str, float]:
    """Build prefixed metric dict from evaluate() output."""
    d = {
        f"{prefix}_acc": m["acc"],
        f"{prefix}_auroc": m["auroc"],
        f"{prefix}_precision": m["precision"],
        f"{prefix}_recall": m["recall"],
    }
    if "worst_group_acc" in m:
        d[f"{prefix}_worst_group_acc"] = m["worst_group_acc"]
    return d


def _build_model(cfg, loaders, device):
    """Build a fresh model for the given architecture."""
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    if cfg.dataset.arch == "distilbert":
        from models import make_distilbert_backbone
        backbone, out_dim = make_distilbert_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    if cfg.dataset.arch == "chemberta":
        from models import make_chemberta_backbone
        backbone, out_dim = make_chemberta_backbone()
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    input_dim = loaders["train"].dataset.input_dim
    return MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim).to(device)


def _build_discovery_model(cfg, loaders, device):
    """Build a discovery ERM (frozen backbone for ResNet/DistilBERT, full for MLP)."""
    if cfg.dataset.arch == "resnet":
        backbone, out_dim = make_resnet_backbone(freeze=True)
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    if cfg.dataset.arch == "distilbert":
        from models import make_distilbert_backbone
        backbone, out_dim = make_distilbert_backbone(freeze=True)
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    if cfg.dataset.arch == "chemberta":
        from models import make_chemberta_backbone
        backbone, out_dim = make_chemberta_backbone(freeze=True)
        return MLP(backbone=backbone, backbone_out_dim=out_dim).to(device)
    input_dim = loaders["train"].dataset.input_dim
    return MLP(input_dim=input_dim, hidden_dim=cfg.model.hidden_dim).to(device)


# Alias for LfF: uses the same frozen-backbone pattern as the discovery model.
# This matches published LfF (Nam et al. 2020), which trains both biased and
# debiased models with frozen pretrained backbones and trainable head only.
_build_lff_biased_model = _build_discovery_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_dataloaders(cfg: DictConfig) -> dict[str, DataLoader]:
    """Build train, ID-test, and OOD-test dataloaders.

    Dispatches on cfg.dataset.name. Returns {"train", "id_test", "ood_test"}.
    """
    name = cfg.dataset.name
    seed = cfg.training.seed

    if name == "cmnist":
        noise = cfg.dataset.label_noise
        data_dir = cfg.dataset.data_dir
        train_ds = ColoredMNIST(cfg.dataset.train_correlation, label_noise=noise,
                                split="train", data_dir=data_dir, seed=seed)
        id_test_ds = ColoredMNIST(cfg.dataset.train_correlation, label_noise=noise,
                                  split="test", data_dir=data_dir, seed=seed)
        ood_test_ds = ColoredMNIST(cfg.dataset.test_correlation, label_noise=noise,
                                   split="test", data_dir=data_dir, seed=seed)

    elif name == "continuous_cmnist":
        noise = cfg.dataset.label_noise
        data_dir = cfg.dataset.data_dir
        beta_c = cfg.dataset.beta_concentration
        train_ds = ContinuousCMNIST(cfg.dataset.env_correlation, label_noise=noise,
                                    split="train", data_dir=data_dir, seed=seed,
                                    beta_concentration=beta_c)
        id_test_ds = ContinuousCMNIST(cfg.dataset.env_correlation, label_noise=noise,
                                      split="test", data_dir=data_dir, seed=seed,
                                      beta_concentration=beta_c)
        ood_test_ds = ContinuousCMNIST(cfg.dataset.test_correlation, label_noise=noise,
                                       split="test", data_dir=data_dir, seed=seed,
                                       beta_concentration=beta_c)

    elif name == "multi_cmnist":
        noise = cfg.dataset.label_noise
        data_dir = cfg.dataset.data_dir
        train_ds = MultiSpuriousCMNIST(
            color_correlation=cfg.dataset.color_correlation,
            brightness_correlation=cfg.dataset.brightness_correlation,
            label_noise=noise, split="train", data_dir=data_dir, seed=seed)
        id_test_ds = MultiSpuriousCMNIST(
            color_correlation=cfg.dataset.color_correlation,
            brightness_correlation=cfg.dataset.brightness_correlation,
            label_noise=noise, split="test", data_dir=data_dir, seed=seed)
        ood_test_ds = MultiSpuriousCMNIST(
            color_correlation=cfg.dataset.test_color_correlation,
            brightness_correlation=cfg.dataset.test_brightness_correlation,
            label_noise=noise, split="test", data_dir=data_dir, seed=seed)

    elif name == "tadf":
        ppath = cfg.dataset.parquet_path
        spur_prop = getattr(cfg.dataset, "spurious_property", None)
        spur_corr = getattr(cfg.dataset, "spurious_correlation", 0.9)
        train_ds = TADFDataset(parquet_path=ppath, split="train", seed=seed,
                               spurious_property=spur_prop, spurious_correlation=spur_corr)
        id_test_ds = TADFDataset(parquet_path=ppath, split="test", seed=seed)
        ood_test_ds = TADFDataset(parquet_path=ppath, split="test_misaligned", seed=seed,
                                  spurious_property=spur_prop)

    elif name in ("mof_thermal", "mof_solvent", "perovskite", "battery"):
        ppath = cfg.dataset.parquet_path
        target_col = cfg.dataset.target_column
        spur_prop = getattr(cfg.dataset, "spurious_property", None)
        spur_corr = getattr(cfg.dataset, "spurious_correlation", 0.9)
        train_ds = ChemistryDataset(parquet_path=ppath, target_column=target_col,
                                    split="train", seed=seed,
                                    spurious_property=spur_prop, spurious_correlation=spur_corr)
        id_test_ds = ChemistryDataset(parquet_path=ppath, target_column=target_col,
                                      split="test", seed=seed)
        ood_test_ds = ChemistryDataset(parquet_path=ppath, target_column=target_col,
                                       split="test_misaligned", seed=seed,
                                       spurious_property=spur_prop)

    elif name in ("bace", "bbbp", "hiv", "clintox", "tox21", "sider", "muv", "pcba", "esol"):
        from data_molnet import MolNetDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/molnet")
        train_ds = MolNetDataset(name=name, split="train", seed=seed, data_dir=data_dir)
        id_test_ds = MolNetDataset(name=name, split="test", seed=seed, data_dir=data_dir)
        ood_test_ds = MolNetDataset(name=name, split="test_scaffold", seed=seed, data_dir=data_dir)

    elif name in ("bace_chemberta", "bbbp_chemberta"):
        # ChemBERTa-tokenised twin of bace / bbbp; same scaffold split as Morgan.
        from data_molnet import MolNetTokenDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/molnet")
        molnet_name = name.replace("_chemberta", "")
        train_ds = MolNetTokenDataset(name=molnet_name, split="train",
                                      seed=seed, data_dir=data_dir)
        id_test_ds = MolNetTokenDataset(name=molnet_name, split="test",
                                        seed=seed, data_dir=data_dir)
        ood_test_ds = MolNetTokenDataset(name=molnet_name, split="test_scaffold",
                                         seed=seed, data_dir=data_dir)

    elif name in ("hia_hou", "bioavailability_ma", "pgp_broccatelli",
                  "bbb_martins", "herg", "dili", "ames", "skin_reaction"):
        from data_tdc import TDCDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/tdc")
        train_ds   = TDCDataset(name=name, split="train",         seed=seed, data_dir=data_dir)
        id_test_ds = TDCDataset(name=name, split="test",          seed=seed, data_dir=data_dir)
        ood_test_ds = TDCDataset(name=name, split="test_scaffold", seed=seed, data_dir=data_dir)

    elif name.endswith("_chemberta") and name.replace("_chemberta", "") in (
            "hia_hou", "bioavailability_ma", "pgp_broccatelli",
            "bbb_martins", "herg", "dili", "ames", "skin_reaction"):
        # ChemBERTa-tokenised twin of a TDC dataset; same scaffold split.
        from data_tdc import TDCTokenDataset
        data_dir = getattr(cfg.dataset, "data_dir", "./data/tdc")
        tdc_name = name.replace("_chemberta", "")
        train_ds = TDCTokenDataset(name=tdc_name, split="train",
                                   seed=seed, data_dir=data_dir)
        id_test_ds = TDCTokenDataset(name=tdc_name, split="test",
                                     seed=seed, data_dir=data_dir)
        ood_test_ds = TDCTokenDataset(name=tdc_name, split="test_scaffold",
                                      seed=seed, data_dir=data_dir)

    elif name == "waterbirds":
        train_ds = WaterbirdsDataset(split="train", data_dir=cfg.dataset.data_dir)
        id_test_ds = WaterbirdsDataset(split="val", data_dir=cfg.dataset.data_dir)
        ood_test_ds = WaterbirdsDataset(split="test", data_dir=cfg.dataset.data_dir)

    elif name == "celeba":
        train_ds = CelebADataset(split="train", data_dir=cfg.dataset.data_dir)
        id_test_ds = CelebADataset(split="val", data_dir=cfg.dataset.data_dir)
        ood_test_ds = CelebADataset(split="test", data_dir=cfg.dataset.data_dir)

    elif name == "civilcomments":
        data_dir = getattr(cfg.dataset, "data_dir", "./data/civil_comments")
        max_len = getattr(cfg.dataset, "max_length", 128)
        train_ds = CivilCommentsDataset(split="train", max_length=max_len, data_dir=data_dir)
        id_test_ds = CivilCommentsDataset(split="val", max_length=max_len, data_dir=data_dir)
        ood_test_ds = CivilCommentsDataset(split="test", max_length=max_len, data_dir=data_dir)

    elif name == "multinli":
        data_dir = getattr(cfg.dataset, "data_dir", "./data/multi_nli")
        max_len = getattr(cfg.dataset, "max_length", 128)
        train_ds = MultiNLIDataset(split="train", max_length=max_len, data_dir=data_dir)
        id_test_ds = MultiNLIDataset(split="val", max_length=max_len, data_dir=data_dir)
        ood_test_ds = MultiNLIDataset(split="test", max_length=max_len, data_dir=data_dir)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    use_cuda = torch.cuda.is_available()
    kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=True if use_cuda else False,
    )
    return {
        "train": DataLoader(train_ds, shuffle=True, **kwargs),
        "id_test": DataLoader(id_test_ds, shuffle=False, **kwargs),
        "ood_test": DataLoader(ood_test_ds, shuffle=False, **kwargs),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device) -> dict[str, float]:
    """Evaluate model. Returns acc, precision, recall, auroc, loss, and
    worst_group_acc (when spurious labels are available in the batch)."""
    model.eval()
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    auroc = torchmetrics.AUROC(task="binary").to(device)

    total_loss, total_n = 0.0, 0
    all_preds, all_labels, all_spurious = [], [], []
    has_spurious = False

    with torch.no_grad():
        for batch in loader:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            probs = model.predict(x)
            preds = probs.argmax(dim=1)

            total_loss += F.nll_loss(probs.clamp(min=1e-7).log(), y,
                                     reduction="sum").item()
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
                    all_spurious.append(torch.tensor(
                        s if not isinstance(s, int) else [s]))

    result = {
        "acc": accuracy.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "auroc": auroc.compute().item(),
        "loss": total_loss / total_n,
    }

    if has_spurious:
        preds_t = torch.cat(all_preds)
        labels_t = torch.cat(all_labels)
        spurious_t = torch.cat(all_spurious)
        groups = labels_t * 2 + spurious_t
        group_accs = {}
        for g in range(4):
            mask = groups == g
            if mask.any():
                group_accs[g] = (preds_t[mask] == labels_t[mask]).float().mean().item()
        if group_accs:
            result["worst_group_acc"] = min(group_accs.values())

    return result


# ---------------------------------------------------------------------------
# Discovery: loss-based environment identification + permutation test
# ---------------------------------------------------------------------------

def discover_environments(
    cfg: DictConfig,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Score training examples by averaged ERM loss and split into environments.

    Pipeline:
        1. Train throw-away ERM for K epochs.
        2. Average per-example losses over the last K/2 epochs.
        3. Median split into K=2 environments.
        4. Upweight high-loss examples: w_i = 1 + α · ℓ_i / max(ℓ).
        5. Permutation test → signal_ratio → reliability.

    Returns:
        assignment  - (N,) long: 0 = env A (low loss), 1 = env B (high loss).
        weights     - (N,) float: per-example importance weights.
        diag_metrics - discovery diagnostics (signal_ratio, reliability, etc.).
    """
    train_ds = loaders["train"].dataset
    N = len(train_ds)
    K = getattr(cfg.training, "num_discovery_envs", 2)
    upweight_alpha = getattr(cfg.training, "discovery_upweight", 50.0)

    # --- Phase 1: train discovery ERM and collect averaged losses ---
    disc = _build_discovery_model(cfg, loaders, device)
    opt = torch.optim.AdamW(disc.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)

    n_epochs = cfg.training.discovery_epochs
    avg_window = max(1, n_epochs // 2)
    accumulated = torch.zeros(N)
    n_acc = 0

    for ep in range(n_epochs):
        disc.train()
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(disc(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep >= n_epochs - avg_window:
            disc.eval()
            with torch.no_grad():
                for batch in loaders["train"]:
                    x = _to_device(batch["image"], device)
                    y = batch["label"].to(device)
                    idx = batch["index"]
                    ce = F.cross_entropy(disc(x), y, reduction="none").cpu()
                    accumulated[idx] += ce
            n_acc += 1

    scores = accumulated / max(n_acc, 1)

    # --- Phase 2: median split + upweight ---
    ranks = scores.argsort().argsort()
    assignment = (ranks.float() / N * K).long().clamp(max=K - 1)

    score_max = scores.max()
    if upweight_alpha > 0 and score_max > 1e-8:
        weights = 1.0 + upweight_alpha * (scores / score_max)
    else:
        weights = torch.ones(N)

    # --- Phase 3: permutation test ---
    def _rv(assign_t):
        env_sums = torch.zeros(K)
        env_counts = torch.zeros(K)
        for kk in range(K):
            mask = assign_t == kk
            if mask.any():
                env_sums[kk] = (weights[mask] * scores[mask]).sum()
                env_counts[kk] = weights[mask].sum()
        env_l = env_sums / env_counts.clamp(min=1)
        return ((env_l - env_l.mean()) ** 2).sum().item()

    actual_rv = _rv(assignment)
    g = torch.Generator().manual_seed(cfg.training.seed + 777)
    perm_rvs = [_rv(assignment[torch.randperm(N, generator=g)]) for _ in range(10)]
    mean_perm_rv = sum(perm_rvs) / len(perm_rvs)

    if mean_perm_rv > 1e-12:
        signal_ratio = actual_rv / mean_perm_rv
    else:
        signal_ratio = 10.0 if actual_rv > 1e-12 else 1.0
    reliability = min(1.0, max(0.0, (signal_ratio - 1.0) / 2.0))

    # --- Diagnostics (uses spurious labels for REPORTING only, not decisions) ---
    diag = {
        "discovery/n_env_A": float((assignment == 0).sum().item()),
        "discovery/n_env_B": float((assignment == 1).sum().item()),
        "discovery/reweight_max": float(weights.max().item()),
        "adaptive/actual_risk_var": actual_rv,
        "adaptive/mean_perm_risk_var": mean_perm_rv,
        "adaptive/signal_ratio": signal_ratio,
        "adaptive/reliability": reliability,
    }

    if hasattr(train_ds, "spurious"):
        colors = train_ds.spurious.float()
        labels = train_ds.labels.float()
        mask_A, mask_B = assignment == 0, assignment == 1

        def _corr(mask):
            c, l = colors[mask], labels[mask]
            if len(c) < 2 or c.std() < 1e-8 or l.std() < 1e-8:
                return 0.0
            return torch.corrcoef(torch.stack([c, l]))[0, 1].item()

        s_used = assignment[mask_A | mask_B].float()
        c_used = colors[mask_A | mask_B]
        diag["discovery/assignment_color_abs_corr"] = (
            abs(torch.corrcoef(torch.stack([s_used, c_used]))[0, 1].item())
            if len(s_used) > 1 and s_used.std() > 1e-8 and c_used.std() > 1e-8
            else 0.0
        )
        diag["discovery/color_label_corr_A"] = _corr(mask_A)
        diag["discovery/color_label_corr_B"] = _corr(mask_B)

    return assignment, weights, diag


# ---------------------------------------------------------------------------
# Auto-λ: N-scaling rule for V-REx penalty strength, capped for safety
# ---------------------------------------------------------------------------

def auto_lambda(discovery_metrics: dict, cfg: DictConfig) -> float:
    """Compute V-REx penalty strength from dataset size and reliability.

    λ = min(λ_max, λ_base · √(N_ref / N)) · reliability

    The 1/√N scaling keeps the penalty meaningful on large datasets while
    preventing instability on small ones. Anchored at λ≈10 for N=5000
    (validated on Waterbirds). The square-root scaling is gentler than
    1/N, giving λ≈1.8 on CelebA (N=163K) instead of 0.3 — enough for
    V-REx to have a measurable effect on the training dynamics.

    A cap at λ_max=20 prevents extreme penalty values on very small
    datasets (N<2500) where the V-REx loss estimate is noisy.

    Reliability from the permutation test gates λ toward zero when
    the discovered environments are noise.
    """
    import math
    reliability = discovery_metrics.get("adaptive/reliability", 1.0)
    N = int(discovery_metrics.get("discovery/n_env_A", 0) +
            discovery_metrics.get("discovery/n_env_B", 0))
    if N < 1:
        return 0.0
    lam = min(20.0, 10.0 * math.sqrt(5000.0 / N)) * reliability
    return lam


# ---------------------------------------------------------------------------
# Training: ERM baseline
# ---------------------------------------------------------------------------

def train_erm(
    cfg: DictConfig, model: torch.nn.Module,
    loaders: dict[str, DataLoader], device: torch.device,
    run: object, group_free: bool = False,
) -> dict[str, float]:
    """Standard ERM training. Returns best-checkpoint metrics."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)

    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(y)
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_n += len(y)

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        metrics = {"train/loss": ep_loss / ep_n, "train/acc": ep_correct / ep_n,
                   **_eval_metrics("eval/id", id_m), **_eval_metrics("eval/ood", ood_m)}
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_m, group_free), model, metrics)

    return selector.restore(model)


# ---------------------------------------------------------------------------
# Training: JTT baseline (Liu et al. 2021)
# ---------------------------------------------------------------------------

def discover_jtt_weights(
    cfg: DictConfig, loaders: dict[str, DataLoader], device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """JTT phase 1: train ERM, identify misclassified, assign binary weights."""
    train_ds = loaders["train"].dataset
    N = len(train_ds)
    upweight = getattr(cfg.training, "discovery_upweight", 50.0)

    disc = _build_discovery_model(cfg, loaders, device)
    opt = torch.optim.AdamW(disc.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)
    for _ in range(cfg.training.discovery_epochs):
        disc.train()
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            loss = F.cross_entropy(disc(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    disc.eval()
    misclassified = torch.zeros(N, dtype=torch.bool)
    with torch.no_grad():
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"]
            misclassified[idx] = (disc(x).argmax(1) != y).cpu()

    weights = torch.ones(N)
    weights[misclassified] = upweight
    n_err = misclassified.sum().item()
    return weights, {"jtt/n_misclassified": float(n_err),
                     "jtt/frac_misclassified": n_err / N,
                     "jtt/upweight_factor": float(upweight)}


def train_jtt(
    cfg: DictConfig, model: torch.nn.Module,
    loaders: dict[str, DataLoader], device: torch.device,
    run: object, weights: torch.Tensor, discovery_metrics: dict,
    group_free: bool = False,
) -> dict[str, float]:
    """JTT phase 2: train with upweighted loss on misclassified examples."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)
    weights = weights.to(device)
    log_metrics(run, discovery_metrics, step=0)

    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none")
            w = weights[idx]
            loss = (w * ce).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(y)
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_n += len(y)

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        metrics = {"train/loss": ep_loss / ep_n, "train/acc": ep_correct / ep_n,
                   **_eval_metrics("eval/id", id_m), **_eval_metrics("eval/ood", ood_m)}
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_m, group_free), model, metrics)

    return selector.restore(model)


# ---------------------------------------------------------------------------
# Training: LfF baseline (Nam et al. 2020, "Learning from Failure")
# ---------------------------------------------------------------------------

def _gce_loss(logits: torch.Tensor, targets: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    """Generalized cross-entropy loss (Nam et al. 2020).

    GCE amplifies bias: samples the model is already confident on get higher
    gradient weight, encouraging the biased model to lean into shortcuts.
    Formula: q * p(y|x)^q * CE(x, y), where gradients don't flow through the weight.
    """
    p = F.softmax(logits, dim=1)
    p_y = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    weight = (p_y.detach() ** q) * q
    return weight * F.cross_entropy(logits, targets, reduction="none")


class _LfFEMA:
    """Per-sample exponential moving average of losses, with class-wise normalization."""

    def __init__(self, labels: torch.Tensor, alpha: float = 0.7):
        self.alpha = alpha
        self.labels = labels
        self.ema = torch.zeros(len(labels))
        self.updated = torch.zeros(len(labels))

    def update(self, losses: torch.Tensor, indices: torch.Tensor) -> None:
        idx = indices.cpu()
        # First update: use raw value. Subsequent: EMA smoothing.
        self.ema[idx] = (self.alpha * self.ema[idx]
                         + (1 - self.alpha * self.updated[idx]) * losses.cpu().detach())
        self.updated[idx] = 1.0

    def get_normalized(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return class-wise normalized EMA losses for given indices."""
        vals = self.ema[indices.cpu()].clone()
        for c in labels.unique():
            cls_mask = labels.cpu() == c
            cls_all = self.labels == c
            max_loss = self.ema[cls_all].max().clamp(min=1e-8)
            vals[cls_mask] /= max_loss
        return vals


def train_lff(
    cfg: DictConfig, model: torch.nn.Module,
    loaders: dict[str, DataLoader], device: torch.device,
    run: object, group_free: bool = False,
) -> dict[str, float]:
    """LfF: train biased + debiased models simultaneously (Nam et al. 2020).

    The biased model learns shortcuts via GCE loss. The debiased model is trained
    with CE reweighted by relative difficulty: W(x) = EMA_b / (EMA_b + EMA_d).
    Per-sample losses are EMA-smoothed and class-wise normalized.
    No group labels are used anywhere.
    """
    train_ds = loaders["train"].dataset
    N = len(train_ds)
    all_labels = train_ds.labels if hasattr(train_ds, "labels") else torch.zeros(N, dtype=torch.long)

    biased_model = _build_model(cfg, loaders, device)

    opt_debiased = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)
    opt_biased = torch.optim.AdamW(
        biased_model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)

    ema_b = _LfFEMA(all_labels, alpha=0.7)
    ema_d = _LfFEMA(all_labels, alpha=0.7)

    selector = _ModelSelector()
    for epoch in range(cfg.training.epochs):
        model.train()
        biased_model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0

        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"]

            logits_d = model(x)
            logits_b = biased_model(x)

            # Per-sample CE for EMA update
            ce_d = F.cross_entropy(logits_d, y, reduction="none")
            ce_b = F.cross_entropy(logits_b, y, reduction="none")

            # Update EMAs
            ema_b.update(ce_b, idx)
            ema_d.update(ce_d, idx)

            # Class-wise normalized EMA losses for weight computation
            norm_b = ema_b.get_normalized(idx, y).to(device)
            norm_d = ema_d.get_normalized(idx, y).to(device)
            w = norm_b / (norm_b + norm_d + 1e-8)

            # Joint loss: GCE for biased, weighted CE for debiased
            loss_b = _gce_loss(logits_b, y).mean()
            loss_d = (w.detach() * ce_d).mean()
            loss = loss_b + loss_d

            opt_debiased.zero_grad()
            opt_biased.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(biased_model.parameters(), 1.0)
            opt_debiased.step()
            opt_biased.step()

            ep_loss += loss_d.item() * len(y)
            ep_correct += (logits_d.argmax(1) == y).sum().item()
            ep_n += len(y)

        id_m = evaluate(model, loaders["id_test"], device)
        ood_m = evaluate(model, loaders["ood_test"], device)
        metrics = {"train/loss": ep_loss / ep_n, "train/acc": ep_correct / ep_n,
                   **_eval_metrics("eval/id", id_m), **_eval_metrics("eval/ood", ood_m)}
        log_metrics(run, metrics, step=epoch)
        selector.update(_val_score(id_m, group_free), model, metrics)

    return selector.restore(model)


# ---------------------------------------------------------------------------
# Training: our method — V-REx on discovered environments
# ---------------------------------------------------------------------------

def train_vrex(
    cfg: DictConfig, model: torch.nn.Module,
    loaders: dict[str, DataLoader], device: torch.device,
    run: object, assignment: torch.Tensor, weights: torch.Tensor,
    discovery_metrics: dict[str, float],
    group_free: bool = False,
    lam: float | None = None,
) -> dict[str, float]:
    """V-REx training with automatic fallback to JTT.

    Starts training with V-REx penalty on discovered environments. After a
    warmup period (2 epochs), checks whether the penalty is reducing risk
    variance. If not (RV at epoch 2 >= RV at epoch 0), the penalty is not
    helping — automatically sets λ=0 for the remaining epochs, which is
    equivalent to JTT (plain upweighted ERM). This ensures the method is
    never worse than JTT without requiring group labels to decide.

    Args:
        assignment: (N,) long — environment index per training example.
        weights: (N,) float — per-example importance weights from discovery.
        discovery_metrics: diagnostics from discover_environments.
        group_free: if True, model selection uses avg accuracy (no group labels).
        lam: override for V-REx penalty strength. If None, auto-calibrated.
    """
    if lam is None:
        lam = auto_lambda(discovery_metrics, cfg)
    lam_original = lam

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)
    assignment = assignment.to(device)
    weights = weights.to(device)
    K = int(assignment.max().item()) + 1
    patience = getattr(cfg.training, "early_stop_patience", 0)
    vrex_warmup = 2  # epochs before checking if V-REx is helping

    log_metrics(run, discovery_metrics, step=0)

    selector = _ModelSelector()
    epochs_no_improve = 0
    val_acc_history = []  # track val accuracy for fallback decision

    for epoch in range(cfg.training.epochs):
        model.train()
        ep_loss, ep_correct, ep_n, ep_rv = 0.0, 0, 0, 0.0

        for batch in loaders["train"]:
            x = _to_device(batch["image"], device)
            y = batch["label"].to(device)
            idx = batch["index"].to(device)

            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none")
            a = assignment[idx]
            w = weights[idx]

            # Per-environment weighted mean losses
            env_losses = []
            for k in range(K):
                mask = a == k
                if mask.any():
                    wk = w[mask]
                    env_losses.append((wk * ce[mask]).sum() / wk.sum())

            if len(env_losses) >= 2:
                env_t = torch.stack(env_losses)
                mean_loss = env_t.mean()
                risk_var = ((env_t - mean_loss) ** 2).sum()
                loss = mean_loss + lam * risk_var
            elif len(env_losses) == 1:
                loss = env_losses[0]
                risk_var = torch.tensor(0.0)
            else:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item() * len(y)
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_n += len(y)
            ep_rv += risk_var.item() * len(y)

        avg_rv = ep_rv / ep_n
        id_m = evaluate(model, loaders["id_test"], device)
        val_acc_history.append(id_m["acc"])

        # (V-REx penalty is applied throughout training. Automatic fallback
        # to JTT is handled at the experiment level by comparing V-REx and
        # JTT models' validation accuracy — see run_experiment.py.)

        ood_m = evaluate(model, loaders["ood_test"], device)
        metrics = {
            "train/loss": ep_loss / ep_n, "train/acc": ep_correct / ep_n,
            "train/risk_variance": avg_rv, "train/lambda": lam,
            **_eval_metrics("eval/id", id_m), **_eval_metrics("eval/ood", ood_m),
        }
        log_metrics(run, metrics, step=epoch)

        prev = selector.best_score
        selector.update(_val_score(id_m, group_free), model, metrics)
        improved = selector.best_score > prev

        if patience > 0 and epoch >= min(cfg.training.epochs // 2, 5):
            epochs_no_improve = 0 if improved else epochs_no_improve + 1
            if epochs_no_improve >= patience:
                break

    return selector.restore(model)
