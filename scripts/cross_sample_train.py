"""Train models under the cross-sample fragility protocol.

Cross-sample protocol
---------------------
A canonical dataset is constructed once at ``--canonical_data_seed`` (default
99), so the test set is byte-identical across every training run. For each
``--train_seed`` we draw a different *bootstrap* of the canonical training
data (size N with replacement) — this simulates "what would have happened
if I had drawn a different sample of size N from the same population."

We measure cross-sample fragility as the per-example argmax disagreement
between the predictions of two models trained at different ``train_seed``s
on the *same* canonical test set.

Modes
-----
``erm``           Single model trained on one bootstrap. Baseline.
``deep_ensemble`` K models on the *same* bootstrap, different inits.
                  Isolates parameter variance (no data variance).
``bagging``       K models on K *independent* bootstraps. Parameter +
                  data variance, no consistency loss.
``twin_indep``    K=2 models on independent bootstraps with a sym-KL
                  consistency penalty on the union of both batches.
                  This is the paper's main method.

All modes save softmax predictions on canonical id_test and ood_test.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._provenance import make_context as _make_provenance  # noqa: E402
from scripts.run_experiment import HPARAMS, _build_model, build_cfg  # noqa: E402
from train import make_dataloaders  # noqa: E402
from utils import set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _move_x(batch_image, device):
    """Move a batch ``"image"`` field to ``device``. Handles dict-batches."""
    if isinstance(batch_image, dict):
        return {k: v.to(device) for k, v in batch_image.items()}
    return batch_image.to(device)


def _predict(model, loader, device):
    """Forward `model` over `loader`, return (probs, labels, indices) np arrays."""
    model.eval()
    probs, labels, idx = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = _move_x(batch["image"], device)
            probs.append(F.softmax(model(x), dim=1).cpu().numpy())
            labels.append(batch["label"].numpy())
            idx.append(batch["index"].numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idx)


def _predict_reg(model, loader, device):
    """Regression analogue of `_predict`: returns raw scalar predictions.

    The regression head outputs shape (B, 1); we ``squeeze(-1)`` to get
    a 1-D array.  No softmax is applied (regression has no class
    probabilities).  Cross-sample fragility for regression is computed
    downstream as ``mean |pred_A(x) - pred_B(x)|`` between two retrainings.

    Returns: ``(preds_1d, labels_1d, indices_1d)`` as ``np.float32`` /
    ``np.int64`` numpy arrays.
    """
    model.eval()
    preds, labels, idx = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = _move_x(batch["image"], device)
            preds.append(model(x).squeeze(-1).cpu().numpy())
            labels.append(batch["label"].numpy())
            idx.append(batch["index"].numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(idx)


def _is_regression(cfg) -> bool:
    """Dispatch helper: true iff the dataset config declares ``task: regression``.

    Used at every method-branch entry point in this script to switch
    between (CE + softmax) and (MSE + raw output).  ``getattr`` with
    default keeps the helper backwards-compatible with classification
    configs that pre-date the ``task`` field.
    """
    return getattr(cfg.dataset, "task", "classification") == "regression"


def _build_mc_dropout_mlp(cfg, canonical_loaders, device, p=0.2):
    """MLP backbone with dropout layers inserted after each ReLU.

    Only supports the default-MLP branch of `_build_model` (i.e. Morgan-
    fingerprint chemistry datasets); raises for resnet/distilbert/chemberta/gin.
    The MC-dropout baseline is reported on the headline architecture only.
    """
    if cfg.dataset.arch in ("resnet", "distilbert", "chemberta", "gin"):
        raise ValueError(
            f"mc_dropout mode supports MLP backbones only; got arch={cfg.dataset.arch}")
    import torch.nn as nn
    from models import MLP
    input_dim = canonical_loaders["train"].dataset.input_dim
    h = cfg.model.hidden_dim
    backbone = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(p),
        nn.Linear(h, h),         nn.ReLU(), nn.Dropout(p),
    )
    return MLP(backbone=backbone, backbone_out_dim=h).to(device)


def _predict_mc(model, loader, device, T=20):
    """T stochastic forward passes with dropout active; return averaged softmax."""
    # Keep dropout active. Our MLP has no BatchNorm, so model.train() is safe.
    model.train()
    probs, labels, idx = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = _move_x(batch["image"], device)
            stack = torch.stack(
                [F.softmax(model(x), dim=1) for _ in range(T)], dim=0)
            probs.append(stack.mean(0).cpu().numpy())
            labels.append(batch["label"].numpy())
            idx.append(batch["index"].numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idx)


def _bootstrap_indices(n_train, seed, size=None):
    """`size` (or `n_train` if None) samples with replacement from {0,..,n_train-1}.

    Note: at the same `seed` value, different modes use slightly different
    seed conventions:
        ERM, twin_indep:  seed = train_seed                  (one or two draws)
        bagging, deep_ensemble: seed = train_seed * 100 + k  (one draw per k)
    Cross-sample metrics in this paper compare runs *across train_seeds*;
    method comparisons within a single train_seed are not exact matched
    pairs because of this convention.  See `paper/sections/scope.tex`.

    `size` lets the caller shrink the bootstrap to a subsample of the
    canonical training set, used by within-dataset N-scaling experiments.
    """
    if size is None:
        size = n_train
    return np.random.default_rng(seed).integers(0, n_train, size=size)


def _make_loader(base_dataset, indices, cfg, canonical_loaders, pool_idx=None):
    """DataLoader over ``Subset(base_dataset, indices)`` with canonical kwargs.

    If `pool_idx` is given, `indices` is interpreted as relative to the pool
    and is mapped through `pool_idx` before subsetting `base_dataset`.
    """
    indices = np.asarray(indices)
    if pool_idx is not None:
        indices = pool_idx[indices]
    return DataLoader(
        Subset(base_dataset, indices.tolist()),
        batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=canonical_loaders["train"].num_workers,
        pin_memory=canonical_loaders["train"].pin_memory,
        collate_fn=canonical_loaders["train"].collate_fn,
    )


# ---------------------------------------------------------------------------
# Training procedures (one per mode)
# ---------------------------------------------------------------------------

def _train_one_model(cfg, train_loader, canonical_loaders, device,
                     model_seed, epochs):
    """Train a single model on ``train_loader`` and return it.

    The loss is dispatched on the dataset's task type:
      - classification: cross-entropy on logits
      - regression:     MSE on the squeezed scalar prediction

    For regression, labels are cast to ``float32`` because Apple MPS
    does not support ``float64`` and the default Python-float collate
    yields ``float64``.  ``set_seed`` controls the model's random
    initialisation (separate from the bootstrap seed used in
    ``_make_loader``).
    """
    set_seed(model_seed)
    model = _build_model(cfg, canonical_loaders, device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)
    regression = _is_regression(cfg)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x = _move_x(batch["image"], device)
            if regression:
                # MPS does not support float64; collate yields float64 from
                # Python-float labels, so cast at load time.
                y = batch["label"].to(device=device, dtype=torch.float32)
            else:
                y = batch["label"].to(device)
            opt.zero_grad()
            if regression:
                F.mse_loss(model(x).squeeze(-1), y).backward()
            else:
                F.cross_entropy(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def _train_ensemble(cfg, canonical_loaders, device, train_seed, epochs, K,
                    independent_bootstraps, pool_idx=None, n_train=None):
    """K-model ensemble.

    independent_bootstraps=False  → 'deep ensemble': all K models share one
                                    bootstrap; only init seeds differ.
    independent_bootstraps=True   → 'bagging': each model gets its own
                                    independent bootstrap.

    `pool_idx` and `n_train` (effective pool size) restrict the bootstrap
    pool when subsampling for within-dataset N-scaling.
    """
    base = canonical_loaders["train"].dataset
    if n_train is None:
        n_train = len(base)
    shared_loader = None
    if not independent_bootstraps:
        shared_loader = _make_loader(
            base, _bootstrap_indices(n_train, train_seed),
            cfg, canonical_loaders, pool_idx=pool_idx)

    models = []
    for k in range(K):
        if independent_bootstraps:
            loader = _make_loader(
                base, _bootstrap_indices(n_train, train_seed * 100 + k),
                cfg, canonical_loaders, pool_idx=pool_idx)
        else:
            loader = shared_loader
        models.append(_train_one_model(
            cfg, loader, canonical_loaders, device,
            model_seed=train_seed * 100 + k, epochs=epochs))
    return models


def _sym_kl_loss(logits_a, logits_b, eps=1e-12):
    """Symmetric KL between two batches of softmax distributions."""
    p_a = F.softmax(logits_a, dim=1)
    p_b = F.softmax(logits_b, dim=1)
    log_p_a = torch.log(p_a + eps)
    log_p_b = torch.log(p_b + eps)
    return 0.5 * ((p_a * (log_p_a - log_p_b)).sum(-1)
                  + (p_b * (log_p_b - log_p_a)).sum(-1)).mean()


def _train_twin_indep(cfg, canonical_loaders, device, train_seed, epochs, lam,
                      pool_idx=None, n_train=None):
    """Two-head twin where each head sees its OWN independent bootstrap.

    This is the paper's main method.  The two heads are full networks
    (model_A, model_B) initialised with consecutive seeds so they
    differ from step zero.  Each network's bootstrap is drawn
    independently from the canonical pool, giving an expected
    inter-network bootstrap overlap of $\\sim 40\\%$ (the operating
    point on the codistillation family that empirically holds
    accuracy across our 10-dataset spectrum).

    Per training step:
      - Pull one batch from each network's loader (independent bootstraps).
      - Both networks forward both batches (4 forward passes).
      - Task loss: each network on its own batch only.
      - Consistency loss: sym-KL between networks (classification) or
        MSE between predictions (regression), evaluated on the union
        of the two batches so the consistency penalty sees both
        networks' bootstrap support.
      - Joint loss = task + lam * consistency, single backward updates
        both networks' parameters.

    The classification consistency uses sym-KL (symmetric KL on softmax
    distributions); the regression consistency uses MSE between scalar
    predictions.  The two have different units so ``lam`` does not
    transfer directly between task types — see ``app:regression`` and
    ``app:gin`` in the paper for rule-selected lambdas per setting.
    """
    base = canonical_loaders["train"].dataset
    if n_train is None:
        n_train = len(base)

    set_seed(train_seed)
    model_A = _build_model(cfg, canonical_loaders, device)
    set_seed(train_seed + 1)
    model_B = _build_model(cfg, canonical_loaders, device)
    opt_A = torch.optim.AdamW(model_A.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)

    rng = np.random.default_rng(train_seed)
    boot_A = rng.integers(0, n_train, size=n_train)
    boot_B = rng.integers(0, n_train, size=n_train)
    loader_A = _make_loader(base, boot_A, cfg, canonical_loaders, pool_idx=pool_idx)
    loader_B = _make_loader(base, boot_B, cfg, canonical_loaders, pool_idx=pool_idx)

    regression = _is_regression(cfg)
    for _ in range(epochs):
        model_A.train(); model_B.train()
        it_A, it_B = iter(loader_A), iter(loader_B)
        for _ in range(max(len(loader_A), len(loader_B))):
            try:    ba = next(it_A)
            except StopIteration: it_A = iter(loader_A); ba = next(it_A)
            try:    bb = next(it_B)
            except StopIteration: it_B = iter(loader_B); bb = next(it_B)

            x_a = _move_x(ba["image"], device)
            x_b = _move_x(bb["image"], device)
            if regression:
                y_a = ba["label"].to(device=device, dtype=torch.float32)
                y_b = bb["label"].to(device=device, dtype=torch.float32)
            else:
                y_a = ba["label"].to(device)
                y_b = bb["label"].to(device)

            out_a_on_a, out_b_on_a = model_A(x_a), model_B(x_a)
            out_a_on_b, out_b_on_b = model_A(x_b), model_B(x_b)

            if regression:
                # MSE task loss + MSE consistency between the two networks'
                # raw predictions on each batch.
                pa_a, pb_a = out_a_on_a.squeeze(-1), out_b_on_a.squeeze(-1)
                pa_b, pb_b = out_a_on_b.squeeze(-1), out_b_on_b.squeeze(-1)
                task_loss = (F.mse_loss(pa_a, y_a) + F.mse_loss(pb_b, y_b))
                cons = 0.5 * (F.mse_loss(pa_a, pb_a) + F.mse_loss(pa_b, pb_b))
            else:
                task_loss = (F.cross_entropy(out_a_on_a, y_a)
                             + F.cross_entropy(out_b_on_b, y_b))
                cons = 0.5 * (_sym_kl_loss(out_a_on_a, out_b_on_a)
                              + _sym_kl_loss(out_a_on_b, out_b_on_b))
            loss = task_loss + lam * cons

            opt_A.zero_grad(); opt_B.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
            opt_A.step(); opt_B.step()
    return model_A, model_B


def _train_codistillation(cfg, canonical_loaders, device, train_seed, epochs,
                          lam, pool_idx=None, n_train=None):
    """Codistillation~\\cite{Anil 2018}: two networks on DISJOINT shards
    of the canonical training pool, with sym-KL agreement on the union of
    each step's two batches.  This is the published method-side neighbour
    to twin-indep; the only mechanical difference is data sharding (no
    overlap) instead of bootstraps with replacement (~40% overlap)."""
    base = canonical_loaders["train"].dataset
    if n_train is None:
        n_train = len(base)
    set_seed(train_seed)
    model_A = _build_model(cfg, canonical_loaders, device)
    set_seed(train_seed + 1)
    model_B = _build_model(cfg, canonical_loaders, device)
    opt_A = torch.optim.AdamW(model_A.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)

    rng = np.random.default_rng(train_seed)
    perm = rng.permutation(n_train)
    half = n_train // 2
    shard_A, shard_B = perm[:half], perm[half:]
    loader_A = _make_loader(base, shard_A, cfg, canonical_loaders, pool_idx=pool_idx)
    loader_B = _make_loader(base, shard_B, cfg, canonical_loaders, pool_idx=pool_idx)

    regression = _is_regression(cfg)
    for _ in range(epochs):
        model_A.train(); model_B.train()
        it_A, it_B = iter(loader_A), iter(loader_B)
        for _ in range(max(len(loader_A), len(loader_B))):
            try:    ba = next(it_A)
            except StopIteration: it_A = iter(loader_A); ba = next(it_A)
            try:    bb = next(it_B)
            except StopIteration: it_B = iter(loader_B); bb = next(it_B)
            x_a = _move_x(ba["image"], device)
            x_b = _move_x(bb["image"], device)
            if regression:
                y_a = ba["label"].to(device=device, dtype=torch.float32)
                y_b = bb["label"].to(device=device, dtype=torch.float32)
            else:
                y_a = ba["label"].to(device)
                y_b = bb["label"].to(device)
            la_a, lb_a = model_A(x_a), model_B(x_a)
            la_b, lb_b = model_A(x_b), model_B(x_b)
            ce = F.cross_entropy(la_a, y_a) + F.cross_entropy(lb_b, y_b)
            cons = 0.5 * (_sym_kl_loss(la_a, lb_a) + _sym_kl_loss(la_b, lb_b))
            loss = ce + lam * cons
            opt_A.zero_grad(); opt_B.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
            opt_A.step(); opt_B.step()
    return model_A, model_B


def _train_distillation_anchor(cfg, canonical_loaders, device, train_seed, epochs,
                               lam, pool_idx=None, n_train=None):
    """Distillation-from-anchor~\\cite{Jiang 2022}: train a fresh student
    with task loss + lambda * KL(student || frozen anchor).  We use an
    ERM model trained at canonical_data_seed (no bootstrap) as the
    anchor, mimicking the deployment-against-incumbent setting."""
    base = canonical_loaders["train"].dataset
    if n_train is None:
        n_train = len(base)

    # Anchor: ERM on a fresh bootstrap drawn at a fixed (train_seed-independent)
    # seed, so every student sees the same incumbent.
    set_seed(0)
    anchor = _build_model(cfg, canonical_loaders, device)
    anchor_loader = _make_loader(
        base, _bootstrap_indices(n_train, 0), cfg, canonical_loaders,
        pool_idx=pool_idx,
    )
    anchor_opt = torch.optim.AdamW(anchor.parameters(), lr=cfg.training.lr,
                                   weight_decay=cfg.training.weight_decay)
    for _ in range(epochs):
        anchor.train()
        for batch in anchor_loader:
            x = _move_x(batch["image"], device)
            y = batch["label"].to(device)
            anchor_opt.zero_grad()
            F.cross_entropy(anchor(x), y).backward()
            torch.nn.utils.clip_grad_norm_(anchor.parameters(), 1.0)
            anchor_opt.step()
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad = False

    # Student: trained on its own bootstrap; KL to frozen anchor on each batch.
    set_seed(train_seed)
    student = _build_model(cfg, canonical_loaders, device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)
    student_loader = _make_loader(
        base, _bootstrap_indices(n_train, train_seed), cfg, canonical_loaders,
        pool_idx=pool_idx,
    )
    for _ in range(epochs):
        student.train()
        for batch in student_loader:
            x = _move_x(batch["image"], device)
            y = batch["label"].to(device)
            student_logits = student(x)
            with torch.no_grad():
                anchor_logits = anchor(x)
            ce = F.cross_entropy(student_logits, y)
            kl = _sym_kl_loss(student_logits, anchor_logits)
            opt.zero_grad()
            (ce + lam * kl).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
    return student, anchor


def _train_twin_indep_shared(cfg, canonical_loaders, device, train_seed, epochs,
                             lam, pool_idx=None, n_train=None):
    """Ablation: twin-indep but both networks see the SAME bootstrap.
    Disagreement comes only from initialisation, so the consistency loss
    penalises init-induced variance with no data axis to vary against."""
    base = canonical_loaders["train"].dataset
    if n_train is None:
        n_train = len(base)
    set_seed(train_seed)
    model_A = _build_model(cfg, canonical_loaders, device)
    set_seed(train_seed + 1)
    model_B = _build_model(cfg, canonical_loaders, device)
    opt_A = torch.optim.AdamW(model_A.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)

    boot = _bootstrap_indices(n_train, train_seed)
    loader_A = _make_loader(base, boot, cfg, canonical_loaders, pool_idx=pool_idx)
    loader_B = _make_loader(base, boot, cfg, canonical_loaders, pool_idx=pool_idx)

    regression = _is_regression(cfg)
    for _ in range(epochs):
        model_A.train(); model_B.train()
        it_A, it_B = iter(loader_A), iter(loader_B)
        for _ in range(max(len(loader_A), len(loader_B))):
            try:    ba = next(it_A)
            except StopIteration: it_A = iter(loader_A); ba = next(it_A)
            try:    bb = next(it_B)
            except StopIteration: it_B = iter(loader_B); bb = next(it_B)
            x_a = _move_x(ba["image"], device)
            x_b = _move_x(bb["image"], device)
            if regression:
                y_a = ba["label"].to(device=device, dtype=torch.float32)
                y_b = bb["label"].to(device=device, dtype=torch.float32)
            else:
                y_a = ba["label"].to(device)
                y_b = bb["label"].to(device)
            la_a, lb_a = model_A(x_a), model_B(x_a)
            la_b, lb_b = model_A(x_b), model_B(x_b)
            ce = F.cross_entropy(la_a, y_a) + F.cross_entropy(lb_b, y_b)
            cons = 0.5 * (_sym_kl_loss(la_a, lb_a) + _sym_kl_loss(la_b, lb_b))
            loss = ce + lam * cons
            opt_A.zero_grad(); opt_B.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
            opt_A.step(); opt_B.step()
    return model_A, model_B


def _train_twin_gradnorm(cfg, canonical_loaders, device, train_seed, epochs,
                         target_ratio=1.0, ema_alpha=0.9,
                         pool_idx=None, n_train=None):
    """Twin-indep with GradNorm-balanced lambda.

    At every step, lambda is chosen so that lam * |∇L_cons| ≈ target_ratio *
    |∇L_CE| (norm taken over `model_A`'s parameters; the symmetry of the
    two-head construction makes the choice of which network's gradient we use
    a wash).  We EMA-smooth across steps to prevent oscillation.

    This is the parameter-free analogue of `_train_twin_indep` and serves as
    a reported design choice we tried and rejected (see appendix:failures).
    """
    base = canonical_loaders["train"].dataset
    if n_train is None:
        n_train = len(base)

    set_seed(train_seed)
    model_A = _build_model(cfg, canonical_loaders, device)
    set_seed(train_seed + 1)
    model_B = _build_model(cfg, canonical_loaders, device)
    opt_A = torch.optim.AdamW(model_A.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay)

    rng = np.random.default_rng(train_seed)
    boot_A = rng.integers(0, n_train, size=n_train)
    boot_B = rng.integers(0, n_train, size=n_train)
    loader_A = _make_loader(base, boot_A, cfg, canonical_loaders, pool_idx=pool_idx)
    loader_B = _make_loader(base, boot_B, cfg, canonical_loaders, pool_idx=pool_idx)

    lam = 1.0  # initial value before the first balanced update
    regression = _is_regression(cfg)
    for _ in range(epochs):
        model_A.train(); model_B.train()
        it_A, it_B = iter(loader_A), iter(loader_B)
        for _ in range(max(len(loader_A), len(loader_B))):
            try:    ba = next(it_A)
            except StopIteration: it_A = iter(loader_A); ba = next(it_A)
            try:    bb = next(it_B)
            except StopIteration: it_B = iter(loader_B); bb = next(it_B)

            x_a = _move_x(ba["image"], device)
            x_b = _move_x(bb["image"], device)
            if regression:
                y_a = ba["label"].to(device=device, dtype=torch.float32)
                y_b = bb["label"].to(device=device, dtype=torch.float32)
            else:
                y_a = ba["label"].to(device)
                y_b = bb["label"].to(device)

            logits_a_on_a, logits_b_on_a = model_A(x_a), model_B(x_a)
            logits_a_on_b, logits_b_on_b = model_A(x_b), model_B(x_b)

            ce = (F.cross_entropy(logits_a_on_a, y_a)
                  + F.cross_entropy(logits_b_on_b, y_b))
            cons = 0.5 * (_sym_kl_loss(logits_a_on_a, logits_b_on_a)
                          + _sym_kl_loss(logits_a_on_b, logits_b_on_b))

            # Compute |∇CE| and |∇cons| separately to balance them.
            ce_grads = torch.autograd.grad(ce, model_A.parameters(),
                                           retain_graph=True)
            cons_grads = torch.autograd.grad(cons, model_A.parameters(),
                                             retain_graph=True)
            ce_norm = torch.cat([g.flatten() for g in ce_grads]).norm().item()
            cons_norm = torch.cat([g.flatten() for g in cons_grads]).norm().item()
            new_lam = (target_ratio * ce_norm) / max(cons_norm, 1e-8)
            lam = ema_alpha * lam + (1 - ema_alpha) * new_lam
            lam = float(np.clip(lam, 1e-3, 1e3))

            loss = ce + lam * cons
            opt_A.zero_grad(); opt_B.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
            opt_A.step(); opt_B.step()
    return model_A, model_B


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.dataset)
    cfg.training.seed = args.canonical_data_seed
    set_seed(args.canonical_data_seed)
    canonical_loaders = make_dataloaders(cfg)
    epochs = args.epochs if args.epochs is not None else HPARAMS[args.dataset]["epochs"]
    n_train_full = len(canonical_loaders["train"].dataset)

    # Optional within-dataset N-scaling: cap the canonical training pool at
    # the first M indices of a deterministic shuffle (via canonical_data_seed).
    # We map subsequent bootstrap indices through `pool_idx`, leaving
    # `canonical_loaders["train"].dataset` untouched so `_build_model` still
    # finds `input_dim` on the underlying dataset.
    pool_idx = None
    if args.subsample_size is not None and args.subsample_size < n_train_full:
        pool_rng = np.random.default_rng(args.canonical_data_seed)
        pool_idx = pool_rng.permutation(n_train_full)[: args.subsample_size]
        n_train = args.subsample_size
        print(f"Subsample: training pool capped at M={n_train} "
              f"(canonical set has {n_train_full} examples).")
    else:
        n_train = n_train_full

    print(f"Canonical test set: id n={len(canonical_loaders['id_test'].dataset)}, "
          f"ood n={len(canonical_loaders['ood_test'].dataset)}")

    # Provenance: hash the canonical pool and id-test indices once. These
    # are byte-identical across train_seeds for a given canonical_data_seed,
    # so any drift in the underlying dataset will surface as a hash change.
    import hashlib as _hashlib

    def _idx_hash(loader):
        ds = loader.dataset
        idxs = getattr(ds, "indices", None)
        if idxs is None:
            idxs = list(range(len(ds)))
        return _hashlib.sha256(
            np.asarray(idxs, dtype=np.int64).tobytes()
        ).hexdigest()[:16]

    canonical_pool_hash = _idx_hash(canonical_loaders["train"])
    id_test_hash = _idx_hash(canonical_loaders["id_test"])
    if pool_idx is not None:
        canonical_pool_hash = _hashlib.sha256(
            np.asarray(pool_idx, dtype=np.int64).tobytes()
        ).hexdigest()[:16]

    def _save(name: str, payload: dict, *, train_seed: int, **extra) -> None:
        """Save NPZ + manifest sidecar with full provenance.

        ``name`` is the file basename (no extension).  ``payload`` is the
        kwargs dict for ``np.savez_compressed``.  ``train_seed`` plus any
        ``**extra`` go into the manifest's ``config`` block (mode-specific
        keys: K, lam, T, target_ratio, task).  Every NPZ produced by this
        script gets a sibling ``<basename>.manifest.json`` and a line
        appended to ``outputs/cross_sample/RUN_LEDGER.jsonl``.
        """
        npz_path = out_root / f"{name}.npz"
        np.savez_compressed(npz_path, **payload)
        config = {
            "dataset": args.dataset,
            "mode": args.mode,
            "canonical_data_seed": args.canonical_data_seed,
            "train_seed": int(train_seed),
            "epochs": epochs,
            "subsample_size": args.subsample_size,
            "n_train": int(n_train),
            "n_train_full": int(n_train_full),
            **extra,
        }
        ctx = _make_provenance(npz_path, config)
        ctx.data_hash = canonical_pool_hash
        ctx.test_hash = id_test_hash
        ctx.write()

    for train_seed in (int(s) for s in args.train_seeds.split(",")):
        print(f"\n=== {args.dataset}  canonical={args.canonical_data_seed}  "
              f"train_seed={train_seed}  mode={args.mode} ===")

        if args.mode == "erm":
            boot_loader = _make_loader(
                canonical_loaders["train"].dataset,
                _bootstrap_indices(n_train, train_seed),
                cfg, canonical_loaders, pool_idx=pool_idx)
            model = _train_one_model(cfg, boot_loader, canonical_loaders,
                                     device, train_seed, epochs)
            if _is_regression(cfg):
                id_p, id_y, id_i = _predict_reg(model, canonical_loaders["id_test"], device)
                ood_p, ood_y, ood_i = _predict_reg(model, canonical_loaders["ood_test"], device)
                print(f"  id_mae={np.abs(id_p-id_y).mean():.4f}  "
                      f"ood_mae={np.abs(ood_p-ood_y).mean():.4f}")
                _save(f"erm_train{train_seed}",
                      dict(id_preds=id_p, id_labels=id_y, id_indices=id_i,
                           ood_preds=ood_p, ood_labels=ood_y, ood_indices=ood_i,
                           canonical_data_seed=args.canonical_data_seed,
                           train_seed=train_seed, mode="erm", task="regression"),
                      train_seed=train_seed, task="regression")
            else:
                id_p, id_y, id_i = _predict(model, canonical_loaders["id_test"], device)
                ood_p, ood_y, ood_i = _predict(model, canonical_loaders["ood_test"], device)
                print(f"  id_acc={(id_p.argmax(1)==id_y).mean():.4f}  "
                      f"ood_acc={(ood_p.argmax(1)==ood_y).mean():.4f}")
                _save(f"erm_train{train_seed}",
                      dict(id_probs=id_p, id_labels=id_y, id_indices=id_i,
                           ood_probs=ood_p, ood_labels=ood_y, ood_indices=ood_i,
                           canonical_data_seed=args.canonical_data_seed,
                           train_seed=train_seed, mode="erm"),
                      train_seed=train_seed)

        elif args.mode == "mc_dropout":
            # Train one MLP-with-dropout per train_seed on a bootstrap of the
            # canonical pool (same data protocol as ERM); at test time average
            # T stochastic forward passes with dropout active.
            boot_loader = _make_loader(
                canonical_loaders["train"].dataset,
                _bootstrap_indices(n_train, train_seed),
                cfg, canonical_loaders, pool_idx=pool_idx)
            set_seed(train_seed)
            model = _build_mc_dropout_mlp(cfg, canonical_loaders, device)
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=cfg.training.lr,
                                    weight_decay=cfg.training.weight_decay)
            for _ in range(epochs):
                model.train()
                for batch in boot_loader:
                    x = _move_x(batch["image"], device)
                    y = batch["label"].to(device)
                    opt.zero_grad()
                    F.cross_entropy(model(x), y).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            id_p, id_y, id_i = _predict_mc(
                model, canonical_loaders["id_test"], device, T=args.K)
            ood_p, ood_y, ood_i = _predict_mc(
                model, canonical_loaders["ood_test"], device, T=args.K)
            print(f"  id_acc={(id_p.argmax(1)==id_y).mean():.4f}  "
                  f"ood_acc={(ood_p.argmax(1)==ood_y).mean():.4f}  T={args.K}")
            _save(f"mc_dropout_train{train_seed}_T{args.K}",
                  dict(id_probs=id_p, id_labels=id_y, id_indices=id_i,
                       ood_probs=ood_p, ood_labels=ood_y, ood_indices=ood_i,
                       canonical_data_seed=args.canonical_data_seed,
                       train_seed=train_seed, T=args.K, mode="mc_dropout"),
                  train_seed=train_seed, T=args.K)

        elif args.mode in ("deep_ensemble", "bagging"):
            models = _train_ensemble(
                cfg, canonical_loaders, device, train_seed, epochs,
                K=args.K, independent_bootstraps=(args.mode == "bagging"),
                pool_idx=pool_idx, n_train=n_train)
            if _is_regression(cfg):
                id_preds, ood_preds = [], []
                for m in models:
                    ip, iy, ii = _predict_reg(m, canonical_loaders["id_test"], device)
                    op, oy, oi = _predict_reg(m, canonical_loaders["ood_test"], device)
                    id_preds.append(ip); ood_preds.append(op)
                id_avg = np.mean(id_preds, axis=0)
                ood_avg = np.mean(ood_preds, axis=0)
                print(f"  id_mae={np.abs(id_avg-iy).mean():.4f}  "
                      f"ood_mae={np.abs(ood_avg-oy).mean():.4f}  K={args.K}")
                _save(f"{args.mode}_train{train_seed}_K{args.K}",
                      dict(id_preds_avg=id_avg, id_labels=iy, id_indices=ii,
                           ood_preds_avg=ood_avg, ood_labels=oy, ood_indices=oi,
                           canonical_data_seed=args.canonical_data_seed,
                           train_seed=train_seed, K=args.K, mode=args.mode, task="regression"),
                      train_seed=train_seed, K=args.K, task="regression")
            else:
                id_probs, ood_probs = [], []
                for m in models:
                    ip, iy, ii = _predict(m, canonical_loaders["id_test"], device)
                    op, oy, oi = _predict(m, canonical_loaders["ood_test"], device)
                    id_probs.append(ip); ood_probs.append(op)
                id_avg = np.mean(id_probs, axis=0)
                ood_avg = np.mean(ood_probs, axis=0)
                print(f"  id_acc={(id_avg.argmax(1)==iy).mean():.4f}  "
                      f"ood_acc={(ood_avg.argmax(1)==oy).mean():.4f}  K={args.K}")
                _save(f"{args.mode}_train{train_seed}_K{args.K}",
                      dict(id_probs_avg=id_avg, id_labels=iy, id_indices=ii,
                           ood_probs_avg=ood_avg, ood_labels=oy, ood_indices=oi,
                           canonical_data_seed=args.canonical_data_seed,
                           train_seed=train_seed, K=args.K, mode=args.mode),
                      train_seed=train_seed, K=args.K)

        elif args.mode == "twin_indep":
            mA, mB = _train_twin_indep(
                cfg, canonical_loaders, device, train_seed, epochs, args.lam,
                pool_idx=pool_idx, n_train=n_train)
            if _is_regression(cfg):
                id_pA, id_y, id_i = _predict_reg(mA, canonical_loaders["id_test"], device)
                id_pB, _, _      = _predict_reg(mB, canonical_loaders["id_test"], device)
                ood_pA, ood_y, ood_i = _predict_reg(mA, canonical_loaders["ood_test"], device)
                ood_pB, _, _        = _predict_reg(mB, canonical_loaders["ood_test"], device)
                id_avg, ood_avg = 0.5 * (id_pA + id_pB), 0.5 * (ood_pA + ood_pB)
                print(f"  id_mae={np.abs(id_avg-id_y).mean():.4f}  "
                      f"ood_mae={np.abs(ood_avg-ood_y).mean():.4f}  λ={args.lam}")
                _save(f"twin_indep_train{train_seed}_lam{args.lam}",
                      dict(id_preds_A=id_pA, id_preds_B=id_pB, id_preds_avg=id_avg,
                           id_labels=id_y, id_indices=id_i,
                           ood_preds_A=ood_pA, ood_preds_B=ood_pB, ood_preds_avg=ood_avg,
                           ood_labels=ood_y, ood_indices=ood_i,
                           canonical_data_seed=args.canonical_data_seed,
                           train_seed=train_seed, lam=args.lam, mode="twin_indep",
                           task="regression"),
                      train_seed=train_seed, lam=args.lam, task="regression")
            else:
                id_pA, id_y, id_i = _predict(mA, canonical_loaders["id_test"], device)
                id_pB, _, _      = _predict(mB, canonical_loaders["id_test"], device)
                ood_pA, ood_y, ood_i = _predict(mA, canonical_loaders["ood_test"], device)
                ood_pB, _, _        = _predict(mB, canonical_loaders["ood_test"], device)
                id_avg, ood_avg = 0.5 * (id_pA + id_pB), 0.5 * (ood_pA + ood_pB)
                print(f"  id_acc={(id_avg.argmax(1)==id_y).mean():.4f}  "
                      f"ood_acc={(ood_avg.argmax(1)==ood_y).mean():.4f}  λ={args.lam}")
                _save(f"twin_indep_train{train_seed}_lam{args.lam}",
                      dict(id_probs_A=id_pA, id_probs_B=id_pB, id_probs_avg=id_avg,
                           id_labels=id_y, id_indices=id_i,
                           ood_probs_A=ood_pA, ood_probs_B=ood_pB, ood_probs_avg=ood_avg,
                           ood_labels=ood_y, ood_indices=ood_i,
                           canonical_data_seed=args.canonical_data_seed,
                           train_seed=train_seed, lam=args.lam, mode="twin_indep"),
                      train_seed=train_seed, lam=args.lam)

        elif args.mode in ("codistillation", "twin_indep_shared"):
            train_fn = (_train_codistillation if args.mode == "codistillation"
                        else _train_twin_indep_shared)
            mA, mB = train_fn(
                cfg, canonical_loaders, device, train_seed, epochs, args.lam,
                pool_idx=pool_idx, n_train=n_train)
            id_pA, id_y, id_i = _predict(mA, canonical_loaders["id_test"], device)
            id_pB, _, _      = _predict(mB, canonical_loaders["id_test"], device)
            ood_pA, ood_y, ood_i = _predict(mA, canonical_loaders["ood_test"], device)
            ood_pB, _, _        = _predict(mB, canonical_loaders["ood_test"], device)
            id_avg, ood_avg = 0.5 * (id_pA + id_pB), 0.5 * (ood_pA + ood_pB)
            print(f"  id_acc={(id_avg.argmax(1)==id_y).mean():.4f}  "
                  f"ood_acc={(ood_avg.argmax(1)==ood_y).mean():.4f}  λ={args.lam}")
            _save(f"{args.mode}_train{train_seed}_lam{args.lam}",
                  dict(id_probs_A=id_pA, id_probs_B=id_pB, id_probs_avg=id_avg,
                       id_labels=id_y, id_indices=id_i,
                       ood_probs_A=ood_pA, ood_probs_B=ood_pB, ood_probs_avg=ood_avg,
                       ood_labels=ood_y, ood_indices=ood_i,
                       canonical_data_seed=args.canonical_data_seed,
                       train_seed=train_seed, lam=args.lam, mode=args.mode),
                  train_seed=train_seed, lam=args.lam)

        elif args.mode == "distillation_anchor":
            student, anchor = _train_distillation_anchor(
                cfg, canonical_loaders, device, train_seed, epochs, args.lam,
                pool_idx=pool_idx, n_train=n_train)
            id_p, id_y, id_i = _predict(student, canonical_loaders["id_test"], device)
            ood_p, ood_y, ood_i = _predict(student, canonical_loaders["ood_test"], device)
            print(f"  id_acc={(id_p.argmax(1)==id_y).mean():.4f}  "
                  f"ood_acc={(ood_p.argmax(1)==ood_y).mean():.4f}  λ={args.lam}")
            _save(f"distillation_anchor_train{train_seed}_lam{args.lam}",
                  dict(id_probs=id_p, id_labels=id_y, id_indices=id_i,
                       ood_probs=ood_p, ood_labels=ood_y, ood_indices=ood_i,
                       canonical_data_seed=args.canonical_data_seed,
                       train_seed=train_seed, lam=args.lam, mode="distillation_anchor"),
                  train_seed=train_seed, lam=args.lam)

        elif args.mode == "twin_gradnorm":
            mA, mB = _train_twin_gradnorm(
                cfg, canonical_loaders, device, train_seed, epochs,
                target_ratio=args.target_ratio,
                pool_idx=pool_idx, n_train=n_train)
            id_pA, id_y, id_i = _predict(mA, canonical_loaders["id_test"], device)
            id_pB, _, _      = _predict(mB, canonical_loaders["id_test"], device)
            ood_pA, ood_y, ood_i = _predict(mA, canonical_loaders["ood_test"], device)
            ood_pB, _, _        = _predict(mB, canonical_loaders["ood_test"], device)
            id_avg, ood_avg = 0.5 * (id_pA + id_pB), 0.5 * (ood_pA + ood_pB)
            print(f"  id_acc={(id_avg.argmax(1)==id_y).mean():.4f}  "
                  f"ood_acc={(ood_avg.argmax(1)==ood_y).mean():.4f}  "
                  f"target_ratio={args.target_ratio}")
            _save(f"twin_gradnorm_train{train_seed}_tr{args.target_ratio}",
                  dict(id_probs_A=id_pA, id_probs_B=id_pB, id_probs_avg=id_avg,
                       id_labels=id_y, id_indices=id_i,
                       ood_probs_A=ood_pA, ood_probs_B=ood_pB, ood_probs_avg=ood_avg,
                       ood_labels=ood_y, ood_indices=ood_i,
                       canonical_data_seed=args.canonical_data_seed,
                       train_seed=train_seed, target_ratio=args.target_ratio,
                       mode="twin_gradnorm"),
                  train_seed=train_seed, target_ratio=args.target_ratio)

        else:
            raise ValueError(f"unknown mode {args.mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--canonical_data_seed", type=int, default=99,
                    help="Seed for canonical (test-set-fixed) dataset construction.")
    ap.add_argument("--train_seeds", default="1,2,3,4,5",
                    help="Comma-separated list of train_seeds (one bootstrap per seed).")
    ap.add_argument("--mode", required=True,
                    choices=["erm", "mc_dropout", "deep_ensemble", "bagging",
                             "twin_indep", "twin_gradnorm", "codistillation",
                             "distillation_anchor", "twin_indep_shared"])
    ap.add_argument("--K", type=int, default=5,
                    help="Ensemble size (deep_ensemble/bagging only).")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="Consistency-loss weight (twin_indep only).")
    ap.add_argument("--target_ratio", type=float, default=1.0,
                    help="GradNorm target ratio |∇cons|/|∇CE| (twin_gradnorm only).")
    ap.add_argument("--subsample_size", type=int, default=None,
                    help="If set, cap the canonical training pool at this "
                         "many examples (for within-dataset N-scaling).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from HPARAMS (smoke tests).")
    ap.add_argument("--output_dir", default="outputs/cross_sample")
    args = ap.parse_args()
    run(args)
