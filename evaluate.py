from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_assignment_correlation(
    assignment_logits: torch.Tensor,
    dataset,
) -> dict[str, float]:
    """Measure how well the learned soft assignment tracks the spurious attribute.

    For a perfect adversarial partition we expect the adversary to have discovered
    the spurious feature — examples with one attribute value routed to head A,
    the other to head B.  This is quantified by Pearson correlation between
    s_i = sigmoid(logit_i) and spurious_i ∈ {0, 1}.

    High |corr| → adversary found the spurious-correlated split.
    |corr| ≈ 0  → partition is orthogonal to the spurious feature.

    The sign is arbitrary (which head gets which group depends on random init),
    so we report both raw and absolute values.

    Args:
        assignment_logits: (N,) learned logits from train_adversarial_split.
        dataset: dataset with a ``.spurious`` attribute of shape (N,).

    Returns:
        dict with:
          "assignment_color_corr"      — signed Pearson r ∈ [-1, 1]
          "assignment_color_abs_corr"  — |r|, the discrimination metric
    """
    with torch.no_grad():
        s = assignment_logits.cpu().float().sigmoid()   # (N,) ∈ (0, 1)

    spurious = dataset.spurious.float().cpu()   # (N,) ∈ {0.0, 1.0}

    # torch.corrcoef expects (n_variables, n_observations)
    corr_matrix = torch.corrcoef(torch.stack([s, spurious]))
    corr = corr_matrix[0, 1].item()

    return {
        "assignment_color_corr": corr,
        "assignment_color_abs_corr": abs(corr),
    }


def compute_assignment_correlation_multi(
    assignment_logits: torch.Tensor,
    dataset,
) -> dict[str, float]:
    """Per-head spurious correlation for K>2 assignments (softmax parameterisation).

    For each head k, computes |Pearson r| between s_ik = softmax(logits_i)[k]
    and spurious_i.  Reports the maximum over heads — the head that most strongly
    captures the spurious feature.

    Args:
        assignment_logits: (N, K) learned logits from train_adversarial_split_multi.
        dataset: dataset with a ``.spurious`` attribute of shape (N,).

    Returns:
        dict with:
          "assignment_color_max_abs_corr" — max |r| over all K heads
          "assignment_color_mean_abs_corr" — mean |r| over all K heads
    """
    with torch.no_grad():
        s = assignment_logits.cpu().float().softmax(dim=1)   # (N, K)

    spurious = dataset.spurious.float().cpu()   # (N,)
    K = s.shape[1]

    abs_corrs = []
    for k in range(K):
        corr_matrix = torch.corrcoef(torch.stack([s[:, k], spurious]))
        abs_corrs.append(abs(corr_matrix[0, 1].item()))

    return {
        "assignment_color_max_abs_corr": max(abs_corrs),
        "assignment_color_mean_abs_corr": sum(abs_corrs) / K,
    }


def compute_stability_scores(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    n_mc_samples: int = 10,
) -> dict[str, torch.Tensor]:
    """Compute multiple per-example stability/uncertainty scores.

    Scores computed:
    1. **confidence**: max(softmax(logits)) — standard softmax confidence.
    2. **entropy**: -Σ p log p — prediction entropy (higher = more uncertain).
    3. **loss**: cross-entropy loss per example (higher = worse fit).
    4. **mc_dropout_var**: variance of predictions under MC dropout
       (higher = more uncertain).  Only computed if the model has dropout layers
       or we enable dropout manually.

    Args:
        model:  Trained model.
        loader: DataLoader for examples to score.
        device: Torch device.
        n_mc_samples: Number of forward passes for MC dropout.

    Returns:
        Dict with tensors, each (N,):
          confidence, entropy, loss, mc_dropout_var, predictions, labels
    """
    model.eval()

    all_confidence = []
    all_entropy = []
    all_loss = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"]
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y)

            logits = model(x)
            probs = logits.softmax(dim=1).clamp(min=1e-7)

            confidence = probs.max(dim=1).values
            entropy = -(probs * probs.log()).sum(dim=1)
            loss = F.cross_entropy(logits, y.to(device), reduction="none")
            predictions = probs.argmax(dim=1)

            all_confidence.append(confidence.cpu())
            all_entropy.append(entropy.cpu())
            all_loss.append(loss.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(y)

    # MC dropout: enable dropout and do multiple forward passes.
    mc_vars = []
    model.train()  # enable dropout
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            mc_preds = []
            for _ in range(n_mc_samples):
                logits = model(x)
                mc_preds.append(logits.softmax(dim=1)[:, 1].cpu())  # P(class=1)
            mc_stack = torch.stack(mc_preds, dim=0)  # (n_mc, B)
            mc_vars.append(mc_stack.var(dim=0))       # (B,)
    model.eval()

    return {
        "confidence": torch.cat(all_confidence),
        "entropy": torch.cat(all_entropy),
        "loss": torch.cat(all_loss),
        "mc_dropout_var": torch.cat(mc_vars),
        "predictions": torch.cat(all_predictions),
        "labels": torch.cat(all_labels),
    }


def disagreement_stability_scores(
    scores_ours: dict[str, torch.Tensor],
    scores_erm: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Stability score = disagreement between ERM and our model.

    |P_erm(class1) - P_ours(class1)| per example.

    Where they agree: prediction doesn't depend on shortcuts → stable.
    Where they disagree: prediction depends on dataset composition → fragile.

    This doesn't require our model to be more accurate — just different in
    the right way (avoiding shortcuts the ERM uses).
    """
    # Confidence is max(softmax), which for binary = max(P(0), P(1)).
    # We need P(class1) specifically. For binary: P(1) = 1 - P(0).
    # But we have entropy and confidence. Let's use predictions and confidence.
    # Actually, the simplest: |confidence_ours - confidence_erm| weighted by
    # whether they predict the same class.

    # Better: use the raw probability of class 1.
    # For binary classifier: P(1) = 1 - confidence if prediction is 0, else confidence.
    p1_ours = torch.where(
        scores_ours["predictions"] == 1,
        scores_ours["confidence"],
        1.0 - scores_ours["confidence"],
    )
    p1_erm = torch.where(
        scores_erm["predictions"] == 1,
        scores_erm["confidence"],
        1.0 - scores_erm["confidence"],
    )
    return (p1_ours - p1_erm).abs()


def adaptive_stability_scores(
    scores_ours: dict[str, torch.Tensor],
    scores_erm: dict[str, torch.Tensor],
    reliability: float,
) -> dict[str, torch.Tensor]:
    """Blend stability scores from our model and ERM based on reliability.

    When reliability is high (permutation test found strong environment signal),
    our V-REx model's scores are better — it avoids the spurious shortcut.
    When reliability is low (environments are noise), ERM's scores are better —
    it's well-calibrated on clean data.

    The reliability comes from the same permutation test that calibrates λ,
    giving a unified adaptive framework for both training and scoring.

    Args:
        scores_ours: Stability scores from the V-REx model.
        scores_erm:  Stability scores from the ERM model.
        reliability: From the permutation test (0 = pure ERM, 1 = pure V-REx).

    Returns:
        Blended scores with the same keys.
    """
    blended = {}
    for key in ["entropy", "loss", "confidence"]:
        if key in scores_ours and key in scores_erm:
            # Normalise both to [0, 1] range before blending.
            s_ours = scores_ours[key]
            s_erm = scores_erm[key]
            # Min-max normalise.
            s_ours_n = (s_ours - s_ours.min()) / (s_ours.max() - s_ours.min() + 1e-8)
            s_erm_n = (s_erm - s_erm.min()) / (s_erm.max() - s_erm.min() + 1e-8)
            blended[key] = reliability * s_ours_n + (1 - reliability) * s_erm_n
    blended["predictions"] = scores_ours["predictions"]
    blended["labels"] = scores_ours["labels"]
    return blended


def evaluate_stability_discrimination(
    scores_ours: dict[str, torch.Tensor],
    scores_erm: dict[str, torch.Tensor],
    id_preds_erm: torch.Tensor,
    ood_preds_erm: torch.Tensor,
    id_preds_ours: torch.Tensor,
    ood_preds_ours: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Comprehensive evaluation of stability scores for predicting OOD flips.

    For each model (ERM and ours), a "flip" is when a test example's prediction
    changes between the ID and OOD test sets.  We measure how well various
    uncertainty scores predict these flips.

    Scores tested (for both ERM and our model):
    - 1 - confidence (softmax)
    - entropy
    - loss
    - MC dropout variance

    Also computes calibration: bin examples by score, measure actual flip rate per bin.

    Returns a comprehensive dict of AUROCs and calibration metrics.
    """
    import torchmetrics

    results: dict[str, float] = {}

    erm_flips = (id_preds_erm != ood_preds_erm).long()
    ours_flips = (id_preds_ours != ood_preds_ours).long()

    results["erm_flip_rate"] = erm_flips.float().mean().item()
    results["ours_flip_rate"] = ours_flips.float().mean().item()
    results["erm_id_acc"] = (id_preds_erm == labels).float().mean().item()
    results["erm_ood_acc"] = (ood_preds_erm == labels).float().mean().item()
    results["ours_id_acc"] = (id_preds_ours == labels).float().mean().item()
    results["ours_ood_acc"] = (ood_preds_ours == labels).float().mean().item()

    # For each score type, compute AUROC for predicting ERM flips.
    score_types = ["entropy", "loss", "mc_dropout_var"]
    # 1 - confidence is equivalent to using entropy direction, but include both.
    score_types_with_conf = ["confidence_inv"] + score_types

    auroc = torchmetrics.AUROC(task="binary")

    for target_name, flips in [("erm_flips", erm_flips), ("ours_flips", ours_flips)]:
        if flips.sum() == 0 or flips.sum() == len(flips):
            for model_name in ["ours", "erm"]:
                for score_name in score_types_with_conf:
                    results[f"auroc_{model_name}_{score_name}_vs_{target_name}"] = 0.5
            continue

        for model_name, scores in [("ours", scores_ours), ("erm", scores_erm)]:
            # 1 - confidence
            auroc.reset()
            results[f"auroc_{model_name}_confidence_inv_vs_{target_name}"] = auroc(
                1.0 - scores["confidence"], flips
            ).item()

            for score_name in score_types:
                if score_name in scores:
                    auroc.reset()
                    results[f"auroc_{model_name}_{score_name}_vs_{target_name}"] = auroc(
                        scores[score_name], flips
                    ).item()

    # Calibration: bin by our model's entropy, measure actual ERM flip rate per bin.
    n_bins = 5
    if erm_flips.sum() > 0 and erm_flips.sum() < len(erm_flips):
        ours_entropy = scores_ours["entropy"]
        quantiles = torch.linspace(0, 1, n_bins + 1)
        bin_edges = torch.quantile(ours_entropy, quantiles)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (ours_entropy >= lo)
            else:
                mask = (ours_entropy >= lo) & (ours_entropy < hi)
            if mask.sum() > 0:
                results[f"calibration_bin{i}_erm_flip_rate"] = erm_flips[mask].float().mean().item()
                results[f"calibration_bin{i}_n"] = mask.sum().item()

    return results
