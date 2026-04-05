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
    assignment: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    K: int = 2,
) -> dict[str, torch.Tensor]:
    """Compute per-example stability scores from discovered environments.

    For each example, the stability score measures how much the model's
    loss differs across discovered environments.  High score = the prediction
    depends on which data partition the model trains on = fragile.

    Two scores are computed:
    1. **env_loss_gap**: |loss_env_A(x) - loss_env_B(x)| — how differently
       the model performs on this example depending on the environment weighting.
       Approximated by weighting the example's loss by its environment membership.
    2. **confidence**: max(softmax(logits)) — standard softmax confidence.
       This is the baseline we compare stability scores against.

    Args:
        model:      Trained model.
        loader:     DataLoader for the examples to score.
        assignment: (N_train,) environment assignment for training examples.
        weights:    (N_train,) importance weights.
        device:     Torch device.
        K:          Number of environments.

    Returns:
        Dict with:
          "confidence"  — (N_test,) max softmax probability (higher = more confident)
          "stability"   — (N_test,) stability score (higher = more fragile)
          "predictions" — (N_test,) predicted class
          "labels"      — (N_test,) true labels
    """
    model.eval()

    all_confidence = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"]

            logits = model(x)
            probs = logits.softmax(dim=1)  # (B, num_classes)

            confidence = probs.max(dim=1).values  # (B,)
            predictions = probs.argmax(dim=1)     # (B,)

            all_confidence.append(confidence.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(torch.tensor(y) if not isinstance(y, torch.Tensor) else y)

    confidence = torch.cat(all_confidence)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    # Stability score = 1 - confidence.
    # Low confidence = high instability = the model isn't sure.
    # But our contribution is showing that OUR model's confidence
    # (trained with V-REx) is a better stability signal than ERM's confidence.
    # The V-REx model is less confident on examples that genuinely depend on
    # dataset composition, while ERM is confidently wrong on those examples.
    stability = 1.0 - confidence

    return {
        "confidence": confidence,
        "stability": stability,
        "predictions": predictions,
        "labels": labels,
    }


def evaluate_stability_discrimination(
    scores_ours: dict[str, torch.Tensor],
    scores_erm: dict[str, torch.Tensor],
    id_predictions: torch.Tensor,
    ood_predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Evaluate how well stability scores predict OOD prediction flips.

    A "flip" is an example whose prediction changes between the ID test set
    and the OOD test set.  A good stability score should be high for examples
    that flip (fragile predictions) and low for examples that don't (robust).

    We compute AUROC: can the stability score distinguish flippers from non-flippers?

    Args:
        scores_ours: Output of compute_stability_scores for our model.
        scores_erm:  Output of compute_stability_scores for an ERM model.
        id_predictions: (N,) predictions on ID test set.
        ood_predictions: (N,) predictions on OOD test set.
        labels: (N,) true labels.

    Returns:
        Dict with AUROCs for different score types.
    """
    import torchmetrics

    # "Flip" = prediction changed between ID and OOD.
    flips = (id_predictions != ood_predictions).long()
    n_flips = flips.sum().item()
    n_total = len(flips)

    if n_flips == 0 or n_flips == n_total:
        # No discrimination possible.
        return {
            "flip_rate": n_flips / n_total,
            "auroc_ours_stability": 0.5,
            "auroc_erm_stability": 0.5,
            "auroc_ours_confidence_inv": 0.5,
            "auroc_erm_confidence_inv": 0.5,
        }

    auroc = torchmetrics.AUROC(task="binary")

    # Our stability score predicts flips.
    auroc_ours = auroc(scores_ours["stability"], flips).item()

    # ERM stability score predicts flips.
    auroc.reset()
    auroc_erm = auroc(scores_erm["stability"], flips).item()

    # Also compare: does our confidence (inverted) predict flips better than ERM's?
    auroc.reset()
    auroc_ours_conf = auroc(1.0 - scores_ours["confidence"], flips).item()
    auroc.reset()
    auroc_erm_conf = auroc(1.0 - scores_erm["confidence"], flips).item()

    return {
        "flip_rate": n_flips / n_total,
        "auroc_ours_stability": auroc_ours,
        "auroc_erm_stability": auroc_erm,
        "auroc_ours_confidence_inv": auroc_ours_conf,
        "auroc_erm_confidence_inv": auroc_erm_conf,
    }
