"""Helpers for analysing the (now obsolete) adversarial-split method.

This file currently contains only the assignment-correlation diagnostics
used by ``train.train_adversarial_split[_multi]``.  Both functions are
slated for deletion in phase 3 of the refactor (see
``experiments/REFACTOR_PLAN.md``) once the adversarial-split methods
themselves are removed from ``train.py``.

For now they remain so that the existing imports in ``train.py`` continue
to resolve while the running CelebA experiment is still in flight.

The stability-score helpers that used to live here were moved to
``legacy/stability.py``.
"""
from __future__ import annotations

import torch


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


# Stability score helpers (compute_stability_scores, disagreement_stability_scores,
# adaptive_stability_scores, evaluate_stability_discrimination) were moved to
# legacy/stability.py — see legacy/README.md for context. They are not on the
# critical path for the paper's main results.
