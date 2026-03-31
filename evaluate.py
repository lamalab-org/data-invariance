from __future__ import annotations

import torch


def compute_assignment_correlation(
    assignment_logits: torch.Tensor,
    dataset,
) -> dict[str, float]:
    """Measure how well the learned soft assignment tracks the spurious colour feature.

    For a perfect adversarial partition we expect the adversary to have discovered
    the colour attribute — examples with one colour routed to head A, the other
    to head B.  This is quantified by Pearson correlation between
    s_i = sigmoid(logit_i) and color_i ∈ {0, 1}.

    High |corr| → adversary found the colour-correlated split.
    |corr| ≈ 0  → partition is orthogonal to colour (random or tracking a
                   different feature — interesting in its own right).

    The sign is arbitrary (which head gets colour=0 depends on random init),
    so we report both raw and absolute values.

    Args:
        assignment_logits: (N,) learned logits from train_adversarial_split.
        dataset: ColoredMNIST instance with a `.colors` attribute of shape (N,).

    Returns:
        dict with:
          "assignment_color_corr"      — signed Pearson r ∈ [-1, 1]
          "assignment_color_abs_corr"  — |r|, the discrimination metric
    """
    with torch.no_grad():
        s = assignment_logits.cpu().float().sigmoid()   # (N,) ∈ (0, 1)

    colors = dataset.colors.float().cpu()   # (N,) ∈ {0.0, 1.0}

    # torch.corrcoef expects (n_variables, n_observations)
    corr_matrix = torch.corrcoef(torch.stack([s, colors]))
    corr = corr_matrix[0, 1].item()

    return {
        "assignment_color_corr": corr,
        "assignment_color_abs_corr": abs(corr),
    }


def compute_assignment_correlation_multi(
    assignment_logits: torch.Tensor,
    dataset,
) -> dict[str, float]:
    """Per-head colour correlation for K>2 assignments (softmax parameterisation).

    For each head k, computes |Pearson r| between s_ik = softmax(logits_i)[k]
    and color_i. Reports the maximum over heads — the head that most strongly
    captures the colour feature.

    Args:
        assignment_logits: (N, K) learned logits from train_adversarial_split_multi.
        dataset: ColoredMNIST instance with a `.colors` attribute of shape (N,).

    Returns:
        dict with:
          "assignment_color_max_abs_corr" — max |r| over all K heads
          "assignment_color_mean_abs_corr" — mean |r| over all K heads
    """
    with torch.no_grad():
        s = assignment_logits.cpu().float().softmax(dim=1)   # (N, K)

    colors = dataset.colors.float().cpu()   # (N,)
    K = s.shape[1]

    abs_corrs = []
    for k in range(K):
        corr_matrix = torch.corrcoef(torch.stack([s[:, k], colors]))
        abs_corrs.append(abs(corr_matrix[0, 1].item()))

    return {
        "assignment_color_max_abs_corr": max(abs_corrs),
        "assignment_color_mean_abs_corr": sum(abs_corrs) / K,
    }
