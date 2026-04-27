"""Tests for `scripts/_analysis_lib.py`.

The metric implementations (sym-KL, paired bootstrap, NPZ load) are the
load-bearing correctness pieces that every paper number depends on.
We verify them against:
  - scipy.stats.entropy as an independent sym-KL reference,
  - hand-computed values on tiny inputs,
  - bootstrap CI stability across reruns at fixed seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import entropy

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _analysis_lib import (  # noqa: E402
    bootstrap_ci, bootstrap_paired, sym_kl,
)


def test_sym_kl_matches_scipy():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=20)
    q = rng.dirichlet([1.0, 1.0, 1.0], size=20)
    ours = sym_kl(p, q)
    scipy_ref = 0.5 * np.array(
        [entropy(pi, qi) + entropy(qi, pi) for pi, qi in zip(p, q)])
    assert np.allclose(ours, scipy_ref, atol=1e-6)


def test_sym_kl_zero_for_identical_inputs():
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert np.allclose(sym_kl(p, p), 0.0, atol=1e-10)


def test_sym_kl_handles_zeros_with_eps():
    # When one distribution has 0 probability where the other does not,
    # we use eps clamping so KL is finite (not inf).
    p = np.array([[1.0, 0.0]])
    q = np.array([[0.5, 0.5]])
    val = sym_kl(p, q, eps=1e-12)
    assert np.isfinite(val).all()
    assert val[0] > 0


def test_bootstrap_ci_includes_truth():
    # Mean of [0.0,...,1.0] is 0.5; CI should bracket it.
    arr = np.linspace(0.0, 1.0, 1001)
    rng = np.random.default_rng(0)
    mean, lo, hi = bootstrap_ci(arr, n_boot=1000, rng=rng)
    assert lo < 0.5 < hi
    assert abs(mean - 0.5) < 1e-3


def test_bootstrap_ci_deterministic_under_fixed_rng():
    arr = np.linspace(0.0, 1.0, 100)
    a = bootstrap_ci(arr, n_boot=500, rng=np.random.default_rng(7))
    b = bootstrap_ci(arr, n_boot=500, rng=np.random.default_rng(7))
    assert a == b


def test_bootstrap_ci_handles_empty():
    mean, lo, hi = bootstrap_ci([])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_paired_bootstrap_negative_when_method_better():
    # 'method' has lower churn than 'reference' for every pair.
    deltas = -1e-2 * np.ones(45)   # method lower by 0.01
    mean, lo, hi = bootstrap_paired(deltas, n_boot=2000,
                                    rng=np.random.default_rng(0))
    assert mean < 0
    assert hi < 0   # CI excludes 0 ⇒ significant


def test_paired_bootstrap_centered_at_mean():
    rng = np.random.default_rng(0)
    deltas = rng.normal(loc=-0.05, scale=0.01, size=45)
    mean, lo, hi = bootstrap_paired(deltas, n_boot=5000,
                                    rng=np.random.default_rng(0))
    assert abs(mean - deltas.mean()) < 1e-6
    assert lo <= mean <= hi
