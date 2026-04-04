from __future__ import annotations

import torch

from data import ColoredMNIST


def make_train_dataset() -> ColoredMNIST:
    return ColoredMNIST(env_correlation=0.9, label_noise=0.25, split="train", seed=42)


def make_test_dataset() -> ColoredMNIST:
    return ColoredMNIST(env_correlation=0.1, label_noise=0.25, split="test", seed=42)


# ---------------------------------------------------------------------------
# Shape and dtype checks
# ---------------------------------------------------------------------------

def test_image_shape():
    ds = make_train_dataset()
    assert ds.images.shape == (60_000, 3, 28, 28), f"unexpected shape: {ds.images.shape}"


def test_label_shape_and_dtype():
    ds = make_train_dataset()
    assert ds.labels.shape == (60_000,)
    assert ds.labels.dtype == torch.int64, "labels must be int64 for F.cross_entropy"


def test_labels_binary():
    ds = make_train_dataset()
    assert ds.labels.min() >= 0 and ds.labels.max() <= 1


def test_colors_binary():
    ds = make_train_dataset()
    assert ds.colors.min() >= 0 and ds.colors.max() <= 1


def test_blue_channel_zero():
    # Channel 2 carries no information by construction — verify this is not broken.
    ds = make_train_dataset()
    assert ds.images[:, 2, :, :].max().item() == 0.0


def test_pixel_range():
    ds = make_train_dataset()
    assert ds.images.min() >= 0.0 and ds.images.max() <= 1.0


# ---------------------------------------------------------------------------
# Semantic checks
# ---------------------------------------------------------------------------

def test_color_label_correlation_train():
    """Train set correlation should be close to 0.9."""
    ds = make_train_dataset()
    empirical = (ds.labels == ds.colors).float().mean().item()
    assert abs(empirical - 0.9) < 0.05, f"train correlation {empirical:.3f} too far from 0.9"


def test_color_label_correlation_test():
    """Test set correlation should be close to 0.1."""
    ds = make_test_dataset()
    empirical = (ds.labels == ds.colors).float().mean().item()
    assert abs(empirical - 0.1) < 0.05, f"test correlation {empirical:.3f} too far from 0.1"


def test_label_balance():
    """After label noise, the class balance should remain near 50/50."""
    ds = make_train_dataset()
    frac_ones = ds.labels.float().mean().item()
    assert abs(frac_ones - 0.5) < 0.05, f"label balance {frac_ones:.3f} unexpectedly skewed"


def test_red_channel_only_when_color_zero():
    """Examples with color=0 should have non-zero red channel and zero green channel."""
    ds = make_train_dataset()
    mask = ds.colors == 0
    # Red channel must carry signal for color=0 examples
    assert ds.images[mask, 0].max().item() > 0
    # Green channel must be zero for color=0 examples
    assert ds.images[mask, 1].max().item() == 0.0


def test_green_channel_only_when_color_one():
    """Examples with color=1 should have non-zero green channel and zero red channel."""
    ds = make_train_dataset()
    mask = ds.colors == 1
    assert ds.images[mask, 1].max().item() > 0
    assert ds.images[mask, 0].max().item() == 0.0


# ---------------------------------------------------------------------------
# __getitem__ contract
# ---------------------------------------------------------------------------

def test_getitem_keys():
    ds = make_train_dataset()
    sample = ds[0]
    assert set(sample.keys()) == {"image", "label", "color", "spurious", "index"}


def test_getitem_index_matches():
    ds = make_train_dataset()
    for i in [0, 1, 100, 999]:
        assert ds[i]["index"] == i


def test_getitem_image_shape():
    ds = make_train_dataset()
    assert ds[0]["image"].shape == (3, 28, 28)


def test_len():
    assert len(make_train_dataset()) == 60_000
    assert len(make_test_dataset()) == 10_000


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_deterministic():
    """Same seed should produce identical datasets."""
    ds1 = ColoredMNIST(env_correlation=0.9, seed=0)
    ds2 = ColoredMNIST(env_correlation=0.9, seed=0)
    assert torch.equal(ds1.labels, ds2.labels)
    assert torch.equal(ds1.colors, ds2.colors)


def test_different_seeds_differ():
    """Different seeds should (almost certainly) produce different color assignments."""
    ds1 = ColoredMNIST(env_correlation=0.9, seed=0)
    ds2 = ColoredMNIST(env_correlation=0.9, seed=1)
    assert not torch.equal(ds1.colors, ds2.colors)
