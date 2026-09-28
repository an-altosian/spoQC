"""The nonzero-only accumulation must be BIT-identical to ovrlpy, not merely close.

Skipping a row whose blurred signal is exactly zero omits `0.0 * factor_c`, which is +-0.0
for finite loadings and leaves the accumulator unchanged. Retained rows go through the same
two rounding steps as ovrlpy's -- one multiply, one add, same values, no reassociation and
no FMA fusion -- so `np.array_equal` is the right assertion and `np.allclose` would be too
weak to catch a real regression.
"""
from __future__ import annotations

from queue import SimpleQueue

import numpy as np
import polars as pl
import pytest

ovrlpy = pytest.importorskip("ovrlpy")
from ovrlpy._utils import _calculate_embedding as ovrlpy_original  # noqa: E402

from spoqc._ovrlpy_fast import _calculate_embedding_sparse  # noqa: E402


def _items(n_genes, side, rng, *, n_min=2, n_max=120, z_center=0.5):
    items = []
    for gene in range(n_genes):
        n = int(rng.integers(n_min, n_max))
        items.append((gene, pl.DataFrame({
            "x_pixel": rng.integers(0, side, n),
            "y_pixel": rng.integers(0, side, n),
            "z": rng.random(n),
            "z_center": np.full(n, z_center),
        })))
    return items


def _run(fn, mask, components, items, **kwargs):
    queue: SimpleQueue = SimpleQueue()
    for item in items:
        queue.put(item)
    return fn(queue, mask, components, bandwidth=1.0, dtype=np.float32, **kwargs)


@pytest.mark.parametrize(
    "n_genes,side,n_components,sparsity",
    [(40, 60, 21, 0.3), (120, 90, 30, 0.3), (60, 80, 30, 0.9)],
)
def test_bit_identical_to_ovrlpy(n_genes, side, n_components, sparsity):
    """The headline claim, across dense and very sparse masks."""
    rng = np.random.default_rng(0)
    mask = rng.random((side, side)) > sparsity
    components = rng.standard_normal((n_components, n_genes))
    items = _items(n_genes, side, rng)

    expected = _run(ovrlpy_original, mask, components, items)
    got = _run(_calculate_embedding_sparse, mask, components, items)

    for want, have in zip(expected, got):
        assert want.shape == have.shape
        assert want.dtype == have.dtype
        assert np.array_equal(want, have), (
            f"max abs diff {np.abs(want - have).max():.3e} over "
            f"{np.count_nonzero(want != have):,} of {want.size:,} entries"
        )


def test_empty_queue_returns_integer_sentinel():
    """ovrlpy's caller checks `isinstance(x, int) and x == 0`, so 0 must stay an int."""
    rng = np.random.default_rng(1)
    mask = rng.random((20, 20)) > 0.3
    top, bottom = _run(_calculate_embedding_sparse, mask, rng.standard_normal((5, 3)), [])
    assert isinstance(top, int) and top == 0
    assert isinstance(bottom, int) and bottom == 0


def test_single_transcript_genes_are_skipped_like_ovrlpy():
    """`len(df) < 2` genes are dropped by ovrlpy; a lone transcript is not negligible."""
    rng = np.random.default_rng(2)
    side = 40
    mask = rng.random((side, side)) > 0.3
    components = rng.standard_normal((8, 6))
    items = _items(6, side, rng, n_min=1, n_max=2)
    assert all(len(df) == 1 for _, df in items), "fixture must be single-transcript"

    assert _run(ovrlpy_original, mask, components, items) == (0, 0)
    assert _run(_calculate_embedding_sparse, mask, components, items) == (0, 0)


def test_z_equal_to_centre_is_dropped_from_both_sides():
    """ovrlpy filters `z > z_center` and `z < z_center`, so equality falls out of both."""
    side = 30
    mask = np.ones((side, side), dtype=bool)
    components = np.ones((4, 2))
    items = [(g, pl.DataFrame({
        "x_pixel": np.array([3, 9, 14]),
        "y_pixel": np.array([4, 8, 12]),
        "z": np.full(3, 0.5),
        "z_center": np.full(3, 0.5),
    })) for g in range(2)]

    assert _run(ovrlpy_original, mask, components, items) == (0, 0)
    assert _run(_calculate_embedding_sparse, mask, components, items) == (0, 0)


def test_all_zero_signal_rows_do_not_change_the_result():
    """A gene confined to one corner leaves most rows zero -- exactly the skipped case."""
    rng = np.random.default_rng(3)
    side = 70
    mask = np.ones((side, side), dtype=bool)
    components = rng.standard_normal((30, 4))
    items = [(g, pl.DataFrame({
        "x_pixel": rng.integers(0, 6, 20),
        "y_pixel": rng.integers(0, 6, 20),
        "z": rng.random(20),
        "z_center": np.full(20, 0.5),
    })) for g in range(4)]

    expected = _run(ovrlpy_original, mask, components, items)
    got = _run(_calculate_embedding_sparse, mask, components, items)
    for want, have in zip(expected, got):
        zero_rows = np.count_nonzero(~np.any(want != 0, axis=1))
        assert zero_rows > side * side // 2, "fixture must leave most rows zero"
        assert np.array_equal(want, have)


def test_non_finite_loading_is_rejected_not_silently_handled():
    """`0.0 * inf` is NaN, which ovrlpy propagates and skipping would not."""
    rng = np.random.default_rng(4)
    side = 25
    mask = np.ones((side, side), dtype=bool)
    components = rng.standard_normal((6, 3))
    components[2, 1] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        _run(_calculate_embedding_sparse, mask, components, _items(3, side, rng))
