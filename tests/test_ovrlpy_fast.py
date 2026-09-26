"""The ovrlpy embedding shim must match ovrlpy's own implementation.

Differential test: build synthetic patches, run ovrlpy's `_calculate_embedding` and the
shim on identical input, and compare. The shim only changes how the per-gene rank-1
updates are accumulated (in-place BLAS `ger` instead of a temporary per gene), so results
must agree to floating-point noise and the `0` sentinel behaviour must be preserved.
"""
from __future__ import annotations

from queue import SimpleQueue

import numpy as np
import polars as pl
import pytest

ovrlpy = pytest.importorskip("ovrlpy")
from ovrlpy._utils import _calculate_embedding as ovrlpy_original  # noqa: E402

from spoqc._ovrlpy_fast import (  # noqa: E402
    SUPPORTED_OVRLPY_VERSIONS,
    _calculate_embedding_fast,
    install,
)


def _patch(n_genes, side, n_components, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((side, side)) > 0.3
    components = rng.standard_normal((n_components, n_genes))  # float64, like PCA
    items = []
    for gene in range(n_genes):
        n = int(rng.integers(2, 120))
        items.append((gene, pl.DataFrame({
            "x_pixel": rng.integers(0, side, n),
            "y_pixel": rng.integers(0, side, n),
            "z": rng.random(n),
            "z_center": np.full(n, 0.5),
        })))
    return mask, components, items


def _run(fn, mask, components, items):
    queue = SimpleQueue()
    for item in items:
        queue.put(item)
    return fn(queue, mask, components, bandwidth=1.0, dtype=np.float32)


@pytest.mark.parametrize("n_genes,side,n_components", [(40, 60, 21), (120, 90, 30)])
def test_matches_ovrlpy(n_genes, side, n_components):
    mask, components, items = _patch(n_genes, side, n_components)
    expected_top, expected_bottom = _run(ovrlpy_original, mask, components, items)
    got_top, got_bottom = _run(_calculate_embedding_fast, mask, components, items)

    for expected, got in ((expected_top, got_top), (expected_bottom, got_bottom)):
        assert expected.shape == got.shape
        assert np.allclose(expected, got, rtol=1e-10, atol=1e-10), (
            f"max abs diff {np.abs(expected - got).max():.3e}"
        )


def test_empty_queue_returns_the_zero_sentinel():
    """ovrlpy's caller does `isinstance(embedding, int) and embedding == 0`."""
    mask, components, _ = _patch(1, 10, 4)
    assert _run(_calculate_embedding_fast, mask, components, []) == (0, 0)
    assert _run(ovrlpy_original, mask, components, []) == (0, 0)


def test_genes_with_fewer_than_two_transcripts_are_skipped():
    """ovrlpy's _gene_embedding returns (None, None) for len(df) < 2."""
    mask, components, _ = _patch(2, 20, 4)
    single = pl.DataFrame({"x_pixel": [1], "y_pixel": [1], "z": [0.9], "z_center": [0.5]})
    assert _run(_calculate_embedding_fast, mask, components, [(0, single)]) == (0, 0)


def test_install_is_version_guarded():
    applied = install()
    assert applied is (ovrlpy.__version__ in SUPPORTED_OVRLPY_VERSIONS)
    if applied:
        from ovrlpy import _ovrlp, _utils
        assert _utils._calculate_embedding is _calculate_embedding_fast
        assert _ovrlp._calculate_embedding is _calculate_embedding_fast
