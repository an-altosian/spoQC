"""_sample_expression_parallel must be EXACTLY equal to ovrlpy's serial implementation.

Each gene's KDE is sampled independently and the per-gene results are reindexed by
`gene_list` before assembly, so neither completion order nor worker count can reach the
values -- unlike compute_VSI, where ovrlpy reduces partial sums in completion order. That
makes exact equality the right bar, on X, obsm and var_names alike.
"""
from __future__ import annotations

import numpy as np
import pytest

ovrlpy = pytest.importorskip("ovrlpy")
pytest.importorskip("pandas")

from ovrlpy import _kde  # noqa: E402

from spoqc._ovrlpy_fast import _sample_expression_parallel  # noqa: E402


def _transcripts(n_genes=40, per_gene=1800, extent=500, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_core = per_gene // 3
    xs, ys, zs, gs = [], [], [], []
    for gene in range(n_genes):
        cx, cy = rng.uniform(0, extent, 2)
        xs.append(np.clip(rng.normal(cx, 35, n_core), 0, extent))
        ys.append(np.clip(rng.normal(cy, 35, n_core), 0, extent))
        xs.append(rng.uniform(0, extent, per_gene - n_core))
        ys.append(rng.uniform(0, extent, per_gene - n_core))
        zs.append(rng.uniform(2.0, 18.0, per_gene))
        gs.append(np.full(per_gene, f"G{gene:03d}"))
    return pd.DataFrame({
        "x": np.concatenate(xs), "y": np.concatenate(ys),
        "z": np.concatenate(zs), "gene": np.concatenate(gs),
    })


@pytest.fixture(scope="module")
def sampled():
    obj = ovrlpy.Ovrlp(_transcripts(), min_distance=6.0, n_components=6, n_workers=3)
    obj.patch_length = 200
    kwargs = dict(
        min_expression=obj._expression_threshold(10),
        kde_bandwidth=obj.KDE_bandwidth,
        genes=sorted(obj.transcripts["gene"].unique()),
        n_workers=3,
        min_pixel_distance=obj.min_distance,
        patch_length=obj.patch_length,
        dtype=obj.dtype,
    )
    return (_kde._sample_expression(obj.transcripts, **kwargs),
            _sample_expression_parallel(obj.transcripts, **kwargs))


def test_shape_and_var_order_match(sampled):
    expected, got = sampled
    assert expected.shape == got.shape
    assert expected.shape[0] > 0, "fixture produced no pseudocells"
    assert list(expected.var_names) == list(got.var_names)


def test_expression_matrix_is_bit_identical(sampled):
    expected, got = sampled
    want, have = np.asarray(expected.X), np.asarray(got.X)
    assert want.dtype == have.dtype
    assert np.array_equal(want, have), (
        f"max abs diff {np.abs(want - have).max():.3e} over "
        f"{np.count_nonzero(want != have):,} of {want.size:,} entries"
    )


def test_spatial_coordinates_are_bit_identical(sampled):
    """obsm rows come from `coords`, appended per patch -- so patch ORDER must survive."""
    expected, got = sampled
    assert np.array_equal(expected.obsm["spatial"], got.obsm["spatial"])
