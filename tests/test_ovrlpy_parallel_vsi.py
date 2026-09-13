"""compute_VSI_parallel must be deterministic and agree with ovrlpy at n_workers=1.

The obvious assertion -- "equals ovrlpy's output" -- is not available: ovrlpy combines
`n_workers` partial sums with `reduce` over `as_completed`, so at n_workers > 1 it does not
reproduce its own result (measured: 114,133 of 394,923 pixels move by up to 3.6e-07 between
two runs of unmodified code). At n_workers=1 there is a single partial sum and no ordering,
so that is the reference here.
"""
from __future__ import annotations

import numpy as np
import pytest

ovrlpy = pytest.importorskip("ovrlpy")
pytest.importorskip("pandas")

from ovrlpy import _ovrlp, _utils  # noqa: E402

from spoqc._ovrlpy_fast import (  # noqa: E402
    _calculate_embedding_sparse,
    compute_VSI_parallel,
)

ORIGINAL_VSI = _ovrlp.Ovrlp.compute_VSI
ORIGINAL_EMBED = _utils._calculate_embedding


def _transcripts(n_genes=30, per_gene=1600, extent=400, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_core = per_gene // 4
    xs, ys, zs, gs = [], [], [], []
    for gene in range(n_genes):
        cx, cy = rng.uniform(0, extent, 2)
        xs.append(np.clip(rng.normal(cx, 40, n_core), 0, extent))
        ys.append(np.clip(rng.normal(cy, 40, n_core), 0, extent))
        xs.append(rng.uniform(0, extent, per_gene - n_core))
        ys.append(rng.uniform(0, extent, per_gene - n_core))
        zs.append(rng.uniform(2.0, 18.0, per_gene))
        gs.append(np.full(per_gene, f"G{gene:03d}"))
    return pd.DataFrame({
        "x": np.concatenate(xs), "y": np.concatenate(ys),
        "z": np.concatenate(zs), "gene": np.concatenate(gs),
    })


@pytest.fixture(scope="module")
def fitted():
    """Fit the upstream state ONCE; every variant runs against this same object.

    Fitting per variant would compare ovrlpy's pseudocell/PCA fit too, which is not
    reproducible -- that mistake made two runs of identical code differ by 0.77.
    """
    obj = ovrlpy.Ovrlp(_transcripts(), min_distance=6.0, n_components=6, n_workers=1)
    obj.patch_length = 150
    obj.process_coordinates(gridsize=1, n_iter=20)
    obj.fit_transcripts(10, genes=None, fit_umap=False)
    yield obj


def _vsi(obj, vsi, embed):
    _utils._calculate_embedding = embed
    _ovrlp._calculate_embedding = embed
    try:
        vsi(obj, min_transcripts=2)
        return obj.integrity_map.copy()
    finally:
        _utils._calculate_embedding = ORIGINAL_EMBED
        _ovrlp._calculate_embedding = ORIGINAL_EMBED


def test_parallel_matches_ovrlpy_serial_at_one_worker(fitted):
    """The only well-defined reference: ovrlpy with a single partial sum."""
    fitted.n_workers = 1
    expected = _vsi(fitted, ORIGINAL_VSI, _calculate_embedding_sparse)
    got = _vsi(fitted, compute_VSI_parallel, _calculate_embedding_sparse)

    assert expected.shape == got.shape
    assert np.count_nonzero(expected) > 0, "fixture produced an all-zero map"
    assert np.array_equal(expected, got), (
        f"max abs diff {np.abs(expected - got).max():.3e} over "
        f"{np.count_nonzero(expected != got):,} pixels"
    )


def test_parallel_is_deterministic_across_runs(fitted):
    """Unlike ovrlpy at n_workers > 1, repeated runs must be byte-for-byte equal."""
    fitted.n_workers = 3
    first = _vsi(fitted, compute_VSI_parallel, _calculate_embedding_sparse)
    second = _vsi(fitted, compute_VSI_parallel, _calculate_embedding_sparse)
    assert np.array_equal(first, second)


def test_parallel_result_is_independent_of_worker_count(fitted):
    """Patches are independent, so the answer must not depend on how many workers ran."""
    fitted.n_workers = 1
    one = _vsi(fitted, compute_VSI_parallel, _calculate_embedding_sparse)
    fitted.n_workers = 4
    four = _vsi(fitted, compute_VSI_parallel, _calculate_embedding_sparse)
    assert np.array_equal(one, four)
