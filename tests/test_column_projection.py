"""Projecting columns before .compute() must not change values, index, or dtype.

spoQC materialises the 42.6M-row transcript table at ~18 sites. Each full
materialisation costs ~8 GB of RSS and several seconds, and most sites read only one
or two of its eight columns. Selecting those columns *before* .compute() pushes the
projection into the reader.

That is only a safe substitution if `ddf[cols].compute()` is indistinguishable from
`ddf.compute()[cols]` -- same values, same index, same dtypes, including for the
categorical and string columns the transcript table carries. These tests pin that
invariant on a synthetic frame shaped like the real one, so they need no test data.
"""
from __future__ import annotations

import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def transcripts():
    """Shaped like sdata.points['transcripts']: mixed dtypes, non-trivial index."""
    rng = np.random.default_rng(0)
    n = 5000
    frame = pd.DataFrame({
        'x': rng.uniform(0, 1000, n),
        'y': rng.uniform(0, 1000, n),
        'z': rng.uniform(0, 20, n),
        'feature_name': pd.Categorical(rng.choice(['GeneA', 'GeneB', 'NegControl'], n)),
        'cell_id': rng.integers(0, 900, n),
        'overlaps_nucleus': rng.integers(0, 2, n).astype(bool),
        'transcript_id': np.arange(n, dtype=np.int64),
        'qv': rng.uniform(0, 40, n),
    })
    return dd.from_pandas(frame, npartitions=7)


@pytest.mark.parametrize("cols", [['qv'], ['x', 'y'], ['feature_name'], ['x', 'y', 'qv']])
def test_projection_matches_full_materialisation(transcripts, cols):
    projected = transcripts[cols].compute()
    full = transcripts.compute()[cols]

    pd.testing.assert_frame_equal(projected, full)


@pytest.mark.parametrize("col", ['qv', 'feature_name', 'x'])
def test_single_column_series_matches(transcripts, col):
    """The call sites use ddf[[col]].compute()[col], i.e. they want a Series back."""
    projected = transcripts[[col]].compute()[col]
    full = transcripts.compute()[col]

    pd.testing.assert_series_equal(projected, full)
    assert projected.index.equals(full.index)
    assert projected.dtype == full.dtype


def test_projection_preserves_index_for_assignment(transcripts):
    """qv_image assigns the projected Series into another frame, so alignment matters."""
    centroids = transcripts[['x', 'y']].compute()
    qv = transcripts[['qv']].compute()['qv']

    centroids['qv'] = qv
    expected = transcripts.compute()
    assert centroids['qv'].equals(expected['qv'])
    assert not centroids['qv'].isna().any(), "index misalignment would introduce NaNs"


def test_categorical_dtype_survives_projection(transcripts):
    """feature_name is categorical; a projection that silently cast it would break masks."""
    projected = transcripts[['feature_name']].compute()['feature_name']
    assert isinstance(projected.dtype, pd.CategoricalDtype)
    assert list(np.array(projected)) == list(np.array(transcripts.compute()['feature_name']))


def test_scatter_density_helper_needs_only_x_y_and_category():
    """negativeprobeqc projects to x/y/feature_name; prove the plot helper needs no more.

    The helper receives the frame wholesale, so the projection is only safe if it reads
    nothing beyond those columns plus the derived category. Exercising it on exactly a
    projected frame is what makes that a check rather than an assumption.
    """
    import matplotlib
    matplotlib.use("Agg")
    from spoqc import helperfuncs

    rng = np.random.default_rng(0)
    projected = pd.DataFrame({
        'x': rng.uniform(0, 100, 500),
        'y': rng.uniform(0, 100, 500),
        'feature_name': pd.Categorical(rng.choice(['NegControlCodeword_1', 'GeneA'], 500)),
    })
    projected['neg_probes'] = [
        n.startswith('NegControlCodeword') for n in projected['feature_name'].astype(str)
    ]

    import tempfile
    with tempfile.TemporaryDirectory() as out:
        helperfuncs.plot_scatter_density_df(
            projected[projected['neg_probes']], out, 'neg_probes', 'neg_probes',
            None, ['black'], 'Density of negative probes',
        )
        assert os.listdir(out), "helper produced no output from the projected frame"
