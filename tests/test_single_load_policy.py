"""The transcript table must be materialised once, served many times, then released.

Policy: load each piece of data once, run every computation that needs it, then unload.
Nothing large stays resident past the phase that needs it.

The transcript table is 42.6M x 8 on a full sample; one materialisation costs ~6 s and
~8.7 GB of RSS, and it used to be computed independently at ten call sites. These tests
pin the three properties the shared loader has to have: it computes once, callers cannot
corrupt each other, and a release genuinely drops the copy so a later load sees fresh
data (which is what ovrlpy's in-place coordinate rewrite depends on).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spoqc import helperfuncs


class FakePoints:
    """Stands in for a dask points frame, counting materialisations AND projections.

    A real dask DataFrame supports `df[['a', 'b']]` and builds only those columns, which
    is the whole point of projecting before `.compute()`: on 674,599,124 real transcripts
    the full 13-column table costs 65.6 GB and takes RssAnon from 2.20 to 90.44 GB, while
    the four columns ovrlpy reads cost 14.8 GB and reach 38.59 GB. `materialised_columns`
    records what each `.compute()` actually built, so an implementation that slices AFTER
    materialising is caught instead of passing silently -- both produce the same values.
    """

    def __init__(self, frame):
        self._frame = frame
        self.computes = 0
        self.materialised_columns: list = []

    def __getitem__(self, columns):
        return FakeProjection(self, list(columns))

    def compute(self):
        self.computes += 1
        self.materialised_columns.append(list(self._frame.columns))
        return self._frame.copy()


class FakeProjection:
    """What selecting columns on a FakePoints returns, as dask would."""

    def __init__(self, parent, columns):
        self._parent = parent
        self._columns = columns

    def compute(self):
        self._parent.computes += 1
        self._parent.materialised_columns.append(list(self._columns))
        return self._parent._frame.loc[:, self._columns].copy()


class FakeSdata:
    def __init__(self, frame):
        self.points = {'transcripts': FakePoints(frame)}

    def __getitem__(self, key):
        return self.points[key]


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        'x': rng.uniform(0, 100, n),
        'y': rng.uniform(0, 100, n),
        'feature_name': pd.Categorical(rng.choice(['GeneA', 'NegControlCodeword_1'], n)),
        'qv': rng.uniform(0, 40, n),
    }, index=pd.RangeIndex(n))


@pytest.fixture(autouse=True)
def _clean_cache():
    helperfuncs.release_transcripts()
    yield
    helperfuncs.release_transcripts()


def test_materialises_only_once_across_many_callers(frame):
    sdata = FakeSdata(frame)
    for _ in range(5):
        helperfuncs.load_transcripts(sdata, 'transcripts')
    helperfuncs.load_transcripts(sdata, 'transcripts', ['x', 'y'])
    helperfuncs.load_transcripts(sdata, 'transcripts', ['qv'])

    assert sdata.points['transcripts'].computes == 1, "table was materialised more than once"


def test_callers_get_independent_frames(frame):
    """Several consumers add columns to what they receive; that must not leak."""
    sdata = FakeSdata(frame)
    first = helperfuncs.load_transcripts(sdata, 'transcripts')
    first['neg_probes'] = True

    second = helperfuncs.load_transcripts(sdata, 'transcripts')
    assert 'neg_probes' not in second.columns, "one consumer's mutation leaked into another"

    third = helperfuncs.load_transcripts(sdata, 'transcripts', ['x'])
    third['x'] = -1.0
    assert (helperfuncs.load_transcripts(sdata, 'transcripts')['x'] > -1.0).all()


def test_projection_from_cache_matches_full(frame):
    sdata = FakeSdata(frame)
    full = helperfuncs.load_transcripts(sdata, 'transcripts')
    for cols in (['x', 'y'], ['qv'], ['feature_name']):
        pd.testing.assert_frame_equal(
            helperfuncs.load_transcripts(sdata, 'transcripts', cols), full[cols]
        )


def test_index_is_preserved(frame):
    sdata = FakeSdata(frame)
    got = helperfuncs.load_transcripts(sdata, 'transcripts', ['qv'])
    assert got.index.equals(frame.index)


def test_release_forces_a_fresh_read(frame):
    """ovrlpy rewrites coordinates in place; after release the next load must see them."""
    sdata = FakeSdata(frame)
    helperfuncs.load_transcripts(sdata, 'transcripts')
    assert sdata.points['transcripts'].computes == 1

    # simulate ovrlpy correcting the coordinates underneath us
    sdata.points['transcripts']._frame['x'] = -99.0

    stale = helperfuncs.load_transcripts(sdata, 'transcripts', ['x'])
    assert (stale['x'] != -99.0).all(), "cache should still be serving the old copy"

    helperfuncs.release_transcripts('transcripts')
    fresh = helperfuncs.load_transcripts(sdata, 'transcripts', ['x'])
    assert (fresh['x'] == -99.0).all(), "after release the corrected values must be read"
    assert sdata.points['transcripts'].computes == 2


def test_release_without_key_clears_everything(frame):
    sdata = FakeSdata(frame)
    helperfuncs.load_transcripts(sdata, 'transcripts')
    helperfuncs.release_transcripts()
    helperfuncs.load_transcripts(sdata, 'transcripts')
    assert sdata.points['transcripts'].computes == 2


def test_projection_happens_before_materialising(frame):
    """The requested columns are the only ones BUILT -- not sliced off afterwards.

    This is the property that removed 126.55 GB of peak on the real dataset
    (165.14 -> 38.59 GB). Slicing a fully materialised frame yields identical values, so
    only a check on what was built can tell the two implementations apart.
    """
    sdata = FakeSdata(frame)
    got = helperfuncs.load_transcripts(sdata, 'transcripts', ['x', 'y'])

    points = sdata.points['transcripts']
    assert points.materialised_columns == [['x', 'y']], (
        f"materialised {points.materialised_columns}, so the full table was built and "
        "then sliced -- the memory is spent either way"
    )
    assert list(got.columns) == ['x', 'y']


def test_projected_values_match_slicing_the_full_table(frame):
    """Projecting early must change the peak, never the data."""
    full = helperfuncs.load_transcripts(FakeSdata(frame), 'transcripts')
    helperfuncs.release_transcripts()
    projected = helperfuncs.load_transcripts(FakeSdata(frame), 'transcripts', ['x', 'y'])
    pd.testing.assert_frame_equal(full.loc[:, ['x', 'y']], projected)


def test_a_wider_request_is_not_served_by_a_narrower_cache(frame):
    """Subset-serving must only go one way: narrow from wide, never wide from narrow."""
    sdata = FakeSdata(frame)
    narrow = helperfuncs.load_transcripts(sdata, 'transcripts', ['x'])
    wider = helperfuncs.load_transcripts(sdata, 'transcripts', ['x', 'y', 'qv'])

    assert list(narrow.columns) == ['x']
    assert list(wider.columns) == ['x', 'y', 'qv']
