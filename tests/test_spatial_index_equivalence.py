"""The two spatial-index rewrites must produce output identical to the loops they replaced.

Both tests reproduce the ORIGINAL implementation verbatim, inline, and compare. That is
the only thing that makes these performance changes rather than behaviour changes: the
point is a cheaper route to the same answer, not a cheaper answer.

Original doublet loop (doublet_score.py):
    for i, doublet in corrected_doublet_df.iterrows():
        x1, y1 = doublet['x'], doublet['y']
        distances = np.sqrt((df['x'] - x1)**2 + (df['y'] - y1)**2)
        transcript_doublet[distances <= distance_thresh] = True

Original nucleus loop (convexity.py):
    for cell in cells.geometry:
        overlaps.append(nucleus[nucleus_centroids.geometry.intersects(cell)].index.tolist())
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from spoqc.metrics.segmentation.convexity import find_overlapping_nuclei
from spoqc.metrics.segmentation.doublet_score import flag_transcripts_near_doublets


# --------------------------------------------------------------------------- doublets

def _original_doublet_mask(transcripts, doublets, thresh):
    mask = np.array([False] * len(transcripts))
    for _, doublet in doublets.iterrows():
        x1, y1 = doublet['x'], doublet['y']
        distances = np.sqrt((transcripts['x'] - x1) ** 2 + (transcripts['y'] - y1) ** 2)
        mask[distances <= thresh] = True
    return mask


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("thresh", [0.5, 2.0, 25.0])
def test_doublet_mask_matches_original_loop(seed, thresh):
    rng = np.random.default_rng(seed)
    transcripts = pd.DataFrame(rng.uniform(0, 50, size=(4000, 2)), columns=['x', 'y'])
    doublets = pd.DataFrame(rng.uniform(0, 50, size=(37, 2)), columns=['x', 'y'])

    new = flag_transcripts_near_doublets(transcripts, doublets, thresh)
    old = _original_doublet_mask(transcripts, doublets, thresh)

    assert new.dtype == old.dtype == np.bool_
    assert np.array_equal(new, old), f"{int((new != old).sum())} transcripts differ"


def test_doublet_mask_exact_boundary_is_inclusive():
    """A transcript exactly distance_thresh away was included by `<=`; it must stay so."""
    transcripts = pd.DataFrame({'x': [0.0, 3.0, 5.0, 5.001], 'y': [0.0, 4.0, 0.0, 0.0]})
    doublets = pd.DataFrame({'x': [0.0], 'y': [0.0]})
    new = flag_transcripts_near_doublets(transcripts, doublets, 5.0)
    assert list(new) == list(_original_doublet_mask(transcripts, doublets, 5.0))
    assert list(new) == [True, True, True, False]


def test_doublet_mask_with_no_doublets():
    transcripts = pd.DataFrame({'x': [1.0, 2.0], 'y': [1.0, 2.0]})
    empty = pd.DataFrame(columns=['x', 'y'], dtype=float)
    new = flag_transcripts_near_doublets(transcripts, empty, 1.0)
    assert new.dtype == np.bool_
    assert not new.any()
    assert np.array_equal(new, _original_doublet_mask(transcripts, empty, 1.0))


def test_doublet_weight_column_is_the_mask_as_int():
    """calc_doublet_score derives wdoublet from the mask; both were set on one condition."""
    transcripts = pd.DataFrame({'x': [0.0, 9.0], 'y': [0.0, 9.0]})
    doublets = pd.DataFrame({'x': [0.0], 'y': [0.0]})
    mask = flag_transcripts_near_doublets(transcripts, doublets, 1.0)
    assert list(mask.astype(np.int64)) == [1, 0]


# --------------------------------------------------------------------------- nuclei

def _original_overlaps(cells, nucleus):
    overlaps = []
    nucleus_centroids = nucleus.geometry.centroid
    for cell in cells.geometry:
        overlaps.append(nucleus[nucleus_centroids.geometry.intersects(cell)].index.tolist())
    return overlaps


def _square(x, y, side=1.0):
    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_overlapping_nuclei_matches_original_loop(seed):
    rng = np.random.default_rng(seed)
    cells = gpd.GeoDataFrame(
        geometry=[_square(*rng.uniform(0, 20, size=2), side=3.0) for _ in range(60)]
    )
    nucleus = gpd.GeoDataFrame(
        geometry=[_square(*rng.uniform(0, 20, size=2), side=0.4) for _ in range(300)]
    )
    assert find_overlapping_nuclei(cells, nucleus) == _original_overlaps(cells, nucleus)


def test_overlapping_nuclei_preserves_nucleus_row_order():
    """Each cell's list was in nucleus row order, not sorted by label."""
    cells = gpd.GeoDataFrame(geometry=[_square(0, 0, side=10.0)])
    nucleus = gpd.GeoDataFrame(
        geometry=[Point(3, 3).buffer(0.1), Point(1, 1).buffer(0.1), Point(2, 2).buffer(0.1)],
        index=[99, 7, 50],
    )
    assert find_overlapping_nuclei(cells, nucleus) == _original_overlaps(cells, nucleus)
    assert find_overlapping_nuclei(cells, nucleus) == [[99, 7, 50]]


def test_overlapping_nuclei_with_non_monotonic_cell_index():
    """The rewrite joins positionally, so odd index labels must not reorder the result."""
    cells = gpd.GeoDataFrame(
        geometry=[_square(0, 0, side=2.0), _square(50, 50, side=2.0), _square(1, 1, side=2.0)],
        index=[5, 5, 1],
    )
    nucleus = gpd.GeoDataFrame(
        geometry=[Point(0.5, 0.5).buffer(0.1), Point(1.5, 1.5).buffer(0.1)], index=['a', 'b']
    )
    got = find_overlapping_nuclei(cells, nucleus)
    assert len(got) == 3
    assert got == _original_overlaps(cells, nucleus)


def test_overlapping_nuclei_when_a_cell_has_none():
    cells = gpd.GeoDataFrame(geometry=[_square(0, 0), _square(100, 100)])
    nucleus = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5).buffer(0.05)])
    got = find_overlapping_nuclei(cells, nucleus)
    assert got == _original_overlaps(cells, nucleus)
    assert got[1] == []
