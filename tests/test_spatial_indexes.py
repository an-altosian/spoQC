"""PR4: two O(n^2) neighbour searches replaced by spatial indexes.

Both tests are *differential*: the previous implementations are reproduced
verbatim below and the new code must agree with them exactly, including index
labels, per-cell ordering, and the self-exclusion rule.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box


# --------------------------- the previous implementations ---------------------
def _orig_find_overlapping_nuclei(cells, nucleus):
    overlaps = []
    centroids = nucleus.geometry.centroid
    for cell in cells.geometry:
        overlaps.append(nucleus[centroids.geometry.intersects(cell)].index.tolist())
    return overlaps


def _orig_points_within_radius(df, radius, num):
    out = []
    for i, point in df.iterrows():
        x1, y1 = point["x"], point["y"]
        d = np.sqrt((df["x"] - x1) ** 2 + (df["y"] - y1) ** 2)
        close = df[d <= radius].index.tolist()
        close.remove(i)
        out.append(len(close) if num else close)
    return out


# --------------------------------- fixtures -----------------------------------
@pytest.fixture
def overlapping_geoms():
    """Cells with 0, 1 and several nuclei, plus an unclaimed nucleus."""
    cells = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 10, 10),      # contains nuclei 0 and 1
            box(20, 20, 30, 30),    # contains nucleus 2
            box(50, 50, 60, 60),    # contains none
        ],
        index=[100, 200, 300],
    )
    nucleus = gpd.GeoDataFrame(
        geometry=[Point(2, 2), Point(8, 8), Point(25, 25), Point(99, 99)],
        index=["n0", "n1", "n2", "n3"],
    )
    return cells, nucleus


class TestFindOverlappingNuclei:
    def test_matches_previous_implementation(self, overlapping_geoms):
        from spoqc.metrics.segmentation.convexity import find_overlapping_nuclei

        cells, nucleus = overlapping_geoms
        assert find_overlapping_nuclei(cells, nucleus) == _orig_find_overlapping_nuclei(
            cells, nucleus
        )

    def test_returns_index_labels_not_positions(self, overlapping_geoms):
        """Callers do nucleus_boundaries['geometry'].loc[idx], so labels matter."""
        from spoqc.metrics.segmentation.convexity import find_overlapping_nuclei

        cells, nucleus = overlapping_geoms
        out = find_overlapping_nuclei(cells, nucleus)
        assert out[0] == ["n0", "n1"]
        assert out[1] == ["n2"]

    def test_one_entry_per_cell_including_empty(self, overlapping_geoms):
        """nuclei_count.py does len(x) on every row, so empties must be lists."""
        from spoqc.metrics.segmentation.convexity import find_overlapping_nuclei

        cells, nucleus = overlapping_geoms
        out = find_overlapping_nuclei(cells, nucleus)
        assert len(out) == len(cells)
        assert out[2] == []
        assert [len(x) for x in out] == [2, 1, 0]

    def test_random_geometries_agree(self):
        from spoqc.metrics.segmentation.convexity import find_overlapping_nuclei

        rng = np.random.default_rng(0)
        n = 300
        cx, cy = rng.uniform(0, 200, n), rng.uniform(0, 200, n)
        cells = gpd.GeoDataFrame(geometry=[box(x, y, x + 12, y + 12) for x, y in zip(cx, cy)])
        nx, ny = rng.uniform(0, 200, n), rng.uniform(0, 200, n)
        nucleus = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(nx, ny)])
        got = find_overlapping_nuclei(cells, nucleus)
        exp = _orig_find_overlapping_nuclei(cells, nucleus)
        assert [list(a) for a in got] == [list(b) for b in exp]


class TestPointsWithinRadius:
    @pytest.fixture
    def pts(self):
        rng = np.random.default_rng(1)
        return pd.DataFrame({"x": rng.uniform(0, 100, 250), "y": rng.uniform(0, 100, 250)})

    @pytest.mark.parametrize("num", [True, False])
    @pytest.mark.parametrize("radius", [5.0, 20.0])
    def test_matches_previous_implementation(self, pts, num, radius):
        from spoqc.helperfuncs import points_within_radius

        got = points_within_radius(pts, radius, num)
        exp = _orig_points_within_radius(pts, radius, num)
        if num:
            assert got == exp
        else:
            assert [list(a) for a in got] == [list(b) for b in exp]

    def test_excludes_the_point_itself(self):
        from spoqc.helperfuncs import points_within_radius

        df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]})
        assert points_within_radius(df, 10.0, False) == [[1], [0]]
        assert points_within_radius(df, 10.0, True) == [1, 1]

    def test_isolated_point_gets_empty_list(self):
        from spoqc.helperfuncs import points_within_radius

        df = pd.DataFrame({"x": [0.0, 1000.0], "y": [0.0, 1000.0]})
        assert points_within_radius(df, 5.0, False) == [[], []]
        assert points_within_radius(df, 5.0, True) == [0, 0]

    def test_honours_non_default_index_labels(self):
        """Callers index into other frames with these values."""
        from spoqc.helperfuncs import points_within_radius

        df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}, index=["a", "b"])
        assert points_within_radius(df, 10.0, False) == [["b"], ["a"]]

    def test_no_longer_iterates_rows(self):
        import inspect

        from spoqc import helperfuncs

        # strip the docstring: it discusses the old iterrows form on purpose
        src = inspect.getsource(helperfuncs.points_within_radius)
        body = src.split('"""')[-1]
        assert "iterrows" not in body, "the per-row loop is still there"
        assert "cKDTree" in body
