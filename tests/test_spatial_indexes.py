"""PR4: helperfuncs.points_within_radius, an O(n^2) neighbour search, replaced
by a spatial index.

The test is *differential*: the previous implementation is reproduced verbatim
below and the new code must agree with it exactly, including index labels,
per-point ordering, and the self-exclusion rule.

find_overlapping_nuclei was also in this PR; #8 owns it now.
"""
import numpy as np
import pandas as pd
import pytest


# --------------------------- the previous implementations ---------------------
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

    def test_duplicate_index_labels_counts_agree(self):
        """num=True is identical even with duplicate labels: both drop exactly one."""
        from spoqc.helperfuncs import points_within_radius

        df = self._duplicate_label_frame()
        assert points_within_radius(df, 1.5, True) == _orig_points_within_radius(
            df, 1.5, True
        )

    def test_duplicate_index_labels_differ_only_in_order(self):
        """With duplicate labels the two self-exclusion rules order differently.

        Self-exclusion changed from by-label to by-position. The old code did
        `close.remove(i)`, and list.remove drops the first element equal to `i` --
        when several rows share label `i` that is some other row, not the query
        point. For the row at x=2.5 (label 'a', position 3) the neighbour labels
        in position order are ['a', 'b', 'a']; the old code removed the 'a' at
        position 1 and kept the query point's own label.

        Because rows sharing a label are indistinguishable in a list of labels,
        the returned multiset is unchanged either way -- so this is an ORDERING
        difference, not a membership one, and the counts are identical. The new
        result is in position order, which is what the old code produced for a
        unique index; the old code's output was not.

        Latent on dev: cell and nucleus tables are keyed by unique ids in
        practice, so this path is only reachable with a non-unique index.
        """
        from spoqc.helperfuncs import points_within_radius

        df = self._duplicate_label_frame()
        old = _orig_points_within_radius(df, 1.5, False)
        new = points_within_radius(df, 1.5, False)

        assert old[:3] == new[:3], "rows 0-2 agree exactly"
        assert sorted(old[3]) == sorted(new[3]), "same labels either way"
        assert old[3] == ["b", "a"] and new[3] == ["a", "b"], "differing only in order"

    @staticmethod
    def _duplicate_label_frame():
        # positions: 0='a'@0.0  1='a'@1.0  2='b'@2.0  3='a'@2.5
        return pd.DataFrame(
            {"x": [0.0, 1.0, 2.0, 2.5], "y": [0.0, 0.0, 0.0, 0.0]},
            index=["a", "a", "b", "a"],
        )

    def test_no_longer_iterates_rows(self):
        import inspect

        from spoqc import helperfuncs

        # strip the docstring: it discusses the old iterrows form on purpose
        src = inspect.getsource(helperfuncs.points_within_radius)
        body = src.split('"""')[-1]
        assert "iterrows" not in body, "the per-row loop is still there"
        assert "cKDTree" in body
