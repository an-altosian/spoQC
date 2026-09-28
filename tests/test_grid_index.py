"""The pixel grid was aligned through one Python tuple per pixel.

All three transcript-density images built `[(x, y) for y in y_idx for x in x_idx]`
and passed it to `pd.MultiIndex.from_tuples`. `pd.MultiIndex.from_product` builds
the identical index in C; the one builder is `spoqc.core.groupreduce.pixel_grid_index`.

The acceptance bar is bit-identity, so these tests assert two things: that the
two indexes are `.equals()` each other, and that the values the callers derive
from them are equal element for element -- for the `mean`, `max` and
`value_counts` paths the three call sites actually use.
"""
import numpy as np
import pandas as pd
import pytest

from spoqc.core import groupreduce
from spoqc.helperfuncs import ImageDimStruct


X0, X1, Y0, Y1 = 0, 120, 0, 90
DIM_X, DIM_Y = Y1 - Y0, X1 - X0  # rows follow y (outer loop), columns follow x


def _old_index(x_idx, y_idx):
    """The previous construction, reproduced verbatim."""
    grid = [(x, y) for y in y_idx for x in x_idx]
    return pd.MultiIndex.from_tuples(grid, names=["x", "y"])


def _inline_from_product(x_idx, y_idx):
    """The inline construction the three call sites used before the shared builder, verbatim."""
    return pd.MultiIndex.from_product([y_idx, x_idx], names=["y", "x"]).swaplevel(0, 1)


def _new_index(x_idx, y_idx):
    """The shared builder, for the bounding box those ranges span (float bounds, as sdata extents)."""
    imagedim = ImageDimStruct(
        np.float64(x_idx[0]), np.float64(y_idx[0]), np.float64(x_idx[-1] + 1), np.float64(y_idx[-1] + 1)
    )
    return groupreduce.pixel_grid_index(imagedim)


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    n = 40_000
    return pd.DataFrame(
        {
            "x": rng.integers(X0, X1, n).astype(float),
            "y": rng.integers(Y0, Y1, n).astype(float),
            "qv": rng.random(n) * 40,
            "morans_I": rng.normal(0, 1, n),
        }
    )


class TestIndexIsTheSameIndex:
    def test_indexes_are_equal(self):
        x_idx, y_idx = range(X0, X1), range(Y0, Y1)
        assert _new_index(x_idx, y_idx).equals(_old_index(x_idx, y_idx))

    def test_level_order_and_names_match(self):
        x_idx, y_idx = range(X0, X1), range(Y0, Y1)
        old, new = _old_index(x_idx, y_idx), _new_index(x_idx, y_idx)
        assert list(new.names) == list(old.names) == ["x", "y"]
        assert new[0] == old[0] and new[-1] == old[-1]

    def test_x_varies_fastest(self):
        """Row-major reshape(dim_x, dim_y) requires x to be the inner loop."""
        new = _new_index(range(X0, X1), range(Y0, Y1))
        assert new[0] == (X0, Y0)
        assert new[1] == (X0 + 1, Y0), "x must advance first"
        assert new[DIM_Y] == (X0, Y0 + 1), "y advances after a full row of x"

    @pytest.mark.parametrize("nx,ny", [(1, 1), (1, 7), (7, 1), (3, 5)])
    def test_equal_for_degenerate_and_small_grids(self, nx, ny):
        x_idx, y_idx = range(X0, X0 + nx), range(Y0, Y0 + ny)
        assert _new_index(x_idx, y_idx).equals(_old_index(x_idx, y_idx))

    @pytest.mark.parametrize("as_list", [False, True])
    @pytest.mark.parametrize("x0,y0", [(0, 0), (-3, 17), (4096, 2048)])
    def test_equal_to_the_inline_forms_it_replaces(self, as_list, x0, y0):
        """ac_image/qv_image passed ranges, transcript_density_image passed lists."""
        x_idx, y_idx = range(x0, x0 + 11), range(y0, y0 + 6)
        inline = _inline_from_product(list(x_idx), list(y_idx)) if as_list else _inline_from_product(x_idx, y_idx)
        new = _new_index(x_idx, y_idx)
        assert new.equals(inline)
        assert new.equals(_old_index(x_idx, y_idx))
        assert list(new.names) == list(inline.names) == ["x", "y"]
        for level_new, level_inline in zip(new.levels, inline.levels):
            assert level_new.dtype == level_inline.dtype
        for codes_new, codes_inline in zip(new.codes, inline.codes):
            np.testing.assert_array_equal(codes_new, codes_inline)


class TestDerivedValuesAreBitIdentical:
    """Each of the three call sites reduces differently; all three are covered."""

    def _reduce(self, gm, index):
        return (
            gm.reindex(index).fillna(0.0).to_numpy().astype("float64")
            .reshape(DIM_X, DIM_Y)
        )

    @pytest.mark.parametrize("column,how", [("qv", "mean"), ("morans_I", "max")])
    def test_reindexed_reduction_is_bit_identical(self, points, column, how):
        gm = getattr(points.groupby(["x", "y"])[column], how)()
        x_idx, y_idx = range(X0, X1), range(Y0, Y1)
        np.testing.assert_array_equal(
            self._reduce(gm, _new_index(x_idx, y_idx)),
            self._reduce(gm, _old_index(x_idx, y_idx)),
        )

    def test_value_counts_path_is_bit_identical(self, points):
        """transcript_density_image uses get_indexer, not reindex."""
        counts = points.value_counts(subset=["x", "y"]).rename("count")
        x_idx, y_idx = range(X0, X1), range(Y0, Y1)

        def dens(index):
            idxer = counts.index.get_indexer(index)
            vals = counts.to_numpy()
            return np.array(np.where(idxer >= 0, vals[idxer], 0)).reshape(DIM_X, DIM_Y)

        np.testing.assert_array_equal(dens(_new_index(x_idx, y_idx)),
                                      dens(_old_index(x_idx, y_idx)))

    def test_empty_pixels_are_zero_not_nan(self, points):
        gm = points.groupby(["x", "y"])["qv"].mean()
        out = self._reduce(gm, _new_index(range(X0, X1), range(Y0, Y1)))
        assert np.isfinite(out).all()
        assert (out == 0).any(), "the fixture must leave some pixels empty"


class TestNoCallSiteBuildsTuples:
    @pytest.mark.parametrize(
        "module",
        ["ac_image", "qv_image", "transcript_density_image"],
    )
    def test_source_no_longer_materialises_a_tuple_per_pixel(self, module):
        import importlib
        import inspect

        mod = importlib.import_module(f"spoqc.metrics.transcript_density.{module}")
        src = inspect.getsource(mod)
        assert "MultiIndex.from_tuples(" not in src, f"{module} still calls from_tuples"
        assert "for y in y_idx for x in x_idx" not in src, (
            f"{module} still materialises a tuple per pixel"
        )
        assert "MultiIndex.from_product(" not in src, f"{module} builds its own grid index"
        assert "groupreduce.pixel_grid_index(imagedim)" in src
