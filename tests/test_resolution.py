"""PR1: the morphology-image pyramid level must be selectable, and the raster
transform must follow it.

Before this change `RESOLUTION` returned the literal 'scale0', so every
pixel-scaled stage ran at native resolution regardless of how small the input
was -- subsetting cells or transcripts does not crop the image.
"""
import numpy as np
import pytest
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box


class TestResolutionFlag:
    def test_default_is_scale2(self):
        from spoqc.cli import build_parser

        args = vars(build_parser().parse_args(["-i", "i", "-o", "o", "-t", "t"]))
        assert args["resolution"] == "scale2"

    def test_scale0_still_selectable(self):
        from spoqc.cli import build_parser

        args = vars(
            build_parser().parse_args(
                ["-i", "i", "-o", "o", "-t", "t", "--resolution", "scale0"]
            )
        )
        assert args["resolution"] == "scale0"

    def test_resolution_constant_reads_args(self):
        import inspect

        from spoqc import cli

        src = inspect.getsource(cli)
        body = src[src.index("def RESOLUTION():"):].split("@constant")[0]
        assert "args[" in body, "RESOLUTION must not return a hardcoded literal"
        assert "'scale0'" not in body


class TestRasterTransformFollowsResolution:
    """The transform must derive pixel size from the world extent.

    Polygon coordinates live in 'global', which spatialdata pins to the scale0
    pixel grid. A hardcoded 1 unit/pixel is only correct at scale0.
    """

    # world extent mimics the 10x Xenium breast sample: 35416 x 25778 global units
    WORLD_W, WORLD_H = 35416.0, 25778.0

    @staticmethod
    def _grid(level_div):
        return int(25778 / level_div), int(35416 / level_div)

    def _cells(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        xs = rng.uniform(0, self.WORLD_W - 40, n)
        ys = rng.uniform(0, self.WORLD_H - 40, n)
        return [box(x, y, x + 35, y + 35) for x, y in zip(xs, ys)]

    def _n_on_raster(self, geoms, transform, shape):
        im = rasterize(
            zip(geoms, np.arange(1, len(geoms) + 1)),
            out_shape=shape, transform=transform, fill=0,
            all_touched=False, dtype="int32",
        )
        present = np.unique(im)
        return int((present > 0).sum())

    def _new_transform(self, h, w):
        return from_origin(0, self.WORLD_H, self.WORLD_W / w, self.WORLD_H / h)

    def test_identical_to_hardcoded_at_scale0(self):
        """At scale0 world extent == pixel count, so the fix must be a no-op."""
        h, w = self._grid(1)
        old = from_origin(0, h, 1, 1)
        new = self._new_transform(h, w)
        assert tuple(new) == pytest.approx(tuple(old)), (
            "the extent-derived transform must reduce exactly to the previous "
            "hardcoded form at scale0, so scale0 output cannot change"
        )

    @pytest.mark.parametrize("div,level", [(4, "scale2"), (8, "scale3")])
    def test_hardcoded_transform_loses_cells_below_scale0(self, div, level):
        """Documents the bug this PR must avoid introducing."""
        geoms = self._cells()
        h, w = self._grid(div)
        kept_old = self._n_on_raster(geoms, from_origin(0, h, 1, 1), (h, w))
        assert kept_old < 0.5 * len(geoms), (
            f"at {level} a 1 unit/px transform should drop most cells; if this "
            f"stops being true the premise of the fix has changed"
        )

    @pytest.mark.parametrize("div", [1, 4, 8, 16])
    def test_extent_derived_transform_keeps_every_cell(self, div):
        geoms = self._cells()
        h, w = self._grid(div)
        kept = self._n_on_raster(geoms, self._new_transform(h, w), (h, w))
        assert kept == len(geoms), (
            f"every cell must land on the raster at pyramid divisor {div}"
        )

    def test_both_transform_sites_are_fixed(self):
        import inspect

        from spoqc.subworkflows import hqcr

        src = inspect.getsource(hqcr)
        # match the assignment, not the explanatory comment that quotes the old form
        assert "transform = from_origin(0, height, 1, 1)" not in src, (
            "a hardcoded 1 unit/pixel transform remains; it is only valid at scale0"
        )
        assert src.count("_world_h / height") == 2, (
            "both create_cell_probability_image and map_values_to_cells must "
            "derive pixel size from the world extent"
        )


class TestTranscriptBinningFollowsResolution:
    """The same bug as the raster transform, in the three *_image.py files.

    They built a grid over the *global* (scale0) extent and reshaped the result
    to the *selected* level's dims. Identical at scale0; at scale2 the array is
    16x too large and the reshape raises -- which is what an end-to-end run at
    scale2 hit in hqtr's structure analysis.
    """

    W, H = 1200, 900

    @staticmethod
    def _dim(w, h):
        from spoqc.helperfuncs import ImageDimStruct

        return ImageDimStruct(0, 0, w, h)

    @staticmethod
    def _original_counts(df, imagedim, dim_x, dim_y):
        """The previous implementation, verbatim."""
        import pandas as pd

        x_idx = [i for i in range(int(imagedim.bb_xmin), int(imagedim.bb_xmax))]
        y_idx = [i for i in range(int(imagedim.bb_ymin), int(imagedim.bb_ymax))]
        counts = df.value_counts(subset=["x", "y"]).rename("count")
        grid = [(x, y) for y in y_idx for x in x_idx]
        mi = pd.MultiIndex.from_tuples(grid, names=["x", "y"])
        ix = counts.index.get_indexer(mi)
        vals = counts.to_numpy()
        return np.array(np.where(ix >= 0, vals[ix], 0)).reshape(dim_x, dim_y)

    def _points(self, n=60_000, seed=0):
        import pandas as pd

        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {"x": rng.integers(0, self.W, n), "y": rng.integers(0, self.H, n)}
        )

    def test_identical_to_the_previous_binning_at_scale0(self):
        from spoqc.metrics.transcript_density import _grid

        df = self._points()
        dim = self._dim(self.W, self.H)
        expected = self._original_counts(df, dim, self.H, self.W)
        got = _grid.bin_counts(
            df["x"].to_numpy(), df["y"].to_numpy(), dim, self.H, self.W
        )
        assert np.array_equal(got, expected)

    def test_previous_binning_raises_below_scale0(self):
        """Documents the failure this fix removes."""
        df = self._points()
        dim = self._dim(self.W, self.H)
        with pytest.raises(ValueError, match="cannot reshape"):
            self._original_counts(df, dim, self.H // 4, self.W // 4)

    @pytest.mark.parametrize("div", [1, 2, 4, 8])
    def test_every_point_is_binned_at_any_level(self, div):
        from spoqc.metrics.transcript_density import _grid

        df = self._points()
        dim = self._dim(self.W, self.H)
        got = _grid.bin_counts(
            df["x"].to_numpy(), df["y"].to_numpy(), dim, self.H // div, self.W // div
        )
        assert got.shape == (self.H // div, self.W // div)
        assert got.sum() == len(df), "no point may be dropped or double-counted"

    def test_downsampling_equals_a_block_sum_of_scale0(self):
        """The correctness bar: a scale2 pixel must total its 4x4 scale0 pixels."""
        from spoqc.metrics.transcript_density import _grid

        df = self._points()
        dim = self._dim(self.W, self.H)
        full = _grid.bin_counts(
            df["x"].to_numpy(), df["y"].to_numpy(), dim, self.H, self.W
        )
        quarter = _grid.bin_counts(
            df["x"].to_numpy(), df["y"].to_numpy(), dim, self.H // 4, self.W // 4
        )
        blocks = full.reshape(self.H // 4, 4, self.W // 4, 4).sum(axis=(1, 3))
        assert np.array_equal(quarter, blocks)

    @pytest.mark.parametrize("how", ["mean", "max", "sum"])
    def test_reductions_shape_and_emptiness(self, how):
        from spoqc.metrics.transcript_density import _grid

        df = self._points(n=5000)
        vals = np.random.default_rng(1).random(len(df))
        dim = self._dim(self.W, self.H)
        out = _grid.bin_reduce(
            df["x"].to_numpy(), df["y"].to_numpy(), vals, dim, 90, 120, how=how
        )
        assert out.shape == (90, 120)
        assert np.isfinite(out).all(), "empty pixels must be 0, not NaN"

    def test_max_reduction_matches_a_groupby_max_at_scale0(self):
        import pandas as pd

        from spoqc.metrics.transcript_density import _grid

        df = self._points(n=20_000)
        df["v"] = np.random.default_rng(2).random(len(df))
        dim = self._dim(self.W, self.H)
        gm = df.groupby(["x", "y"])["v"].max()
        grid = [(x, y) for y in range(self.H) for x in range(self.W)]
        expected = (
            gm.reindex(pd.MultiIndex.from_tuples(grid, names=["x", "y"]))
            .fillna(0.0)
            .to_numpy()
            .reshape(self.H, self.W)
        )
        got = _grid.bin_reduce(
            df["x"].to_numpy(), df["y"].to_numpy(), df["v"].to_numpy(),
            dim, self.H, self.W, how="max",
        )
        assert np.allclose(got, expected)

    def test_degenerate_extent_raises(self):
        from spoqc.metrics.transcript_density import _grid

        with pytest.raises(ValueError, match="degenerate"):
            _grid.bin_counts(np.array([0]), np.array([0]), self._dim(0, 0), 4, 4)

    def test_no_tuple_grid_remains_in_the_image_modules(self):
        import inspect

        from spoqc.metrics.transcript_density import ac_image, qv_image
        from spoqc.metrics.transcript_density import transcript_density_image as tdi

        for mod in (tdi, qv_image, ac_image):
            src = inspect.getsource(mod)
            assert "MultiIndex.from_tuples" not in src, (
                f"{mod.__name__} still builds a tuple per pixel over the scale0 extent"
            )
