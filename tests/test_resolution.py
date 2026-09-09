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
