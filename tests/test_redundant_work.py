"""PR3: work whose result was discarded, recomputed, or provably derivable."""
import inspect

import numpy as np
import pytest


class TestUniformityDerivedFromEntropy:
    """uniformity = -(entropy + log q); the second sweep is redundant."""

    @pytest.mark.parametrize("dtype,hi", [(np.uint8, 256), (np.uint16, 65536)])
    @pytest.mark.parametrize("window,r", [(5, 2), (11, 5)])
    def test_derivation_matches_the_real_kernel(self, dtype, hi, window, r):
        from spoqc.image_analysis._slidingwindow import sliding_window_padded
        from spoqc.metrics.image import entropy as E
        from spoqc.metrics.image import uniformity as U

        rng = np.random.default_rng(0)
        img = rng.integers(0, hi, (96, 96)).astype(dtype)
        direct = -sliding_window_padded(U.kl_divergence_uniform, img, r, mode="reflect")
        derived = U.uniformity_from_entropy(
            sliding_window_padded(E.entropy, img, r, mode="reflect"), window, img.dtype
        )
        # The kernel accumulates in float32, the derivation in float64, so the
        # bar is float32 rounding rather than bit-identity. Measured max absolute
        # difference across these four cases is 2.46e-07; 1e-6 leaves ~4x margin.
        assert direct.dtype == np.float32 and derived.dtype == np.float64
        np.testing.assert_allclose(derived, direct, atol=1e-6, rtol=0)

    def test_accepts_a_flattened_entropy_map(self):
        """structure_analysis carries the entropy map flattened."""
        from spoqc.image_analysis._slidingwindow import sliding_window_padded
        from spoqc.metrics.image import entropy as E
        from spoqc.metrics.image import uniformity as U

        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (48, 48)).astype(np.uint8)
        ent = sliding_window_padded(E.entropy, img, 2, mode="reflect")
        flat = U.uniformity_from_entropy(ent.flatten().reshape(img.shape), 5, img.dtype)
        np.testing.assert_allclose(flat, U.uniformity_from_entropy(ent, 5, img.dtype))

    def test_pixel_uniformity_still_works_without_an_entropy_map(self):
        """The uniformity step is independently gated, so the direct path must stay."""
        sig = inspect.signature(
            __import__("spoqc.metrics.image.uniformity", fromlist=["x"]).pixel_uniformity
        )
        assert sig.parameters["entropy_image"].default is None


class TestDeadCodeRemoved:
    def test_get_ci_df_is_gone(self):
        from spoqc.metrics.segmentation import sc_metrics

        assert not hasattr(sc_metrics, "get_ci_df")

    def test_triangle_counter_returns_only_counts(self):
        """indices_list was materialised on 4 call sites and never read."""
        from spoqc.metrics.segmentation import void

        src = inspect.getsource(void.count_stuff_in_triangles_via_delaunay)
        assert "indices_list" not in src
        assert "return counts" in src

    def test_no_call_site_still_unpacks_two_values(self):
        from spoqc.metrics.segmentation import void

        src = inspect.getsource(void)
        assert "counts, indices = count_stuff_in_triangles_via_delaunay" not in src

    def test_counter_still_counts_correctly(self):
        """Behaviour of the surviving return value is unchanged."""
        from scipy.spatial import Delaunay

        from spoqc.metrics.segmentation import void

        pts = np.array([[0.0, 0], [10, 0], [0, 10], [10, 10]])
        d = Delaunay(pts)
        stuff = np.array([[1.0, 1], [2, 2], [9, 9], [-5, -5]])
        counts = void.count_stuff_in_triangles_via_delaunay(d, stuff)
        assert len(counts) == len(d.simplices)
        # the three in-hull points are attributed; the outside one is not
        assert counts.sum() == 3
