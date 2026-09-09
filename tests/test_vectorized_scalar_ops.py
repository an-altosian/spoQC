"""PR2: four places where a scalar or a count was computed by walking the whole
array through the Python interpreter, or by materializing a copy of it.

Each test pins behaviour (identical result) and, where the win is memory rather
than time, pins the absence of the copy.
"""
import numpy as np
import pytest


class TestRelevanceMax:
    """relevance.py:33 -- max(arr.flatten()) boxed every pixel as a Python int."""

    def test_source_no_longer_uses_builtin_max(self):
        import inspect

        from spoqc.metrics.image import relevance

        assert "max(xy_intensities.flatten())" not in inspect.getsource(relevance)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_result_identical(self, dtype):
        rng = np.random.default_rng(0)
        a = (rng.random((128, 128)) * 1000).astype(dtype)
        assert a.max() == max(a.flatten())


class TestPixelCountFromShape:
    """pixel_scoring_dask.py:103 -- len(img.values[0].flatten()) read the whole
    image and copied it again, to compute a number available from .shape."""

    def test_source_no_longer_flattens(self):
        import inspect

        from spoqc.image_analysis import pixel_scoring_dask

        assert ".image.values[0].flatten()" not in inspect.getsource(pixel_scoring_dask)

    @pytest.mark.parametrize("shape", [(7, 13), (1, 1), (2048, 4096)])
    def test_count_identical(self, shape):
        a = np.zeros(shape, dtype=np.uint16)
        assert int(np.prod(a.shape[-2:])) == a.size == len(a.flatten())

    def test_works_on_a_channel_dim(self):
        """The real array is (c, y, x); the count must be y*x, not c*y*x."""
        a = np.zeros((1, 64, 32), dtype=np.uint16)
        assert int(np.prod(a.shape[-2:])) == 64 * 32


class TestSilhouetteSubsampled:
    """helperfuncs.py:965 -- exhaustive silhouette is O(n^2) and runs 145 times
    in the resolution sweep (29 resolutions x 5 inits)."""

    def test_call_passes_sample_size(self):
        import inspect

        from spoqc import helperfuncs

        src = inspect.getsource(helperfuncs)
        i = src.index("silhouette_score(adata_test.obsm['X_umap']")
        assert "sample_size" in src[i:i + 260]

    def test_clamped_form_is_exact_below_the_cap(self):
        from sklearn.metrics import silhouette_score

        rng = np.random.default_rng(0)
        X = rng.random((60, 2))
        labels = rng.integers(0, 3, 60)
        assert silhouette_score(
            X, labels, sample_size=min(10_000, len(X)), random_state=0
        ) == pytest.approx(silhouette_score(X, labels))

    def test_subsampled_score_tracks_the_exhaustive_one(self):
        """On separated clusters the subsample must reach the same conclusion."""
        from sklearn.metrics import silhouette_score

        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(c, 0.25, (6000, 2)) for c in (0, 8, 16)])
        labels = np.repeat([0, 1, 2], 6000)
        full = silhouette_score(X, labels)
        sub = silhouette_score(X, labels, sample_size=10_000, random_state=0)
        assert sub == pytest.approx(full, abs=0.02)
        assert sub > 0.8, "well-separated clusters must still score high"


class TestBackgroundExcludedFromLabelStats:
    """hqcr.py:313 -- `flat_index >= 0` is always true for a rasterize(fill=0)
    index map, so it copied the full image while removing nothing, and folded
    background into polygon id 0."""

    def test_source_uses_strict_greater_than(self):
        import inspect

        from spoqc.subworkflows import hqcr

        src = inspect.getsource(hqcr)
        assert "flat_index >= 0" not in src
        assert "flat_index > 0" in src

    def test_old_mask_removed_nothing(self):
        index_map = np.array([[0, 0, 1], [0, 2, 2]])
        assert (index_map.ravel() >= 0).all()

    def test_new_mask_keeps_only_in_cell_pixels(self):
        index_map = np.array([[0, 0, 1], [0, 2, 2]])
        assert (index_map.ravel() > 0).sum() == 3

    def test_background_no_longer_pollutes_label_means(self):
        from scipy import ndimage

        index_map = np.array([[0, 1], [2, 2]])
        values = np.array([[100.0, 5.0], [7.0, 9.0]])
        fi, fl = index_map.ravel(), values.ravel()
        m = fi > 0
        means = ndimage.mean(input=fl[m], labels=fi[m], index=np.array([1, 2]))
        assert means[0] == pytest.approx(5.0)
        assert means[1] == pytest.approx(8.0)
