"""Differential tests: pixel scoring (hqpr/hqtr clustering, scores, priors, mask_raw) against
the verbatim origin/dev db00d98 code in tests/reference_pixel_scoring_db00d98.py.

Everything must match bit for bit, dtypes included: background histogram, feature matrix,
structure scores, min-max normalisation, the mask_raw parquet parts (values, schema and
pandas metadata) and the beliefs handed to refinement.
"""
import os
import shutil

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr
from dask_ml.preprocessing import MinMaxScaler

import reference_pixel_scoring_db00d98 as reference
from spoqc import helperfuncs
from spoqc.image_analysis import pixel_scoring_dask
from spoqc.metrics.image import pixel_score, utility

METRICS = ['intensity', 'lbp', 'edge_strength', 'energy', 'relevance', 'entropy', 'uniformity', 'homogenity']
METRIC_DTYPES = [np.float64, np.float64, np.float64, np.float64, np.uint8, np.float32, np.float32, np.float32]


def bits_equal(a, b):
    a, b = np.atleast_1d(np.asarray(a)), np.atleast_1d(np.asarray(b))
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def fake_sdata(image):
    data = da.from_array(image[None], chunks=(1, 97, 131))
    ds = xr.Dataset({"image": xr.DataArray(data, dims=("c", "y", "x"))})
    return {"morphology_focus": {"scale0": ds}}


def make_image(shape, rng):
    # Blobs over a noisy background, so k-means, the GMM prior and the mask all have structure.
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    signal = 900 * (np.sin(yy / 17.0) * np.cos(xx / 11.0) > 0.4)
    return (150 + signal + rng.poisson(40, shape)).astype(np.uint16)


def write_metrics(folder, suffix, n, rng, names=METRICS, dtypes=METRIC_DTYPES):
    """Metric parquets as structure analysis writes them, and the arrays written."""
    os.makedirs(folder, exist_ok=True)
    arrays = {}
    for name, dtype in zip(names, dtypes):
        if dtype == np.uint8:
            values = rng.integers(0, 2, n).astype(np.uint8)
        else:
            values = (rng.gamma(2.0, 1.0, n) * rng.integers(1, 4, n)).astype(dtype)
        arrays[name] = values
        pq.write_table(pa.Table.from_arrays([pa.array(values)], names=[name]),
                       f"{folder}/{name}_output_{suffix}.parquet", row_group_size=7_777)
    return arrays


def parquet_parts(path):
    return {f: pq.read_table(f"{path}/{f}") for f in sorted(os.listdir(path))}


def assert_same_parquet_dir(a, b):
    pa_, pb_ = parquet_parts(a), parquet_parts(b)
    assert list(pa_) == list(pb_)
    for f in pa_:
        assert pa_[f].schema.equals(pb_[f].schema, check_metadata=True), f
        assert pa_[f].equals(pb_[f]), f


class TestBackgroundIntensity:
    @pytest.mark.parametrize("numba_threads", [1, 3], indirect=True)
    @pytest.mark.parametrize("kind", ["poisson", "full_range", "constant", "two_values", "sparse"])
    def test_matches_dask_histogram(self, kind, numba_threads):
        rng = np.random.default_rng(1)
        shape = (211, 173)
        image = {
            "poisson": lambda: make_image(shape, rng),
            "full_range": lambda: rng.integers(0, 65536, shape).astype(np.uint16),
            "constant": lambda: np.full(shape, 77, np.uint16),
            "two_values": lambda: rng.choice(np.array([3, 60000], np.uint16), shape),
            "sparse": lambda: (rng.random(shape) < 0.01).astype(np.uint16) * 4095,
        }[kind]()
        expected = reference.estimate_background_intensity_dask(fake_sdata(image), "morphology_focus", "scale0", "0")
        got = utility.estimate_background_intensity(image)
        for e, g in zip(expected, got):
            assert bits_equal(e, g), kind

    def test_flipped_view_counts_the_same(self):
        image = make_image((64, 50), np.random.default_rng(2))
        assert all(bits_equal(a, b) for a, b in zip(utility.estimate_background_intensity(image),
                                                    utility.estimate_background_intensity(np.flipud(image))))

    def test_rejects_non_uint16(self):
        with pytest.raises(TypeError):
            utility.estimate_background_intensity(np.zeros((4, 4), np.float32))


class TestPixelFeatures:
    @pytest.fixture
    def metric_files(self, tmp_path):
        arrays = write_metrics(str(tmp_path), "hqpr_0", 50_003, np.random.default_rng(3))
        return pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqpr_0"), arrays

    def expected(self, files):
        return reference.read_data_as_ddf(files, 10_000).compute()

    @pytest.mark.parametrize("threads", [1, 4])
    def test_parquet_read_matches_reference(self, metric_files, threads):
        files, _ = metric_files
        got = helperfuncs.read_pixel_features(files, threads)
        assert bits_equal(np.ascontiguousarray(got), self.expected(files))

    def test_in_memory_handoff_matches_reference_and_is_consumed(self, metric_files, monkeypatch):
        files, arrays = metric_files
        monkeypatch.setattr(helperfuncs, "PIXEL_FEATURES", {})
        for f in files:
            name = os.path.basename(f).split("_output_")[0]
            helperfuncs.nparr_to_parquet(arrays[name], name, os.path.dirname(f) + "/", "hqpr_0")  # trailing / as structure analysis
        assert len(helperfuncs.PIXEL_FEATURES) == len(files)
        got = helperfuncs.read_pixel_features(files, 2)
        assert bits_equal(np.ascontiguousarray(got), self.expected(files))
        assert helperfuncs.PIXEL_FEATURES == {}

    def test_feature_order_is_listdir_order(self, tmp_path):
        write_metrics(str(tmp_path), "hqtr", 10, np.random.default_rng(4))
        (tmp_path / "unrelated_output_hqpr_0.parquet").touch()
        expected = [f"{tmp_path}/{f}" for f in os.listdir(tmp_path) if f.endswith("hqtr.parquet")]
        assert pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqtr") == expected


class TestScores:
    @pytest.mark.parametrize("chunk_size", [1_000, 4_096, 60_000])
    def test_summed_scores_match_dask_summify(self, tmp_path, chunk_size):
        write_metrics(str(tmp_path), "hqpr_0", 20_011, np.random.default_rng(5))
        files = pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqpr_0")
        names = [os.path.basename(f).split("_output_")[0] for f in files]
        features = helperfuncs.read_pixel_features(files, 3)
        for metrics in (pixel_score.STRUCTURE_METRICS, pixel_score.ANTI_STRUCTURE_METRICS):
            expected = reference.dask_summify(str(tmp_path), "hqpr_0", metrics, chunk_size).compute()
            assert bits_equal(pixel_score.summed_score(features, names, metrics, chunk_size, 3), expected)

    @pytest.mark.parametrize("kind", ["spread", "constant", "tiny"])
    def test_min_max_matches_dask_ml(self, kind):
        rng = np.random.default_rng(6)
        values = {"spread": rng.gamma(1.0, 3e-3, 30_001), "constant": np.full(30_001, 0.37),
                  "tiny": 1e-300 * rng.random(30_001)}[kind]
        ddf = dd.from_dask_array(da.from_array(values, chunks=4_000), columns=["p"])
        expected = MinMaxScaler().fit_transform(ddf[["p"]]).iloc[:, 0].compute().to_numpy()
        assert bits_equal(pixel_scoring_dask.min_max_normalize(values), expected)


def run_pixel_qc(impl, root, modality, image, seed, **extra):
    staining = "0" if modality == "hqpr" else None
    figures = f"{root}/figs"
    os.makedirs(f"{figures}/{modality}/{modality}_clustering/{staining or ''}", exist_ok=True)
    np.random.seed(123)  # the GMM prior draws from the global RNG
    return impl.start_pixel_qc(
        fake_sdata(image), figures, f"{root}/tmp", modality, "morphology_focus", "scale0",
        image.shape[0], image.shape[1], helperfuncs.ImageDimStruct(0, 0, image.shape[1], image.shape[0]),
        seed, 3, chunk_size=7_000, sample_size=20_000, staining=staining, nstds_p=6, **extra)


def prepare(tmp_path, modality, shape, rng):
    """One shared metrics folder (so both runs see one os.listdir order) and one tmp folder per run."""
    suffix = "hqpr_0" if modality == "hqpr" else "hqtr"
    metrics_rel = "metrices/hqpr/0" if modality == "hqpr" else "metrices/hqtr"
    names, dtypes = METRICS, METRIC_DTYPES
    if modality == "hqtr":
        names, dtypes = ["transcript_density"] + METRICS[1:], [np.float64] + METRIC_DTYPES[1:]
    arrays = write_metrics(str(tmp_path / "shared" / metrics_rel), suffix, shape[0] * shape[1], rng, names, dtypes)
    priors = {prior: rng.random(shape[0] * shape[1]) for prior in ("qv", "ac")}
    roots = {}
    for run in ("ref", "new"):
        root = tmp_path / run
        (root / "tmp" / metrics_rel).parent.mkdir(parents=True)
        os.symlink(tmp_path / "shared" / metrics_rel, root / "tmp" / metrics_rel)
        if modality == "hqtr":
            for prior, values in priors.items():
                ddf = dd.from_dask_array(da.from_array(values, chunks=3_001), columns=[f"norm_p_{prior}_density"])
                helperfuncs.ddf_to_parquet(ddf, "hqtr", str(root / "tmp"), [], f"{prior}_prob")
        roots[run] = str(root)
    return roots, arrays, suffix, metrics_rel


@pytest.mark.parametrize("modality", ["hqpr", "hqtr"])
@pytest.mark.parametrize("handoff", ["parquet", "in_memory"])
def test_start_pixel_qc_is_bit_identical_to_reference(tmp_path, modality, handoff, monkeypatch):
    rng = np.random.default_rng(7)
    image = make_image((151, 203), rng)
    roots, arrays, suffix, metrics_rel = prepare(tmp_path, modality, image.shape, rng)
    prefix = "hqpr_0" if modality == "hqpr" else "hqtr"

    run_pixel_qc(reference, roots["ref"], modality, image, seed=11)
    expected_beliefs = reference.read_mask_raw_beliefs(f"{roots['ref']}/tmp", prefix)

    monkeypatch.setattr(helperfuncs, "PIXEL_FEATURES", {})
    if handoff == "in_memory":
        for name, values in arrays.items():
            helperfuncs.nparr_to_parquet(values, name, f"{roots['new']}/tmp/{metrics_rel}/", suffix)
    extra = {}
    if modality == "hqpr" and handoff == "in_memory":
        extra["background_intensity"] = utility.estimate_background_intensity(np.flipud(image))[0]
    beliefs = run_pixel_qc(pixel_scoring_dask, roots["new"], modality, image, seed=11, **extra)

    assert_same_parquet_dir(f"{roots['ref']}/tmp/{prefix}_output_mask_raw", f"{roots['new']}/tmp/{prefix}_output_mask_raw")
    assert bits_equal(beliefs, expected_beliefs)
    assert helperfuncs.PIXEL_FEATURES == {}
    labels = pq.read_table(f"{roots['new']}/tmp/{prefix}_output_mask_raw").column("cluster").to_numpy()
    assert len(np.unique(labels)) > 10  # the clustering is not degenerate
