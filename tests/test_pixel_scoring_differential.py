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

import numba
import reference_pixel_scoring_db00d98 as reference
from spoqc import helperfuncs
from spoqc.core import groupreduce
from spoqc.image_analysis import pixel_scoring_dask
from spoqc.metrics.image import pixel_score, utility

METRICS = ['intensity', 'lbp', 'edge_strength', 'energy', 'relevance', 'entropy', 'uniformity', 'homogenity']
METRIC_DTYPES = [np.float64, np.float64, np.float64, np.float64, np.uint8, np.float32, np.float32, np.float32]


def bits_equal(a, b):
    a, b = np.ascontiguousarray(np.atleast_1d(a)), np.ascontiguousarray(np.atleast_1d(b))
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

    @pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32, np.float64])
    def test_other_dtypes_match_dask_histogram(self, dtype):
        rng = np.random.default_rng(9)
        image = (rng.gamma(2.0, 40.0, (123, 77))).astype(dtype)
        if np.issubdtype(dtype, np.floating):
            image[3, 4] = np.nan
        expected = reference.estimate_background_intensity_dask(fake_sdata(image), "morphology_focus", "scale0", "0")
        got = utility.estimate_background_intensity(image)
        for e, g in zip(expected, got):
            assert bits_equal(e, g), dtype

    def test_all_nan_raises_like_dask(self):
        image = np.full((5, 5), np.nan, np.float32)
        with pytest.raises(ValueError):
            utility.estimate_background_intensity(image)


class TestPixelFeatures:
    @pytest.fixture
    def metric_files(self, tmp_path):
        arrays = write_metrics(str(tmp_path), "hqpr_0", 50_003, np.random.default_rng(3))
        return pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqpr", "hqpr_0"), arrays

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

    @pytest.mark.parametrize("modality,suffix", [("hqpr", "hqpr_0"), ("hqtr", "hqtr")])
    def test_feature_order_is_fixed_whatever_the_write_order(self, tmp_path, modality, suffix):
        names = pixel_scoring_dask.PIXEL_FEATURE_NAMES[modality]
        for name in names[::-1]:
            (tmp_path / f"{name}_output_{suffix}.parquet").touch()
        (tmp_path / "unrelated_output_other.parquet").touch()
        assert pixel_scoring_dask.pixel_feature_files(str(tmp_path), modality, suffix) == [
            f"{tmp_path}/{name}_output_{suffix}.parquet" for name in names]

    def test_missing_feature_raises(self, tmp_path):
        for name in pixel_scoring_dask.PIXEL_FEATURE_NAMES["hqpr"][1:]:
            (tmp_path / f"{name}_output_hqpr_0.parquet").touch()
        with pytest.raises(ValueError, match="missing"):
            pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqpr", "hqpr_0")

    def test_stray_feature_file_raises(self, tmp_path):
        for name in pixel_scoring_dask.PIXEL_FEATURE_NAMES["hqpr"] + ["cluster"]:
            (tmp_path / f"{name}_output_hqpr_0.parquet").touch()
        with pytest.raises(ValueError, match="cluster_output_hqpr_0"):
            pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqpr", "hqpr_0")

    def test_shorter_parquet_raises(self, metric_files):
        files, _ = metric_files
        short = files[3]
        table = pq.read_table(short)
        pq.write_table(table.slice(0, table.num_rows - 1), short)
        with pytest.raises(ValueError, match="pixels"):
            helperfuncs.read_pixel_features(files, 2)

    def test_shorter_in_memory_column_raises(self, metric_files, monkeypatch):
        files, arrays = metric_files
        name = os.path.basename(files[5]).split("_output_")[0]
        monkeypatch.setattr(helperfuncs, "PIXEL_FEATURES", {os.path.abspath(files[5]): arrays[name][:-1].astype(np.float32)})
        with pytest.raises(ValueError, match="pixels"):
            helperfuncs.read_pixel_features(files, 2)


class TestScores:
    @pytest.mark.parametrize("chunk_size", [1_000, 4_096, 60_000])
    def test_summed_scores_match_dask_summify(self, tmp_path, chunk_size):
        write_metrics(str(tmp_path), "hqpr_0", 20_011, np.random.default_rng(5))
        files = pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqpr", "hqpr_0")
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


@pytest.mark.parametrize("shape", [(151, 203), (1, 14_001)], ids=["151x203", "one_row_partition"])
@pytest.mark.parametrize("modality", ["hqpr", "hqtr"])
@pytest.mark.parametrize("handoff", ["parquet", "in_memory"])
def test_start_pixel_qc_is_bit_identical_to_reference(tmp_path, modality, handoff, shape, monkeypatch):
    # 14_001 pixels = 2 chunks of 7_000 plus a one-row partition (a repeated last division).
    rng = np.random.default_rng(7)
    image = make_image(shape, rng)
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


@pytest.mark.parametrize("chunk_size", [7_000, 50_000])
def test_pixel_frame_partitions_match_origin_dev_frame(chunk_size):
    """The cluster-mean groupby gets exactly origin/dev's partitions (values, dtypes, index).

    The means themselves are not compared: origin/dev's dask groupby-mean (a disk shuffle whose
    combine order follows task completion) is not deterministic in its last bits, run to run.
    """
    rng = np.random.default_rng(8)
    n = 200_003
    clusters = rng.integers(0, 100, n).astype(np.int32)
    s_score = rng.gamma(2.0, 50.0, n).astype(np.float32)
    as_score = rng.standard_normal(n).astype(np.float32)
    intensity = rng.integers(0, 65536, n).astype(np.uint16)
    # origin/dev's frame: zeros from_dask_array, then the columns assigned as chunk_size dask arrays.
    ref_ddf = dd.from_dask_array(da.zeros(n, chunks=chunk_size), columns=['cluster'])
    ref_ddf = ref_ddf.assign(cluster=da.from_array(clusters, chunks=chunk_size))
    ref_ddf = ref_ddf.assign(s_score=da.from_array(s_score, chunks=chunk_size), as_score=da.from_array(as_score, chunks=chunk_size))
    ref_ddf = ref_ddf.assign(intensity=da.from_array(intensity, chunks=chunk_size))
    frame = pixel_score.pixel_frame({'cluster': clusters, 's_score': s_score, 'as_score': as_score, 'intensity': intensity},
                                    pixel_score.row_divisions(n, chunk_size))
    assert frame.divisions == ref_ddf.divisions
    for i in range(ref_ddf.npartitions):
        expected, got = ref_ddf.partitions[i].compute(), frame.partitions[i].compute()
        assert got.index.equals(expected.index), i
        for column in expected.columns:
            assert bits_equal(got[column].to_numpy(), expected[column].to_numpy()), (i, column)


class TestGroupSum:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(10)
        n = 3 * groupreduce.GROUP_SUM_CHUNK + 12_345
        groups = rng.integers(0, 100, n).astype(np.int32)
        groups[groups == 42] = 41  # an empty group
        values = (rng.gamma(2.0, 50.0, n) * rng.choice([1e-3, 1.0, 1e3], n)).astype(np.float32)
        return groups, values

    @pytest.mark.parametrize("numba_threads", [1, 2, 4], indirect=True)
    def test_same_bits_for_any_thread_count(self, data, numba_threads):
        groups, values = data
        sums, counts = groupreduce.group_sum(groups, values, 100)
        reference_sums = np.zeros(100)
        for c in range(0, len(groups), groupreduce.GROUP_SUM_CHUNK):  # fixed order: rows in a chunk, then chunks
            part = np.zeros(100)
            for g, v in zip(groups[c:c + groupreduce.GROUP_SUM_CHUNK].tolist(), values[c:c + groupreduce.GROUP_SUM_CHUNK].astype(np.float64).tolist()):
                part[g] += v
            reference_sums += part
        assert bits_equal(sums, reference_sums)
        assert bits_equal(counts, np.bincount(groups, minlength=100).astype(np.int64))

    def test_repeatable(self, data):
        groups, values = data
        runs = [groupreduce.group_sum(groups, values, 100)[0] for _ in range(3)]
        assert all(bits_equal(runs[0], r) for r in runs[1:])
