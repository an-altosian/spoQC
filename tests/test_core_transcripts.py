"""core.transcripts equals the dask element cli.py sets up, and reads each column once."""

from __future__ import annotations

import polars as pl
import spatialdata as sd
from conftest import N_PARTS, assert_same_array, cli_sdata

from spoqc.core import transcripts

COLUMNS = [
    "x",
    "y",
    "z",
    "feature_name",
    "cell_id",
    "transcript_id",
    "overlaps_nucleus",
    "qv",
]


def test_parquet_parts_sort_numerically(synthetic_zarr):
    names = [p.rsplit("/", 1)[1] for p in transcripts.parquet_parts(synthetic_zarr)]
    assert names == [f"part.{i}.parquet" for i in range(N_PARTS)]


def test_load_equals_cli_dask_compute(sdata):
    expected = sdata.points["transcripts"].compute()
    got = transcripts.load_transcripts(sdata, COLUMNS).to_pandas()
    assert expected.index.equals(got.index), "row order is not the deduplicated index"
    for column in COLUMNS:
        if column == "feature_name":
            assert list(got[column].cat.categories) == list(
                expected[column].cat.categories
            )
            assert (
                "NaN" in got[column].cat.categories
                and "UNUSED" not in got[column].cat.categories
            )
            assert_same_array(got[column].cat.codes, expected[column].cat.codes, column)
        else:
            assert_same_array(got[column], expected[column], column)
    assert (got["cell_id"] == -1).any() and (got["cell_id"] >= 0).any()


def test_frame_is_one_chunk_and_enum(sdata):
    frame = transcripts.load_transcripts(sdata, ["x", "feature_name"])
    assert frame.n_chunks() == 1
    assert isinstance(frame.schema["feature_name"], pl.Enum)


def test_each_column_is_read_once(sdata, monkeypatch):
    reads = []
    real = pl.read_parquet
    monkeypatch.setattr(
        pl,
        "read_parquet",
        lambda parts, columns: reads.append(columns) or real(parts, columns=columns),
    )
    transcripts.load_transcripts(sdata, ["x", "y"])
    transcripts.load_transcripts(sdata, ["y", "qv"])
    transcripts.load_transcripts(sdata, ["qv", "x", "y"])
    assert reads == [["x", "y"], ["qv"]]
    transcripts.release(sdata)
    transcripts.load_transcripts(sdata, ["x"])
    assert reads[-1] == ["x"]


def test_global_coordinates_equal_get_centroids(sdata):
    expected = sd.get_centroids(
        sdata["transcripts"], coordinate_system="global"
    ).compute()
    frame = transcripts.global_coordinates(sdata)
    assert transcripts.global_coordinates(sdata) is frame, "computed once per run"
    got = frame.to_pandas()
    assert list(got.columns) == list(expected.columns)
    for column in expected.columns:
        assert_same_array(got[column], expected[column], column)
    assert_same_array(
        got.astype(int).to_numpy(), expected.astype(int).to_numpy(), "astype(int)"
    )


def test_cropped_sdata_equals_its_dask_compute(synthetic_zarr):
    cropped = cli_sdata(synthetic_zarr, crop=(10, 10, 60, 50))
    assert cropped.path is None
    expected = cropped.points["transcripts"].compute()
    got = transcripts.load_transcripts(cropped, COLUMNS).to_pandas()
    assert 0 < len(got) < 6_000
    for column in COLUMNS:
        if column == "feature_name":
            assert list(got[column].cat.categories) == list(
                expected[column].cat.categories
            )
            assert_same_array(got[column].cat.codes, expected[column].cat.codes, column)
        else:
            assert_same_array(got[column], expected[column], column)
    expected_xyz = sd.get_centroids(
        cropped["transcripts"], coordinate_system="global"
    ).compute()
    got_xyz = transcripts.global_coordinates(cropped).to_pandas()
    for column in expected_xyz.columns:
        assert_same_array(got_xyz[column], expected_xyz[column], column)
