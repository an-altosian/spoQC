"""core.raster.read_pixel_columns vs the reader it replaced, verbatim below (perf/integration2 1cd6406).

The old reader opened every part twice per column (a serial metadata pass, then a per-row-group
thread pool); the new one scans all requested columns of every part once in Arrow's C++ scanner.
Both are compared byte for byte, with their dtypes, on the layouts spoQC writes (core.parquet's
dask-identical parts, uneven dask partitions, a many-row-group single file, parts larger than one
scan batch) and on nulls. Mutants of the new reader must fail.
"""
import inspect
import os
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dask.utils import natural_sort_key

from spoqc.core import parquet, raster


def read_pixel_column(path, column, threads):
    """Read one column of a per-pixel parquet (a single file, or a dask directory of part.N.parquet)
    into a 1-D numpy array in the row order dask.dataframe.read_parquet returns.

    Row groups are decoded in parallel into one preallocated array. Nulls come out as dask 2026.1
    returns them: NaN in a float column, and a null anywhere in an integer column promotes the whole
    column to float64 with NaN (the parquet statistics' null counts decide this up front). Raises when
    there are no part files, when the parts disagree on the column's type, or when a non-numeric
    column holds nulls (dask would return an object array).
    """
    if os.path.isdir(path):
        # dask orders its part files naturally (part.2 before part.10), not lexicographically.
        names = sorted(
            (n for n in os.listdir(path) if n.endswith(".parquet")),
            key=natural_sort_key,
        )
        files = [os.path.join(path, n) for n in names]
        if not files:
            raise FileNotFoundError(f"no .parquet part files in {path}")
    else:
        files = [path]
    pieces, sizes, types, nulls = [], [], set(), 0
    for file in files:
        meta = pq.ParquetFile(file)
        types.add(meta.schema_arrow.field(column).type)
        index = meta.schema_arrow.get_field_index(column)
        for group in range(meta.metadata.num_row_groups):
            pieces.append((file, group))
            sizes.append(meta.metadata.row_group(group).num_rows)
            nulls += meta.metadata.row_group(group).column(index).statistics.null_count
    if len(types) != 1:
        raise ValueError(f"parts of {path} disagree on the type of {column}: {sorted(map(str, types))}")
    dtype = np.dtype(types.pop().to_pandas_dtype())
    if nulls and dtype.kind in "iu":
        dtype = np.dtype(np.float64)  # pandas' integer-with-NaN promotion, as dask applies it
    elif nulls and dtype.kind != "f":
        raise ValueError(f"{nulls} nulls in the {dtype} column {column} of {path}")
    starts = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    out = np.empty(int(starts[-1]), dtype=dtype)

    def read(i):
        file, group = pieces[i]
        values = (
            pq.ParquetFile(file)
            .read_row_group(group, columns=[column], use_threads=False)
            .column(0)
        )
        out[starts[i] : starts[i + 1]] = values.to_numpy()  # nulls: NaN (float64 for integers)

    with ThreadPoolExecutor(threads) as executor:
        list(executor.map(read, range(len(pieces))))
    return out


def write_pixel_parts(path, n_rows, starts, seed):
    """A mask_raw-like directory through core.parquet (the writer of every spoQC pixel directory)."""
    rng = np.random.default_rng(seed)
    columns = {
        "mask": rng.integers(0, 2, n_rows).astype(np.int8),
        "beliefs": rng.random(n_rows),
        "beliefs_smoothed": rng.random(n_rows).astype(np.float32),
        "cluster": rng.integers(0, 100, n_rows).astype(np.int32),
        "label": rng.integers(0, 3, n_rows).astype(np.int64),
    }
    parquet.write_parts(path, n_rows, parquet.columns_of(columns), starts, 3)
    return columns


def assert_same_as_old(path, columns, n_rows):
    new = raster.read_pixel_columns(path, columns, n_rows)
    assert list(new) == list(columns)
    for column in columns:
        old = read_pixel_column(path, column, 4)
        assert new[column].dtype == old.dtype, column
        assert new[column].tobytes() == old.tobytes(), column


ALL = ["mask", "beliefs", "beliefs_smoothed", "cluster", "label"]


@pytest.mark.parametrize("columns", [["mask", "beliefs"], ["beliefs_smoothed"], ALL, ["label", "mask"]])
def test_many_small_parts_like_hqtr(tmp_path, columns):
    """hqtr's layout: equal parts (here 500 rows; 10,000 in production), more than 10 of them."""
    n_rows = 23 * 500 + 137
    write_pixel_parts(f"{tmp_path}/d", n_rows, range(0, n_rows, 500), seed=1)
    assert len(os.listdir(f"{tmp_path}/d")) == 24
    assert_same_as_old(f"{tmp_path}/d", columns, n_rows)


def test_uneven_dask_partitions(tmp_path):
    n_rows = 10_007
    write_pixel_parts(f"{tmp_path}/d", n_rows, parquet.from_pandas_starts(n_rows, 13), seed=2)
    assert_same_as_old(f"{tmp_path}/d", ALL, n_rows)


SMALL_BATCH_ROWS = 1_000  # so a few-thousand-row part spans several scan batches
SMALL_FILES_PER_DATASET = 5  # so a directory spans several file groups


@pytest.mark.parametrize("files_per_dataset", [1, 5, 23, 24, 1000])
def test_file_groups_keep_the_part_order(tmp_path, monkeypatch, files_per_dataset):
    monkeypatch.setattr(raster, "_SCAN_FILES_PER_DATASET", files_per_dataset)
    n_rows = 23 * 500 + 137
    write_pixel_parts(f"{tmp_path}/d", n_rows, range(0, n_rows, 500), seed=9)
    assert_same_as_old(f"{tmp_path}/d", ["mask", "beliefs"], n_rows)


def test_parts_larger_than_one_scan_batch(tmp_path, monkeypatch):
    """Arrow's scanner splits a row group into batches of raster._SCAN_BATCH_ROWS; rows stay in order."""
    monkeypatch.setattr(raster, "_SCAN_BATCH_ROWS", SMALL_BATCH_ROWS)
    part = SMALL_BATCH_ROWS * 2 + 3
    n_rows = 2 * part
    write_pixel_parts(f"{tmp_path}/d", n_rows, [0, part], seed=3)
    assert_same_as_old(f"{tmp_path}/d", ["beliefs", "mask"], n_rows)


def test_single_file_with_many_row_groups_like_hqcr(tmp_path):
    rng = np.random.default_rng(4)
    n_rows = 50_003
    table = pa.table({"hqcr_mask": rng.integers(0, 2, n_rows).astype(np.int8), "hqcr_beliefs": rng.random(n_rows)})
    pq.write_table(table, f"{tmp_path}/hqcr.parquet", row_group_size=1_000)
    assert pq.ParquetFile(f"{tmp_path}/hqcr.parquet").metadata.num_row_groups == 51
    assert_same_as_old(f"{tmp_path}/hqcr.parquet", ["hqcr_mask", "hqcr_beliefs"], n_rows)


@pytest.mark.parametrize("type_", [pa.int8(), pa.int64(), pa.uint16(), pa.float32(), pa.float64()])
@pytest.mark.parametrize("null_part", [0, 7, 11])
def test_nulls_in_one_part(tmp_path, type_, null_part):
    """A null in an early or a late part: NaN for floats, float64 with NaN for the whole integer column."""
    rng = np.random.default_rng(5)
    for i in range(12):
        values = rng.integers(0, 50, 40).tolist()
        if i == null_part:
            values[3] = values[17] = None
        pq.write_table(
            pa.table({"v": pa.array(values, type=type_), "w": pa.array(rng.random(40))}),
            f"{tmp_path}/part.{i}.parquet",
            row_group_size=16,
        )
    new = raster.read_pixel_columns(str(tmp_path), ["v", "w"], 12 * 40)
    for column in ("v", "w"):
        old = read_pixel_column(str(tmp_path), column, 3)
        assert new[column].dtype == old.dtype and new[column].tobytes() == old.tobytes()
    assert np.isnan(new["v"]).sum() == 2


@pytest.mark.parametrize("n_rows", [12 * 40 - 1, 12 * 40 + 1])
def test_a_wrong_row_count_raises(tmp_path, n_rows):
    write_pixel_parts(f"{tmp_path}/d", 12 * 40, range(0, 12 * 40, 40), seed=6)
    with pytest.raises(ValueError, match="expected"):
        raster.read_pixel_columns(f"{tmp_path}/d", ["mask"], n_rows)


def mutant(old, new):
    source = inspect.getsource(raster)
    assert source.count(old) == 1, old
    clone = types.ModuleType("raster_mutant")
    clone.__package__ = raster.__package__
    exec(compile(source.replace(old, new), raster.__file__, "exec"), clone.__dict__)
    clone._SCAN_BATCH_ROWS = SMALL_BATCH_ROWS
    clone._SCAN_FILES_PER_DATASET = SMALL_FILES_PER_DATASET
    return clone


MUTANTS = {
    "lexicographic part order": ("key=natural_sort_key,", "key=str,"),
    "batches out of order": (
        "        ).scan_batches()\n",
        "        ).scan_batches()[::-1] if False else reversed(list(dataset.scanner(columns=columns).scan_batches()))\n",
    ),
    "file groups out of order": (
        "range(0, len(files), _SCAN_FILES_PER_DATASET)",
        "reversed(range(0, len(files), _SCAN_FILES_PER_DATASET))",
    ),
    "no integer null promotion": ("        if rows:  # pandas'", "        if False:  # pandas'"),
    "nulls left as fill values": ("out[column][np.concatenate(rows)] = np.nan", "pass"),
    "first batch only": ("        start = stop\n", "        start = stop\n        break\n"),
}


def mutant_differs(tmp_path, module):
    part = SMALL_BATCH_ROWS * 3 + 1
    n_rows = 2 * part + 29 * 500
    write_pixel_parts(f"{tmp_path}/d", n_rows, [0, part, *range(2 * part, n_rows, 500)], seed=8)
    os.makedirs(f"{tmp_path}/n")
    for i in range(12):
        values = list(range(40))
        if i == 7:
            values[3] = None
        pq.write_table(pa.table({"v": pa.array(values, type=pa.int32())}), f"{tmp_path}/n/part.{i}.parquet")
    for path, columns, n in ((f"{tmp_path}/d", ["mask", "beliefs"], n_rows), (f"{tmp_path}/n", ["v"], 480)):
        try:
            new = module.read_pixel_columns(path, columns, n)
        except Exception:
            return True
        for column in columns:
            old = read_pixel_column(path, column, 4)
            if new[column].dtype != old.dtype or new[column].tobytes() != old.tobytes():
                return True
    return False


def test_the_mutant_check_passes_the_real_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(raster, "_SCAN_BATCH_ROWS", SMALL_BATCH_ROWS)
    monkeypatch.setattr(raster, "_SCAN_FILES_PER_DATASET", SMALL_FILES_PER_DATASET)
    assert not mutant_differs(tmp_path, raster)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_caught(tmp_path, name):
    module = mutant(*MUTANTS[name])
    module.types_ = types
    assert mutant_differs(tmp_path, module), f"mutant {name!r} survived"
