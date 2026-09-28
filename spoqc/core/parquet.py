"""The writer of spoQC's per-pixel parquet directories.

`write_parts` writes the files dask writes for
    dd.from_dask_array(da.from_array(column, chunks=chunk_size)) (one per column, assigned side by side)
    .to_parquet(path, engine="pyarrow", write_index=True, overwrite=True)
byte for byte (helperfuncs.ddf_to_parquet's call), without a dask graph: part.{i}.parquet holds
rows i * chunk_size up to (i + 1) * chunk_size with their row positions as the int64 index
'__null_dask_index__', so dd.read_parquet(path, calculate_divisions=True) sees the same
partitions and divisions. The parts are built and written on `threads` threads.
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

INDEX_NAME = "__null_dask_index__"  # what dask names an unnamed index it writes


def write_parts(
    path: str, n_rows: int, part_columns: Callable, chunk_size: int, threads: int
) -> None:
    """
    Writes rows 0..n_rows-1 as dask's partitioned parquet directory at `path` (replacing it).

    part_columns(start, stop) returns the partition's columns, in order, as a dict of 1-D numpy
    arrays of length stop - start; every call must give the same names and dtypes.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    starts = range(0, n_rows, chunk_size)
    if not starts:
        return
    first = part_columns(0, min(chunk_size, n_rows))
    empty = pd.DataFrame({name: values[:0] for name, values in first.items()})
    empty.index = pd.Index(np.arange(0, dtype=np.int64), name=INDEX_NAME)
    schema = pa.Table.from_pandas(empty, nthreads=1, preserve_index=True).schema

    def write(i):
        start = starts[i]
        stop = min(start + chunk_size, n_rows)
        columns = first if i == 0 else part_columns(start, stop)
        arrays = [pa.array(values) for values in columns.values()]
        arrays.append(pa.array(np.arange(start, stop, dtype=np.int64)))
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, f"{path}/part.{i}.parquet", compression="snappy")

    with ThreadPoolExecutor(threads) as executor:
        list(executor.map(write, range(len(starts))))
