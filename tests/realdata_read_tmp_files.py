"""Real-data check (not collected by pytest): read_sdata_parquet_tmp_files on a real run's tmp files.

Usage:
    python tests/realdata_read_tmp_files.py <spatialdata.zarr> <tmp_folder>

<tmp_folder> holds a run's *_hqcr.parquet files. Checks, against the verbatim original:
- a step run on its own (obs without the steps' columns) reads every file, as the original did;
- in -s all the steps' columns are already in obs: the fixed reader skips every file and
  leaves obs as it was, which is what the original's swallowed failure also left.
"""

import sys

import anndata as ad
import pandas as pd

from spoqc import helperfuncs
from test_read_sdata_parquet_tmp_files import original_read_sdata_parquet_tmp_files

zarr_path, tmp_folder = sys.argv[1], sys.argv[2]


def table():
    adata = ad.read_zarr(f"{zarr_path}/tables/table")
    adata.obs.index = [int(i) for i in range(adata.n_obs)]  # cli.py
    adata.obs.index = adata.obs.index.astype(str)
    adata.obs.index.name = "index"
    return {"table": adata}


old, new = table(), table()
original_read_sdata_parquet_tmp_files(old, tmp_folder, "hqcr")
helperfuncs.read_sdata_parquet_tmp_files(new, tmp_folder, "hqcr")
pd.testing.assert_frame_equal(new["table"].obs, old["table"].obs)
print(
    f"on its own: {new['table'].obs.shape[1]} obs columns, identical to the original's read"
)

in_memory = new["table"].obs.copy()
old = {"table": ad.AnnData(obs=in_memory.copy())}
new = {"table": ad.AnnData(obs=in_memory.copy())}
original_read_sdata_parquet_tmp_files(old, tmp_folder, "hqcr")
helperfuncs.read_sdata_parquet_tmp_files(new, tmp_folder, "hqcr")
pd.testing.assert_frame_equal(new["table"].obs, old["table"].obs)
pd.testing.assert_frame_equal(new["table"].obs, in_memory)
print(
    "in-process (-s all): every file skipped, obs unchanged and identical to the original's"
)
