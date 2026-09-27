"""The single thread budget of a spoQC run.

`configure(n)` must run before numpy, numba, polars, dask or ovrlpy are imported:
polars sizes its pool at first import, numba reads NUMBA_NUM_THREADS at import,
OpenBLAS/MKL/OpenMP read their variables when the library loads, and dask reads
DASK_* variables into its config at import.
Pool sizes elsewhere in spoQC come from `N`.
"""

from __future__ import annotations

import os
import sys

ENV_VARS = (
    "POLARS_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLOSC_NTHREADS",
    "DASK_NUM_WORKERS",
)
_FIXED_AT_IMPORT = ("numpy", "numba", "polars", "dask", "ovrlpy")

N: int | None = None


def configure(n: int) -> None:
    global N
    if n < 1:
        raise ValueError(f"thread count must be >= 1, got {n}")
    imported = [m for m in _FIXED_AT_IMPORT if m in sys.modules]
    if imported:
        raise RuntimeError(f"threads.configure must run before importing {imported}")
    for var in ENV_VARS:
        os.environ[var] = str(n)
    N = n
