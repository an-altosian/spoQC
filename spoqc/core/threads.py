"""The single thread budget of a spoQC run.

`configure(n)` must run before the libraries below are imported: polars sizes its
pool at first import, numba reads NUMBA_NUM_THREADS at import, OpenBLAS/MKL/OpenMP
read their variables when the library loads (pyarrow's CPU pool follows
OMP_NUM_THREADS), dask and zarr read DASK_* / ZARR_* variables into their config at
import, and pyarrow reads ARROW_IO_THREADS when its I/O pool starts.
OpenCV and numcodecs' blosc keep their own pools, which are set by call; numcodecs
does not follow BLOSC_NTHREADS (measured), so it is set explicitly.
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
    "ZARR_THREADING__MAX_WORKERS",
    "ARROW_IO_THREADS",
)
_FIXED_AT_IMPORT = (
    "numpy",
    "numba",
    "polars",
    "dask",
    "ovrlpy",
    "pyarrow",
    "zarr",
    "numcodecs",
    "cv2",
)

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
    # imported only now, so numpy (which both load) starts under the variables above
    import cv2
    import numcodecs.blosc

    cv2.setNumThreads(n)  # OpenCV otherwise starts one thread per host CPU
    numcodecs.blosc.set_nthreads(n)
    N = n
