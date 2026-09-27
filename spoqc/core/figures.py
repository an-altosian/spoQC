"""The one figure-writing path: save_figure() hands a finished figure to a worker process.

The figure is pickled at the call site and a worker renders and writes every requested file, so
the main thread goes straight on computing. Rendering (Agg rasterising, PDF path serialisation,
kaleido export) is 70-95% of figure time and holds the GIL, so it needs processes, not threads.

start(n) opens the pool, wait() blocks until every submitted figure is written and re-raises the
first worker error (call it before anything reads a figure file back), stop() waits and closes
the pool. Only figure files go through here; no computed data is touched.
"""

import multiprocessing
import os
import pickle
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as wait_futures

import matplotlib
from matplotlib.collections import Collection, QuadMesh
from matplotlib.figure import Figure

# Collections with at least this many elements are rasterised in PDF output. Vector PDF costs
# ~0.1 ms per marker (44 s and 92 MB for the 410k-point doublet 3D scatter); rasterised it is
# 5 s and 0.3 MB, drawn at the savefig dpi like the PNG. Axes and text stay vector.
RASTERIZE_MIN_ELEMENTS = 10_000
# Figures submitted but not yet written, per worker, before save_figure blocks (bounds the
# pickled bytes held in memory).
MAX_PENDING_PER_WORKER = 2

_executor = None
_max_pending = 0
_pending = set()


def _element_count(collection):
    if isinstance(collection, QuadMesh):
        return collection.get_coordinates().size // 2
    return max(len(collection.get_offsets()), len(collection.get_paths()))


def _write(blob, rc, paths, kwargs):
    fig = pickle.loads(blob)
    if rc is None:  # plotly
        for path in paths:
            fig.write_image(path, **kwargs)
        return
    with matplotlib.rc_context(rc):
        for path in paths:
            if os.fspath(path).lower().endswith(".pdf"):
                for collection in fig.findobj(Collection):
                    if _element_count(collection) >= RASTERIZE_MIN_ELEMENTS:
                        collection.set_rasterized(True)  # no effect on the Agg PNG
            fig.savefig(path, **kwargs)


def save_figure(fig, *paths, **kwargs):
    """Write a matplotlib or plotly `fig` to each of `paths` (format from the extension).

    `kwargs` go to every savefig / write_image call. Returns once the figure is pickled; the
    files exist after wait().
    """
    rc = dict(matplotlib.rcParams) if isinstance(fig, Figure) else None
    blob = pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL)
    done = {f for f in _pending if f.done()}
    if len(_pending) >= _max_pending:
        done, _ = wait_futures(_pending, return_when=FIRST_COMPLETED)
    for future in done:
        _pending.discard(future)
        future.result()
    _pending.add(_executor.submit(_write, blob, rc, paths, kwargs))


def start(workers):
    """Open a pool of `workers` spawned processes for save_figure()."""
    global _executor, _max_pending
    _executor = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
    _max_pending = workers * MAX_PENDING_PER_WORKER


def wait():
    """Block until every submitted figure is written; re-raise the first worker error."""
    while _pending:
        _pending.pop().result()


def stop():
    """wait(), then shut the pool down."""
    global _executor
    try:
        wait()
    finally:
        _executor.shutdown()
        _executor = None
