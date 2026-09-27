"""Write figures in worker processes while the main thread carries on computing.

``start(workers)`` routes matplotlib ``Figure.savefig`` / ``pyplot.savefig`` and plotly
``BaseFigure.write_image`` to a spawn-context ``ProcessPoolExecutor``: the finished figure is
pickled at the call site and a worker renders and writes the file. Rendering (Agg rasterising,
PDF path serialisation, kaleido export) is 70-95% of figure time and holds the GIL, so it needs
processes, not threads.

``wait()`` blocks until every submitted figure is written and re-raises the first worker error;
call it before anything reads a figure file back. ``stop()`` waits and restores the originals.
Only figure files go through here; no computed data is touched.
"""

import multiprocessing
import os
import pickle
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as wait_futures

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import Collection, QuadMesh
from matplotlib.figure import Figure
from plotly.basedatatypes import BaseFigure

# Collections with at least this many elements are rasterised in PDF output. Vector PDF costs
# ~0.1 ms per marker (44 s and 92 MB for the 410k-point doublet 3D scatter); rasterised it is
# 5 s and 0.3 MB, drawn at the savefig dpi exactly like the PNG. Axes and text stay vector.
RASTERIZE_MIN_ELEMENTS = 10_000
# Figures submitted but not yet written, per worker, before savefig blocks (bounds the pickled
# bytes held in memory).
MAX_PENDING_PER_WORKER = 2

_ORIGINALS = {
    "Figure.savefig": Figure.savefig,
    "pyplot.savefig": plt.savefig,
    "BaseFigure.write_image": BaseFigure.write_image,
}
_executor = None
_max_pending = 0
_pending = set()


def _element_count(collection):
    if isinstance(collection, QuadMesh):
        return collection.get_coordinates().size // 2
    return max(len(collection.get_offsets()), len(collection.get_paths()))


def _is_pdf(fname, kwargs):
    fmt = (
        kwargs.get("format")
        or os.path.splitext(os.fspath(fname))[1][1:]
        or matplotlib.rcParams["savefig.format"]
    )
    return fmt.lower() == "pdf"


def _write_matplotlib(blob, rc, fname, args, kwargs):
    fig = pickle.loads(blob)
    with matplotlib.rc_context(rc):
        if _is_pdf(fname, kwargs):
            for collection in fig.findobj(Collection):
                if _element_count(collection) >= RASTERIZE_MIN_ELEMENTS:
                    collection.set_rasterized(True)
        fig.savefig(fname, *args, **kwargs)


def _write_plotly(blob, file, args, kwargs):
    pickle.loads(blob).write_image(file, *args, **kwargs)


def _submit(fn, *args):
    done = {f for f in _pending if f.done()}
    if len(_pending) >= _max_pending:
        done, _ = wait_futures(_pending, return_when=FIRST_COMPLETED)
    for future in done:
        _pending.discard(future)
        future.result()
    _pending.add(_executor.submit(fn, *args))


def _savefig(fig, fname, *args, **kwargs):
    if not isinstance(fname, (str, os.PathLike)):
        # A buffer is read by the caller right away, so it must be filled in this process.
        return _ORIGINALS["Figure.savefig"](fig, fname, *args, **kwargs)
    blob = pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL)
    _submit(_write_matplotlib, blob, dict(matplotlib.rcParams), fname, args, kwargs)


def _pyplot_savefig(*args, **kwargs):
    # pyplot.savefig adds canvas.draw_idle(), which on Agg is a second full render on the main
    # thread; it only matters for interactive backends.
    return plt.gcf().savefig(*args, **kwargs)


def _write_image(fig, file, *args, **kwargs):
    if not isinstance(file, (str, os.PathLike)):
        return _ORIGINALS["BaseFigure.write_image"](fig, file, *args, **kwargs)
    _submit(
        _write_plotly,
        pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL),
        file,
        args,
        kwargs,
    )


def start(workers):
    """Route figure writes to `workers` spawned processes until stop()."""
    global _executor, _max_pending
    _executor = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
    _max_pending = workers * MAX_PENDING_PER_WORKER
    Figure.savefig = _savefig
    plt.savefig = _pyplot_savefig
    BaseFigure.write_image = _write_image


def wait():
    """Block until every submitted figure is written; re-raise the first worker error."""
    while _pending:
        _pending.pop().result()


def stop():
    """wait(), shut the workers down and restore synchronous figure writing."""
    global _executor
    Figure.savefig = _ORIGINALS["Figure.savefig"]
    plt.savefig = _ORIGINALS["pyplot.savefig"]
    BaseFigure.write_image = _ORIGINALS["BaseFigure.write_image"]
    try:
        wait()
    finally:
        _executor.shutdown()
        _executor = None
