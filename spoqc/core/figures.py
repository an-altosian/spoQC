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
from matplotlib.image import AxesImage

# Collections with at least this many elements are rasterised in PDF output. Vector PDF costs
# ~0.1 ms per marker (44 s and 92 MB for the 410k-point doublet 3D scatter); rasterised it is
# 5 s and 0.3 MB, drawn at the savefig dpi like the PNG. Axes and text stay vector.
RASTERIZE_MIN_ELEMENTS = 10_000
# Images with more samples than output pixels are decimated (by stride) to this many samples per
# output pixel before pickling; the renderer antialiases what is left down to output pixels (4
# keeps a noise-texture image within 8/255 mean of the full-res render; 2 was 13/255). A 913 Mpx
# imshow otherwise pickles ~4-7 GB per figure and spends minutes in _resample for PNG and PDF each.
IMAGE_SAMPLES_PER_PIXEL = 4
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


def _decimate_images(fig, dpi):
    for image in fig.findobj(AxesImage):
        ax = image.axes
        left, right, bottom, top = image.get_extent()
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        rows, cols = image.get_array().shape[:2]
        visible_cols = cols * (min(x1, max(left, right)) - max(x0, min(left, right))) / abs(right - left)
        visible_rows = rows * (min(y1, max(bottom, top)) - max(y0, min(bottom, top))) / abs(top - bottom)
        box = ax.get_position()
        out_cols = box.width * fig.get_figwidth() * dpi
        out_rows = box.height * fig.get_figheight() * dpi
        step = int(min(visible_cols / out_cols, visible_rows / out_rows) / IMAGE_SAMPLES_PER_PIXEL)
        if step > 1:  # the norm keeps the vmin/vmax imshow took from the full array
            image.set_data(image.get_array()[::step, ::step])


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
    files exist after wait(). The figure is taken as finished: images in it larger than the
    output are decimated in place.
    """
    rc = None
    if isinstance(fig, Figure):
        rc = dict(matplotlib.rcParams)
        dpi = kwargs.get("dpi", rc["savefig.dpi"])
        _decimate_images(fig, fig.dpi if dpi == "figure" else dpi)
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
