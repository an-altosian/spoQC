"""The one figure-writing path: save_figure() hands a finished figure to a worker process.

The figure is pickled at the call site and a worker renders and writes every requested file, so
the main thread goes straight on computing. Rendering (Agg rasterising, PDF path serialisation,
kaleido export) is 70-95% of figure time and holds the GIL, so it needs processes, not threads.

start(n) opens the pool; without it save_figure writes synchronously in this process.
wait() blocks until every submitted figure is written and re-raises the first worker error: call
it before anything lists, moves or reads a figure file. stop() waits and closes the pool;
abort() cancels queued writes on the error path. Only figure files go through here; no computed
data is touched.

CPU budget: each worker is one render process plus, for plotly, its kaleido Chromium; the two run
in turn, so a worker keeps about one core busy. The caller sizes the pool as a share of its
thread budget (spoqc.cli: THREADS // THREADS_PER_FIGURE_WORKER) and should give the compute
threads the rest.
"""

import io
import multiprocessing
import os
import pickle
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as wait_futures

import matplotlib
import numpy as np
from matplotlib.collections import Collection, QuadMesh
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from spoqc.core import threads

# Threads of the run's budget per figure worker (see the CPU budget note above).
THREADS_PER_FIGURE_WORKER = 4
# Collections with at least this many elements are rasterised in PDF output. Vector PDF costs
# ~0.1 ms per marker (44 s and 92 MB for the 410k-point doublet 3D scatter); rasterised it is
# 5 s and 0.3 MB, drawn at the savefig dpi like the PNG. Axes and text stay vector.
RASTERIZE_MIN_ELEMENTS = 10_000
# Float images with more samples than output pixels are block-averaged down to this many samples
# per output pixel (per axis) in the pickled copy; the renderer antialiases the rest down to
# output pixels. A 913 Mpx imshow otherwise pickles ~4 GB per figure and spends minutes in
# _resample for PNG and PDF each. Integer, bool and label images and 'nearest'/'none'
# interpolation are never reduced: averaging would change what they show.
IMAGE_SAMPLES_PER_PIXEL = 4
# Pickled figure bytes submitted but not yet written before save_figure blocks. Each blob is held
# about twice (here and in the pipe to the worker). A single larger figure is still submitted.
MAX_PENDING_BYTES = 2 * 1024**3

# mplot3d warns that set_rasterized on its collections "will be ignored", but Axes3D composites
# them rasterised all the same: the 410k-point doublet 3D PDF goes from 92 MB to 0.3 MB.
warnings.filterwarnings(
    "ignore", message="Rasterization of .*Path3DCollection", category=UserWarning
)

_executor = None
_pending = {}  # future -> pickled bytes


def _element_count(collection):
    if isinstance(collection, QuadMesh):
        return collection.get_coordinates().size // 2
    return max(len(collection.get_offsets()), len(collection.get_paths()))


def _block_mean(a, k):
    """Mean over k x k blocks (the last block per axis may be smaller), masked samples excluded."""
    starts = [np.arange(0, n, k) for n in a.shape[:2]]

    def block_sum(x):
        return np.add.reduceat(
            np.add.reduceat(x, starts[0], axis=0, dtype=np.float64), starts[1], axis=1
        )

    data = np.ma.getdata(a)
    if np.ma.getmask(a) is np.ma.nomask:
        sizes = [np.diff(np.append(s, n)) for s, n in zip(starts, a.shape[:2])]
        count = np.multiply.outer(*sizes).reshape(
            len(sizes[0]), len(sizes[1]), *([1] * (a.ndim - 2))
        )
        return (block_sum(data) / count).astype(a.dtype)
    valid = ~np.ma.getmaskarray(a)
    count = block_sum(valid)
    mean = block_sum(np.where(valid, data, 0)) / np.maximum(count, 1)
    return np.ma.masked_array(mean.astype(a.dtype), mask=count == 0)


def _reduced_images(fig, dpi):
    """{id(per-pixel array): block-averaged copy} for float images far denser than the output.

    Every per-pixel array of an image is reduced with the same blocks: the data (with its mask,
    and RGB(A) channels) and an array alpha. Extent (fixed by imshow), norm and clim are not
    per-pixel and stay as they are. Only plain AxesImage: NonUniformImage/PcolorImage carry
    per-pixel coordinate arrays and are left alone.
    """
    reduced = {}
    for image in fig.findobj(lambda artist: type(artist) is AxesImage):
        a = image.get_array()
        if not np.issubdtype(a.dtype, np.floating) or image.get_interpolation() in (
            "nearest",
            "none",
        ):
            continue
        # Display extent of the whole image through its own transform (world or pixel units,
        # zoomed or translated alike), at the figure dpi; the output is at `dpi`.
        box = image.get_window_extent()
        scale = dpi / fig.dpi
        samples_per_pixel = min(
            a.shape[1] / (abs(box.width) * scale),
            a.shape[0] / (abs(box.height) * scale),
        )
        k = int(round(samples_per_pixel / IMAGE_SAMPLES_PER_PIXEL, 6))  # display extents carry float noise
        if k > 1:  # the norm keeps the vmin/vmax imshow took from the full array
            reduced[id(a)] = _block_mean(a, k)
            alpha = image.get_alpha()
            if np.ndim(alpha) > 0:  # e.g. ovrlpy's signal-faded integrity map
                reduced[id(alpha)] = _block_mean(np.asarray(alpha, dtype=np.float64), k)
    return reduced


def _same(obj):
    return obj


class _SubstitutingPickler(pickle.Pickler):
    """Pickles `replacements[id(obj)]` in place of obj, leaving the caller's figure untouched."""

    def __init__(self, file, replacements):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._replacements = replacements

    def reducer_override(self, obj):
        if id(obj) in self._replacements:
            return _same, (self._replacements[id(obj)],)
        return NotImplemented


def _pickle(fig, exact, kwargs):
    if exact or not isinstance(fig, Figure):
        return pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL)
    dpi = kwargs.get("dpi", matplotlib.rcParams["savefig.dpi"])
    replacements = _reduced_images(fig, fig.dpi if dpi == "figure" else dpi)
    if not replacements:
        return pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL)
    buffer = io.BytesIO()
    _SubstitutingPickler(buffer, replacements).dump(fig)
    return buffer.getvalue()


def _write(blob, rc, paths, exact, kwargs):
    fig = pickle.loads(blob)
    if rc is None:  # plotly
        for path in paths:
            fig.write_image(path, **kwargs)
        return
    with matplotlib.rc_context(rc):
        for path in paths:
            if not exact and os.fspath(path).lower().endswith(".pdf"):
                for collection in fig.findobj(Collection):
                    if _element_count(collection) >= RASTERIZE_MIN_ELEMENTS:
                        collection.set_rasterized(True)  # no effect on the Agg PNG
            fig.savefig(path, **kwargs)


def save_figure(fig, *paths, exact=False, **kwargs):
    """Write a matplotlib or plotly `fig` to each of `paths` (format from the extension).

    `kwargs` go to every savefig / write_image call. With a pool, this returns once the figure is
    pickled and the files exist after wait(); the caller's figure is never modified.
    `exact=True` writes the figure as savefig would, without image reduction or PDF
    rasterising: use it for files that are read back as data.
    """
    rc = dict(matplotlib.rcParams) if isinstance(fig, Figure) else None
    blob = _pickle(fig, exact, kwargs)
    if _executor is None:
        _write(blob, rc, paths, exact, kwargs)
        return
    for future in [f for f in _pending if f.done()]:
        del _pending[future]
        future.result()
    while _pending and sum(_pending.values()) + len(blob) > MAX_PENDING_BYTES:
        done, _ = wait_futures(_pending, return_when=FIRST_COMPLETED)
        for future in done:
            del _pending[future]
            future.result()
    _pending[_executor.submit(_write, blob, rc, paths, exact, kwargs)] = len(blob)


def start(workers):
    """Open a pool of `workers` spawned processes for save_figure()."""
    global _executor
    # Each worker runs single-threaded: threads.configure(1) is the initializer. Unpickling it
    # imports only spoqc.core.threads, so it runs before the worker imports numpy, polars or
    # numba (the task function lives in this module, which is imported after it).
    _executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=threads.configure,
        initargs=(1,),
    )


def wait():
    """Block until every submitted figure is written; re-raise the first worker error."""
    while _pending:
        future = next(iter(_pending))
        del _pending[future]
        future.result()


def stop():
    """wait(), then shut the pool down."""
    global _executor
    try:
        wait()
    finally:
        _executor.shutdown()
        _executor = None


def abort():
    """On the error path: drop queued writes, let running ones finish, close the pool."""
    global _executor
    _pending.clear()
    _executor.shutdown(cancel_futures=True)
    _executor = None
