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
in turn, so a worker keeps about one core busy. start(threads) gets the run's whole budget.
While the main thread computes, THREADS // THREADS_PER_FIGURE_WORKER background workers write
figures. While it is blocked in wait() or stop(), a drain pool of the remaining workers takes
figures too, so the figure pools use every core of the budget.
"""

import collections
import io
import multiprocessing
import os
import pickle
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
from matplotlib.collections import Collection, QuadMesh
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numba import njit, prange

from spoqc.core import threads

# Threads of the run's budget per figure worker (see the CPU budget note above).
THREADS_PER_FIGURE_WORKER = 4
# Collections with at least this many elements are rasterised in PDF output. Vector PDF costs
# ~0.1 ms per marker (44 s and 92 MB for the 410k-point doublet 3D scatter); rasterised it is
# 5 s and 0.3 MB, drawn at the savefig dpi like the PNG. Axes and text stay vector.
RASTERIZE_MIN_ELEMENTS = 10_000
# Float images with more samples than output pixels are reduced to this many samples per output
# pixel (per axis) in the pickled copy; the renderer antialiases the rest down to output pixels.
# Scalar images are coloured first and the colours averaged (premultiplied by alpha), which is
# what matplotlib's own antialiasing of a downsampled image does. Measured on the hqtr metric set
# at full-scale density (18 samples/px) against matplotlib's full-resolution render: 1.1-3.3/255
# mean at 2 samples/px, where averaging the values instead was up to 11/255 (LBP) even at 4.
# Integer, bool and label images, 'nearest'/'none' interpolation and norms other than a plain
# linear Normalize are never reduced.
IMAGE_SAMPLES_PER_PIXEL = 2
# Pickled figure bytes submitted but not yet written before save_figure blocks. Each blob is held
# about twice (here and in the pipe to the worker). A single larger figure is still submitted.
MAX_PENDING_BYTES = 2 * 1024**3

# mplot3d warns that set_rasterized on its collections "will be ignored", but Axes3D composites
# them rasterised all the same: the 410k-point doublet 3D PDF goes from 92 MB to 0.3 MB.
warnings.filterwarnings(
    "ignore", message="Rasterization of .*Path3DCollection", category=UserWarning
)

_executor = None  # background pool: figures written while the main thread computes
_drain = None  # the rest of the thread budget: used only while the main thread waits
_workers = {}  # pool -> worker count
# Figures are handed to a pool only when it has an idle worker, so none sits in a busy pool's
# queue when wait() opens the drain pool. Guarded by _state; pool callbacks run on the pools'
# manager threads.
_held = collections.deque()  # (the _write arguments, pickled bytes), not yet in a pool
_in_flight = {}  # future -> (pool, pickled bytes)
_errors = []  # worker exceptions, re-raised on the main thread
_draining = False
_state = threading.Condition(threading.RLock())


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


@njit(parallel=True, cache=True)
def _colour_blocks(data, mask, has_mask, vmin, vmax, clip, lut, n_colours, k):
    """Colour each sample like Normalize + Colormap.__call__, then average k x k blocks of
    premultiplied colours (the last block per axis may be smaller); returns uint8 RGBA."""
    rows, cols = data.shape
    out_rows, out_cols = (rows + k - 1) // k, (cols + k - 1) // k
    out = np.empty((out_rows, out_cols, 4), np.uint8)
    i_under, i_over, i_bad = n_colours, n_colours + 1, n_colours + 2
    for block_row in prange(out_rows):
        acc = np.zeros((out_cols, 4))
        count = np.zeros(out_cols)
        for r in range(block_row * k, min(rows, block_row * k + k)):
            for c in range(cols):
                x = data[r, c]
                if (has_mask and mask[r, c]) or np.isnan(x):
                    i = i_bad
                else:
                    v = 0.0 if vmin == vmax else (x - vmin) / (vmax - vmin)
                    if clip:
                        v = min(max(v, 0.0), 1.0)
                    v *= n_colours
                    if v == n_colours:
                        v = n_colours - 1
                    if v < 0:
                        i = i_under
                    elif v >= n_colours:
                        i = i_over
                    else:
                        i = int(v)
                j = c // k
                alpha = lut[i, 3]
                acc[j, 0] += lut[i, 0] * alpha
                acc[j, 1] += lut[i, 1] * alpha
                acc[j, 2] += lut[i, 2] * alpha
                acc[j, 3] += alpha
                count[j] += 1
        for j in range(out_cols):
            weight = acc[j, 3]
            for ch in range(3):
                out[block_row, j, ch] = int(acc[j, ch] / weight * 255 + 0.5) if weight > 0 else 0
            out[block_row, j, 3] = int(weight / count[j] * 255 + 0.5)
    return out


def _colour_average(image, k):
    """Block-averaged uint8 RGBA of a scalar image, through its own norm and colormap."""
    a, cmap, norm = image.get_array(), image.get_cmap(), image.norm
    lut = np.vstack([cmap(np.arange(cmap.N)), [cmap.get_under(), cmap.get_over(), cmap.get_bad()]])
    has_mask = np.ma.getmask(a) is not np.ma.nomask
    mask = np.ma.getmaskarray(a) if has_mask else np.zeros((1, 1), bool)
    return _colour_blocks(np.ma.getdata(a), mask, has_mask, float(norm.vmin), float(norm.vmax),
                          bool(norm.clip), lut, cmap.N, k)


def _reduced_images(fig, dpi):
    """{id(per-pixel array): reduced copy} for float images far denser than the output.

    Every per-pixel array of an image is reduced with the same blocks: the data (scalar data is
    coloured and colour-averaged, RGB(A) data is averaged; masked samples count as the bad
    colour or are excluded) and an array alpha. Extent (fixed by imshow), norm and clim are not
    per-pixel; the colorbar keeps using the image's norm and colormap. Only plain AxesImage:
    NonUniformImage/PcolorImage carry per-pixel coordinate arrays and are left alone.
    """
    reduced = {}
    for image in fig.findobj(lambda artist: type(artist) is AxesImage):
        a = image.get_array()
        if not np.issubdtype(a.dtype, np.floating) or image.get_interpolation() in (
            "nearest",
            "none",
        ):
            continue
        if a.ndim == 2 and type(image.norm) is not Normalize:
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
            reduced[id(a)] = _colour_average(image, k) if a.ndim == 2 else _block_mean(a, k)
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
    with _state:
        _raise_worker_error()
        while (_held or _in_flight) and _pending_bytes() + len(blob) > MAX_PENDING_BYTES:
            _state.wait()
            _raise_worker_error()
        _held.append(((blob, rc, paths, exact, kwargs), len(blob)))
        _dispatch()


def _pending_bytes():
    return sum(n for _, n in _held) + sum(n for _, n in _in_flight.values())


def _raise_worker_error():
    if _errors:
        error = _errors[0]
        _errors.clear()
        raise error


def _dispatch():
    """Hand held figures to pools with an idle worker (the drain pool only while draining)."""
    while _held:
        pools = [_executor] + ([_drain] if _draining and _drain is not None else [])
        busy = collections.Counter(pool for pool, _ in _in_flight.values())
        pool = next((p for p in pools if busy[p] < _workers[p]), None)
        if pool is None:
            return
        args, nbytes = _held.popleft()
        future = pool.submit(_write, *args)
        _in_flight[future] = (pool, nbytes)
        future.add_done_callback(_finished)


def _finished(future):
    with _state:
        del _in_flight[future]
        if not future.cancelled() and future.exception() is not None:
            _errors.append(future.exception())
        _dispatch()
        _state.notify_all()


def _pool(workers):
    # Each worker runs single-threaded: threads.configure(1) is the initializer. Unpickling it
    # imports only spoqc.core.threads, so it runs before the worker imports numpy, polars or
    # numba (the task function lives in this module, which is imported after it).
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=threads.configure,
        initargs=(1,),
    )
    _workers[pool] = workers
    return pool


def start(threads_budget):
    """Open the pools for save_figure(); `threads_budget` is the run's thread count.

    The drain pool's processes are started now, each with a no-op task from this module so it
    imports matplotlib and the rest here, and wait() does not pay that start-up; they idle,
    without figure data, until the main thread waits.
    """
    global _executor, _drain
    background = max(1, threads_budget // THREADS_PER_FIGURE_WORKER)
    _executor = _pool(background)
    if threads_budget > background:
        _drain = _pool(threads_budget - background)
        for _ in range(threads_budget - background):
            _drain.submit(_same, None)


def wait():
    """Block until every submitted figure is written; re-raise the first worker error.

    The main thread is idle meanwhile, so held figures also go to the drain pool, and the
    figure pools use the whole thread budget.
    """
    global _draining
    with _state:
        _draining = True
        try:
            _dispatch()
            while _held or _in_flight:
                _state.wait()
        finally:
            _draining = False
        _raise_worker_error()


def _close(cancel):
    global _executor, _drain
    pools = [p for p in (_executor, _drain) if p is not None]
    _executor = _drain = None
    for pool in pools:  # outside _state: shutdown joins the threads that run _finished
        pool.shutdown(cancel_futures=cancel)
        del _workers[pool]


def stop():
    """wait(), then shut the pools down."""
    try:
        wait()
    finally:
        _close(cancel=False)


def abort():
    """On the error path: drop queued writes, let running ones finish, close the pools."""
    with _state:
        _held.clear()
        _errors.clear()
    _close(cancel=True)
