"""Whole-image raster primitives: load a modality's intensity image, read a per-pixel parquet column,
dilate a binary mask with a disk, and box its 8-connected components.

Every function is exact (bit-identical to the skimage/dask code it replaces; for dilate_disk while
the disk radius is below about the image size, see there) and splits its work
over `threads` (row blocks or dask chunks; parquet reads use the pyarrow pool); the numba kernels use the numba
pool, which spoqc.core.threads.configure sizes (NUMBA_NUM_THREADS) to the run's thread budget.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import dask
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from dask.utils import natural_sort_key
from numba import njit, prange
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import disk

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
# read_pixel_columns scans at most this many files per Arrow dataset. A dataset keeps ~24 KB per
# fragment it has scanned until it is dropped: one dataset over hqtr's 91,296 parts peaked 2.13 GB
# above its output, datasets of 2,048 parts 0.07 GB (and 13.2 s -> 12.7 s).
_SCAN_FILES_PER_DATASET = 2048
_SCAN_BATCH_ROWS = 131_072  # Arrow's default batch size, named so tests can shrink it
# pre_buffer (Arrow's default) caches whole column chunks ahead of decoding: on hqcr's single
# 913M-row file it peaked 1.57 GB above the output; without it, 0.0 GB and 2.66 s -> 2.01 s.
_SCAN_FORMAT = ds.ParquetFileFormat(
    default_fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False)
)


def _scan_batches(files, columns):
    """The files' batches in order, _SCAN_FILES_PER_DATASET files per Arrow dataset."""
    for first in range(0, len(files), _SCAN_FILES_PER_DATASET):
        dataset = ds.dataset(files[first : first + _SCAN_FILES_PER_DATASET], format=_SCAN_FORMAT)
        yield from dataset.scanner(
            columns=columns, use_threads=True, batch_size=_SCAN_BATCH_ROWS
        ).scan_batches()


def read_pixel_columns(path, columns, n_rows):
    """Read columns of a per-pixel parquet (a single file, or a dask directory of part.N.parquet)
    into 1-D numpy arrays of n_rows values each, in the row order dask.dataframe.read_parquet
    returns, as {column: array}.

    Every file is opened and decoded once for all `columns`, by Arrow's C++ dataset scanner on the
    pyarrow CPU pool (sized by spoqc.core.threads.configure). spoQC's hqtr masks are 91,296
    ten-thousand-row parts per directory, and at that size the cost is per file, not per byte:
    opening each file from Python (a serial metadata pass, then again per column in a thread
    pool) held the GIL for most of it and kept the pool near 1.9 of 4 cores. ScanBatches yields
    batches in dataset (file list) order and row order within each file (arrow/dataset/scanner.h);
    the files are scanned in consecutive groups so the per-fragment state Arrow keeps is released.

    Nulls come out as dask 2026.1 returns them: NaN in a float column, and a null anywhere in an
    integer column promotes the whole column to float64 with NaN. Raises when there are no part
    files, when a part's type for a column differs from the first part's, when a non-numeric
    column holds nulls (dask would return an object array), or when the files do not hold
    exactly n_rows rows.
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
    types = {column: pq.read_schema(files[0]).field(column).type for column in columns}
    out = {
        column: np.empty(n_rows, dtype=np.dtype(type_.to_pandas_dtype()))
        for column, type_ in types.items()
    }
    null_rows = {column: [] for column in columns}  # integer columns: rows to set NaN at the end
    start, checked = 0, None
    for tagged in _scan_batches(files, columns):
        if tagged.fragment.path != checked:  # the first batch of each file
            checked = tagged.fragment.path
            schema = tagged.fragment.physical_schema
            for column, type_ in types.items():
                if schema.field(column).type != type_:
                    raise ValueError(
                        f"parts of {path} disagree on the type of {column}: "
                        f"{type_} in {files[0]}, {schema.field(column).type} in {checked}"
                    )
        batch = tagged.record_batch
        stop = start + batch.num_rows
        if stop > n_rows:
            raise ValueError(f"{path} holds more than the {n_rows} rows expected")
        for column in columns:
            values = batch.column(column)
            if values.null_count:
                kind = out[column].dtype.kind
                if kind in "iu":
                    null_rows[column].append(
                        start + np.flatnonzero(values.is_null().to_numpy(zero_copy_only=False))
                    )
                    values = values.fill_null(0)
                elif kind != "f":
                    raise ValueError(
                        f"{values.null_count} nulls in the {out[column].dtype} column {column} of {path}"
                    )
            out[column][start:stop] = values.to_numpy(zero_copy_only=False)  # float nulls: NaN
        start = stop
    if start != n_rows:
        raise ValueError(f"{path} holds {start} rows, {n_rows} expected")
    for column, rows in null_rows.items():
        if rows:  # pandas' integer-with-NaN promotion, as dask applies it
            out[column] = out[column].astype(np.float64)
            out[column][np.concatenate(rows)] = np.nan
    return out


def load_intensity_image(
    sdata,
    spoqc_tmp_folder,
    modality,
    image_type,
    resolution,
    dim_x,
    dim_y,
    threads,
    staining=None,
):
    """The modality's intensity image as a (dim_x, dim_y) array, flipped upside down (row 0 = top).

    hqtr: the transcript density image that structure_analysis saved as
    `{spoqc_tmp_folder}/metrices/hqtr/transcript_density_output_hqtr.parquet` (already flipped), read
    back rather than regenerated. Other modalities: channel `staining` (0 when None) of the image,
    loading only that channel.
    """
    if modality == "hqtr":
        density_file = (
            f"{spoqc_tmp_folder}/metrices/hqtr/transcript_density_output_hqtr.parquet"
        )
        return read_pixel_columns(density_file, ["transcript_density"], dim_x * dim_y)[
            "transcript_density"
        ].reshape(dim_x, dim_y)
    channel = int(staining) if staining else 0
    with dask.config.set(scheduler="threads", num_workers=threads):
        image = sdata[image_type][resolution].image[channel].values
    return np.flipud(image)


@njit(parallel=True, fastmath=False)
def _row_gaps(image, cap, gaps):
    """gaps[y, x] = min(cap, distance along row y to its nearest nonzero pixel); returns the
    number of pixels that are neither 0 nor 1."""
    n_rows, n_cols = image.shape
    not_binary = 0
    for y in prange(n_rows):
        gap = cap
        for x in range(n_cols):
            value = image[y, x]
            if value != 0:
                gap = 0
                if value != 1:
                    not_binary += 1
            elif gap < cap:
                gap += 1
            gaps[y, x] = gap
        gap = cap
        for x in range(n_cols - 1, -1, -1):
            if image[y, x] != 0:
                gap = 0
            elif gap < cap:
                gap += 1
            if gap < gaps[y, x]:
                gaps[y, x] = gap
    return not_binary


@njit(parallel=True, fastmath=False)
def _dilate_rows(gaps, half_widths, out):
    """out[y, x] = 1 if some footprint row k (row offset k - r, half width half_widths[k]) reaches
    a nonzero pixel, i.e. gaps[y + k - r, x] <= half_widths[k]; else 0."""
    n_rows, n_cols = gaps.shape
    radius = (len(half_widths) - 1) // 2
    for y in prange(n_rows):
        hit = np.zeros(n_cols, dtype=np.uint8)
        for k in range(len(half_widths)):
            source = y + k - radius
            if source < 0 or source >= n_rows:
                continue
            width = half_widths[k]
            for x in range(n_cols):
                hit[x] |= gaps[source, x] <= width
        for x in range(n_cols):
            out[y, x] = hit[x]


def dilate_disk(image, radius):
    """skimage.morphology.dilation(image, disk(radius)) for a 0/1 image, as a row-parallel kernel.

    Bit-identical to skimage for radii below about the image size (the pipeline uses 10 and 1 on
    full images). Beyond that skimage/scipy return wrong results, and this kernel still returns the
    correct dilation.

    The disk is a stack of centred row segments, so a pixel is set when, for some row offset, the
    nearest nonzero pixel along that row is within the segment's half width. Pixels outside the
    image never change a maximum (skimage's 'reflect' border only repeats pixels the footprint
    already covers), so they are skipped. Raises ValueError when image is not 0/1.
    """
    footprint = disk(radius).astype(bool)
    half_widths = ((footprint.sum(axis=1) - 1) // 2).astype(np.int64)
    columns = np.arange(-radius, radius + 1)
    assert np.array_equal(
        footprint, np.abs(columns)[None, :] <= half_widths[:, None]
    ), "disk rows are not centred segments"
    assert radius + 1 <= np.iinfo(np.uint8).max

    gaps = np.empty(image.shape, dtype=np.uint8)
    not_binary = _row_gaps(image, radius + 1, gaps)
    if not_binary:
        raise ValueError(
            f"dilate_disk needs a 0/1 image; {not_binary} pixels are neither"
        )
    out = np.empty_like(image)
    _dilate_rows(gaps, half_widths, out)
    return out


def component_boxes(image, threads):
    """Bounding boxes of the 8-connected nonzero components of `image`, as an (n, 4) int64 array of
    [min_row, min_col, max_row, max_col) rows in skimage.measure.label order (scan order of each
    component's first pixel), i.e. the region.bbox values of regionprops(label(image)).

    Row blocks are labelled in parallel; pieces that touch across a block seam are joined with a
    sparse-graph connected-components pass, and a component takes the rank of its first piece.
    """
    n_rows, n_cols = image.shape
    edges = np.unique(np.linspace(0, n_rows, min(threads, n_rows) + 1).astype(np.int64))

    def label_block(i):
        top, bottom = edges[i], edges[i + 1]
        labels, _ = ndi.label(image[top:bottom], structure=_EIGHT_CONNECTED)
        slices = ndi.find_objects(labels)
        boxes = np.array(
            [
                [s[0].start + top, s[1].start, s[0].stop + top, s[1].stop]
                for s in slices
            ],
            dtype=np.int64,
        ).reshape(-1, 4)
        return boxes, labels[0].copy(), labels[-1].copy()

    with ThreadPoolExecutor(threads) as executor:
        blocks = list(executor.map(label_block, range(len(edges) - 1)))
    if not blocks:
        return np.empty((0, 4), dtype=np.int64)

    boxes = np.concatenate([b[0] for b in blocks])
    n_pieces = len(boxes)
    if n_pieces == 0:
        return boxes
    # Global piece id = block offset + local label - 1; local labels are in scan order within a
    # block and blocks are in row order, so piece ids are in global scan order of first pixels.
    offsets = np.concatenate([[0], np.cumsum([len(b[0]) for b in blocks])])
    sources, targets = [], []
    for i in range(1, len(blocks)):
        above = blocks[i - 1][2].astype(np.int64)
        below = blocks[i][1].astype(np.int64)
        for shift in (-1, 0, 1):  # 8-connectivity: above[x] touches below[x + shift]
            a = above[max(0, -shift) : n_cols - max(0, shift)]
            b = below[max(0, shift) : n_cols - max(0, -shift)]
            touching = (a > 0) & (b > 0)
            sources.append(a[touching] - 1 + offsets[i - 1])
            targets.append(b[touching] - 1 + offsets[i])
    sources = np.concatenate(sources) if sources else np.empty(0, dtype=np.int64)
    targets = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(sources), dtype=np.int8), (sources, targets)),
        shape=(n_pieces, n_pieces),
    )
    n_components, component = connected_components(graph, directed=False)

    first_piece = np.full(n_components, n_pieces, dtype=np.int64)
    np.minimum.at(first_piece, component, np.arange(n_pieces))
    merged = np.empty((n_components, 4), dtype=np.int64)
    merged[:, :2] = np.iinfo(np.int64).max
    merged[:, 2:] = np.iinfo(np.int64).min
    np.minimum.at(merged[:, 0], component, boxes[:, 0])
    np.minimum.at(merged[:, 1], component, boxes[:, 1])
    np.maximum.at(merged[:, 2], component, boxes[:, 2])
    np.maximum.at(merged[:, 3], component, boxes[:, 3])
    return merged[np.argsort(first_piece)]
