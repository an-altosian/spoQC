"""Whole-image raster primitives: load a modality's intensity image, read a per-pixel parquet column,
dilate a binary mask with a disk, and box its 8-connected components.

Every function is exact (bit-identical to the skimage/dask code it replaces) and splits its work
over `threads` (row blocks, parquet row groups or dask chunks); the numba kernels use the numba
pool, which cli.run sizes to CONST.THREADS.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import dask
import numpy as np
import pyarrow.parquet as pq
from dask.utils import natural_sort_key
from numba import njit, prange
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import disk

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def read_pixel_column(path, column, threads):
    """Read one column of a per-pixel parquet (a single file, or a dask directory of part.N.parquet)
    into a 1-D numpy array in the row order dask.dataframe.read_parquet returns.

    Row groups are decoded in parallel into one preallocated array.
    """
    if os.path.isdir(path):
        # dask orders its part files naturally (part.2 before part.10), not lexicographically.
        names = sorted(
            (n for n in os.listdir(path) if n.endswith(".parquet")),
            key=natural_sort_key,
        )
        files = [os.path.join(path, n) for n in names]
    else:
        files = [path]
    pieces, sizes, dtype = [], [], None
    for file in files:
        meta = pq.ParquetFile(file)
        dtype = np.dtype(meta.schema_arrow.field(column).type.to_pandas_dtype())
        for group in range(meta.metadata.num_row_groups):
            pieces.append((file, group))
            sizes.append(meta.metadata.row_group(group).num_rows)
    starts = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    out = np.empty(int(starts[-1]), dtype=dtype)

    def read(i):
        file, group = pieces[i]
        values = (
            pq.ParquetFile(file)
            .read_row_group(group, columns=[column], use_threads=False)
            .column(0)
        )
        out[starts[i] : starts[i + 1]] = values.to_numpy()

    with ThreadPoolExecutor(threads) as executor:
        list(executor.map(read, range(len(pieces))))
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
        return read_pixel_column(density_file, "transcript_density", threads).reshape(
            dim_x, dim_y
        )
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
