"""Bin point coordinates onto the pixel grid of a chosen pyramid level.

Transcript and cell coordinates live in the 'global' coordinate system, which
spatialdata pins to the FULL-resolution (scale0) pixel grid. The image being
binned onto, however, is whichever pyramid level was selected, and a lower level
covers that same world extent with fewer pixels.

The previous code built a Python tuple per grid cell over the *global* extent and
aligned counts to it with a MultiIndex, then reshaped the result to the
*selected* level's dims. Those two agree only at scale0; at scale2 the array is
16x too large and the reshape raises. Deriving the scale from
(pixel count / world extent) is correct at every level and reduces to the
identity at scale0.

It is also far cheaper: the per-pixel Python tuple and the
MultiIndex.get_indexer over the whole grid both disappear. The reductions
themselves are left as pandas groupby / numpy ufunc.at so that every output value
is bit-identical to the code this replaces.
"""

import numpy as np
import pandas as pd


def flat_pixel_index(x, y, imagedim, dim_x, dim_y):
    """Map world (global) coordinates to a flat index into a dim_x by dim_y grid.

    dim_x is the number of rows (image y axis), dim_y the number of columns
    (image x axis), matching how the callers unpack the image dims. The flat
    layout is row-major (row * dim_y + col), which is what a subsequent
    reshape(dim_x, dim_y) expects -- the same ordering the previous
    y-outer/x-inner tuple grid produced.
    """
    world_w = float(imagedim.bb_xmax) - float(imagedim.bb_xmin)
    world_h = float(imagedim.bb_ymax) - float(imagedim.bb_ymin)
    if world_w <= 0 or world_h <= 0:
        raise ValueError(f"degenerate image extent: {world_w} x {world_h}")

    # pixels per world unit; exactly 1.0 at scale0
    sx = dim_y / world_w
    sy = dim_x / world_h

    col = np.floor((np.asarray(x, dtype=np.float64) - float(imagedim.bb_xmin)) * sx)
    row = np.floor((np.asarray(y, dtype=np.float64) - float(imagedim.bb_ymin)) * sy)
    np.clip(col, 0, dim_y - 1, out=col)
    np.clip(row, 0, dim_x - 1, out=row)
    return (row.astype(np.int64) * dim_y + col.astype(np.int64))


def bin_counts(x, y, imagedim, dim_x, dim_y):
    """Per-pixel point counts as a dim_x by dim_y array."""
    flat = flat_pixel_index(x, y, imagedim, dim_x, dim_y)
    return np.bincount(flat, minlength=dim_x * dim_y).reshape(dim_x, dim_y)


def bin_reduce(x, y, values, imagedim, dim_x, dim_y, how="mean"):
    """Per-pixel reduction of `values`, as a dim_x by dim_y array.

    how='mean' averages the points falling in each pixel; how='max' takes the
    maximum. Empty pixels are 0 either way, as in the previous implementation.

    Both are BIT-IDENTICAL to the code this replaces, which is the acceptance
    bar. The expensive part of the old path was never the reduction -- it was
    building one Python tuple per grid cell and aligning through
    pd.MultiIndex.from_tuples. So the reduction is still pandas' own groupby,
    just keyed on the flat pixel index instead of an (x, y) MultiIndex: the same
    values are grouped together in the same order, so the accumulation order and
    therefore the last bit of every float is unchanged.

    An np.bincount-based mean is ~4x faster again but differs by up to 3 ULP,
    because it accumulates in a different order. That is not acceptable here.
    """
    flat = flat_pixel_index(x, y, imagedim, dim_x, dim_y)
    n = dim_x * dim_y
    vals = np.asarray(values, dtype=np.float64)

    if how == "max":
        # np.maximum.at selects an existing value, so it is exact by construction.
        out = np.zeros(n, dtype=np.float64)
        np.maximum.at(out, flat, vals)
        return out.reshape(dim_x, dim_y)
    if how == "mean":
        grouped = pd.Series(vals).groupby(flat).mean()
        out = np.zeros(n, dtype=np.float64)
        out[grouped.index.to_numpy()] = grouped.to_numpy()
        return out.reshape(dim_x, dim_y)
    raise ValueError(f"unknown reduction {how!r}; expected mean or max")
