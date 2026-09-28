import numpy as np

from ... import helperfuncs
from ...core.threads import map_slices

# elementwise work runs on slices of this many pixels, one per thread task: 2 MB float64
# temporaries, measured 2x faster than 4M-pixel slices on a 64 Mpx image
ROWS_PER_TASK = 1 << 18
# Rows per part file of the prior parquets. origin/dev wrote 10,000-row parts (91,295 files per
# prior on a 913 Mpx slide); measured on a 16 Mpx crop on NFS: 1,600 files 2.4 s -> 16 files 0.35 s
# to write, 3.0 s -> 0.22 s to read, while a part stays 24 MB in memory.
PART_ROWS = 1_000_000


def calc_prob_pixel_stuff_v2(values, figure_path, thresh, std, tail, col, threads):
    """
    The prior of every pixel value in `values` (a 1-D float64 array): a Gaussian density at
    `thresh`, set to its peak on the `tail` side, subtracted from the peak (d_{col}), then
    min-max scaled to [0, 1] as dask_ml's MinMaxScaler scales a column (norm_p_{col}).

    Returns (norm_p, part_columns): the norm_p_{col} array, and part_columns(start, stop), the
    columns {col, norm_p_{col}} (what hqtr clustering and the per-cell analysis read) of pixels
    start..stop-1 for core.parquet.write_parts.
    Elementwise work runs on slices on `threads` threads; every value is computed exactly as
    it is for the whole array.
    """

    if std <= 0:
        raise ValueError("std must be > 0")

    inv_std = 1.0 / std
    norm_const = inv_std / np.sqrt(2.0 * np.pi)

    def _part(x):
        # Gaussian PDF centered at `thresh`
        z = (x - thresh) * inv_std
        pdf = norm_const * np.exp(-0.5 * z * z)

        # Tail overwrite to norm_const (then we'll invert below).
        # You have to use norm_const because it is ultimately where the peak height of the Guassian is.
        # Do not use np.max(pdf) here because each slice has its own distribution.
        # Thus the constant here is given by the Gaussian shape.
        if tail == "left":
            pdf = np.where(x < thresh, norm_const, pdf)
        elif tail == "right":
            pdf = np.where(x > thresh, norm_const, pdf)
        # else: no tail tweak

        # Because the tailing sets values to norm_const we have substract norm_const
        # to create 0 which is the extreme case of the worst probability.
        # Keep in mind that you deal with pdfs here not probabilities.
        return norm_const - pdf

    def bounds(s):
        d = _part(values[s])
        return np.nanmin(d), np.nanmax(d)

    # Min-Max normalize as dask_ml's MinMaxScaler (feature_range (0, 1)): min and max skip NaN,
    # a zero range scales by 1, and x * scale + min_.
    slice_bounds = map_slices(bounds, len(values), ROWS_PER_TASK, threads)
    data_min = np.nanmin([b[0] for b in slice_bounds])
    data_max = np.nanmax([b[1] for b in slice_bounds])
    data_range = data_max - data_min
    scale = (1 - 0) / (data_range if data_range != 0 else np.float64(1))
    min_ = 0 - data_min * scale

    norm_p = np.empty(len(values), dtype=np.float64)

    def scale_slice(s):
        norm_p[s] = _part(values[s]) * scale + min_

    map_slices(scale_slice, len(values), ROWS_PER_TASK, threads)

    def part_columns(start, stop):
        x = values[start:stop]
        # d_{col} has no reader, so it is no longer written
        return {col: x, f"norm_p_{col}": norm_p[start:stop]}

    helperfuncs.plot_histogram_for_array(
        values,
        100,
        figure_path,
        f"{col}: t={np.round(thresh, 3)} with {1} x {np.round(std, 3)} std",
        f"{col}_prior",
        t=thresh,
        std=std,
        nstds=1,
    )

    return norm_p, part_columns
