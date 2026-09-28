"""Per-pixel image metrics and the one driver that plots and flattens them.

Every kernel takes a 2D image and returns a tuple of 2D metric images.
`pixel_metric` runs a kernel once, plots each result and returns them flattened.
The window-histogram kernel (entropy, uniformity, homogeneity) is
`image_analysis._slidingwindow.texture_metrics`.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern

from ... import helperfuncs

ROW_CHUNKS_PER_THREAD = 4  # load balance for the row-chunked kernels


def pixel_metric(kernel, image, figure_path, imagedim, plots, **kernel_args):
    """Run `kernel(image, **kernel_args)` once; plot each 2D result and return them flattened
    (as views where the result is C-contiguous, so no second full-image copy is made).

    plots: one (suffix, title, legend_dict) per kernel result; legend_dict None means no legend.
    """
    results = kernel(image, **kernel_args)
    flat = []
    for result, (suffix, title, legend_dict) in zip(results, plots, strict=True):
        helperfuncs.plot_pixels(
            figure_path,
            result,
            imagedim,
            suffix,
            title,
            "hot",
            False,
            legend_dict is not None,
            legend_dict=legend_dict,
        )
        flat.append(result.ravel())
    return flat


def _row_chunked(func, image, halo, threads):
    """float64 func(image) computed on row chunks in `threads` threads; exact when an output row
    depends only on input rows within `halo` of it (image borders are kept as borders)."""
    n = image.shape[0]
    out = np.empty(image.shape, dtype=np.float64)
    bounds = np.linspace(0, n, threads * ROW_CHUNKS_PER_THREAD + 1).astype(int)

    def one(k):
        a, b = bounds[k], bounds[k + 1]
        lo, hi = max(a - halo, 0), min(b + halo, n)
        out[a:b] = func(image[lo:hi])[a - lo : b - lo]

    with ThreadPoolExecutor(threads) as executor:
        list(executor.map(one, range(len(bounds) - 1)))
    return out


def signal_noise_ratio(xy_intensities, background_intensity):
    """Log2 signal-noise ratio: is the pixel noise or true positive?"""
    return (np.log2((xy_intensities + 1) / background_intensity),)


def lbp(xy_intensities, n_points, radius, threads):
    """Local binary pattern ("uniform"); skimage reads samples up to ceil(radius) rows away."""
    return (
        _row_chunked(
            lambda rows: local_binary_pattern(rows, n_points, radius, method="uniform"),
            xy_intensities,
            math.ceil(radius) + 1,
            threads,
        ),
    )


def edge_strength(xy_intensities):
    """Log10 Sobel gradient magnitude."""
    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)
    grad_x = cv2.Sobel(xy_intensities, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x
    grad_y = cv2.Sobel(xy_intensities, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y
    return (np.log10(np.sqrt(grad_x**2 + grad_y**2) + 1),)


def energy(xy_intensities, window_size, threads):
    """Log10 local energy: gaussian-weighted mean of squared intensities (radius window_size)."""
    squared_image = xy_intensities.astype(np.float64) ** 2
    energy_image = _row_chunked(
        lambda rows: gaussian_filter(rows, sigma=1, radius=window_size),
        squared_image,
        window_size,
        threads,
    )
    return (np.log10(energy_image + 1),)


def relevance(xy_intensities, background_intensity):
    """1 where the blurred pixel is above the Otsu threshold, else 0."""
    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)
    blur = cv2.GaussianBlur(xy_intensities, (5, 5), 0)
    _, segmented_image = cv2.threshold(
        blur,
        background_intensity,
        xy_intensities.max(),
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return ((segmented_image > 0).astype(np.uint8),)

