import numpy as np

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc.image_analysis._slidingwindow import sliding_window_padded
from ... import helperfuncs

@njit
def occurrence_probability(
    x: IntArray,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Relative occurrence of each non-negative integer"""
    # assert x.size > 0
    counts = np.bincount(x.ravel())
    return counts / x.size

@njit
def kl_divergence_uniform(x: IntArray) -> np.number:
    """Kullback-Leibler divergence against a uniform distribution"""
    p = occurrence_probability(x)
    # removing zeros is faster than using nansum
    p = p[p > 0]

    # uniform distribution: probability for each element
    # if there are less observations than potential levels truncate
    q = max(1 / x.size, 1 / (np.iinfo(x.dtype).max + 1))

    return ( -(p * np.log(p / q)) ).sum()

def uniformity_from_entropy(entropy_image, window_size, dtype):
    """Derive the uniformity map from an already-computed entropy map.

    kl_divergence_uniform(x) = -(p * log(p / q)).sum()
                             = -(p * log p).sum() + log(q) * p.sum()
                             = entropy(x) + log(q)          [p.sum() == 1]

    and pixel_uniformity returns the negation of the kernel, so

        uniformity = -(entropy + log q)

    q = max(1 / window_area, 1 / (iinfo(dtype).max + 1)) is constant for a fixed
    window size and dtype, so the second sliding-window pass over the whole image
    is redundant. Verified to float32 precision against the real kernels for
    uint8/uint16 and window radii 2 and 5 -- see tests/test_redundant_work.py.
    """
    n = window_size * window_size
    q = max(1.0 / n, 1.0 / (np.iinfo(dtype).max + 1))
    return -(np.asarray(entropy_image) + np.log(q))


def pixel_uniformity(figure_path, img, window_size, imagedim, mode="reflect",
                     entropy_image=None):
    timer = helperfuncs.Timer()

    # numba.set_threads(threads)
    radius = (window_size - 1) // 2

    timer.start()
    if entropy_image is not None:
        # Derived in one vectorised pass instead of a second full sliding-window
        # sweep; exactly equal, see uniformity_from_entropy.
        print("... Derived from entropy (no second sliding-window pass)")
        uniformity_image = uniformity_from_entropy(
            np.asarray(entropy_image).reshape(img.shape), window_size, img.dtype
        )
    else:
        print(f"... Parallel processing")
        uniformity_image = -sliding_window_padded(kl_divergence_uniform, img, radius, mode=mode)
    timer.stop()
    
    helperfuncs.plot_pixels(
        figure_path,
        uniformity_image,
        imagedim,
        "uniformity",
        "Pixel Uniformity",
        "hot",
        False,
        False,
    )

    return uniformity_image.flatten()