import numba
import numpy as np
import plotly.express as px

from ... import helperfuncs
from spoqc.core.figures import save_figure

def turn_into_uint8(arr):
    # normalize to 0–1 if needed
    if int(arr.min()) != 0 or int(arr.max()) != 1:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    # scale to 0–255 and convert to uint8
    uint8_arr = (arr * 255).astype(np.uint8)
    return uint8_arr

def pixel_intensity_qc(figure_path, intensities, background_intensity, hist, bin_edges, dim_x, dim_y, imagedim):

    timer = helperfuncs.Timer()

    figures = []

    # When you plot a histogram via plotly, it stores all the orginal data in the json file 
    # and makes the bins and counts on the javascript side. 
    # Thus the plot get quite large.
    # Use therefore the precomupted histogram data from numpy.
    bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    print("[NOTE] Barplot")
    timer.start()
    fig = px.bar(x=bins, y=hist, labels={'x':'intensity', 'y':'count'})
    fig.update_layout(
        title=f"Total distribution intensity with backkground intensity {background_intensity}"
    )
    timer.stop()
    helperfuncs.apply_general_plotly_layout(fig, True)
    figures.append(fig)
    save_figure(fig, f"{figure_path}/histogram_intensity.png", f"{figure_path}/histogram_intensity.pdf", scale=3)

    with open(f'{figure_path}/histogram_intensity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    signal_noise_ratio_log2fc = np.log2( (intensities + 1) / background_intensity )

    helperfuncs.plot_pixels(
        figure_path,
        np.array(signal_noise_ratio_log2fc).reshape(dim_x, dim_y),
        imagedim,
        'snr', 
        'Log2 Signal-Noise-Ratio', 
        'hot',
        False,
        False
    )
    
    return signal_noise_ratio_log2fc


UINT16_VALUES = 1 << 16


@numba.njit(parallel=True, cache=True)
def _uint16_counts(image, n_threads):
    """Exact np.bincount(image.ravel(), minlength=65536) of a 2-D uint16 image, one partial count per thread."""
    n_rows, n_cols = image.shape
    partial = np.zeros((n_threads, UINT16_VALUES), dtype=np.int64)
    for t in numba.prange(n_threads):
        for i in range(t * n_rows // n_threads, (t + 1) * n_rows // n_threads):
            for j in range(n_cols):
                partial[t, image[i, j]] += 1
    counts = np.zeros(UINT16_VALUES, dtype=np.int64)
    for t in range(n_threads):
        counts += partial[t]
    return counts


def estimate_background_intensity(image, nbins=100):
    """Background intensity of a 2-D uint16 image: the centre of the most populated of `nbins`
    equal bins over [min, max]. Returns (background, hist, bin_edges).

    One pass over the pixels counts every uint16 value; the histogram is then binned from those
    counts with the same np.histogram(bins=nbins, range=(min, max)) arithmetic that
    da.histogram applied per chunk, so hist and bin_edges are exactly origin/dev's.
    """
    if image.dtype != np.uint16:
        raise TypeError(f"estimate_background_intensity needs a uint16 image, got {image.dtype}")
    counts = _uint16_counts(image, numba.get_num_threads())
    values = np.flatnonzero(counts)
    vmin, vmax = values[0], values[-1]
    if vmin == vmax:
        vmax = vmin + 1.0
    range_ = (float(vmin), float(vmax))
    hist = np.histogram(values.astype(np.uint16), bins=nbins, range=range_, weights=counts[values])[0]
    bin_edges = np.linspace(range_[0], range_[1], num=nbins + 1)

    max_bin_idx = int(np.argmax(hist))
    # center of the winning bin
    background = np.round((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) * 0.5, 3)
    return background, hist, bin_edges
