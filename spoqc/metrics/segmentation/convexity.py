import numpy as np
import plotly.express as px
import shapely
import geopandas as gpd

from concurrent.futures import ThreadPoolExecutor

from ... import helperfuncs

def convexity_metrics(polygons: np.ndarray, threads: int) -> np.ndarray:
    """
    Measures the convexity of every polygon's exterior ring at once.

    For each ring (closing vertex included) the cross product of each pair of
    consecutive edges is computed; the metric is the larger of the number of
    positive and negative cross products divided by the number of vertices
    (range [0, 1]; 1 is fully convex).

    Parameters:
        polygons (np.ndarray): Array of shapely Polygons.
        threads (int): Number of threads; each handles a contiguous chunk of polygons.

    Returns:
        np.ndarray: float64 convexity metric per polygon.
    """
    with ThreadPoolExecutor(threads) as executor:
        return np.concatenate(list(executor.map(_ring_convexity, np.array_split(polygons, threads))))


def _ring_convexity(polygons: np.ndarray) -> np.ndarray:
    coords, ring = shapely.get_coordinates(shapely.get_exterior_ring(polygons), return_index=True)
    n = np.bincount(ring, minlength=len(polygons))
    if (n < 3).any():
        raise ValueError("A polygon must have at least three vertices.")

    start = np.repeat(np.cumsum(n) - n, n)
    local = np.arange(len(coords)) - start
    n_ring = n[ring]
    # Three consecutive points per vertex, wrapping around within each ring
    p2 = coords[start + (local + 1) % n_ring]
    p3 = coords[start + (local + 2) % n_ring]
    v1 = p2 - coords
    v2 = p3 - p2
    cross_products = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    pos = np.bincount(ring[cross_products > 0], minlength=len(polygons))
    neg = np.bincount(ring[cross_products < 0], minlength=len(polygons))

    # Just take the maximum amount of consistent angles
    return np.maximum(neg, pos) / n


# Function to assign nulcei to cell
def find_overlapping_nuclei(cells: gpd.GeoDataFrame, nucleus: gpd.GeoDataFrame, threads: int):
    """
    Returns the (cell position, nucleus position) pairs whose nucleus centroid
    intersects the cell, sorted by cell then nucleus position.
    """
    print("[NOTE] Find overlapping nuceli for cells")
    timer = helperfuncs.Timer()
    timer.start()
    cell_geometries = np.asarray(cells.geometry.values)
    nucleus_centroids = np.asarray(nucleus.geometry.centroid.values)
    tree = shapely.STRtree(nucleus_centroids)

    def overlapping_pairs(chunk):
        # Bounding-box candidates, then the exact predicate on each candidate pair
        cell_pos, nucleus_pos = tree.query(cell_geometries[chunk])
        cell_pos = chunk[cell_pos]
        hit = shapely.intersects(nucleus_centroids[nucleus_pos], cell_geometries[cell_pos])
        return cell_pos[hit], nucleus_pos[hit]

    chunks = np.array_split(np.arange(len(cell_geometries)), threads)
    with ThreadPoolExecutor(threads) as executor:
        pairs = list(executor.map(overlapping_pairs, chunks))
    cell_pos = np.concatenate([p[0] for p in pairs])
    nucleus_pos = np.concatenate([p[1] for p in pairs])
    order = np.lexsort((nucleus_pos, cell_pos))
    timer.stop()
    return cell_pos[order], nucleus_pos[order]


def calc_convexity(sdata, figure_path, threads):

    timer = helperfuncs.Timer()

    # Convexity calculation for cell polygon
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    cell_convexity_metric = convexity_metrics(np.asarray(sdata['cell_boundaries'].geometry.values), threads)
    timer.stop()

    sdata['table'].obs['convexity_cell'] = cell_convexity_metric > 0.5
    sdata['table'].obs['convexity_metric_cell'] = cell_convexity_metric

    # Find nuceli cell overlaps
    n_cells = len(sdata['cell_boundaries'])
    cell_pos, nucleus_pos = find_overlapping_nuclei(sdata['cell_boundaries'], sdata['nucleus_boundaries'], threads)
    bounds = np.searchsorted(cell_pos, np.arange(n_cells + 1))
    nucleus_labels = sdata['nucleus_boundaries'].index.values[nucleus_pos].tolist()
    sdata['table'].obs['nuclei_idxs'] = [nucleus_labels[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    # Convexity calcualteion for nuclei associated with cell
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    nuclei_convexity = convexity_metrics(np.asarray(sdata['nucleus_boundaries'].geometry.values)[nucleus_pos], threads)
    n_nuclei = np.diff(bounds)
    # Cells without nuclei get 0 (an integer column if no cell has a nucleus)
    nulcei_convexity_metric = np.zeros(n_cells, dtype=float if len(nucleus_pos) else int)
    min_convexity_metric = nulcei_convexity_metric.copy()
    # Since we might have more then one nuclei in a cell we take the mean.
    # Cells with the same nucleus count k form a (cells, k) matrix; numpy
    # reduces each row exactly as it reduces a 1-D array of length k.
    for k in np.unique(n_nuclei[n_nuclei > 0]):
        cells_k = np.flatnonzero(n_nuclei == k)
        convexities = nuclei_convexity[bounds[cells_k][:, None] + np.arange(k)]
        nulcei_convexity_metric[cells_k] = np.mean(convexities, axis=1)
        min_convexity_metric[cells_k] = np.min(convexities, axis=1)
    timer.stop()

    sdata['table'].obs['convexity_mean_nuceli'] = nulcei_convexity_metric
    sdata['table'].obs['convexity_min_nuceli'] = min_convexity_metric
    sdata['table'].obs['convexity_nuclei'] = min_convexity_metric > 0.5

    plotcats = [['convexity_cell', 'convexity_metric_cell'], 
                ['convexity_nuclei', 'convexity_mean_nuceli']]

    figures = []
    for i, cat in enumerate(plotcats):

        helperfuncs.plot_scatter(sdata['table'], figure_path, cat[0], None, 
                                cat[0], ['black', 'lightblue'], None)
        helperfuncs.plot_scatter_density(sdata['table'], figure_path, f'{cat[0]}_{cat[1]}', 
                                         cat[0], cat[1], ['black', 'lightblue'], '' \
                                         f'Density of {cat[1]}')
        
        fig = px.histogram(sdata['table'].obs, x=cat[1], nbins=100, width=800, height=800)
        fig.update_layout(
            title=f"Total distribution {cat[1]} for cells of all samples"
        )
        helperfuncs.apply_general_plotly_layout(fig, True)
        figures.append(fig)
        fig.write_image(f"{figure_path}/histogram_{cat[1]}.png", scale=3)
        fig.write_image(f"{figure_path}/histogram_{cat[1]}.pdf", scale=3)

    with open(f'{figure_path}/convexity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))