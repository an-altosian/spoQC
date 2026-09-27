from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ... import helperfuncs
from ...core import spatial


def get_border_scores(points, radius, step, threads):
    """
    Border score of every point: over rotations by multiples of `step` degrees, the largest
    |log2((1 + #neighbours right of the point) / (1 + #neighbours left of it))|, counting the
    points within `radius` (the point itself sits at 0 and counts on neither side).
    """
    n_points = len(points)
    point_pos, neighbour_pos = spatial.pairs_within(points, points, radius, threads)
    diffs = points[neighbour_pos] - points[point_pos]  # (pairs, 2)

    angles = np.radians(np.arange(0, 360, step))
    rotation_matrices = np.stack([
        np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for a in angles
    ])

    def rotation_scores(rotation_matrix):
        x_coords = (diffs @ rotation_matrix)[:, 0]
        # Add one to both sides to avoid inf; both sides are treated equally.
        num_left = np.bincount(point_pos[x_coords > 0], minlength=n_points) + 1
        num_right = np.bincount(point_pos[x_coords < 0], minlength=n_points) + 1
        # Only the magnitude matters, not the direction. The few distinct ratios go through
        # the scalar log2, as the per-point original did.
        ratios, ratio_idx = np.unique(num_left / num_right, return_inverse=True)
        return np.array([abs(np.log2(ratio)) for ratio in ratios])[ratio_idx]

    # numpy releases the GIL in the matmul, comparisons and bincounts: one rotation per thread.
    with ThreadPoolExecutor(threads) as executor:
        return np.max(list(executor.map(rotation_scores, rotation_matrices)), axis=0)


def define_border_cells(sdata: dict, figure_path: str, thresh: float,
                        radius: float, stepsize: float, threads: int) -> None:
    
    """
    Identifies and annotates border cells in spatial transcriptomics data.
    
    Args:
        sdata (dict): A dictionary containing spatial transcriptomics data. 
                      Assumes 'table' key includes an `obsm` attribute with spatial coordinates.
        figure_path (str): Path to save the visualization of border cells.
        thresh (float): Threshold for classifying cells as border cells based on scores.
        radius (float): Radius used for calculating border scores.
        stepsize (float): Step size used in the border score calculation.
        threads (int): Number of threads to use for parallel processing.

    Returns:
        None: Modifies the `sdata` object in-place by adding:
              - `border_cell`: A boolean column in `obs` indicating whether each cell is a border cell.
              - `border_scores`: A column in `obs` with the border scores for each cell.
              Additionally, saves a scatter plot visualization to the specified path.

    Notes:
        - The border scores are computed using `get_border_scores`.
        - A scatter plot of the border cells is generated using `helperfuncs.plot_scatter`.
    """

    border_scores = get_border_scores(np.ascontiguousarray(sdata['table'].obsm['spatial'][:, :2]), radius, stepsize, threads)

    border_cells = border_scores >= thresh

    sdata['table'].obs['border_cell'] = border_cells
    sdata['table'].obs['border_scores'] = border_scores

    # Plot for border cells
    helperfuncs.plot_scatter(sdata['table'], figure_path, 'border_cell', None,
                             'border_cell', ['lightblue', 'red'], 'Border Cells')