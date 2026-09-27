"""Spatial-index queries returning (left, right) position pair arrays.

The index only proposes candidates by bounding box; the exact predicate of
the original per-object code decides, so results are identical to a full scan.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely


def polygons_containing(polygons: np.ndarray, points: np.ndarray, threads: int):
    """
    Finds every (polygon, point) pair where shapely.intersects(point, polygon),
    i.e. the point lies inside or on the boundary of the polygon.

    Parameters:
        polygons (np.ndarray): shapely polygons.
        points (np.ndarray): shapely points.
        threads (int): number of threads; each handles a contiguous chunk of polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: polygon positions and point positions,
        sorted by polygon position, then point position.
    """
    tree = shapely.STRtree(points)

    def pairs(chunk):
        polygon_pos, point_pos = tree.query(polygons[chunk])
        polygon_pos = chunk[polygon_pos]
        hit = shapely.intersects(points[point_pos], polygons[polygon_pos])
        return polygon_pos[hit], point_pos[hit]

    with ThreadPoolExecutor(threads) as executor:
        chunk_pairs = list(executor.map(pairs, np.array_split(np.arange(len(polygons)), threads)))
    polygon_pos = np.concatenate([p[0] for p in chunk_pairs])
    point_pos = np.concatenate([p[1] for p in chunk_pairs])
    order = np.lexsort((point_pos, polygon_pos))
    return polygon_pos[order], point_pos[order]
