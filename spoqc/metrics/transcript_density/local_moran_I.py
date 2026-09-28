import numpy as np
import polars as pl
import pandas as pd
import concurrent.futures
import scipy.sparse as sp

from libpysal.weights import KNN

from ... import helperfuncs
from ...core import groupreduce, spatial, transcripts

# Cells within this distance of a cell form its Moran's I neighbourhood.
NEIGHBOURHOOD_RADIUS = 100
# Transcripts outside cells take the local Moran's I of the nearest in-cell transcript
# of the same gene, if one lies closer than this.
OUTSIDE_NEIGHBOUR_DISTANCE = 100.0


def fill_outside_from_nearest_inside(coords, feat, local_I, outside_mask, threads):
    """
    Sets local_I of every outside transcript to that of the nearest inside transcript of the
    same feature closer than OUTSIDE_NEIGHBOUR_DISTANCE, else to 0.0. Modifies local_I in place.

    Transcripts are grouped by feature once (a stable sort keeps each group's positions
    ascending, as np.flatnonzero gave them); features then run on `threads` threads, one
    nearest query each (KD-tree builds and queries release the GIL). Features only read
    inside values and write disjoint outside positions, so the order does not matter.
    """
    codes, features = pd.factorize(feat)
    order = np.argsort(codes, kind="stable")
    offsets = groupreduce.group_offsets(codes[order], len(features))

    def nearest_inside_values(members):
        out_idx = members[outside_mask[members]]
        in_idx = members[~outside_mask[members]]
        values = np.zeros(out_idx.size, dtype=local_I.dtype)
        if out_idx.size and in_idx.size:
            nn, dists = spatial.nearest(coords[out_idx], coords[in_idx], 1, distance_upper_bound=OUTSIDE_NEIGHBOUR_DISTANCE)
            has_neighbor = np.isfinite(dists) & (nn < in_idx.size)
            values[has_neighbor] = local_I[in_idx[nn[has_neighbor]]]
        return out_idx, values

    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        for out_idx, values in executor.map(nearest_inside_values, groupreduce.ragged_lists(order, offsets)):
            local_I[out_idx] = values
    return local_I


# Vectorized Moran-I for all genes in a neighborhood
def moran_I_all_genes(X_dense: np.ndarray, w) -> np.ndarray:
    """
    X_dense: (n, num_genes) float array for selected cells
    w: libpysal weights object
    returns: (num_genes,) Moran's I per gene
    """
    n = X_dense.shape[0]
    if n < 3:
        return np.full((X_dense.shape[1],), -1.0, dtype=np.float32)

    # ensure sparse CSR for W
    # NOTE: w.sparse is typically CSR; w.transform='r' row-standardizes
    w.transform = "r"
    weights = w.sparse  # scipy sparse

    # S0 for row-standardized weights is just sum(W)
    row_standardized_weights = weights.sum()
    if row_standardized_weights == 0:
        return np.full((X_dense.shape[1],), -1.0, dtype=np.float32)

    # center (do NOT standardize by std unless you want "z-scores"; Moran uses mean-centering)
    z = X_dense - X_dense.mean(axis=0, keepdims=True)

    # sparse matmul releases the GIL and is fast
    z_weights = weights @ z

    num = np.einsum("ij,ij->j", z, z_weights)         # sum over rows
    den = np.einsum("ij,ij->j", z, z)

    # protect against constant genes in the neighborhood
    out = np.full((X_dense.shape[1],), -1.0, dtype=np.float32)
    ok = den > 0
    out[ok] = (n / row_standardized_weights) * (num[ok] / den[ok])
    return out

# Choose a fast weights builder for points
def build_weights(coords_subset: np.ndarray, k):
    # fixed K neighbors
    # Take Minimum of k (30) cells if there are that many cells.
    w = KNN.from_array(coords_subset, k=k)  # tune k
    return w

# Core computation per i (no sdata['table'] slicing, no GeoPandas)
# This calculate all Moran'Is for all genes for one cell.
def compute_one_i(i: int, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X, k=30):
    idx = distance_matrix[i]
    m = len(idx)
    center_cell_id = int(center_cell_ids[i])

    if m <= k:
        return center_cell_id, np.full((num_genes,), -1.0, dtype=np.float32)

    coords = coords_all[idx, :]

    # Extract expression for just those cells.
    # If rna_X is sparse: this makes a dense (m, num_genes) only for the neighborhood (cheap-ish).
    X_sub = rna_X[idx, :].toarray() if sp.issparse(rna_X) else np.asarray(rna_X[idx, :])

    w = build_weights(coords, k)
    I_all = moran_I_all_genes(X_sub, w)
    return center_cell_id, I_all


# Chunked worker: write into preallocated output
def worker_chunk(i_start: int, i_end: int, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X):
    ids = np.empty((i_end - i_start,), dtype=np.int64)
    I_block = np.empty((i_end - i_start, num_genes), dtype=np.float32)
    for t, i in enumerate(range(i_start, i_end)):
        cid, I_all = compute_one_i(i, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X)
        ids[t] = cid
        I_block[t, :] = I_all
    return i_start, ids, I_block


def calculate_local_moran_I_values(sdata, threads):

    # ----------------------------
    # Precompute once (avoid .todense())
    # ----------------------------
    genes_list = np.array(sdata['table'].var_names)
    num_genes = len(genes_list)

    # keep sparse if possible
    rna_X = sdata['table'].X  # typically CSR/CSC
    coords_all = np.asarray(sdata['table'].obsm['spatial'], dtype=np.float64)
    center_cell_ids = sdata['table'].obs.index.to_numpy()

    distance_matrix = spatial.neighbour_lists(coords_all, NEIGHBOURHOOD_RADIUS, threads)

    # ----------------------------
    # Parallel execution
    # ----------------------------
    n = len(distance_matrix)
    chunk_size = 128  # bigger is usually better after vectorization
    chunks = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]

    # Preallocate final result: (n_cells, n_genes)
    # If you need mapping by center_cell_id, keep ids separately (returned).
    all_ids = np.empty((n,), dtype=np.int64)
    all_I = np.empty((n, num_genes), dtype=np.float32)

    timer = helperfuncs.Timer()
    timer.start()

    # After vectorization, threads often work well because numpy/scipy sparse releases GIL.
    # If weight-building dominates and is pure Python, try ProcessPoolExecutor instead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(worker_chunk, a, b, num_genes, distance_matrix, 
                             center_cell_ids, coords_all, rna_X) for (a, b) in chunks]
        for k, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            i_start, ids_block, I_block = fut.result()
            i_end = i_start + len(ids_block)
            all_ids[i_start:i_end] = ids_block
            all_I[i_start:i_end, :] = I_block
            if k % 5 == 0 or k == len(futures):
                print(f"done chunks {k}/{len(futures)}")

    timer.stop()

    # Now you have:
    #   all_ids: (n,) center cell ids
    #   all_I:   (n, num_genes) Moran's I per center cell and gene
    
    return local_moran_I_per_transcript(sdata, all_ids, all_I, threads)


def local_moran_I_per_transcript(sdata, all_ids, all_I, threads):
    # Make sure dtypes match your all_ids / var_names
    transcripts_df = transcripts.load_transcripts(sdata, ['x', 'y', 'cell_id', 'feature_name'])
    transcripts_cell_id = transcripts_df["cell_id"]
    transcripts_feature = transcripts_df["feature_name"].to_physical().to_numpy()  # Enum codes

    # Build fast maps -> indices
    cell_to_row = {cid: i for i, cid in enumerate(all_ids)}
    gene_to_col = {g: j for j, g in enumerate(sdata['table'].var_names)}

    # Unmatched cells become null and unmatched genes -1, as NaN did in the pandas map.
    cell_rows = transcripts_cell_id.replace_strict(cell_to_row, default=None, return_dtype=pl.Int64)
    valid_cell = cell_rows.is_not_null().to_numpy()
    cell_rows = cell_rows.fill_null(-1).to_numpy()
    gene_cols = transcripts.lookup_by_code(transcripts_df["feature_name"], gene_to_col, -1, np.int64)[transcripts_feature]

    # Initialize output
    loca_morans_I_array = np.full(len(transcripts_feature), -1.0, dtype=np.float32)

    # Valid rows are those that found both a cell and a gene
    valid = valid_cell & (gene_cols != -1)

    # One shot gather
    loca_morans_I_array[valid] = all_I[cell_rows[valid], gene_cols[valid]]

    # --------------------------------------------------------
    # Now I have to take care of the transcripts outside cells
    # --------------------------------------------------------
    # For those transcripts I take the nearest transcripts with the same feature name.
    # If none can be found the local Moran's I will be set to 0.0.

    outside_mask = (loca_morans_I_array == -1)

    # Pull arrays once (avoid repeated pandas overhead)
    x = transcripts_df["x"].cast(pl.Float64).to_numpy()
    y = transcripts_df["y"].cast(pl.Float64).to_numpy()
    coords = np.column_stack((x, y))

    local_I = loca_morans_I_array
    fill_outside_from_nearest_inside(coords, transcripts_feature, local_I, outside_mask, threads)

    print('... done calculating local morans I')
    return local_I