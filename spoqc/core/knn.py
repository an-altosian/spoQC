"""The kNN graph that scanpy's Leiden clustering reads: sc.pp.neighbors, minus pynndescent's unused work.

For n_obs >= 8192 (euclidean), sc.pp.neighbors builds a pynndescent.PyNNDescentTransformer with UMAP's
defaults and calls its fit_transform. Two parts of that call cost time without changing the
graph (pynndescent 0.6.0, scanpy 1.12):

1. fit_transform ends with index_.compress_index(), which builds the index's search trees
   for later queries. scanpy never queries the index, and compress_index() runs after the
   graph was already returned. On sparse input it also allocates
   (n_nodes, 2, n_obs) float32 hyperplanes (convert_tree_format is passed indptr.shape[0] - 1
   as data_dim) and fills half of them: 24.7 GB allocated for hqcr's 167,780 cells.
2. The random-projection forest is built with joblib n_jobs=settings.n_jobs (1). Each tree's
   rng state is drawn before the joblib call (rp_trees.make_forest) and the tree builders are
   nogil numba functions returned in order, so any n_jobs gives the same trees. Only the
   forest gets more threads: NN-descent still runs on settings.n_jobs threads, because its
   candidate sampling splits its rng stream by thread count (utils.new_build_candidates).
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import scanpy as sc
from pynndescent import PyNNDescentTransformer, pynndescent_
from pynndescent.rp_trees import make_forest

# scanpy.neighbors.Neighbors._handle_transformer: for the euclidean metric with knn=True, scanpy
# uses sklearn brute force below this many cells, and pynndescent from it on
SCANPY_PYNNDESCENT_MIN_OBS = 8192


class _GraphOnlyTransformer(PyNNDescentTransformer):
    """PyNNDescentTransformer.fit_transform without compress_index(); the forest on more threads.

    __init__ is inherited (scanpy reads get_params()), so forest_threads is set after construction.
    """

    forest_threads = 1

    def fit_transform(self, X, y=None, **fit_params):
        def forest(data, n_neighbors, n_trees, leaf_size, rng_state, random_state, n_jobs, *args, **kwds):
            return make_forest(
                data, n_neighbors, n_trees, leaf_size, rng_state, random_state, self.forest_threads, *args, **kwds
            )

        with mock.patch.object(pynndescent_, "make_forest", forest):
            self.fit(X, compress_index=False)
        return self.transform(X=None)


def neighbors(adata, n_neighbors: int, random_state: int, threads: int) -> None:
    """sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=random_state), same outputs."""
    if adata.n_obs < SCANPY_PYNNDESCENT_MIN_OBS:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=random_state)
        return
    transformer = _GraphOnlyTransformer(
        # the keywords scanpy passes when transformer is None (_handle_transformer)
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds={},
        random_state=random_state,
        n_jobs=sc.settings.n_jobs,
        n_trees=min(64, 5 + round((adata.n_obs) ** 0.5 / 20.0)),
        n_iters=max(5, round(np.log2(adata.n_obs))),
    )
    transformer.forest_threads = threads
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        random_state=random_state,
        transformer=transformer,
    )
