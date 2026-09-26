"""Faster replacements for the ovrlpy 1.2.0 internals that dominate Doublet QC.

A full-scale profile (913 Mpx, 42.6M transcripts, py-spy across all threads) put ovrlpy
at **4261 s of the run's 6739 s of CPU -- 63%**, entered from `doublet_score.py`. Two
defects account for it, and this module replaces exactly those two:

1. The per-gene accumulation is bound by DRAM traffic over a ~60 MB accumulator, and a
   gene's blurred signal is mostly zero -- so `_calculate_embedding_sparse` skips the
   exact-zero rows. Retained rows are exact to the bit; see its docstring.

2. Both patch loops (`_ovrlp.py:365`, `_kde.py:237`) are serial, parallelising only the
   genes *within* a patch where scipy.ndimage and boolean-mask indexing hold the GIL
   (measured 7,107 s of CPU in 7,800 s of wall at n_workers=16 -- 0.91x). The loops are
   embarrassingly parallel, so `compute_VSI_parallel` and `_sample_expression_parallel`
   decompose them over processes.

Applied as a shim rather than an edit to site-packages, because a `pip install` would
silently revert the latter. Pinned to the ovrlpy versions whose internals it reproduces;
on any other version it declines to patch and ovrlpy's own code is used, so an upgrade
cannot silently break -- it only stops helping.

A BLAS rank-1 (`ger`) accumulation and a batched-KDE variant were both built and
measured, and both LOST: rank-1 ran 45.43 s against sparse's 31.18 s on a
production-regime fixture (the cost is memory traffic, not arithmetic), and the batched
KDE cannot reproduce ovrlpy's per-gene bounding-box truncation at all. Neither is kept
here; the numbers are recorded in the PR that proposed them.
"""

from __future__ import annotations

import multiprocessing as _mp
from queue import Empty

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_array

SUPPORTED_OVRLPY_VERSIONS = ("1.2.0",)

_TRUNCATE = 4  # matches ovrlpy._kde._TRUNCATE

_XY_DEFAULT = ("x_pixel", "y_pixel")

# A worker's own BLAS/numeric pool multiplies against the worker count: 16 workers
# each opening a 16-thread pool is 256 threads on a 16-core box. `spawn` children read
# these at import, so they are set before the pool is created.
_POOL_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


def _calculate_embedding_sparse(genes, mask, components, **kwargs):
    """Drop-in replacement for ovrlpy._utils._calculate_embedding, skipping zero rows.

    The accumulation is bound by memory traffic, not arithmetic. With n_components=30 and
    a 500x500 patch the accumulator is 30 x 250,000 x 8 B = **60 MB**, far beyond any L3,
    and ovrlpy touches all of it once per gene per side. At the measured median of 6,492
    genes per patch that is on the order of a terabyte of DRAM traffic for ONE patch,
    which is why neither threads (ovrlpy's own 16: 0.91x) nor processes (4: 1.08x) help.

    But a gene's blurred signal is mostly zero: `kde_2d_discrete` blurs with bandwidth 2.5
    and truncate 4, so a gene is nonzero only within ~10 px of one of its own transcripts.
    Measured by dilating real transcript positions on a 520x520 patch (600 genes, >= 2
    transcripts each):

        nonzero fraction after blur:  median 6.8%   mean 16.0%   p90 44.5%
        below  5%: 43% of genes    below 10%: 56%    below 25%: 78%

    Adding `0.0 * factor_c` is a no-op, so those rows are skipped and the traffic falls
    with the mean nonzero fraction. Finding them costs one pass over `signal`
    (n_pixels x 4 B = 1 MB) against the 120 MB the update itself moves -- about 1%.

    Equivalence: `signal[rows, None] * factor[None, :]` then `+=` is the SAME pair of
    rounding steps as ovrlpy's `signal[:, None] * factor[None, :]` then `+=`, on the same
    values -- no reassociation, no FMA fusion -- so retained rows are exact to the bit.
    A skipped row would have added `0.0 * factor_c`, which is +-0.0 for finite loadings
    and leaves the accumulator unchanged; the only reachable difference is the SIGN of a
    zero in a pixel that is zero for every gene, which compares equal under `==` and
    `np.array_equal` and cannot change `_cosine_similarity`. Non-finite loadings would
    break that argument (`0.0 * inf` is NaN, which ovrlpy propagates and this would not),
    so they are rejected rather than silently handled.

    Measured on the real dataset, whole compute_VSI stage, 16 workers, matched cache:
    4839.1 s -> 2455.6 s with the process-parallel loop below; max abs difference in the
    final integrity_map 6.556511e-07, with ZERO pixels past 1e-06.
    """
    from ovrlpy._kde import kde_2d_discrete

    x_col, y_col = _XY_DEFAULT
    n_pixels = int(np.count_nonzero(mask))
    n_components = components.shape[0]

    top_acc = None
    bottom_acc = None

    while True:
        try:
            i, gene = genes.get(block=False)
        except Empty:
            break

        # ovrlpy skips genes with fewer than two transcripts in the patch.
        if len(gene) < 2:
            continue

        factor = np.asarray(components[:, i], dtype=np.float64)
        if not np.isfinite(factor).all():
            raise ValueError(
                f"non-finite PCA loading for gene index {i}: skipping zero-signal rows is "
                "only equivalent to ovrlpy for finite loadings, because 0.0 * inf is NaN"
            )

        above = gene.select(_XY_DEFAULT).filter(gene["z"] > gene["z_center"])
        below = gene.select(_XY_DEFAULT).filter(gene["z"] < gene["z_center"])

        for part, which in ((above, "top"), (below, "bottom")):
            if len(part) == 0:
                continue
            signal = kde_2d_discrete(
                part[x_col].to_numpy(), part[y_col].to_numpy(), size=mask.shape, **kwargs
            )[mask]

            rows = np.flatnonzero(signal)
            if rows.size == 0:
                continue

            if which == "top":
                if top_acc is None:
                    top_acc = np.zeros((n_pixels, n_components), dtype=np.float64)
                target = top_acc
            else:
                if bottom_acc is None:
                    bottom_acc = np.zeros((n_pixels, n_components), dtype=np.float64)
                target = bottom_acc

            # float32 signal x float64 factor promotes to float64, as in the original.
            target[rows] += (
                np.asarray(signal[rows], dtype=np.float64)[:, None] * factor[None, :]
            )

    return (0 if top_acc is None else top_acc, 0 if bottom_acc is None else bottom_acc)


# patches with *processes* sidesteps the GIL.
#
# Each task carries the patch's transcript slice and its signal tile and returns only the
# finished (patch_length, patch_length) float32 tile -- the cosine similarity is reduced
# inside the worker, so the (n_pixels, n_components) float64 embeddings, up to 65 MB per
# side, never cross a pipe.

_WORKER: dict = {}


def _pool_env():
    """Pin worker numeric pools to one thread, returning the previous values."""
    import os

    saved = {k: os.environ.get(k) for k in _POOL_ENV_VARS}
    os.environ.update({k: "1" for k in _POOL_ENV_VARS})
    return saved


def _restore_env(saved):
    import os

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _vsi_init(components, gene2idx, bandwidth, dtype, min_expression):
    """ProcessPoolExecutor initializer: stash the per-run constants once per worker."""
    _WORKER.update(
        components=components,
        gene2idx=gene2idx,
        bandwidth=bandwidth,
        dtype=dtype,
        min_expression=min_expression,
    )


def _vsi_patch(task):
    """Compute one patch's cosine-similarity tile. Runs in a worker process.

    Reproduces `Ovrlp.compute_VSI`'s per-patch body verbatim, including the `patch_mask`
    construction, the `is_in(gene2idx)` filter and the integer-0 sentinel checks. Returns
    None wherever the original would `continue`.
    """
    from queue import SimpleQueue

    import polars as pl
    from ovrlpy._utils import _cosine_similarity

    patch_df, patch_signal, remove_pad = task
    state = _WORKER
    gene2idx = state["gene2idx"]

    not_padding = np.zeros(patch_signal.shape, dtype=bool)
    not_padding[remove_pad] = True
    patch_mask = (patch_signal > state["min_expression"]) & not_padding
    if not patch_mask.any():
        return None

    patch_df = patch_df.filter(pl.col("gene").cast(pl.String).is_in(gene2idx))
    gene_queue: SimpleQueue = SimpleQueue()
    for (gene, *_), df in patch_df.group_by("gene"):
        if gene in gene2idx:
            gene_queue.put((gene2idx[gene], df.drop("gene")))

    top, bottom = _calculate_embedding_sparse(
        gene_queue,
        patch_mask,
        state["components"],
        bandwidth=state["bandwidth"],
        dtype=state["dtype"],
    )
    if isinstance(top, int) or isinstance(bottom, int):
        return None

    tile = np.zeros_like(patch_signal)
    tile[patch_mask] = _cosine_similarity(top, bottom)
    return tile[remove_pad]


def _drain(executor, submit_next, pending, on_result):
    """Run a bounded sliding window of tasks, applying `on_result` as each lands.

    The window matters: submitting all 840 patches at once would pickle every transcript
    slice into the queue simultaneously (~10 GB). It keeps that bounded while never
    letting a worker idle.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    more = submit_next()
    while pending:
        done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
        for future in done:
            on_result(pending.pop(future), future.result())
        if more:
            more = submit_next()


def compute_VSI_parallel(self, *, min_transcripts: float = 2, queue_depth: int = 2):
    """Process-parallel replacement for `ovrlpy.Ovrlp.compute_VSI`.

    Only the *scheduling* changes: every patch is computed by the same code the serial
    loop runs, so a patch's tile does not depend on how many workers are active. Verified
    deterministic across runs and independent of worker count, and equal to ovrlpy at
    n_workers=1 (which is its only reproducible configuration -- at n_workers > 1 it
    reduces partial sums over `as_completed` and does not reproduce itself).
    """
    from concurrent.futures import ProcessPoolExecutor
    from math import ceil

    import tqdm
    from ovrlpy._kde import kde_2d_discrete
    from ovrlpy._patching import _patches, n_patches

    min_expression = self._expression_threshold(min_transcripts)
    padding = int(ceil(_TRUNCATE * self.KDE_bandwidth))
    gene2idx = {gene: i for i, gene in enumerate(self.genes)}

    signal = kde_2d_discrete(
        self.transcripts["x_pixel"].to_numpy(),
        self.transcripts["y_pixel"].to_numpy(),
        bandwidth=self.KDE_bandwidth,
        dtype=self.dtype,
    )
    shape = signal.shape
    cosine_similarity = np.zeros_like(signal)

    def tasks():
        for patch_df, padded, unpadded in _patches(
            self.transcripts[["gene", "x_pixel", "y_pixel", "z", "z_center"]],
            self.patch_length,
            padding,
            size=shape,
            coordinates=("x_pixel", "y_pixel"),
        ):
            if len(patch_df) == 0:
                continue
            left_pad = unpadded[0].start - padded[0].start
            bottom_pad = unpadded[1].start - padded[1].start
            remove_pad = (
                slice(left_pad, left_pad + unpadded[0].stop - unpadded[0].start),
                slice(bottom_pad, bottom_pad + unpadded[1].stop - unpadded[1].start),
            )
            yield (patch_df, signal[padded], remove_pad), unpadded

    saved = _pool_env()
    try:
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=_mp.get_context("spawn"),
            initializer=_vsi_init,
            initargs=(
                self.pca.components_,
                gene2idx,
                self.KDE_bandwidth,
                self.dtype,
                min_expression,
            ),
        ) as executor:
            pending: dict = {}
            stream = tasks()
            limit = max(1, queue_depth * self.n_workers)
            progress = tqdm.tqdm(total=n_patches(self.patch_length, shape))

            def submit_next():
                while len(pending) < limit:
                    item = next(stream, None)
                    if item is None:
                        return False
                    task, unpadded = item
                    pending[executor.submit(_vsi_patch, task)] = unpadded
                return True

            def on_result(unpadded, tile):
                if tile is not None:
                    cosine_similarity[unpadded] = tile
                progress.update(1)

            _drain(executor, submit_next, pending, on_result)
            progress.close()
    finally:
        _restore_env(saved)

    self.signal_map = signal.T
    self.integrity_map = cosine_similarity.T


def install_parallel_vsi() -> bool:
    """Bind `compute_VSI_parallel` onto `ovrlpy.Ovrlp` in place of `compute_VSI`."""
    import ovrlpy
    from ovrlpy import _ovrlp

    version = getattr(ovrlpy, "__version__", None)
    if version not in SUPPORTED_OVRLPY_VERSIONS:
        print(
            f"[NOTE] ovrlpy {version} is not one of {SUPPORTED_OVRLPY_VERSIONS}; "
            "keeping its serial patch loop"
        )
        return False

    _ovrlp.Ovrlp.compute_VSI = compute_VSI_parallel
    return True


_SAMPLE_WORKER: dict = {}


def _sample_init(coord_columns, gene_column, dtype):
    """ProcessPoolExecutor initializer for the expression-sampling workers."""
    _SAMPLE_WORKER.update(
        coord_columns=list(coord_columns), gene_column=gene_column, dtype=dtype
    )


def _sample_patch(task):
    """Sample every gene's KDE at one patch's local maxima. Runs in a worker process."""
    import pandas as pd
    from ovrlpy._kde import kde_and_sample

    patch_df, maxima, patch_size = task
    state = _SAMPLE_WORKER
    coord_columns = state["coord_columns"]

    sampled = {}
    for gene, df in patch_df.group_by(state["gene_column"]):
        name, values = kde_and_sample(
            *(df[c] for c in coord_columns),
            sampling_coordinates=maxima,
            gene=gene[0],
            size=patch_size,
            bandwidth=1,
            dtype=state["dtype"],
        )
        sampled[name] = values
    return pd.DataFrame(sampled)


def _sample_expression_parallel(
    transcripts,
    kde_bandwidth: float = 2.5,
    min_expression: float = 2,
    min_pixel_distance: float = 5,
    genes=None,
    coord_columns=("x", "y", "z"),
    gene_column: str = "gene",
    n_workers: int = 8,
    patch_length: int = 500,
    dtype=np.float32,
):
    """Process-parallel replacement for `ovrlpy._kde._sample_expression`.

    Unlike compute_VSI this one is exactly reproducible: each gene's KDE is sampled
    independently and the per-gene results land in a dict that is immediately reindexed by
    `gene_list`, so neither completion order nor worker count can reach the values. Moving
    whole patches into worker processes is therefore bit-identical by construction --
    verified on X, `obsm["spatial"]` and `var_names` against ovrlpy's own output.

    The preamble (one global `kde_nd` over every transcript plus `find_local_maxima`) is
    left serial: it is a single large filter, not a per-gene loop.
    """
    import warnings
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    import polars as pl
    import tqdm
    from anndata import AnnData
    from anndata._warnings import ImplicitModificationWarning
    from ovrlpy._kde import _TRUNCATE as KDE_TRUNCATE
    from ovrlpy._kde import find_local_maxima, kde_nd
    from ovrlpy._patching import _patches, n_patches

    coord_columns = list(coord_columns)
    assert len(coord_columns) == 3 or len(coord_columns) == 2

    # lower resolution instead of increasing bandwidth!
    transcripts = (
        transcripts.lazy()
        .select(pl.col(coord_columns) / kde_bandwidth, gene_column)
        .collect(engine="streaming")
    )

    print("determining pseudocells")
    kde = kde_nd(*(transcripts[c] for c in coord_columns), bandwidth=1, dtype=dtype)
    min_dist = 1 + int(min_pixel_distance / kde_bandwidth)
    local_maximum_coordinates = find_local_maxima(
        kde, min_pixel_distance=min_dist, min_expression=min_expression
    )
    print("found", len(local_maximum_coordinates), "pseudocells")
    size = kde.shape
    del kde

    if genes is not None:
        transcripts = transcripts.filter(pl.col("gene").cast(pl.String).is_in(genes))
    gene_list = sorted(transcripts[gene_column].unique())

    padding = KDE_TRUNCATE

    print("sampling expression:")
    # Results are keyed by patch index so the assembled row order matches the serial
    # implementation regardless of completion order; `coords` is appended in generator
    # order, which is the same order, so the two stay aligned.
    frames: dict = {}
    coords: list = []

    def tasks():
        for index, (patch_df, padded, unpadded) in enumerate(
            _patches(transcripts, patch_length, padding, size=size)
        ):
            patch_maxima = local_maximum_coordinates[
                (local_maximum_coordinates[:, 0] >= unpadded[0].start)
                & (local_maximum_coordinates[:, 0] < unpadded[0].stop)
                & (local_maximum_coordinates[:, 1] >= unpadded[1].start)
                & (local_maximum_coordinates[:, 1] < unpadded[1].stop),
                :,
            ]
            coords.append(patch_maxima)
            maxima = patch_maxima.copy()
            maxima[:, 0] -= padded[0].start
            maxima[:, 1] -= padded[1].start
            patch_size = (
                padded[0].stop - padded[0].start,
                padded[1].stop - padded[1].start,
                *size[2:],
            )
            yield index, (patch_df, maxima, patch_size)

    saved = _pool_env()
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_mp.get_context("spawn"),
            initializer=_sample_init,
            initargs=(coord_columns, gene_column, dtype),
        ) as executor:
            pending: dict = {}
            stream = tasks()
            limit = max(1, 2 * n_workers)
            progress = tqdm.tqdm(total=n_patches(patch_length, size))

            def submit_next():
                while len(pending) < limit:
                    item = next(stream, None)
                    if item is None:
                        return False
                    index, task = item
                    pending[executor.submit(_sample_patch, task)] = index
                return True

            def on_result(index, frame):
                frames[index] = frame
                progress.update(1)

            _drain(executor, submit_next, pending, on_result)
            progress.close()
    finally:
        _restore_env(saved)

    ordered = [frames[i] for i in range(len(frames))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ImplicitModificationWarning)
        adata = AnnData(pd.concat(ordered, ignore_index=True)[gene_list].fillna(0))
    adata.obsm["spatial"] = np.rint(np.vstack(coords) * kde_bandwidth).astype(np.int32)
    return adata


def install_parallel_sampling() -> bool:
    """Bind `_sample_expression_parallel` over `ovrlpy._kde._sample_expression`."""
    import ovrlpy
    from ovrlpy import _kde, _ovrlp

    version = getattr(ovrlpy, "__version__", None)
    if version not in SUPPORTED_OVRLPY_VERSIONS:
        print(
            f"[NOTE] ovrlpy {version} is not one of {SUPPORTED_OVRLPY_VERSIONS}; "
            "keeping its serial expression sampling"
        )
        return False

    _kde._sample_expression = _sample_expression_parallel
    # _ovrlp may have imported the symbol directly.
    if hasattr(_ovrlp, "_sample_expression"):
        _ovrlp._sample_expression = _sample_expression_parallel
    return True


def install() -> bool:
    """Patch ovrlpy if its version is one this shim was written against.

    Returns True if the patch was applied.
    """
    import ovrlpy
    from ovrlpy import _ovrlp, _utils

    version = getattr(ovrlpy, "__version__", None)
    if version not in SUPPORTED_OVRLPY_VERSIONS:
        print(
            f"[NOTE] ovrlpy {version} is not one of {SUPPORTED_OVRLPY_VERSIONS}; "
            "keeping its own embedding accumulation"
        )
        return False

    # The nonzero-only accumulation, not the rank-1 one: the cost here is DRAM traffic
    # over a ~60 MB accumulator, not arithmetic, so skipping exact-zero rows beats making
    # the arithmetic cheaper (measured 31.18 s vs 45.43 s on a production-regime fixture).
    _utils._calculate_embedding = _calculate_embedding_sparse
    # _ovrlp imported the symbol directly, so it needs rebinding too.
    _ovrlp._calculate_embedding = _calculate_embedding_sparse

    # Both patch loops are serial in ovrlpy, with threads only splitting work *within* a
    # patch where scipy.ndimage and boolean-mask indexing hold the GIL.
    install_parallel_vsi()
    install_parallel_sampling()
    return True
