"""A cheaper accumulation for ovrlpy's per-gene embedding.

ovrlpy 1.2.0 builds the top/bottom embeddings one gene at a time:

    signal_top = kde_2d_discrete(...)[mask]              # (n_pixels,)  float32
    signal_top = signal_top[:, None] * factor[None, :]   # (n_pixels, n_components)
    embedding_top += top

A full-scale profile (913 Mpx, 42.6M transcripts, py-spy across all threads) put
ovrlpy at **4261 s of the run's 6739 s of CPU -- 63%**, entered from
`doublet_score.py`, and those two multiply lines alone at 1520 s of it.

The accumulator is (n_pixels x n_components) x 8 B -- 60 MB for a 500 px patch at
n_components=30, far past any L3 -- and ovrlpy allocates a fresh one of those per
gene as a temporary, then touches all of it. `_calculate_embedding_sparse`
updates only the rows a gene is actually nonzero in, which removes the temporary
and most of the traffic. It is bit-identical; see its docstring.

Applied as a shim rather than an edit to site-packages, because a `pip install`
would silently revert the latter. Pinned to the ovrlpy versions whose internals
it reproduces; on any other version it declines to patch and ovrlpy's own code
runs, so an upgrade cannot break -- it only stops helping.
"""

from __future__ import annotations

from queue import Empty

import numpy as np

SUPPORTED_OVRLPY_VERSIONS = ("1.2.0",)

_XY_DEFAULT = ("x_pixel", "y_pixel")


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


def install() -> bool:
    """Patch ovrlpy's embedding accumulation if its version is one we reproduce.

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

    _utils._calculate_embedding = _calculate_embedding_sparse
    # _ovrlp imported the symbol directly, so it needs rebinding too.
    _ovrlp._calculate_embedding = _calculate_embedding_sparse
    return True
