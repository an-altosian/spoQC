"""Faster accumulation for ovrlpy's per-gene embedding.

ovrlpy 1.2.0 builds the top/bottom embeddings one gene at a time:

    signal_top = kde_2d_discrete(...)[mask]              # (n_pixels,)  float32
    signal_top = signal_top[:, None] * factor[None, :]   # (n_pixels, n_components)
    ...
    embedding_top += top

A full-scale profile put those two multiply lines (`_utils.py:173` and `:181`) at
**1520 s of the 4261 s** ovrlpy spends on a 913 Mpx sample. The reason is that the sum
over genes of `outer(signal_g, factor_g)` is a matrix product written out by hand: every
gene allocates a fresh `(n_pixels, n_components)` temporary purely to add it into the
accumulator, so each gene costs three passes over that array plus an allocation.

BLAS has an operation for exactly this -- `ger`, a rank-1 update, `A := alpha*x*y' + A`
-- which updates the accumulator in place with no temporary and one pass. Measured with
n_components=21 and 300 genes:

    n_pixels   per-gene outer   in-place ger
      50,000          0.57 s        0.02 s   (32.6x)
     800,000          8.87 s        0.38 s   (23.6x)

with a maximum absolute difference of 7e-14, i.e. float noise -- the arithmetic and its
order are unchanged, only the temporaries are gone.

This is applied as a shim rather than a patch to the installed package, because a
`pip install` would silently revert an edit inside site-packages. It is pinned to the
ovrlpy versions whose internals it reproduces; on any other version it declines to patch
and the original implementation is used, so a future upgrade cannot silently break.
"""

from __future__ import annotations

from queue import Empty

import numpy as np
from scipy.linalg.blas import get_blas_funcs

SUPPORTED_OVRLPY_VERSIONS = ("1.2.0",)

_XY_DEFAULT = ("x_pixel", "y_pixel")


def _calculate_embedding_fast(genes, mask, components, **kwargs):
    """Drop-in replacement for ovrlpy._utils._calculate_embedding.

    Same inputs, same outputs (including the integer 0 sentinel ovrlpy's caller checks
    for when a worker received no genes), but accumulates with an in-place BLAS rank-1
    update instead of allocating a temporary per gene.
    """
    from ovrlpy._kde import kde_2d_discrete

    x_col, y_col = _XY_DEFAULT
    n_pixels = int(np.count_nonzero(mask))
    n_components = components.shape[0]

    # Accumulators are held transposed and Fortran-ordered because that is what `ger`
    # updates in place; they are transposed back on return, which is a view.
    top_acc = None
    bottom_acc = None
    ger = None

    while True:
        try:
            i, gene = genes.get(block=False)
        except Empty:
            break

        # ovrlpy skips genes with fewer than two transcripts in the patch.
        if len(gene) < 2:
            continue

        factor = np.asarray(components[:, i], dtype=np.float64)
        above = gene.select(_XY_DEFAULT).filter(gene["z"] > gene["z_center"])
        below = gene.select(_XY_DEFAULT).filter(gene["z"] < gene["z_center"])

        for part, which in ((above, "top"), (below, "bottom")):
            if len(part) == 0:
                continue
            signal = kde_2d_discrete(
                part[x_col].to_numpy(), part[y_col].to_numpy(), size=mask.shape, **kwargs
            )[mask]
            # float32 signal x float64 factor promotes to float64 in the original.
            signal = np.asarray(signal, dtype=np.float64)

            if which == "top":
                if top_acc is None:
                    top_acc = np.zeros((n_components, n_pixels), dtype=np.float64, order="F")
                target = top_acc
            else:
                if bottom_acc is None:
                    bottom_acc = np.zeros((n_components, n_pixels), dtype=np.float64, order="F")
                target = bottom_acc

            if ger is None:
                (ger,) = get_blas_funcs(("ger",), (target, factor))
            # target += outer(factor, signal), in place, no temporary
            ger(1.0, factor, signal, a=target, overwrite_a=1)

    return (
        0 if top_acc is None else top_acc.T,
        0 if bottom_acc is None else bottom_acc.T,
    )


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

    _utils._calculate_embedding = _calculate_embedding_fast
    # _ovrlp imported the symbol directly, so it needs rebinding too.
    _ovrlp._calculate_embedding = _calculate_embedding_fast
    return True
