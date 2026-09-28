"""Reductions over ragged groups stored as one flat array.

A group layout is given by `offsets` (length n_groups + 1): group g owns the
flat elements offsets[g]:offsets[g + 1].
"""

from typing import Callable

import numba
import numpy as np

# Rows per partial sum in group_sum. Fixed, so the summation order never depends on the thread count.
GROUP_SUM_CHUNK = 1 << 20


def group_offsets(group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Offsets of each group in a flat array whose group_idx is sorted."""
    return np.searchsorted(group_idx, np.arange(n_groups + 1))


def cyclic_shift(offsets: np.ndarray, shift: int) -> np.ndarray:
    """Flat index of the element `shift` places after each element, wrapping around within its group."""
    sizes = np.diff(offsets)
    start = np.repeat(offsets[:-1], sizes)
    return start + (np.arange(offsets[-1]) - start + shift) % np.repeat(sizes, sizes)


def group_count(group_idx: np.ndarray, mask: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of elements per group where mask is True."""
    return np.bincount(group_idx[mask], minlength=n_groups)


def ragged_reduce(values: np.ndarray, offsets: np.ndarray, reducer: Callable, out: np.ndarray) -> np.ndarray:
    """
    Writes reducer(values[offsets[g]:offsets[g + 1]]) into out[g] for every non-empty group.

    Groups of equal size k are reduced together as one (groups, k) matrix with
    reducer(..., axis=1); numpy reduces each row exactly as it reduces the 1-D
    group, so the result is bit-identical to calling reducer per group.
    Empty groups keep their value in `out`.
    """
    sizes = np.diff(offsets)
    for k in np.unique(sizes[sizes > 0]):
        groups = np.flatnonzero(sizes == k)
        out[groups] = reducer(values[offsets[groups][:, None] + np.arange(k)], axis=1)
    return out


def ragged_lists(values: list, offsets: np.ndarray) -> list:
    """One Python list per group."""
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


@numba.njit(parallel=True)
def _chunk_group_sums(group_idx, values, n_groups, chunk):
    n_chunks = (len(group_idx) + chunk - 1) // chunk
    sums = np.zeros((n_chunks, n_groups))
    counts = np.zeros((n_chunks, n_groups), dtype=np.int64)
    for c in numba.prange(n_chunks):
        for i in range(c * chunk, min(len(group_idx), (c + 1) * chunk)):
            sums[c, group_idx[i]] += values[i]
            counts[c, group_idx[i]] += 1
    return sums, counts


def group_sum(group_idx: np.ndarray, values: np.ndarray, n_groups: int):
    """Per-group float64 sum of values and element count, in a fixed order.

    Each GROUP_SUM_CHUNK rows are summed in row order, then the chunk sums are added in chunk
    order, so the result is identical for any thread count and run to run.
    """
    sums, counts = _chunk_group_sums(group_idx, values, n_groups, GROUP_SUM_CHUNK)
    total, count = sums[0].copy(), counts[0].copy()
    for c in range(1, len(sums)):
        total += sums[c]
        count += counts[c]
    return total, count
