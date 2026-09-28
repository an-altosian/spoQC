"""`install()` must patch exactly what it claims, and only on a version it was written for.

The equivalence of the replacement itself is covered by `test_ovrlpy_sparse.py`.
This file is only about the binding.
"""
from __future__ import annotations

import pytest

ovrlpy = pytest.importorskip("ovrlpy")

from spoqc._ovrlpy_fast import SUPPORTED_OVRLPY_VERSIONS, install  # noqa: E402


def test_install_is_version_guarded():
    from spoqc._ovrlpy_fast import _calculate_embedding_sparse

    applied = install()
    assert applied is (ovrlpy.__version__ in SUPPORTED_OVRLPY_VERSIONS)
    if applied:
        from ovrlpy import _ovrlp, _utils
        assert _utils._calculate_embedding is _calculate_embedding_sparse
        assert _ovrlp._calculate_embedding is _calculate_embedding_sparse


def test_only_the_accumulation_is_patched():
    """Nothing else is touched, so nothing else can regress runtime or memory.

    Process-parallel patch loops were measured and are NOT shipped: they reached
    1.97x end to end but took the full-scale tree peak from 24.63 GB to 31.50 GB.
    A change that costs 6.87 GB of peak is not an optimisation.
    """
    import spoqc._ovrlpy_fast as fast

    for name in ("compute_VSI_parallel", "_sample_expression_parallel",
                 "install_parallel_vsi", "install_parallel_sampling",
                 "_calculate_embedding_fast", "_calculate_embedding_batched"):
        assert not hasattr(fast, name), f"{name} must not ship"
