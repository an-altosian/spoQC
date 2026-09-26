"""`install()` must patch exactly what it claims, and only on a version it was written for.

The equivalence of the replacements themselves is covered elsewhere:
`test_ovrlpy_sparse.py` for the accumulation, `test_ovrlpy_parallel_vsi.py` and
`test_ovrlpy_parallel_sampling.py` for the two patch loops. This file is only about the
binding.
"""
from __future__ import annotations

import pytest

ovrlpy = pytest.importorskip("ovrlpy")

from spoqc._ovrlpy_fast import SUPPORTED_OVRLPY_VERSIONS, install  # noqa: E402


def test_install_is_version_guarded():
    """install() binds the nonzero-only accumulation and BOTH process-parallel loops.

    It binds `_calculate_embedding_sparse` and nothing else: a BLAS rank-1 accumulation
    was built and measured at 45.43 s against sparse's 31.18 s on a production-regime
    fixture, because the accumulation is bound by DRAM traffic over a ~60 MB accumulator
    rather than by arithmetic. It is not kept in the module.
    """
    from spoqc._ovrlpy_fast import (
        _calculate_embedding_sparse,
        _sample_expression_parallel,
        compute_VSI_parallel,
    )

    applied = install()
    assert applied is (ovrlpy.__version__ in SUPPORTED_OVRLPY_VERSIONS)
    if applied:
        from ovrlpy import _kde, _ovrlp, _utils
        assert _utils._calculate_embedding is _calculate_embedding_sparse
        assert _ovrlp._calculate_embedding is _calculate_embedding_sparse
        assert _ovrlp.Ovrlp.compute_VSI is compute_VSI_parallel
        assert _kde._sample_expression is _sample_expression_parallel


def test_rejected_variants_are_not_shipped():
    """Minimum change: a measured-and-rejected alternative is not carried as dead code."""
    import spoqc._ovrlpy_fast as fast

    for name in ("_calculate_embedding_fast", "_calculate_embedding_batched"):
        assert not hasattr(fast, name), f"{name} lost on measurement and must not ship"
