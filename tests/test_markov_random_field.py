"""Golden-output tests for the loopy belief propagation MRF.

The expected SHA-256 digests were produced by the tiled zarr/memmap version
(origin/dev db00d98) that this in-RAM version replaces, so they pin the
beliefs and labels bit for bit.
"""
import hashlib

import numba
import numpy as np
import pytest

from spoqc.hqr.markov_random_field_zarr_parallel import (
    first_version_loopy_belief_propagation_parallel as lbp,
)

GOLDEN = {
    "total": (dict(beta=1.5, max_iter=15),
              "7af9f2fb3278cd52684308786aae96080234eba27e163fdc615d41efc63b34c2",
              "677d9e07da9d94428dfc21f50347b00a4b801b1ba065a388af22479c978acfbb", 4411),
    "min": (dict(beta=1.0, max_iter=20),
            "7e1e4fc7b11385ff365bc03a0eb411b8943f33dc971f45feb47248a0ebe8f61d",
            "23618efc681d83f1df40039f416bc74c256b97bfb2239ef56c1ace3e54cb754d", 4403),
}


@pytest.fixture
def prob_map():
    rng = np.random.default_rng(20260927)
    yy, xx = np.mgrid[:67, :131]
    noisy = 0.5 + 0.45 * np.sin(yy / 5.0) * np.cos(xx / 3.0) + 0.1 * rng.standard_normal((67, 131))
    return np.clip(noisy, 0, 1)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("normalize", sorted(GOLDEN))
def test_lbp_matches_golden_output_bit_for_bit(prob_map, normalize, threads):
    kwargs, beliefs_sha, labels_sha, n_label1 = GOLDEN[normalize]
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    beliefs, labels = lbp(prob_map, normalize=normalize, **kwargs)
    assert beliefs.dtype == np.float32 and beliefs.shape == prob_map.shape
    assert labels.dtype == np.int8 and labels.shape == prob_map.shape
    assert int(labels.sum()) == n_label1
    assert _sha(beliefs) == beliefs_sha
    assert _sha(labels) == labels_sha


def test_lbp_float32_and_float64_inputs_agree(prob_map):
    b64, l64 = lbp(prob_map, beta=1.5, max_iter=5, normalize="total")
    b32, l32 = lbp(prob_map.astype(np.float32), beta=1.5, max_iter=5, normalize="total")
    assert np.array_equal(b64, b32) and np.array_equal(l64, l32)


def test_lbp_rejects_unknown_normalization(prob_map):
    with pytest.raises(SystemExit):
        lbp(prob_map, normalize="median")
