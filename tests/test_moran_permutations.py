"""Model QC: Moran's I and its permutation variance per PC, vs the verbatim origin/dev loop.

origin/dev (tests/legacy/qc_model.py, db00d98) rebuilt the same Queen weights for every PC
and ran esda's serial 999-permutation loop. The new code builds the weights once and
evaluates the permutations on a thread pool, drawing them from the global RNG in esda's
order. The acceptance bar is bit-identity of every number the step reports (moran_I,
spatial_variance, variance_explained) and of the global RNG state it leaves behind.
"""

import types

import numpy as np
import pandas as pd
import pytest
from esda.moran import Moran
from libpysal.weights import Queen
import geopandas as gpd

from spoqc import helperfuncs
from spoqc.subworkflows import qc_model

from conftest import load_legacy

legacy = load_legacy("qc_model", "spoqc.subworkflows")

N_POINTS = 2_500
N_PCS = 4
THREADS = 4


def _frame(dtype, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 5_000, N_POINTS), rng.uniform(0, 5_000, N_POINTS)
    df = pd.DataFrame({"x": x, "y": y})
    for i in range(N_PCS):
        signal = np.sin(x / (300 + 100 * i)) * np.cos(y / 400)
        df[f"PC{i}"] = (signal * (i % 2) + rng.normal(size=N_POINTS)).astype(dtype)
    return df


class _Figure:
    """Stands in for make_subplots(); records what the step would plot."""

    def __init__(self, record):
        self.record = record

    def add_trace(self, trace, secondary_y):
        self.record.append((trace.name, np.asarray(trace.y).copy()))

    def update_layout(self, **kw):
        pass

    def update_yaxes(self, **kw):
        pass

    def write_html(self, *a, **kw):
        pass

    def write_image(self, *a, **kw):
        pass


def _run(module, monkeypatch, df, seed, **kw):
    """The step's reported traces and the global RNG state it leaves."""
    record = []
    monkeypatch.setattr(module, "make_subplots", lambda **k: _Figure(record))
    monkeypatch.setattr(helperfuncs, "apply_general_plotly_layout", lambda fig, b: None)
    if hasattr(module, "save_figure"):
        monkeypatch.setattr(module, "save_figure", lambda *a, **k: None)
    sdata = {
        "table": types.SimpleNamespace(uns={"pca": {"variance": np.linspace(3, 1, 10)}})
    }
    np.random.seed(seed)
    module.plot_spatial_vs_exression_variance(sdata, "unused", df.copy(), N_PCS, **kw)
    return record, np.random.get_state()


def _assert_same(a, b):
    (rec_a, st_a), (rec_b, st_b) = a, b
    assert [n for n, _ in rec_a] == [n for n, _ in rec_b]
    for (name, ya), (_, yb) in zip(rec_a, rec_b):
        assert ya.dtype == yb.dtype, name
        np.testing.assert_array_equal(ya, yb, err_msg=name, strict=True)
        assert ya.tobytes() == yb.tobytes(), name  # bit-identical, incl. signed zeros
    assert st_a[0] == st_b[0] and st_a[2:] == st_b[2:]
    np.testing.assert_array_equal(st_a[1], st_b[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [123, 7])
def test_step_numbers_and_rng_state_match_origin_dev(monkeypatch, dtype, seed):
    df = _frame(dtype)
    old = _run(legacy, monkeypatch, df, seed)
    new = _run(qc_model, monkeypatch, df, seed, threads=THREADS)
    _assert_same(old, new)


def test_single_thread_matches_too(monkeypatch):
    df = _frame(np.float32, seed=3)
    _assert_same(
        _run(legacy, monkeypatch, df, 123),
        _run(qc_model, monkeypatch, df, 123, threads=1),
    )


def test_one_moran_matches_esda_attributes():
    """I and VI_sim against esda's own object, with a chunk count that is not a multiple."""
    df = _frame(np.float32, seed=5)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    np.random.seed(11)
    ref = Moran(df["PC1"], Queen.from_dataframe(gdf), permutations=999)
    np.random.seed(11)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(THREADS) as pool:
        I, VI = qc_model.moran_I_and_sim_variance(
            df["PC1"], Queen.from_dataframe(gdf), 999, pool, THREADS
        )
    assert I == ref.I and VI == ref.VI_sim


def test_queen_weights_consume_no_rng():
    """Hoisting Queen out of the PC loop is only exact if building it draws nothing."""
    df = _frame(np.float64)
    np.random.seed(1)
    Queen.from_dataframe(gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y)))
    after = np.random.get_state()[1].copy()
    np.random.seed(1)
    np.testing.assert_array_equal(after, np.random.get_state()[1])


# ----------------------------------------------------------------- mutants must fail


def _mutant_fresh_rng(z, weights, scale, z2ss, perms):
    rs = np.random.RandomState(0)
    return qc_model_permuted_I(
        z, weights, scale, z2ss, [rs.permutation(len(z)) for _ in perms]
    )


def _mutant_scaled_z(z, weights, scale, z2ss, perms):
    return qc_model_permuted_I(z / z.std(), weights, scale, z2ss, perms)


def _mutant_binary_weights(z, weights, scale, z2ss, perms):
    return qc_model_permuted_I(z, (weights > 0).astype(float), scale, z2ss, perms)


def _mutant_drop_last(z, weights, scale, z2ss, perms):
    out = qc_model_permuted_I(z, weights, scale, z2ss, perms)
    out[-1] = out[0]
    return out


qc_model_permuted_I = qc_model._permuted_I


@pytest.mark.parametrize(
    "mutant",
    [_mutant_fresh_rng, _mutant_scaled_z, _mutant_binary_weights, _mutant_drop_last],
)
def test_mutants_are_caught(monkeypatch, mutant):
    df = _frame(np.float32)
    old = _run(legacy, monkeypatch, df, 123)
    monkeypatch.setattr(qc_model, "_permuted_I", mutant)
    new = _run(qc_model, monkeypatch, df, 123, threads=THREADS)
    with pytest.raises(AssertionError):
        _assert_same(old, new)


def test_mutant_skipped_draw_is_caught(monkeypatch):
    """Drawing one permutation too many shifts every later PC's RNG stream."""
    df = _frame(np.float32)
    old = _run(legacy, monkeypatch, df, 123)
    real = qc_model.moran_I_and_sim_variance

    def extra_draw(*a, **k):
        np.random.permutation(3)
        return real(*a, **k)

    monkeypatch.setattr(qc_model, "moran_I_and_sim_variance", extra_draw)
    with pytest.raises(AssertionError):
        _assert_same(old, _run(qc_model, monkeypatch, df, 123, threads=THREADS))
