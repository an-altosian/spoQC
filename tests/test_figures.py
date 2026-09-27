import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from anndata import AnnData

from spoqc import helperfuncs
from spoqc.core import figures
from spoqc.core.figures import save_figure

WORKERS = 2


@pytest.fixture
def pool():
    figures.start(WORKERS)
    yield figures
    figures.stop()


def _scatter(n=100, seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots()
    ax.scatter(rng.random(n), rng.random(n), s=1)
    return fig


def _adata(n=500, seed=0):
    rng = np.random.default_rng(seed)
    adata = AnnData(np.zeros((n, 1), dtype=np.float32))
    adata.obsm["spatial"] = rng.random((n, 2)) * 1000
    adata.obs["celltype"] = pd.Categorical(rng.choice(["a", "b", "c"], n))
    adata.obs["score"] = rng.random(n)
    return adata


class TestWritesFiles:
    def test_matplotlib_figure_is_written_to_every_path(self, pool, tmp_path):
        fig = _scatter()
        save_figure(
            fig, tmp_path / "a.png", f"{tmp_path}/a.pdf", dpi=50, bbox_inches="tight"
        )
        plt.close(fig)
        pool.wait()
        assert (tmp_path / "a.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")

    def test_plotly_figure_is_written_to_every_path(self, pool, tmp_path):
        fig = px.scatter(x=[1, 2, 3], y=[3, 1, 2])
        save_figure(fig, f"{tmp_path}/p.png", f"{tmp_path}/p.pdf", scale=1)
        pool.wait()
        assert (tmp_path / "p.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "p.pdf").read_bytes().startswith(b"%PDF")

    def test_many_figures_beyond_the_pending_bound_all_land(self, pool, tmp_path):
        n = WORKERS * figures.MAX_PENDING_PER_WORKER * 3
        for i in range(n):
            fig = _scatter(seed=i)
            save_figure(fig, tmp_path / f"{i}.png", dpi=20)
            plt.close(fig)
        pool.wait()
        assert sorted(os.listdir(tmp_path)) == sorted(f"{i}.png" for i in range(n))

    def test_figure_changed_after_save_is_written_as_it_was_saved(self, pool, tmp_path):
        fig = _scatter()
        fig.axes[0].set_title("saved")
        with plt.rc_context({"svg.fonttype": "none"}):  # text as text; also checks rcParams reach the worker
            save_figure(fig, tmp_path / "a.svg")
        fig.axes[0].set_title("changed afterwards")
        pool.wait()
        svg = (tmp_path / "a.svg").read_text()
        assert "saved" in svg and "changed afterwards" not in svg

    def test_real_site_plot_scatter_writes_png_and_pdf(self, pool, tmp_path):
        helperfuncs.plot_scatter(
            _adata(), str(tmp_path), "x", None, "celltype", None, "title"
        )
        pool.wait()
        for ext in ["png", "pdf"]:
            assert (tmp_path / f"scatterplot_x.{ext}").stat().st_size > 0


class TestErrors:
    def test_worker_error_is_raised_by_wait(self, pool, tmp_path):
        save_figure(_scatter(), tmp_path / "missing_dir" / "a.png")
        with pytest.raises(FileNotFoundError):
            pool.wait()

    def test_worker_error_is_raised_by_the_next_save(self, pool, tmp_path):
        save_figure(_scatter(), tmp_path / "missing_dir" / "a.png")
        next(iter(pool._pending)).exception()  # let the failing write finish
        with pytest.raises(FileNotFoundError):
            save_figure(_scatter(), tmp_path / "b.png")

    def test_worker_error_is_raised_by_stop(self, tmp_path):
        figures.start(WORKERS)
        save_figure(
            px.scatter(x=[1, 2], y=[1, 2]), f"{tmp_path}/p.png", format="no-such-format"
        )
        with pytest.raises(ValueError):
            figures.stop()


class TestDataUnchanged:
    def test_obs_and_obsm_are_identical_after_plotting(self, pool, tmp_path):
        adata = _adata()
        obs, spatial = adata.obs.copy(), adata.obsm["spatial"].copy()
        helperfuncs.plot_scatter(
            adata, str(tmp_path), "x", None, "celltype", None, None
        )
        helperfuncs.plot_scatter_density(
            adata, str(tmp_path), "y", "celltype", "score", None, None
        )
        pool.wait()
        pd.testing.assert_frame_equal(adata.obs, obs, check_exact=True)
        np.testing.assert_array_equal(adata.obsm["spatial"], spatial)

    def test_large_collection_is_rasterised_only_in_the_worker_pdf(
        self, pool, tmp_path
    ):
        fig = _scatter(n=figures.RASTERIZE_MIN_ELEMENTS)
        save_figure(fig, tmp_path / "big.pdf")
        pool.wait()
        assert b"/Subtype /Image" in (tmp_path / "big.pdf").read_bytes()
        assert not fig.axes[0].collections[0].get_rasterized()

    def test_small_collection_stays_vector_in_pdf(self, pool, tmp_path):
        save_figure(_scatter(n=100), tmp_path / "small.pdf")
        pool.wait()
        assert b"/Subtype /Image" not in (tmp_path / "small.pdf").read_bytes()


class TestWorkers:
    def test_worker_count_is_respected(self, pool, tmp_path):
        for i in range(WORKERS * 4):
            fig = _scatter(seed=i)
            save_figure(fig, tmp_path / f"{i}.png", dpi=20)
            plt.close(fig)
        pool.wait()
        assert pool._executor._max_workers == WORKERS
        assert 0 < len(pool._executor._processes) <= WORKERS
