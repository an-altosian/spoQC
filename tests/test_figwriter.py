import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from plotly.basedatatypes import BaseFigure

from spoqc import figwriter, helperfuncs

WORKERS = 2


@pytest.fixture
def writer():
    figwriter.start(WORKERS)
    yield figwriter
    if figwriter._executor is not None:
        figwriter.stop()


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
    def test_pyplot_and_figure_savefig_write_png_and_pdf(self, writer, tmp_path):
        fig = _scatter()
        plt.savefig(tmp_path / "a.png", dpi=50)
        plt.savefig(f"{tmp_path}/a.pdf")
        fig.savefig(os.path.join(tmp_path, "b.png"), bbox_inches="tight")
        plt.close(fig)
        writer.wait()
        for name in ["a.png", "a.pdf", "b.png"]:
            assert (tmp_path / name).stat().st_size > 0, name

    def test_plotly_write_image_writes_png_and_pdf(self, writer, tmp_path):
        fig = px.scatter(x=[1, 2, 3], y=[3, 1, 2])
        fig.write_image(f"{tmp_path}/p.png", scale=1)
        fig.write_image(f"{tmp_path}/p.pdf", scale=1)
        writer.wait()
        assert (tmp_path / "p.png").stat().st_size > 0
        assert (tmp_path / "p.pdf").read_bytes().startswith(b"%PDF")

    def test_many_figures_beyond_the_pending_bound_all_land(self, writer, tmp_path):
        n = WORKERS * figwriter.MAX_PENDING_PER_WORKER * 3
        for i in range(n):
            fig = _scatter(seed=i)
            fig.savefig(tmp_path / f"{i}.png", dpi=20)
            plt.close(fig)
        writer.wait()
        assert sorted(os.listdir(tmp_path)) == sorted(f"{i}.png" for i in range(n))

    def test_buffer_target_is_filled_before_savefig_returns(self, writer):
        buf = io.BytesIO()
        _scatter().savefig(buf, format="png")
        assert buf.getvalue().startswith(b"\x89PNG")

    def test_real_site_plot_scatter_writes_both_files(self, writer, tmp_path):
        helperfuncs.plot_scatter(
            _adata(), str(tmp_path), "x", None, "celltype", None, "title"
        )
        writer.wait()
        for ext in ["png", "pdf"]:
            assert (tmp_path / f"scatterplot_x.{ext}").stat().st_size > 0


class TestErrors:
    def test_worker_error_is_raised_by_wait(self, writer, tmp_path):
        _scatter().savefig(tmp_path / "missing_dir" / "a.png")
        with pytest.raises(FileNotFoundError):
            writer.wait()

    def test_worker_error_is_raised_by_the_next_savefig(self, writer, tmp_path):
        _scatter().savefig(tmp_path / "missing_dir" / "a.png")
        writer._pending.copy().pop().exception()  # let the failing write finish
        with pytest.raises(FileNotFoundError):
            _scatter().savefig(tmp_path / "b.png")

    def test_worker_error_is_raised_by_stop(self, tmp_path):
        figwriter.start(WORKERS)
        fig = px.scatter(x=[1, 2], y=[1, 2])
        fig.write_image(f"{tmp_path}/p.png", format="no-such-format")
        with pytest.raises(ValueError):
            figwriter.stop()


class TestDataUnchanged:
    def test_obs_and_obsm_are_identical_after_plotting(self, writer, tmp_path):
        adata = _adata()
        obs, spatial = adata.obs.copy(), adata.obsm["spatial"].copy()
        helperfuncs.plot_scatter(
            adata, str(tmp_path), "x", None, "celltype", None, None
        )
        helperfuncs.plot_scatter_density(
            adata, str(tmp_path), "y", "celltype", "score", None, None
        )
        writer.wait()
        pd.testing.assert_frame_equal(adata.obs, obs, check_exact=True)
        np.testing.assert_array_equal(adata.obsm["spatial"], spatial)

    def test_large_collection_is_rasterised_only_in_the_worker_pdf(
        self, writer, tmp_path
    ):
        fig = _scatter(n=figwriter.RASTERIZE_MIN_ELEMENTS)
        fig.savefig(tmp_path / "big.pdf")
        writer.wait()
        assert b"/Subtype /Image" in (tmp_path / "big.pdf").read_bytes()
        assert not fig.axes[0].collections[0].get_rasterized()


class TestWorkers:
    def test_worker_count_is_respected(self, writer, tmp_path):
        for i in range(WORKERS * 4):
            fig = _scatter(seed=i)
            fig.savefig(tmp_path / f"{i}.png", dpi=20)
            plt.close(fig)
        writer.wait()
        assert writer._executor._max_workers == WORKERS
        assert 0 < len(writer._executor._processes) <= WORKERS

    def test_stop_restores_synchronous_writing(self, tmp_path):
        figwriter.start(WORKERS)
        figwriter.stop()
        assert Figure.savefig is figwriter._ORIGINALS["Figure.savefig"]
        assert plt.savefig is figwriter._ORIGINALS["pyplot.savefig"]
        assert BaseFigure.write_image is figwriter._ORIGINALS["BaseFigure.write_image"]
        _scatter().savefig(tmp_path / "sync.png")
        assert (tmp_path / "sync.png").stat().st_size > 0
