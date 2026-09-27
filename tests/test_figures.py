import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from matplotlib.transforms import Affine2D
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

    def test_many_figures_beyond_the_pending_bound_all_land(self, pool, tmp_path, monkeypatch):
        monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 1)
        n = WORKERS * 6
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


class TestImageReduction:
    """Output: 2 in x 2 in figure at 50 dpi, axes filling it, so the image covers 100 x 100 px."""

    def _imshow(self, data, **kwargs):
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(data, **kwargs)
        return fig, ax

    def _reduced_shape(self, fig):
        reduced = figures._reduced_images(fig, 50)
        return None if not reduced else next(iter(reduced.values())).shape

    def test_float_image_is_block_averaged_to_the_output_density(self):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        # 40 samples per output pixel -> blocks of 40 // 4 = 10
        assert self._reduced_shape(fig) == (400, 400)

    def test_one_pixel_lines_survive_block_averaging(self):
        data = np.zeros((4000, 4000), dtype=np.float32)
        data[::97, :] = 1.0  # 1-px lines, never aligned to a block
        fig, _ = self._imshow(data)
        reduced = next(iter(figures._reduced_images(fig, 50).values()))
        assert (reduced.max(axis=1) > 0).sum() == len(range(0, 4000, 97))

    @pytest.mark.parametrize("dtype", [np.uint8, np.int32, bool])
    def test_integer_bool_and_label_images_are_never_reduced(self, dtype):
        data = np.zeros((4000, 4000), dtype=dtype)
        data[::97, :] = 1
        fig, _ = self._imshow(data)
        assert self._reduced_shape(fig) is None

    @pytest.mark.parametrize("interpolation", ["nearest", "none"])
    def test_nearest_and_none_interpolation_are_never_reduced(self, interpolation):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.float32), interpolation=interpolation)
        assert self._reduced_shape(fig) is None

    def test_world_unit_extent_translated_far_from_origin(self):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.float32), extent=(1e5, 1e5 + 50, 2e5, 2e5 + 50))
        assert self._reduced_shape(fig) == (400, 400)

    def test_image_with_its_own_transform(self):
        # spatialdata-plot style: pixel-unit extent, world units via the image transform
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(np.zeros((4000, 4000), dtype=np.float32),
                  transform=Affine2D().scale(0.25).translate(300, 700) + ax.transData)
        ax.set_xlim(300, 1300)
        ax.set_ylim(1700, 700)
        assert self._reduced_shape(fig) == (400, 400)

    def test_zoomed_in_image_is_reduced_less(self):
        fig, ax = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        ax.set_xlim(0, 1000)  # 4x zoom: 10 samples per output pixel -> blocks of 2
        ax.set_ylim(1000, 0)
        assert self._reduced_shape(fig) == (2000, 2000)

    def test_strongly_zoomed_image_keeps_full_resolution(self):
        fig, ax = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        ax.set_xlim(0, 50)
        ax.set_ylim(50, 0)
        assert self._reduced_shape(fig) is None

    def test_masked_samples_are_excluded_from_block_means(self):
        data = np.ma.masked_array(np.full((8, 8), 2.0), mask=np.zeros((8, 8), bool))
        data[:4, :4] = np.ma.masked
        data[4:, 4:] = 6.0
        reduced = figures._block_mean(data, 4)
        assert reduced.mask.tolist() == [[True, False], [False, False]]
        assert reduced[1, 1] == 6.0

    def test_callers_figure_is_not_modified_and_colour_scale_is_kept(self, pool, tmp_path):
        data = np.zeros((4000, 4000), dtype=np.float32)
        data[1, 1] = 7.0
        fig, ax = self._imshow(data, cmap="viridis")
        image = ax.images[0]
        before = image.get_array()
        save_figure(fig, tmp_path / "img.png", tmp_path / "img.pdf", dpi=50)
        pool.wait()
        assert image.get_array() is before and before.shape == (4000, 4000)
        assert (image.norm.vmin, image.norm.vmax) == (0.0, 7.0)
        assert (tmp_path / "img.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "img.pdf").read_bytes().startswith(b"%PDF")

    def test_exact_writes_what_savefig_writes(self, pool, tmp_path):
        rng = np.random.default_rng(0)
        fig, ax = self._imshow(rng.random((2000, 2000)).astype(np.float32))
        ax.scatter(rng.random(figures.RASTERIZE_MIN_ELEMENTS) * 2000, rng.random(figures.RASTERIZE_MIN_ELEMENTS) * 2000, s=1)
        metadata = {"Software": None}
        save_figure(fig, tmp_path / "exact.png", exact=True, metadata=metadata)
        save_figure(fig, tmp_path / "exact.pdf", exact=True)
        pool.wait()
        fig.savefig(tmp_path / "direct.png", metadata=metadata)
        assert (tmp_path / "exact.png").read_bytes() == (tmp_path / "direct.png").read_bytes()
        assert b"/Subtype /Image" in (tmp_path / "exact.pdf").read_bytes()  # the imshow itself
        assert (tmp_path / "exact.pdf").read_bytes().count(b"/Subtype /Image") == 1  # scatter stays vector


class TestReadBacks:
    def test_sort_files_moves_every_figure_even_while_writes_are_in_flight(self, pool, tmp_path):
        names = [f"{prefix}_{i}.png" for prefix in ("umap", "barplot", "violin") for i in range(4)]
        for name in names:
            fig = _scatter(n=20_000, seed=len(name))
            save_figure(fig, tmp_path / name, dpi=150)
            plt.close(fig)
        helperfuncs.sort_files(str(tmp_path), "prefix", ["res.txt", "done.txt"])
        assert sorted(p.name for p in tmp_path.iterdir()) == ["barplot", "umap", "violin"]
        assert sorted(p.name for p in tmp_path.glob("*/*.png")) == sorted(names)

    def test_stripe_thickness_reads_the_figure_after_it_lands_and_matches_serial(self, tmp_path):
        from spoqc.subworkflows import qc_wsi

        def thickness(out):
            out.mkdir()
            rng = np.random.default_rng(0)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(rng.random((600, 600)).astype(np.float32) > 0.7, cmap="viridis")
            ax.axis("off")
            save_figure(fig, out / "input.png", exact=True, bbox_inches="tight")
            plt.close(fig)
            return qc_wsi.measure_stripe_thickness_and_black_area(str(out / "input.png"), np.array([68, 1, 84]), str(out))

        serial = thickness(tmp_path / "serial")
        figures.start(WORKERS)
        try:
            pooled = thickness(tmp_path / "pooled")
        finally:
            figures.stop()
        assert pooled == serial


class TestPoolLifecycle:
    def test_without_start_figures_are_written_before_save_returns(self, tmp_path):
        assert figures._executor is None
        save_figure(_scatter(), tmp_path / "sync.png")
        assert (tmp_path / "sync.png").read_bytes().startswith(b"\x89PNG")

    def test_without_start_worker_errors_raise_at_the_call(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            save_figure(_scatter(), tmp_path / "missing_dir" / "a.png")

    def test_abort_cancels_queued_writes_and_closes_the_pool(self, tmp_path):
        figures.start(1)
        for i in range(12):
            save_figure(_scatter(n=50_000, seed=i), tmp_path / f"{i}.png", dpi=200)
        figures.abort()
        assert figures._executor is None and not figures._pending
        assert len(list(tmp_path.glob("*.png"))) < 12

    def test_pending_bytes_stay_under_the_cap(self, pool, tmp_path, monkeypatch):
        blob = len(pickle.dumps(_scatter(n=5_000), protocol=pickle.HIGHEST_PROTOCOL))
        monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 3 * blob)
        for i in range(20):
            save_figure(_scatter(n=5_000, seed=i), tmp_path / f"{i}.png", dpi=50)
            assert sum(figures._pending.values()) <= 3 * blob
        pool.wait()
        assert len(list(tmp_path.glob("*.png"))) == 20


class TestCli:
    def _main(self, monkeypatch, tmp_path, run, calls):
        from spoqc import cli

        monkeypatch.setattr(cli.figures, "start", lambda n: calls.append(("start", n)))
        monkeypatch.setattr(cli.figures, "stop", lambda: calls.append(("stop",)))
        monkeypatch.setattr(cli.figures, "abort", lambda: calls.append(("abort",)))
        monkeypatch.setattr(cli, "run", run)
        cli.main(["-i", str(tmp_path), "-o", str(tmp_path), "-t", str(tmp_path), "-n", "12"])

    def test_pool_gets_a_share_of_the_thread_budget_and_is_stopped(self, monkeypatch, tmp_path):
        calls = []
        self._main(monkeypatch, tmp_path, lambda CONST: None, calls)
        assert calls == [("start", 12 // figures.THREADS_PER_FIGURE_WORKER), ("stop",)]

    def test_a_failing_step_aborts_the_pool_and_propagates(self, monkeypatch, tmp_path):
        def run(CONST):
            raise RuntimeError("step failed")

        calls = []
        with pytest.raises(RuntimeError, match="step failed"):
            self._main(monkeypatch, tmp_path, run, calls)
        assert calls == [("start", 12 // figures.THREADS_PER_FIGURE_WORKER), ("abort",)]
