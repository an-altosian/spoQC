import os
import pickle
import time

import matplotlib
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


def _until(condition, timeout=60):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


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
        _until(lambda: figures._errors)  # the failing write has finished
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
        background = max(1, WORKERS // figures.THREADS_PER_FIGURE_WORKER)
        assert pool._executor._max_workers == background
        assert 0 < len(pool._executor._processes) <= background
        assert background + figures._workers[figures._drain] == WORKERS


class TestImageReduction:
    """Output: 2 in x 2 in figure at 50 dpi, axes filling it, so the image covers 100 x 100 px."""

    def _imshow(self, data, **kwargs):
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(data, **kwargs)
        return fig, ax

    def _reduced_shape(self, fig):
        reduced = figures._reduced_images(fig, 50)
        return None if not reduced else next(iter(reduced.values())).shape[:2]

    @staticmethod
    def _side(samples_per_pixel, n=4000):
        """Reduced side for `n` samples at `samples_per_pixel` per output pixel."""
        k = int(samples_per_pixel / figures.IMAGE_SAMPLES_PER_PIXEL)
        return -(-n // k)

    def test_float_image_is_block_averaged_to_the_output_density(self):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        # 40 samples per output pixel
        assert self._reduced_shape(fig) == (self._side(40),) * 2

    def test_one_pixel_lines_survive_block_averaging(self):
        data = np.zeros((4000, 4000), dtype=np.float32)
        data[::97, :] = 1.0  # 1-px lines, never aligned to a block
        fig, _ = self._imshow(data)
        reduced = next(iter(figures._reduced_images(fig, 50).values())).astype(int)
        background = reduced[-1, -1]  # a block with no line (4000 - 1 is not a multiple of 97)
        rows_with_line = (np.abs(reduced - background).sum(axis=-1) > 0).any(axis=1)
        assert rows_with_line.sum() == len(range(0, 4000, 97))

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
        assert self._reduced_shape(fig) == (self._side(40),) * 2

    def test_image_with_its_own_transform(self):
        # spatialdata-plot style: pixel-unit extent, world units via the image transform
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(np.zeros((4000, 4000), dtype=np.float32),
                  transform=Affine2D().scale(0.25).translate(300, 700) + ax.transData)
        ax.set_xlim(300, 1300)
        ax.set_ylim(1700, 700)
        assert self._reduced_shape(fig) == (self._side(40),) * 2

    def test_zoomed_in_image_is_reduced_less(self):
        fig, ax = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        ax.set_xlim(0, 1000)  # 4x zoom: 10 samples per output pixel
        ax.set_ylim(1000, 0)
        assert self._reduced_shape(fig) == (self._side(10),) * 2

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
        assert figures._executor is None and not figures._held and not figures._in_flight
        assert len(list(tmp_path.glob("*.png"))) < 12

    def test_pending_bytes_stay_under_the_cap(self, pool, tmp_path, monkeypatch):
        blob = len(pickle.dumps(_scatter(n=5_000), protocol=pickle.HIGHEST_PROTOCOL))
        monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 3 * blob)
        for i in range(20):
            save_figure(_scatter(n=5_000, seed=i), tmp_path / f"{i}.png", dpi=50)
            assert figures._pending_bytes() <= 3 * blob
        pool.wait()
        assert len(list(tmp_path.glob("*.png"))) == 20


class TestCli:
    def _main(self, monkeypatch, tmp_path, run, calls):
        from spoqc import cli
        from spoqc.cli_args import build_parser
        from spoqc.core import threads

        # the entry point would have run threads.configure(12); numpy is already loaded here
        monkeypatch.setattr(threads, "N", 12)
        monkeypatch.setattr(cli.figures, "start", lambda n: calls.append(("start", n)))
        monkeypatch.setattr(cli.figures, "stop", lambda: calls.append(("stop",)))
        monkeypatch.setattr(cli.figures, "abort", lambda: calls.append(("abort",)))
        monkeypatch.setattr(cli, "run", run)
        cli.main(build_parser().parse_args(["-i", str(tmp_path), "-o", str(tmp_path), "-t", str(tmp_path), "-n", "12"]))

    def test_pool_gets_the_thread_budget_and_is_stopped(self, monkeypatch, tmp_path):
        calls = []
        self._main(monkeypatch, tmp_path, lambda CONST: None, calls)
        assert calls == [("start", 12), ("stop",)]

    def test_a_failing_step_aborts_the_pool_and_propagates(self, monkeypatch, tmp_path):
        def run(CONST):
            raise RuntimeError("step failed")

        calls = []
        with pytest.raises(RuntimeError, match="step failed"):
            self._main(monkeypatch, tmp_path, run, calls)
        assert calls == [("start", 12), ("abort",)]


def _png(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGBA")).astype(int)


class TestPerPixelAttributes:
    """Everything per-pixel must be reduced with the data, or the worker's render breaks."""

    SHAPE = (3000, 2000)

    def _smooth(self, channels=None):
        yy, xx = np.mgrid[0 : self.SHAPE[0], 0 : self.SHAPE[1]]
        base = (np.sin(xx / 300) * np.cos(yy / 400) + 1) / 2
        return base if channels is None else np.stack([base ** (i + 1) for i in range(channels)], axis=-1)

    def _reduced_vs_exact(self, tmp_path, draw):
        """Save the same figure reduced and exact; return (number of reduced arrays, mean |diff|)."""
        fig = plt.figure(figsize=(3, 2), dpi=100)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        draw(ax)
        n_reduced = len(figures._reduced_images(fig, 100))
        save_figure(fig, tmp_path / "reduced.png", tmp_path / "reduced.pdf")
        save_figure(fig, tmp_path / "exact.png", exact=True)
        plt.close(fig)
        assert (tmp_path / "reduced.pdf").read_bytes().startswith(b"%PDF")
        return n_reduced, np.abs(_png(tmp_path / "reduced.png") - _png(tmp_path / "exact.png")).mean()

    def test_array_alpha_is_reduced_with_the_data(self, tmp_path):
        # ovrlpy._plot_signal_integrity: imshow(integrity, alpha=(signal / t).clip(0, 1) ** 2)
        alpha = self._smooth() ** 2
        n, diff = self._reduced_vs_exact(
            tmp_path, lambda ax: ax.imshow(self._smooth(), alpha=alpha, vmin=0, vmax=1, origin="lower")
        )
        assert n == 2 and diff < 2

    def test_array_alpha_with_a_pool_writes_png_and_pdf_and_leaves_the_caller_alone(self, pool, tmp_path):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        alpha = self._smooth()
        image = ax.imshow(self._smooth().astype(np.float32), alpha=alpha)
        save_figure(fig, tmp_path / "a.png", tmp_path / "a.pdf")
        pool.wait()
        assert image.get_alpha() is alpha and image.get_array().shape == self.SHAPE
        assert (tmp_path / "a.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")

    @pytest.mark.parametrize("channels", [3, 4])
    def test_rgb_and_rgba_float_images(self, tmp_path, channels):
        n, diff = self._reduced_vs_exact(tmp_path, lambda ax: ax.imshow(self._smooth(channels)))
        assert n == 1 and diff < 2

    def test_rgba_float_image_with_array_alpha(self, tmp_path):
        n, diff = self._reduced_vs_exact(tmp_path, lambda ax: ax.imshow(self._smooth(4), alpha=self._smooth()))
        assert n == 2 and diff < 2

    def test_default_extent_and_clim_keep_their_place(self, tmp_path):
        def draw(ax):
            ax.imshow(self._smooth(), cmap="magma").set_clim(0.2, 0.8)

        n, diff = self._reduced_vs_exact(tmp_path, draw)
        assert n == 1 and diff < 2

    def test_masked_image(self, tmp_path):
        data = np.ma.masked_less(self._smooth(), 0.3)
        n, diff = self._reduced_vs_exact(tmp_path, lambda ax: ax.imshow(data))
        assert n == 1 and diff < 3

    def test_non_uniform_image_is_never_reduced(self):
        from matplotlib.image import NonUniformImage

        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        image = NonUniformImage(ax)
        image.set_data(np.linspace(0, 1, 2000), np.linspace(0, 1, 3000), self._smooth())
        ax.add_image(image)
        assert figures._reduced_images(fig, 100) == {}


@pytest.fixture(scope="module")
def small_ovrlp():
    """A real ovrlpy analysis on 60k synthetic transcripts over 1500 x 1500 um (about 10 s)."""
    ovrlpy = pytest.importorskip("ovrlpy")
    rng = np.random.default_rng(0)
    n, side = 60_000, 1500.0
    x, y = rng.random(n) * side, rng.random(n) * side
    domain = (x // 500).astype(int) + 3 * (y // 500).astype(int)
    genes = np.array([f"g{i}" for i in range(12)])
    df = pd.DataFrame({"gene": genes[(domain % 4) * 3 + rng.integers(0, 3, n)], "x": x, "y": y, "z": rng.normal(5, 1, n)})
    ovrlp = ovrlpy.Ovrlp(df, min_distance=8, n_components=3, n_workers=2, random_state=0)
    ovrlp.analyse()
    return ovrlpy, ovrlp


class TestOvrlpyFigures:
    def test_region_of_interest_with_a_dense_integrity_map_is_written(self, pool, small_ovrlp, tmp_path):
        # Regression: doublet_score.py doublet_case_*_zoomed crashed in the worker with
        # "operands could not be broadcast" because the array alpha was not reduced.
        ovrlpy, ovrlp = small_ovrlp
        fig = ovrlpy.plot_region_of_interest(ovrlp, 750, 750, window_size=750, figsize=(6, 4))
        assert len(figures._reduced_images(fig, 100)) == 2  # the integrity data and its alpha
        save_figure(fig, tmp_path / "roi.png", tmp_path / "roi.pdf")
        pool.wait()
        assert (tmp_path / "roi.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "roi.pdf").read_bytes().startswith(b"%PDF")

    def test_signal_integrity_map_is_written(self, pool, small_ovrlp, tmp_path):
        ovrlpy, ovrlp = small_ovrlp
        fig = ovrlpy.plot_signal_integrity(ovrlp, signal_threshold=2)
        save_figure(fig, tmp_path / "map.png", tmp_path / "map.pdf")
        pool.wait()
        assert (tmp_path / "map.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "map.pdf").read_bytes().startswith(b"%PDF")


def _worker_thread_pools():
    import cv2
    import numba
    import polars
    from threadpoolctl import threadpool_info

    from spoqc.core import threads

    return {
        "N": threads.N,
        "polars": polars.thread_pool_size(),
        "numba": numba.config.NUMBA_NUM_THREADS,
        "cv2": cv2.getNumThreads(),
        "native": sorted({p["num_threads"] for p in threadpool_info()}),
    }


def test_figure_workers_run_single_threaded(pool):
    report = pool._executor.submit(_worker_thread_pools).result()
    assert report == {"N": 1, "polars": 1, "numba": 1, "cv2": 1, "native": [1]}


class TestColourAverage:
    def test_kernel_matches_matplotlib_colours_averaged_premultiplied(self):
        rng = np.random.default_rng(0)
        data = np.ma.masked_array(rng.normal(0.5, 0.4, (37, 53)), mask=rng.random((37, 53)) < 0.1)
        cmap = plt.get_cmap("hot").with_extremes(under="blue", over="green", bad=(1, 0, 0, 0.5))
        for clip in (False, True):
            fig, ax = plt.subplots()
            image = ax.imshow(data, cmap=cmap, norm=matplotlib.colors.Normalize(0.1, 0.9, clip=clip))
            got = figures._colour_average(image, 4).astype(float) / 255
            rgba = image.to_rgba(data)  # matplotlib's own colours, bad/under/over included
            rgba[..., :3] *= rgba[..., 3:]
            starts = [np.arange(0, n, 4) for n in data.shape]
            sums = np.add.reduceat(np.add.reduceat(rgba, starts[0], axis=0), starts[1], axis=1)
            counts = np.multiply.outer(*[np.diff(np.append(s, n)) for s, n in zip(starts, data.shape)])
            want = np.concatenate([sums[..., :3] / sums[..., 3:], (sums[..., 3] / counts)[..., None]], axis=-1)
            np.testing.assert_allclose(got, want, atol=0.5 / 255 + 1e-9)
            plt.close(fig)

    def test_non_linear_norm_is_never_reduced(self):
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(np.random.default_rng(0).random((4000, 4000)) + 0.1, norm=matplotlib.colors.LogNorm())
        assert figures._reduced_images(fig, 50) == {}

    def test_noisy_texture_looks_like_the_full_resolution_render(self, tmp_path):
        # LBP-like per-pixel noise at the full-scale hqtr density (18 samples per output pixel):
        # averaging values would draw it as one flat mid colour; averaging colours does not.
        codes = np.random.default_rng(0).integers(0, 102, (3600, 3600)).astype(np.float64)
        fig = plt.figure(figsize=(2, 2), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(codes, cmap="hot")
        save_figure(fig, tmp_path / "reduced.png")
        save_figure(fig, tmp_path / "exact.png", exact=True)
        plt.close(fig)
        assert np.abs(_png(tmp_path / "reduced.png") - _png(tmp_path / "exact.png")).mean() < 3


class TestDrain:
    def _slow_figures(self, tmp_path, n):
        for i in range(n):
            fig = _scatter(n=100_000, seed=i)
            save_figure(fig, tmp_path / f"{i}.png", dpi=150)
            plt.close(fig)

    def test_wait_moves_queued_figures_to_a_drain_pool_of_the_rest_of_the_budget(self, tmp_path, monkeypatch):
        pools = []
        real_pool = figures._pool
        monkeypatch.setattr(figures, "_pool", lambda n: pools.append(n) or real_pool(n))
        figures.start(4)
        in_flight = []
        real_dispatch = figures._dispatch
        monkeypatch.setattr(figures, "_dispatch", lambda: real_dispatch() or in_flight.append(len(figures._in_flight)))
        try:
            self._slow_figures(tmp_path, 8)
            computing = max(in_flight)
            drain_submit = figures._drain.submit
            moved = []
            monkeypatch.setattr(figures._drain, "submit", lambda *a: moved.append(a) or drain_submit(*a))
            figures.wait()
        finally:
            figures.stop()
        assert pools == [1, 3]  # background 4 // THREADS_PER_FIGURE_WORKER, drain the other 3
        assert computing == 1  # while the main thread computes: the background worker only
        assert max(in_flight) == 4 and moved  # while it waits: the whole budget
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{i}.png" for i in range(8))

    def test_a_budget_of_one_thread_has_no_drain_pool(self, tmp_path, monkeypatch):
        pools = []
        real_pool = figures._pool
        monkeypatch.setattr(figures, "_pool", lambda n: pools.append(n) or real_pool(n))
        figures.start(1)
        try:
            self._slow_figures(tmp_path, 3)
        finally:
            figures.stop()
        assert pools == [1]
        assert len(list(tmp_path.glob("*.png"))) == 3

    def test_drain_pool_errors_propagate_from_wait(self, tmp_path):
        figures.start(4)
        try:
            self._slow_figures(tmp_path, 3)
            for i in range(3):
                save_figure(_scatter(), tmp_path / "missing_dir" / f"{i}.png")
            with pytest.raises(FileNotFoundError):
                figures.wait()
        finally:
            figures.abort()
