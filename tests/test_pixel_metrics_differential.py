"""Differential test: structural image analysis (hqpr/hqtr pixel metrics) against the verbatim original.

tests/reference_pixel_metrics_fca01f5.py holds perf/integration fca01f5's metric wrappers and
start_image_struc_analyis. Both versions run on the same synthetic images; every parquet written
to the metrices folder (each is a pixel-clustering feature) must match byte for byte, with the
same file names, and every plot_pixels call must receive the same array, dtype and arguments.
"""

import os
from types import SimpleNamespace

import cv2
import dask.array as da
import numpy as np
import pyarrow.parquet as pq
import pytest

import reference_pixel_metrics_fca01f5 as reference
from spoqc import helperfuncs
from spoqc.image_analysis import _slidingwindow, structure_analysis
from spoqc.metrics.image import pixel_metrics, utility
from spoqc.metrics.transcript_density import transcript_density_image

IMAGE_TYPE, RESOLUTION = "morphology_focus", "s0"
SHAPES = [(1, 9), (9, 1), (2, 3), (5, 5), (7, 4), (4, 9), (37, 23), (1030, 517)]
HQPR_KINDS = ["tissue", "full_range", "constant", "edges"]
HQTR_KINDS = ["density", "zeros"]


def make_image(kind, shape, rng):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    if (
        kind == "tissue"
    ):  # realistic range with blobs, noise and a zero background patch
        img = (
            90
            + 800 * (np.sin(yy / 5.0) * np.cos(xx / 3.0)) ** 2
            + rng.gamma(2.0, 30.0, shape)
        )
        img[: shape[0] // 3, : shape[1] // 3] = 0
        return img.astype(np.uint16)
    if kind == "full_range":  # hits 0 and 65535 (the uint16 +1 wraps)
        img = rng.integers(0, 65536, shape, dtype=np.uint16)
        img.flat[0], img.flat[-1] = 0, 65535
        return img
    if kind == "constant":
        return np.full(shape, 7, dtype=np.uint16)
    if kind == "edges":
        return np.where((yy // 3 + xx // 2) % 2 == 0, 3000, 12).astype(np.uint16)
    if kind == "density":  # hqtr: int64 transcript densities, including values > 255
        img = rng.poisson(3.0, shape).astype(np.int64)
        img[shape[0] // 2 :, shape[1] // 2 :] *= 120
        return img
    if kind == "zeros":
        return np.zeros(shape, dtype=np.int64)
    raise ValueError(kind)


class FakeSdata(dict):
    """The parts of a SpatialData object start_image_struc_analyis reads."""

    def __init__(self, image):
        image3d = image[None]
        super().__init__(
            {
                IMAGE_TYPE: {
                    RESOLUTION: SimpleNamespace(
                        image=SimpleNamespace(
                            values=image3d,
                            data=da.from_array(image3d, chunks=(1, 256, 256)),
                        )
                    )
                }
            }
        )
        self.points = {"transcripts": SimpleNamespace(compute=lambda: None)}


def record_calls(monkeypatch, density=None):
    """Record plot_pixels / save_figure calls instead of drawing; serve `density` as the hqtr image."""
    calls = []

    def plot_pixels(
        figure_path, image, imagedim, suffix, title, cmap, axis_off, baroff, **kwargs
    ):
        image = np.asarray(image)
        kwargs = {"points": None, "legend_dict": None, "flip": False, **kwargs}  # plot_pixels defaults
        calls.append(
            (
                "plot_pixels",
                suffix,
                title,
                cmap,
                axis_off,
                baroff,
                kwargs,
                image.dtype,
                image.copy(),
            )
        )

    monkeypatch.setattr(helperfuncs, "plot_pixels", plot_pixels)
    monkeypatch.setattr(
        helperfuncs,
        "plot_scatter_by_category",
        lambda *a, **k: calls.append(("scatter",)),
    )
    for module in (reference, utility):
        monkeypatch.setattr(
            module,
            "save_figure",
            lambda fig, *paths, **k: calls.append(("save_figure", paths)),
        )
    if density is not None:
        monkeypatch.setattr(
            transcript_density_image,
            "generate_transcript_density_image",
            lambda *a, **k: density.flatten(),
        )
    return calls


def run(func, base, image, modality, monkeypatch, **kwargs):
    """Run one start_image_struc_analyis; return (plot calls, {file: parquet table}, figure files)."""
    staining = "0" if modality == "hqpr" else None
    metrices = (
        f"{base}/tmp/metrices/{modality}/{staining}"
        if staining
        else f"{base}/tmp/metrices/{modality}"
    )
    figures = (
        f"{base}/fig/{modality}/{modality}_metrices/{staining}"
        if staining
        else f"{base}/fig/{modality}/{modality}_metrices"
    )
    os.makedirs(metrices)
    os.makedirs(figures)
    calls = record_calls(monkeypatch, density=image if modality == "hqtr" else None)
    dim_x, dim_y = image.shape
    func(
        FakeSdata(image), f"{base}/fig", f"{base}/tmp", modality, IMAGE_TYPE, RESOLUTION,
        None, dim_x, dim_y, True, **kwargs, **({"staining": staining} if staining else {}),
    )  # fmt: skip
    tables = {f: pq.read_table(f"{metrices}/{f}") for f in sorted(os.listdir(metrices))}
    calls = [
        tuple(
            str(c).replace(base, "") if isinstance(c, (str, tuple)) else c for c in call
        )
        for call in calls
    ]
    return calls, tables, sorted(os.listdir(figures))


def assert_same_calls(new, old):
    assert len(new) == len(old), (len(new), len(old))
    for n, o in zip(new, old):
        if n[0] != "plot_pixels":
            assert n == o
            continue
        assert n[:8] == o[:8], (n[:8], o[:8])
        assert n[8].shape == o[8].shape and n[8].tobytes() == o[8].tobytes(), n[1]


def assert_same_tables(new, old):
    assert list(new) == list(old)
    for name in new:
        assert new[name].schema.equals(old[name].schema, check_metadata=True), name
        assert new[name].equals(old[name]), name
        a, b = new[name].column(0).to_numpy(), old[name].column(0).to_numpy()
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name


def compare(image, modality, tmp_path, monkeypatch, threads):
    # cv2.GaussianBlur on uint16 (relevance) varies call to call on small images at >= 32 cv2 threads
    # (cv2's default here is every host CPU), in the original too; at <= 8 threads it matches 1 thread.
    previous = cv2.getNumThreads()
    cv2.setNumThreads(threads)
    try:
        return _compare(image, modality, tmp_path, monkeypatch, threads)
    finally:
        cv2.setNumThreads(previous)


def _compare(image, modality, tmp_path, monkeypatch, threads):
    old = run(
        reference.start_image_struc_analyis,
        f"{tmp_path}/old",
        image,
        modality,
        monkeypatch,
    )
    monkeypatch.undo()
    new = run(structure_analysis.start_image_struc_analyis, f"{tmp_path}/new", image, modality, monkeypatch,
              threads=threads)  # fmt: skip
    assert_same_calls(new[0], old[0])
    assert_same_tables(new[1], old[1])
    assert new[2] == old[2]
    return new


@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", HQPR_KINDS)
def test_hqpr_matches_original(kind, shape, numba_threads, tmp_path, monkeypatch):
    image = make_image(kind, shape, np.random.default_rng(sum(shape)))
    _, tables, _ = compare(image, "hqpr", tmp_path, monkeypatch, numba_threads)
    expected = [
        "edge_strength",
        "energy",
        "entropy",
        "homogenity",
        "intensity",
        "lbp",
        "relevance",
        "uniformity",
    ]
    assert list(tables) == [f"{m}_output_hqpr_0.parquet" for m in expected]


@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", HQTR_KINDS)
def test_hqtr_matches_original(kind, shape, numba_threads, tmp_path, monkeypatch):
    image = make_image(kind, shape, np.random.default_rng(sum(shape)))
    _, tables, _ = compare(image, "hqtr", tmp_path, monkeypatch, numba_threads)
    expected = ["edge_strength", "energy", "entropy", "homogenity", "lbp", "relevance", "transcript_density",
                "uniformity"]  # fmt: skip
    assert list(tables) == [f"{m}_output_hqtr.parquet" for m in expected]


def test_texture_metrics_constant_window_is_perfectly_homogeneous():
    entropy, uniformity, homogeneity = _slidingwindow.texture_metrics(
        np.full((6, 6), 3, np.uint8), 5
    )
    assert (
        (homogeneity == 1).all()
        and (entropy == 0).all()
        and entropy.dtype == np.float32
    )
    assert (uniformity == np.float32(np.log(25))).all()  # kl = -(1 * log(1 / (1/25)))


def test_row_chunked_matches_whole_image_with_more_chunks_than_rows():
    image = np.arange(3 * 11, dtype=np.float64).reshape(3, 11) ** 2
    whole = pixel_metrics.lbp(image, 100, 3, 1)[0]
    assert pixel_metrics.lbp(image, 100, 3, 4)[0].tobytes() == whole.tobytes()
