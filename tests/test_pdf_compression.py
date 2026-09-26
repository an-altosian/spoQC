"""PDF stream compression must change the file and never the rendered page.

`pdf.compression` selects the zlib level for Flate-encoded content streams. Flate is
lossless, so the rendered page cannot change -- but "cannot in principle" is not
evidence, so these tests rasterise both PDFs and compare pixels.
"""
from __future__ import annotations

import shutil
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import spoqc  # noqa: F401  -- importing sets the rcParam

pdftoppm = pytest.mark.skipif(
    shutil.which("pdftoppm") is None, reason="needs poppler's pdftoppm to rasterise"
)


@pytest.fixture
def panel():
    """Raster image plus text, so both the image and font paths are exercised."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(rng.random((150, 150)), cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("panel 0.42")
    ax.set_xlabel("x")
    yield fig
    plt.close(fig)


def _write(panel, path, level):
    with matplotlib.rc_context({"pdf.compression": level}):
        panel.savefig(path, bbox_inches="tight", dpi=150)


def _raster(path, tmp_path):
    stem = tmp_path / f"{path.stem}-raster"
    subprocess.run(
        ["pdftoppm", "-r", "120", "-png", "-singlefile", str(path), str(stem)], check=True
    )
    with Image.open(f"{stem}.png") as handle:
        return np.asarray(handle.convert("RGB")).astype(int)


def test_rcparam_is_set():
    assert matplotlib.rcParams["pdf.compression"] == 0


@pdftoppm
@pytest.mark.parametrize("level", [1, 6, 9])
def test_rendered_page_identical_to_compressed(panel, tmp_path, level):
    """The whole justification: level 0 must rasterise identically to a compressed PDF."""
    fast = tmp_path / "fast.pdf"
    reference = tmp_path / f"ref{level}.pdf"
    _write(panel, fast, 0)
    _write(panel, reference, level)

    a, b = _raster(fast, tmp_path), _raster(reference, tmp_path)
    assert a.shape == b.shape, "page geometry changed"
    assert np.array_equal(a, b), f"level 0 differs from level {level}, max {np.abs(a - b).max()}"


def test_uncompressed_is_larger(panel, tmp_path):
    """Sanity check that the rcParam is actually taking effect on the bytes."""
    fast = tmp_path / "fast.pdf"
    small = tmp_path / "small.pdf"
    _write(panel, fast, 0)
    _write(panel, small, 6)
    assert fast.stat().st_size > small.stat().st_size
