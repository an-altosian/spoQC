"""The PNG compress-level default must change the file and never the image.

These are differential tests: each one reproduces the ORIGINAL behaviour (matplotlib's
own savefig, reached through the shim's `__wrapped__`) and asserts the patched path
produces byte-identical *pixels*. That is the only property that makes this a
performance change rather than an output change.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import spoqc  # noqa: F401  -- importing installs the shim
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


@pytest.fixture
def panel():
    """A figure shaped like spoQC's image panels: imshow + colorbar + title."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(rng.random((200, 200)), cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("panel")
    yield fig
    plt.close(fig)


def _pixels(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGBA"))


def test_shim_is_installed():
    assert hasattr(Figure.savefig, "__wrapped__"), "spoqc import should wrap Figure.savefig"


def test_png_pixels_identical_to_matplotlib_default(panel, tmp_path):
    """The whole justification for the change: same pixels, different bytes."""
    patched = tmp_path / "patched.png"
    original = tmp_path / "original.png"

    panel.savefig(patched, dpi=150)
    # Reach the unpatched implementation to reproduce matplotlib's level-6 default.
    Figure.savefig.__wrapped__(panel, original, dpi=150)

    assert np.array_equal(_pixels(patched), _pixels(original)), "compress_level changed pixels"


@pytest.mark.parametrize("level", [0, 1, 3, 6, 9])
def test_pixels_identical_at_every_compress_level(panel, tmp_path, level):
    reference = tmp_path / "ref.png"
    Figure.savefig.__wrapped__(panel, reference, dpi=150)
    candidate = tmp_path / f"c{level}.png"
    panel.savefig(candidate, dpi=150, pil_kwargs={"compress_level": level})
    assert np.array_equal(_pixels(candidate), _pixels(reference))


@pytest.fixture
def observed(monkeypatch):
    """Capture the kwargs that actually reach matplotlib's real savefig.

    The spy must sit INSIDE the shim, not outside it: a spy installed over the shim
    records what the caller passed, which proves nothing about what the shim did.
    So substitute the spy for the raw implementation, then re-install the shim on top.
    """
    seen = {}
    raw = Figure.savefig.__wrapped__

    def spy(self, fname, **kwargs):
        seen.update(kwargs.get("pil_kwargs") or {})
        return raw(self, fname, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    spoqc._install_png_compress_default()
    yield seen


def test_default_is_applied_to_png(panel, tmp_path, observed):
    """A call site passing no pil_kwargs still reaches matplotlib at the cheap level."""
    panel.savefig(tmp_path / "x.png", dpi=100)
    assert observed.get("compress_level") == 1 == spoqc._PNG_COMPRESS_LEVEL


def test_explicit_caller_level_wins(panel, tmp_path, observed):
    """A call site that deliberately asks for level 9 must still reach level 9."""
    panel.savefig(tmp_path / "y.png", dpi=100, pil_kwargs={"compress_level": 9})
    assert observed["compress_level"] == 9


def test_caller_pil_kwargs_dict_not_mutated(panel, tmp_path):
    caller = {}
    panel.savefig(tmp_path / "z.png", dpi=100, pil_kwargs=caller)
    assert caller == {}, "the shim must not write into the caller's dict"


def test_pdf_is_untouched(panel, tmp_path):
    """PDF compression is the separate pdf.compression rcParam; pil_kwargs would raise."""
    panel.savefig(tmp_path / "a.pdf")  # would TypeError if pil_kwargs were injected
    assert (tmp_path / "a.pdf").stat().st_size > 0
