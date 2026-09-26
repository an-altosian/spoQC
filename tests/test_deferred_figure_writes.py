"""Deferred figure writes must be invisible to everything except timing.

The hazard this guards: a deferred write is not on disk yet, so any code that reads a
figure back sees a missing file. spoQC does that in one place --
qc_wsi.generate_input writes input_domain_thickness_analysis.png and
measure_stripe_thickness_and_black_area then cv2.imread's it. Without a barrier,
imread returns None and OpenCV fails with "!_src.empty() in function 'inRange'".

That regression reached a full-scale run before it was caught, which is why it is
pinned here.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

import spoqc  # noqa: F401  -- installs the queuing shim
import matplotlib.pyplot as plt  # noqa: E402
from spoqc import helperfuncs  # noqa: E402


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(rng.random((80, 80)), cmap="viridis")
    yield fig
    plt.close(fig)


@pytest.fixture(autouse=True)
def _drain():
    helperfuncs.flush_figures()
    yield
    helperfuncs.shutdown_figure_pool()


def _md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def test_ensure_written_makes_a_queued_figure_readable(panel, tmp_path, monkeypatch):
    """The exact qc_wsi pattern: write a figure, then read it back in the same stage."""
    monkeypatch.setenv("SPOQC_DEFER_FIGURES", "1")
    path = tmp_path / "input_domain_thickness_analysis.png"
    panel.savefig(path)

    # Precondition: the write really was deferred. Without this the test would pass
    # trivially whenever the figure fell back to an inline write, which is how an
    # earlier version of it failed to catch the very bug it was written for.
    assert helperfuncs._FIG_QUEUE, "figure was not queued; deferral is not active"
    assert not path.exists(), "deferred figure should not be on disk yet"

    helperfuncs.ensure_written(path)
    assert path.exists(), "ensure_written must flush the queue"
    assert os.path.getsize(path) > 0
    # cv2.imread returns None on a missing file; prove a real decoder can read it
    cv2 = pytest.importorskip("cv2")
    assert cv2.imread(str(path)) is not None


def test_deferred_output_is_byte_identical_to_direct(panel, tmp_path, monkeypatch):
    monkeypatch.setenv("SPOQC_DEFER_FIGURES", "0")
    direct = tmp_path / "direct.png"
    panel.savefig(direct, dpi=150, bbox_inches="tight")

    monkeypatch.setenv("SPOQC_DEFER_FIGURES", "1")
    deferred = tmp_path / "deferred.png"
    panel.savefig(deferred, dpi=150, bbox_inches="tight")
    helperfuncs.flush_figures()

    assert _md5(direct) == _md5(deferred)


def test_disabling_deferral_writes_immediately(panel, tmp_path, monkeypatch):
    monkeypatch.setenv("SPOQC_DEFER_FIGURES", "0")
    path = tmp_path / "immediate.png"
    panel.savefig(path)
    assert path.exists(), "with deferral off the file must exist without a flush"


def test_flush_reports_how_many_it_wrote(panel, tmp_path, monkeypatch):
    monkeypatch.setenv("SPOQC_DEFER_FIGURES", "1")
    for i in range(3):
        panel.savefig(tmp_path / f"f{i}.png")
    assert helperfuncs.flush_figures() == 3
    assert all((tmp_path / f"f{i}.png").exists() for i in range(3))


def test_ensure_written_enables_a_save_then_move(panel, tmp_path, monkeypatch):
    """qc_model's pattern: scanpy saves via matplotlib, then shutil.move runs at once."""
    import shutil

    monkeypatch.setenv("SPOQC_DEFER_FIGURES", "1")
    source = tmp_path / "pca_variance_ratio.png"
    destination = tmp_path / "moved.png"
    panel.savefig(source)

    assert helperfuncs._FIG_QUEUE, "deferral is not active"
    helperfuncs.ensure_written(source)
    shutil.move(str(source), str(destination))  # FileNotFoundError without the barrier
    assert destination.exists()
