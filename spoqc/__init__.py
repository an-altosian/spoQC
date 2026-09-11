from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

_PNG_COMPRESS_LEVEL = 1


def _install_png_compress_default() -> None:
    """Default PNG writes to zlib level 1 instead of matplotlib's level 6.

    `compress_level` changes the FILE, not the IMAGE: the pixels decode bit-for-bit
    identically at every level (asserted in tests/test_png_compress_level.py). It is
    therefore a cheaper way to emit the same output, not a cheaper output.

    Why it is worth doing: profiling a full `-s all` run put `PIL _encode_tile` -- zlib
    compressing the PNGs -- at 9.8% of wall clock, and level 6 is ~30% of savefig time.
    Level 1 cuts savefig by ~1.2-1.4x for ~5% larger files. Level 0 is *slower* than
    level 1, because doubling the bytes costs more in write time than it saves in zlib.

    Applied here rather than at the ~45 `savefig` call sites so that the default cannot
    drift between them, and only for PNG -- PDF compression is governed by the separate
    `pdf.compression` rcParam and is untouched.
    """
    import os
    from matplotlib.figure import Figure

    original = Figure.savefig

    def savefig(self, fname, **kwargs):
        name = fname if isinstance(fname, (str, bytes, os.PathLike)) else ""
        if str(name).lower().endswith(".png"):
            pil_kwargs = dict(kwargs.get("pil_kwargs") or {})
            pil_kwargs.setdefault("compress_level", _PNG_COMPRESS_LEVEL)
            kwargs["pil_kwargs"] = pil_kwargs

        # Defer the write to a process pool. Agg does not release the GIL, so threads
        # cap out at ~1.4x while processes reach 7.3x with byte-identical output.
        # Only for real paths: a file object or buffer cannot be handed to a worker.
        if os.environ.get("SPOQC_DEFER_FIGURES", "1") != "0" and isinstance(
            fname, (str, os.PathLike)
        ):
            from . import helperfuncs

            try:
                helperfuncs.queue_figure(self, fname, **kwargs)
                return
            except Exception as exc:  # not every figure is picklable
                # Write it here instead, and say so -- a silently dropped figure would
                # be far worse than losing the parallelism for this one plot.
                print(f"[NOTE] figure {fname} is not picklable ({type(exc).__name__}); "
                      "writing it inline")
        return original(self, fname, **kwargs)

    savefig.__wrapped__ = original
    Figure.savefig = savefig


_install_png_compress_default()

# PDF stream compression. matplotlib defaults to zlib level 6; the streams are Flate,
# i.e. lossless, so the level changes the FILE and never the rendered page. Verified by
# rasterising both with pdftoppm at 150 dpi: pixel-identical, max channel difference 0
# (tests/test_pdf_compression.py).
#
# Level 0 is ~1.5x faster on a PDF savefig. Level 1 is NOT worth setting -- measured at
# 1625 ms/figure-pair against level 6's 1629 ms, i.e. no gain, so the choice is
# effectively compress or do not.
#
# COST: uncompressed streams make PDFs ~3.6-4.2x larger (a run's PDF output goes from
# ~131 MB to roughly 0.5 GB). Set this back to 6 if output size matters more than time.
matplotlib.rcParams["pdf.compression"] = 0

try:
    from importlib.metadata import version as _version
    __version__ = _version("spoqc")
except Exception:  # pragma: no cover
    __version__ = "0.0.1"

__all__ = ["__version__"]