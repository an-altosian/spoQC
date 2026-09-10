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
        return original(self, fname, **kwargs)

    savefig.__wrapped__ = original
    Figure.savefig = savefig


_install_png_compress_default()

try:
    from importlib.metadata import version as _version
    __version__ = _version("spoqc")
except Exception:  # pragma: no cover
    __version__ = "0.0.1"

__all__ = ["__version__"]