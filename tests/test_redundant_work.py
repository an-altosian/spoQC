"""PR3: work whose result was discarded, recomputed, or provably derivable."""
import inspect

import numpy as np
import pytest


class TestPdfTwinsAreOptIn:
    """Every figure was written as PNG and vector PDF; nothing reads the PDF."""

    def test_final_report_embeds_only_png(self):
        """The premise for gating PDF writes: the report never consumes a .pdf."""
        import inspect

        from spoqc.subworkflows import final_report

        src = inspect.getsource(final_report)
        assert ".pdf" not in src, "final_report must not reference a .pdf"
        assert ".png" in src

    def test_gate_exists_and_defaults_off(self):
        from spoqc import helperfuncs

        assert helperfuncs.WRITE_PDF is False

    def test_no_ungated_pdf_writes_remain(self):
        """Paren-aware scan: every savefig/write_image producing a .pdf is gated."""
        import io
        import pathlib
        import tokenize

        bad = []
        for path in pathlib.Path("spoqc").rglob("*.py"):
            text = path.read_text()
            # drop comments so prose mentioning the old form is not matched
            stripped = []
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type != tokenize.COMMENT:
                    stripped.append(tok)
            code = tokenize.untokenize(stripped)
            for fn in ("savefig", "write_image"):
                idx = 0
                while (i := code.find(fn + "(", idx)) != -1:
                    idx = i + 1
                    if code[max(0, i - 4):i].endswith("_pdf"):
                        continue
                    depth, j = 0, code.index("(", i)
                    while j < len(code):
                        if code[j] == "(":
                            depth += 1
                        elif code[j] == ")":
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    if ".pdf" in code[i:j]:
                        bad.append(f"{path}: {code[i:j][:70]}")
        assert not bad, f"ungated PDF writes: {bad}"

    def test_scanpy_pdf_save_is_gated(self):
        """scanpy writes figures itself, so savefig-based gating misses it."""
        import inspect

        from spoqc.subworkflows import qc_model

        src = inspect.getsource(qc_model)
        i = src.index("save='.pdf'")
        window = src[max(0, i - 200):i]
        assert "WRITE_PDF" in window, "the scanpy .pdf save must sit behind the gate"

    def test_helpers_are_noops_when_disabled(self, tmp_path):
        from spoqc import helperfuncs

        target = tmp_path / "nope.pdf"
        helperfuncs.savefig_pdf(str(target), dpi=300)
        assert not target.exists(), "savefig_pdf must not write while WRITE_PDF is False"

    def test_helper_writes_when_enabled(self, tmp_path, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from spoqc import helperfuncs

        monkeypatch.setattr(helperfuncs, "WRITE_PDF", True)
        plt.figure()
        plt.plot([0, 1], [0, 1])
        target = tmp_path / "yes.pdf"
        helperfuncs.savefig_pdf(str(target))
        plt.close()
        assert target.exists() and target.stat().st_size > 0


class TestUniformityDerivedFromEntropy:
    """uniformity = -(entropy + log q); the second sweep is redundant."""

    @pytest.mark.parametrize("dtype,hi", [(np.uint8, 256), (np.uint16, 65536)])
    @pytest.mark.parametrize("window,r", [(5, 2), (11, 5)])
    def test_derivation_matches_the_real_kernel(self, dtype, hi, window, r):
        from spoqc.image_analysis._slidingwindow import sliding_window_padded
        from spoqc.metrics.image import entropy as E
        from spoqc.metrics.image import uniformity as U

        rng = np.random.default_rng(0)
        img = rng.integers(0, hi, (96, 96)).astype(dtype)
        direct = -sliding_window_padded(U.kl_divergence_uniform, img, r, mode="reflect")
        derived = U.uniformity_from_entropy(
            sliding_window_padded(E.entropy, img, r, mode="reflect"), window, img.dtype
        )
        np.testing.assert_allclose(derived, direct, atol=1e-4)

    def test_accepts_a_flattened_entropy_map(self):
        """structure_analysis carries the entropy map flattened."""
        from spoqc.image_analysis._slidingwindow import sliding_window_padded
        from spoqc.metrics.image import entropy as E
        from spoqc.metrics.image import uniformity as U

        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (48, 48)).astype(np.uint8)
        ent = sliding_window_padded(E.entropy, img, 2, mode="reflect")
        flat = U.uniformity_from_entropy(ent.flatten().reshape(img.shape), 5, img.dtype)
        np.testing.assert_allclose(flat, U.uniformity_from_entropy(ent, 5, img.dtype))

    def test_pixel_uniformity_still_works_without_an_entropy_map(self):
        """The uniformity step is independently gated, so the direct path must stay."""
        sig = inspect.signature(
            __import__("spoqc.metrics.image.uniformity", fromlist=["x"]).pixel_uniformity
        )
        assert sig.parameters["entropy_image"].default is None


class TestDeadCodeRemoved:
    def test_get_ci_df_is_gone(self):
        from spoqc.metrics.segmentation import sc_metrics

        assert not hasattr(sc_metrics, "get_ci_df")

    def test_triangle_counter_returns_only_counts(self):
        """indices_list was materialised on 4 call sites and never read."""
        from spoqc.metrics.segmentation import void

        src = inspect.getsource(void.count_stuff_in_triangles_via_delaunay)
        assert "indices_list" not in src
        assert "return counts" in src

    def test_no_call_site_still_unpacks_two_values(self):
        from spoqc.metrics.segmentation import void

        src = inspect.getsource(void)
        assert "counts, indices = count_stuff_in_triangles_via_delaunay" not in src

    def test_counter_still_counts_correctly(self):
        """Behaviour of the surviving return value is unchanged."""
        from scipy.spatial import Delaunay

        from spoqc.metrics.segmentation import void

        pts = np.array([[0.0, 0], [10, 0], [0, 10], [10, 10]])
        d = Delaunay(pts)
        stuff = np.array([[1.0, 1], [2, 2], [9, 9], [-5, -5]])
        counts = void.count_stuff_in_triangles_via_delaunay(d, stuff)
        assert len(counts) == len(d.simplices)
        # the three in-hull points are attributed; the outside one is not
        assert counts.sum() == 3
