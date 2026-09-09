"""PR6: the leiden resolution sweep fans out over processes, not GIL-bound threads.

Measured on an 8-resolution sweep over 8,000 cells: the thread pool went
70.68 s -> 74.08 s from 1 to 8 workers (0.95x, i.e. slower) while peak RSS rose
410.9 -> 1252.4 MB, because every worker's adata.copy() was resident in the same
process. joblib loky reached 3.76x with a flat parent resident set.
"""
import inspect

import numpy as np
import pytest


class TestSweepUsesProcesses:
    def test_no_thread_pool_in_the_sweep(self):
        from spoqc import helperfuncs

        src = inspect.getsource(helperfuncs.test_resolutions_leiden)
        # the measurement table lives in a comment, so strip comments first
        code = "\n".join(
            ln.split("#")[0] for ln in src.splitlines()
        )
        assert "ThreadPoolExecutor" not in code, (
            "the sweep is GIL-bound; a thread pool costs memory and buys no speedup"
        )
        assert "Parallel(" in code and "delayed(" in code

    def test_uses_the_loky_backend(self):
        from spoqc import helperfuncs

        code = "\n".join(
            ln.split("#")[0]
            for ln in inspect.getsource(helperfuncs.test_resolutions_leiden).splitlines()
        )
        assert 'backend="loky"' in code, "processes, not threads"

    def test_n_jobs_is_bounded_by_the_work_available(self):
        """Spawning more workers than resolutions wastes memory for no gain."""
        from spoqc import helperfuncs

        code = "\n".join(
            ln.split("#")[0]
            for ln in inspect.getsource(helperfuncs.test_resolutions_leiden).splitlines()
        )
        assert "min(" in code and "len(resolutions)" in code


class TestJoblibSemanticsMatchTheOldPool:
    """The two constructs must be interchangeable for this workload."""

    @staticmethod
    def _work(x, i):
        return [i, x * 2]

    def test_results_are_reassembled_by_index_not_completion_order(self):
        """The old code indexed out[results[0]]; order must not depend on timing."""
        from joblib import Parallel, delayed

        items = list(range(8))
        got = Parallel(n_jobs=4, backend="loky")(
            delayed(self._work)(x, i) for i, x in enumerate(items)
        )
        out = [None] * len(items)
        for r in got:
            out[r[0]] = r[1:]
        assert out == [[x * 2] for x in items]

    def test_parallel_preserves_input_order(self):
        from joblib import Parallel, delayed

        got = Parallel(n_jobs=4, backend="loky")(
            delayed(self._work)(x, i) for i, x in enumerate(range(12))
        )
        assert [r[0] for r in got] == list(range(12))


class TestSweepStillWorksEndToEnd:
    @pytest.fixture(scope="class")
    def adata(self):
        import anndata as ad
        import scanpy as sc

        rng = np.random.default_rng(0)
        X = np.vstack(
            [rng.normal(c, 1.0, (400, 25)) for c in (0, 6, 12)]
        ).astype(np.float32)
        a = ad.AnnData(X)
        sc.pp.pca(a, n_comps=10)
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.umap(a)
        return a

    def test_returns_a_resolution_in_range(self, adata, tmp_path):
        from spoqc import helperfuncs

        win = helperfuncs.test_resolutions_leiden(
            adata, str(tmp_path), 4, k=20, steps=4, end=1.5, start=0.0
        )
        assert 0.0 <= float(win) <= 1.5

    def test_single_worker_and_many_workers_agree(self, adata, tmp_path):
        """n_jobs must not change the answer."""
        from spoqc import helperfuncs

        da, db = tmp_path / "a", tmp_path / "b"
        da.mkdir(); db.mkdir()
        a = helperfuncs.test_resolutions_leiden(
            adata, str(da), 1, k=20, steps=4, end=1.5, start=0.0
        )
        b = helperfuncs.test_resolutions_leiden(
            adata, str(db), 8, k=20, steps=4, end=1.5, start=0.0
        )
        assert float(a) == pytest.approx(float(b))
