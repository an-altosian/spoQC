"""PR5: the binary MRF is solved exactly by a graph cut instead of approximated
by loopy belief propagation."""
import numpy as np
import pytest

BETA = 1.5


def _energy(labels, p, beta):
    """The Potts energy both solvers claim to minimise."""
    lab = np.asarray(labels).astype(np.int8)
    eps = 1e-8
    unary = np.where(lab == 0, -np.log(1.0 - p + eps), -np.log(p + eps)).sum()
    pair = (lab[:, :-1] != lab[:, 1:]).sum() + (lab[:-1, :] != lab[1:, :]).sum()
    return float(unary + beta * pair)


class TestLabelPolarity:
    """The source/sink capacity order is a convention. Getting it backwards
    silently inverts every mask, so it is pinned rather than assumed."""

    def test_high_probability_labels_good(self):
        from spoqc.hqr.graphcut import solve_binary_mrf

        labels = solve_binary_mrf(np.full((32, 32), 0.9), BETA)
        assert (labels == 1).all(), "p=0.9 must label 1 (good) in spoQC's convention"

    def test_low_probability_labels_bad(self):
        from spoqc.hqr.graphcut import solve_binary_mrf

        labels = solve_binary_mrf(np.full((32, 32), 0.1), BETA)
        assert (labels == 0).all(), "p=0.1 must label 0 (bad)"

    def test_regions_are_recovered_in_the_right_place(self):
        from spoqc.hqr.graphcut import solve_binary_mrf

        p = np.full((64, 64), 0.9)
        p[:16, :16] = 0.05
        labels = solve_binary_mrf(p, BETA)
        assert (labels[:16, :16] == 0).all()
        assert (labels[32:, 32:] == 1).all()

    def test_labels_are_only_zero_or_one(self):
        from spoqc.hqr.graphcut import solve_binary_mrf

        rng = np.random.default_rng(0)
        labels = solve_binary_mrf(rng.random((48, 48)), BETA)
        assert set(np.unique(labels)) <= {0, 1}


class TestSmoothing:
    def test_beta_zero_keeps_isolated_noise(self):
        """With no pairwise penalty the solution is the per-pixel argmin."""
        from spoqc.hqr.graphcut import solve_binary_mrf

        p = np.full((64, 64), 0.85)
        rng = np.random.default_rng(0)
        idx = rng.choice(64 * 64, 200, replace=False)
        p.flat[idx] = 0.05
        assert (solve_binary_mrf(p, 0.0) == 0).sum() == 200

    def test_positive_beta_removes_isolated_noise(self):
        from spoqc.hqr.graphcut import solve_binary_mrf

        p = np.full((64, 64), 0.85)
        rng = np.random.default_rng(0)
        p.flat[rng.choice(64 * 64, 200, replace=False)] = 0.05
        assert (solve_binary_mrf(p, BETA) == 0).sum() == 0


class TestExactnessVersusLbp:
    def test_graphcut_energy_is_not_worse_than_lbp(self):
        """The whole justification: the cut is the exact global minimum, so it
        can never score worse on the shared objective."""
        from spoqc.hqr import graphcut
        from spoqc.hqr import markov_random_field_zarr_parallel as lbp

        rng = np.random.default_rng(0)
        blk = rng.random((8, 8)).astype(np.float32)
        p = np.kron(blk, np.ones((32, 32), dtype=np.float32))

        gc = graphcut.solve_binary_mrf(p, BETA)
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            _, ll = lbp.first_version_loopy_belief_propagation_parallel(
                p, td, "t", beta=BETA, max_iter=15, normalize="total"
            )
            lbp_labels = np.asarray(ll[:]).reshape(p.shape)

        assert _energy(gc, p, BETA) <= _energy(lbp_labels, p, BETA)


class TestDropInContract:
    def test_signature_matches_the_lbp_entry_point(self):
        import inspect

        from spoqc.hqr import graphcut
        from spoqc.hqr import markov_random_field_zarr_parallel as lbp

        a = inspect.signature(graphcut.first_version_loopy_belief_propagation_parallel)
        b = inspect.signature(lbp.first_version_loopy_belief_propagation_parallel)
        assert list(a.parameters) == list(b.parameters)

    def test_returns_beliefs_and_labels(self, tmp_path):
        from spoqc.hqr import graphcut

        p = np.full((32, 32), 0.8, dtype=np.float32)
        beliefs, labels = graphcut.first_version_loopy_belief_propagation_parallel(
            p, str(tmp_path), "m", beta=BETA, max_iter=15, normalize="total"
        )
        assert labels.shape == p.shape
        assert beliefs.shape == p.shape
        assert 0.0 <= float(beliefs.min()) and float(beliefs.max()) <= 1.0, (
            "the soft channel must be a genuine value in [0, 1]"
        )

    def test_writes_nothing_to_the_temp_folder(self, tmp_path):
        """LBP needs a 32 B/pixel message memmap; the cut needs no temp file."""
        from spoqc.hqr import graphcut

        graphcut.first_version_loopy_belief_propagation_parallel(
            np.full((32, 32), 0.8, dtype=np.float32), str(tmp_path), "m", beta=BETA
        )
        assert list(tmp_path.iterdir()) == []

    def test_exposes_the_lbp_visualiser(self):
        from spoqc.hqr import graphcut

        assert callable(graphcut.visualize_markov_calculation)


class TestSolverSelection:
    def test_default_is_graphcut(self):
        from spoqc import hqr

        assert hqr.solver().__name__.endswith("graphcut")

    def test_lbp_remains_selectable(self):
        from spoqc import hqr

        assert hqr.solver("lbp").__name__.endswith("markov_random_field_zarr_parallel")

    def test_unknown_solver_raises(self):
        from spoqc import hqr

        with pytest.raises(ValueError, match="unknown mrf solver"):
            hqr.solver("nope")

    def test_cli_default_and_choices(self):
        from spoqc.cli import build_parser

        args = vars(build_parser().parse_args(["-i", "i", "-o", "o", "-t", "t"]))
        assert args["mrf_solver"] == "graphcut"
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["-i", "i", "-o", "o", "-t", "t", "--mrf_solver", "bogus"]
            )


class TestMemoryGuard:
    """The cut trades disk traffic for resident memory, so the limit is checked."""

    def test_fires_when_the_graph_cannot_fit(self):
        from spoqc.hqr.graphcut import check_memory

        with pytest.raises(MemoryError, match="graph cut needs"):
            check_memory((25778, 35416), available_gb=64.0)

    def test_message_is_actionable(self):
        from spoqc.hqr.graphcut import check_memory

        with pytest.raises(MemoryError) as e:
            check_memory((25778, 35416), available_gb=64.0)
        msg = str(e.value)
        assert "--resolution" in msg and "--mrf_solver lbp" in msg

    def test_passes_at_scale2_on_a_modest_box(self):
        from spoqc.hqr.graphcut import check_memory

        assert check_memory((6444, 8854), available_gb=64.0) < 20.0

    def test_estimate_is_linear_in_nodes(self):
        from spoqc.hqr.graphcut import estimate_memory_gb

        assert estimate_memory_gb(2_000_000) == pytest.approx(
            2 * estimate_memory_gb(1_000_000)
        )
