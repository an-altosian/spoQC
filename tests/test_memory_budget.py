"""PR7: a memory budget that derives the tunables and bounds real peak RSS.

Motivation, concretely: three concurrent full-resolution runs (each measured at
11-14 GB, one unbounded) exhausted a host and killed the session. Nothing in
spoQC observed or limited its own memory use -- the chunk size, the k-means
subsample and the worker count were fixed defaults, so peak RSS was whatever
those happened to produce on the data at hand.
"""

import multiprocessing as mp
import os
import time

import pytest

from spoqc import memory as M

GB = 10**9


class TestBudgetResolution:
    def test_explicit_mem_is_taken_at_face_value(self):
        """A scheduler allocation may differ from what is free right now."""
        assert M.budget_bytes(16) == 16 * GB

    def test_default_is_a_fraction_of_available(self):
        budget = M.budget_bytes(None)
        available = M.available_bytes()
        assert budget == pytest.approx(available * M.DEFAULT_FRACTION, rel=0.05)
        assert budget < available, "must leave headroom for the rest of the machine"

    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    def test_non_positive_mem_is_rejected(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            M.budget_bytes(bad)

    def test_available_bytes_is_plausible(self):
        assert 10**8 < M.available_bytes() < 10**15


class TestRssAccounting:
    """The reading must include children -- that is where the big allocations are."""

    def test_self_rss_is_nonzero_and_plausible(self):
        rss = M.current_rss_bytes(include_children=False)
        assert 10**6 < rss < 10**12

    def test_child_allocation_is_counted(self):
        """A self-only reading reports a flat, reassuring, wrong number."""
        before = M.current_rss_bytes()
        proc = mp.Process(target=_hog, args=(600, 2.0))
        proc.start()
        try:
            time.sleep(1.0)
            with_child = M.current_rss_bytes()
        finally:
            proc.join()
        assert with_child - before > 400 * 10**6, (
            "the child's ~600 MB must appear in the process-group total"
        )

    def test_self_only_reading_misses_the_child(self):
        """Pins why include_children defaults to True."""
        proc = mp.Process(target=_hog, args=(600, 2.0))
        proc.start()
        try:
            time.sleep(1.0)
            self_only = M.current_rss_bytes(include_children=False)
            group = M.current_rss_bytes(include_children=True)
        finally:
            proc.join()
        assert group > self_only + 400 * 10**6

    def test_dead_pid_contributes_zero_not_an_error(self):
        assert M._rss_of(999_999_999) == 0


class TestDerivedTunables:
    def test_chunk_size_holds_the_headroom_fraction(self):
        for gb in (1, 2, 4):
            budget = gb * GB
            rows = M.derive_chunk_size(budget, bytes_per_row=200)
            assert rows * 200 == pytest.approx(budget * M.ALLOCATION_HEADROOM, rel=0.01)

    def test_chunk_size_scales_with_the_budget(self):
        small = M.derive_chunk_size(1 * GB, bytes_per_row=200)
        big = M.derive_chunk_size(2 * GB, bytes_per_row=200)
        assert big == pytest.approx(2 * small, rel=0.01)

    def test_chunk_size_is_clamped_at_both_ends(self):
        assert M.derive_chunk_size(1, bytes_per_row=10**9) == 10_000
        assert M.derive_chunk_size(10**15, bytes_per_row=1) == 5_000_000

    def test_chunk_size_rejects_nonsense_row_size(self):
        with pytest.raises(ValueError, match="bytes_per_row"):
            M.derive_chunk_size(GB, bytes_per_row=0)

    @pytest.mark.parametrize(
        "budget_gb,per_worker_gb,requested,expected",
        [
            (4, 3, 32, 1),  # the OOM case: a big -n must not mean a big fan-out
            (8, 3, 32, 2),
            (16, 3, 32, 5),
            (256, 3, 32, 32),  # ample budget: honour the request
            (256, 3, 4, 4),  # never exceed what was asked for
        ],
    )
    def test_worker_cap(self, budget_gb, per_worker_gb, requested, expected):
        assert (
            M.derive_max_workers(budget_gb * GB, per_worker_gb * GB, requested)
            == expected
        )

    def test_worker_cap_never_returns_zero(self):
        """A budget smaller than one worker must still run, serially."""
        assert M.derive_max_workers(10**6, 3 * GB, 32) == 1

    def test_fits_matches_the_graph_cut_figures(self):
        """~260 B/node measured: scale2 fits in 16 GB, scale0 does not."""
        budget = 16 * GB
        assert M.fits(budget, int(57.1e6 * 260))
        assert not M.fits(budget, int(913e6 * 260))


class TestWatchdogEnforcement:
    """Deriving is advisory; this is the part that actually bounds peak RSS."""

    def test_breach_fires_when_over_budget(self):
        fired = []
        w = M.Watchdog(
            budget=10**6,
            interval=0.05,
            on_breach=lambda peak, budget: fired.append(peak),
        )
        w.sample()
        assert len(fired) == 1
        assert fired[0] > 10**6

    def test_no_breach_when_under_budget(self):
        fired = []
        w = M.Watchdog(
            budget=500 * GB,
            interval=0.05,
            on_breach=lambda peak, budget: fired.append(peak),
        )
        w.sample()
        assert fired == []

    def test_live_thread_catches_a_child_overrun(self):
        """The exact failure this PR exists to prevent."""
        caught = []
        w = M.Watchdog(
            budget=700 * 10**6,
            interval=0.1,
            on_breach=lambda peak, budget: caught.append(peak),
        )
        w.start()
        proc = mp.Process(target=_hog, args=(1200, 1.5))
        proc.start()
        try:
            time.sleep(1.0)
        finally:
            proc.join()
            peak = w.stop()
        assert caught, "a child exceeding the budget must be caught"
        assert peak > 700 * 10**6

    def test_peak_is_monotonic(self):
        w = M.Watchdog(budget=500 * GB, interval=0.05)
        first = w.sample()
        assert w.sample() >= first

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError, match="must be positive"):
            M.Watchdog(budget=0)

    def test_cannot_start_twice(self):
        w = M.Watchdog(budget=500 * GB, interval=5).start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                w.start()
        finally:
            w.stop()

    def test_context_manager_reports_the_peak(self):
        import io

        buf = io.StringIO()
        with M.Watchdog(budget=500 * GB, interval=5, stream=buf) as w:
            w.sample()
        out = buf.getvalue()
        assert "Peak memory" in out and "budget" in out

    def test_default_diagnostic_names_every_escape_hatch(self, monkeypatch):
        """An overrun message must say what to change, not just that it failed."""
        import io

        buf = io.StringIO()
        monkeypatch.setattr(os, "kill", lambda *a, **k: None)
        w = M.Watchdog(budget=GB, interval=5, stream=buf)
        w._default_on_breach(3 * GB, GB)
        msg = buf.getvalue()
        for flag in (
            "--mem",
            "--resolution",
            "--pixel_qc_chunk_size",
            "--mrf_solver lbp",
        ):
            assert flag in msg, f"the diagnostic must mention {flag}"
        assert "3.0 GB" in msg and "1.0 GB" in msg

    def test_breach_signals_the_process(self, monkeypatch):
        """Enforcement means terminating, not just logging."""
        import io

        signalled = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signalled.append((pid, sig)))
        w = M.Watchdog(budget=GB, interval=5, stream=io.StringIO())
        w._default_on_breach(3 * GB, GB)
        assert signalled and signalled[0][0] == os.getpid()


class TestCliWiring:
    def test_mem_flag_exists_and_defaults_to_auto(self):
        from spoqc.cli import build_parser

        args = vars(build_parser().parse_args(["-i", "i", "-o", "o", "-t", "t"]))
        assert args["mem"] is None, "no --mem means derive from available memory"

    def test_mem_flag_is_parsed_as_gb(self):
        from spoqc.cli import build_parser

        args = vars(
            build_parser().parse_args(["-i", "i", "-o", "o", "-t", "t", "--mem", "16"])
        )
        assert args["mem"] == 16.0

    def test_tunable_flags_default_to_none_so_derivation_can_apply(self):
        """They must be distinguishable from an explicit value."""
        from spoqc.cli import build_parser

        args = vars(build_parser().parse_args(["-i", "i", "-o", "o", "-t", "t"]))
        assert args["pixel_qc_chunk_size"] is None
        assert args["kmeans_sample_size"] is None

    def test_explicit_tunable_still_wins(self):
        from spoqc.cli import build_parser

        args = vars(
            build_parser().parse_args(
                [
                    "-i",
                    "i",
                    "-o",
                    "o",
                    "-t",
                    "t",
                    "--mem",
                    "8",
                    "--pixel_qc_chunk_size",
                    "123456",
                ]
            )
        )
        assert args["pixel_qc_chunk_size"] == 123456

    def test_cli_starts_a_watchdog(self):
        import inspect

        from spoqc import cli

        src = inspect.getsource(cli)
        assert "Watchdog(" in src and ".start()" in src


def _hog(megabytes, hold_seconds):
    """Allocate and touch pages so they become resident, then hold."""
    buf = bytearray(megabytes * 1024 * 1024)
    buf[::4096] = b"x" * len(buf[::4096])
    time.sleep(hold_seconds)
