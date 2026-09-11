"""Memory budget for a spoQC run: derive the tunables from it, and enforce it.

Two jobs, and they are different:

1. **Derive.** Peak resident set is set by a handful of knobs -- the dask row-chunk
   size, the k-means fitting subsample, how many worker processes run at once, and
   whether a graph cut of the chosen pyramid level fits at all. Those are currently
   independent flags with fixed defaults, so a run's memory use is whatever those
   defaults happen to produce on the data in front of it. Given a budget, they can
   all be derived instead.

2. **Enforce.** Deriving is advisory: a wrong estimate still overruns. A watchdog
   samples the process group's real RSS and terminates the run with a diagnostic
   the moment the budget is exceeded, so the failure is spoQC's, loud, and
   attributable -- not the kernel OOM killer taking out the shell, the session, or a
   co-tenant job.

Why not RLIMIT_AS: it caps *virtual* address space. spoQC memory-maps its LBP
message store (32 bytes per pixel, hundreds of GB of address space at full
resolution) and dask reserves freely, so an RLIMIT_AS low enough to be a useful RSS
bound would break normal operation. RSS sampling measures the thing we actually care
about.

Linux-only, by design: it reads /proc. spoQC ships a Linux container.
"""

from __future__ import annotations

import os
import signal
import sys
import threading

#: Fraction of detected available memory used when --mem is not given.
DEFAULT_FRACTION = 0.8

#: Fraction of the budget a single derived allocation is allowed to claim, leaving
#: room for the interpreter, BLAS scratch, GEOS, and the figure pipeline.
ALLOCATION_HEADROOM = 0.25

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def available_bytes() -> int:
    """Memory actually available now, from MemAvailable in /proc/meminfo."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def budget_bytes(mem_gb: float | None = None) -> int:
    """Resolve the run's budget in bytes.

    An explicit --mem is taken at face value: the caller may deliberately want a
    budget larger or smaller than what is free right now (a scheduler allocation,
    for instance). Without one, take DEFAULT_FRACTION of what is available.
    """
    if mem_gb is not None:
        if mem_gb <= 0:
            raise ValueError(f"--mem must be positive, got {mem_gb}")
        return int(mem_gb * 1e9)
    return int(available_bytes() * DEFAULT_FRACTION)


def _descendant_pids(pid: int) -> list[int]:
    """pid plus every descendant, via /proc/<pid>/task/*/children."""
    found, stack = [], [pid]
    while stack:
        current = stack.pop()
        found.append(current)
        task_dir = f"/proc/{current}/task"
        try:
            tasks = os.listdir(task_dir)
        except OSError:
            continue
        for task in tasks:
            try:
                with open(f"{task_dir}/{task}/children") as fh:
                    stack.extend(int(p) for p in fh.read().split())
            except OSError:
                continue
    return found


def _rss_of(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm") as fh:
            return int(fh.read().split()[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return 0  # the process exited between listing and reading


def current_rss_bytes(pid: int | None = None, include_children: bool = True) -> int:
    """Resident set of the process, optionally summed over its descendants.

    Children matter: the leiden sweep and any joblib fan-out put the large
    allocations in worker processes, so a self-only reading would miss them
    entirely and report a flat, reassuring, wrong number.
    """
    root = os.getpid() if pid is None else pid
    if not include_children:
        return _rss_of(root)
    return sum(_rss_of(p) for p in _descendant_pids(root))


def derive_chunk_size(
    budget: int, bytes_per_row: int, *, minimum: int = 10_000, maximum: int = 5_000_000
) -> int:
    """Row-chunk size that keeps one chunk inside the budget's headroom."""
    if bytes_per_row <= 0:
        raise ValueError(f"bytes_per_row must be positive, got {bytes_per_row}")
    rows = int(budget * ALLOCATION_HEADROOM / bytes_per_row)
    return max(minimum, min(maximum, rows))


def derive_max_workers(budget: int, per_worker_bytes: int, requested: int) -> int:
    """Cap worker count so the fan-out cannot exceed the budget.

    Process parallelism multiplies memory by the number of workers; this is what
    keeps a bigger -n from turning a runtime win into an OOM.
    """
    if per_worker_bytes <= 0:
        raise ValueError(f"per_worker_bytes must be positive, got {per_worker_bytes}")
    return max(1, min(int(requested), int(budget / per_worker_bytes)))


def fits(budget: int, needed_bytes: int) -> bool:
    return needed_bytes <= budget


def format_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


class BudgetExceeded(MemoryError):
    """Raised, or reported by the watchdog, when the run overruns its budget."""


class Watchdog:
    """Samples process-group RSS and terminates the run if the budget is exceeded.

    Enforcement rather than advice. The default action prints a diagnostic naming
    the budget, the observed peak, and the flags that reduce it, then exits 137 --
    the same code a kernel OOM kill produces, so schedulers classify it the same
    way, but with an explanation and without collateral damage.
    """

    def __init__(
        self, budget: int, *, interval: float = 2.0, on_breach=None, stream=None
    ):
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self.budget = budget
        self.interval = interval
        self.peak = 0
        self._on_breach = on_breach or self._default_on_breach
        self._stream = stream or sys.stderr
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _default_on_breach(self, peak: int, budget: int) -> None:
        print(
            f"\n[ERROR] spoQC exceeded its memory budget: using {format_gb(peak)} "
            f"of {format_gb(budget)}.\n"
            f"        Raise it with --mem, or lower peak usage with a coarser "
            f"--resolution (each level is 4x fewer pixels), a smaller "
            f"--pixel_qc_chunk_size, fewer threads via -n, or "
            f"--mrf_solver lbp (streams instead of holding the graph in RAM).\n"
            f"        Terminating deliberately rather than risking an OOM kill.",
            file=self._stream,
            flush=True,
        )
        os.kill(os.getpid(), signal.SIGTERM)

    def sample(self) -> int:
        """Take one reading; returns the new peak. Also the unit-test entry point."""
        rss = current_rss_bytes()
        self.peak = max(self.peak, rss)
        if rss > self.budget:
            self._on_breach(rss, self.budget)
        return self.peak

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> "Watchdog":
        if self._thread is not None:
            raise RuntimeError("watchdog already started")
        self._thread = threading.Thread(
            target=self._loop, name="spoqc-memory-watchdog", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> int:
        """Stop sampling and return the observed peak."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None
        return self.peak

    def __enter__(self) -> "Watchdog":
        return self.start()

    def __exit__(self, *exc) -> bool:
        peak = self.stop()
        print(
            f"[NOTE] Peak memory: {format_gb(peak)} of "
            f"{format_gb(self.budget)} budget "
            f"({100 * peak / self.budget:.0f}%)",
            file=self._stream,
            flush=True,
        )
        return False
