"""core.threads: the `-n` budget reaches every library before it is imported."""

from __future__ import annotations

import json
import subprocess
import sys

# Runs the real console entry point, with spoqc.cli replaced by a probe that imports the
# real cli (and with it polars, numba, numpy, dask, ovrlpy) and reports the pool sizes.
PROBE = r"""
import json, sys, types

def probe(args):
    heavy = sorted(m for m in ("numpy", "numba", "polars", "dask", "ovrlpy") if m in sys.modules)
    del sys.modules["spoqc.cli"]
    import spoqc.cli
    import dask, matplotlib, numba, polars
    from threadpoolctl import threadpool_info
    from spoqc.core import threads
    print(json.dumps({
        "imported_before_cli": heavy,
        "ovrlpy_imported": "ovrlpy" in sys.modules,
        "N": threads.N,
        "polars": polars.thread_pool_size(),
        "numba_max": numba.config.NUMBA_NUM_THREADS,
        "numba": numba.get_num_threads(),
        "dask": dask.config.get("num_workers"),
        "blas": sorted({pool["num_threads"] for pool in threadpool_info()}),
        "backend": matplotlib.get_backend().lower(),
    }))

fake = types.ModuleType("spoqc.cli")
fake.main = probe
sys.modules["spoqc.cli"] = fake
from spoqc.__main__ import main
main(sys.argv[1:])
"""


def _run(*argv: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", PROBE, "-i", "in", "-o", "out", "-t", "tmp", *argv],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_entry_point_sets_every_pool_to_n():
    report = _run("-n", "3")
    assert report["imported_before_cli"] == []
    assert report["ovrlpy_imported"]
    assert report["N"] == 3
    assert report["polars"] == 3
    assert report["numba_max"] == 3 and report["numba"] == 3
    assert report["dask"] == 3
    assert report["blas"] == [3]
    assert report["backend"] == "agg"


def test_dev_test_budget_is_eight():
    assert _run("-n", "3", "--dev_test")["N"] == 8


def test_configure_after_numpy_import_fails():
    code = "import numpy\nfrom spoqc.core import threads\nthreads.configure(2)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "must run before importing ['numpy']" in result.stderr
