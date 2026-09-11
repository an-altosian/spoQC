"""Shared test setup.

spoqc defers `savefig` to a process pool, so a test that writes a figure and then inspects
the file on disk sees nothing until a flush. Most tests here assert properties OF THE FILE
(compression level, pixel identity, rasterised PDF content) and are unrelated to how the
write is scheduled, so deferral is disabled by default and the tests that specifically
exercise it opt back in by setting the variable themselves.

Without this, adding deferral turned 13 pre-existing tests red for a reason that had
nothing to do with what they were testing.
"""
import pytest


@pytest.fixture(autouse=True)
def figures_written_synchronously(monkeypatch, request):
    if "deferred_figure_writes" in request.node.nodeid:
        return  # that module drives SPOQC_DEFER_FIGURES itself
    monkeypatch.setenv("SPOQC_DEFER_FIGURES", "0")
