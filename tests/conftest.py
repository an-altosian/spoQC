import numba
import pytest


@pytest.fixture
def numba_threads(request):
    """Run the test with request.param numba threads, restoring the old count after.

    Skips when the numba pool cannot provide that many threads, so a
    multi-thread case never silently runs single-threaded.
    """
    wanted = request.param
    if wanted > numba.config.NUMBA_NUM_THREADS:
        pytest.skip(f"needs {wanted} numba threads, pool has {numba.config.NUMBA_NUM_THREADS}")
    previous = numba.get_num_threads()
    numba.set_num_threads(wanted)
    assert numba.get_num_threads() == wanted
    yield wanted
    numba.set_num_threads(previous)
