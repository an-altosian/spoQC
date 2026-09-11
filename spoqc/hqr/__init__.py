from . import combine_masks
from . import combine_masks_zoom
from . import markov_random_field_zarr_parallel
from . import graphcut

#: Which solver the MRF call sites use. Set once from the CLI (--mrf_solver).
SOLVER = "graphcut"

_SOLVERS = {
    "graphcut": graphcut,
    "lbp": markov_random_field_zarr_parallel,
}


def solver(name=None):
    """Return the module implementing the MRF solve.

    Both expose first_version_loopy_belief_propagation_parallel(...) with the
    same signature and visualize_markov_calculation(...), so the call sites are
    identical either way.
    """
    key = SOLVER if name is None else name
    if key not in _SOLVERS:
        raise ValueError(f"unknown mrf solver {key!r}; expected one of {sorted(_SOLVERS)}")
    return _SOLVERS[key]
