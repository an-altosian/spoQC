"""Exact solver for the binary MRF that loopy belief propagation approximates.

The energy minimised by `markov_random_field_zarr_parallel` is

    E(x) = sum_i  [ x_i = 0 ? -log(1 - p_i + eps) : -log(p_i + eps) ]
         + beta * sum_(i,j) in E4  1[x_i != x_j]

i.e. an Ising/Potts model on a 4-connected pixel grid with the pairwise matrix
[[0, beta], [beta, 0]]. That pairwise term is submodular for beta >= 0:

    V(0,0) + V(1,1) = 0  <=  V(0,1) + V(1,0) = 2*beta

By Greig-Porteous-Seheult (1989) and Kolmogorov-Zabih (2004), a binary energy
with submodular pairwise terms is minimised **exactly and globally** by a single
s-t minimum cut. Loopy BP is an iterative *approximation* to a problem that has
a polynomial exact solution, so the graph cut is both faster and strictly more
accurate.

Label convention matches the rest of spoQC: 0 = bad, 1 = good. The source/sink
capacity order below is calibrated so a field of p=0.9 labels 1 and p=0.1
labels 0; see tests/test_graphcut.py, which pins it. Getting that order wrong
silently inverts every mask, so it is asserted rather than assumed.
"""

import os

import numpy as np

# reused verbatim so graphcut is a drop-in for the LBP module at every call site
from .markov_random_field_zarr_parallel import visualize_markov_calculation  # noqa: F401

EPS = np.float64(1e-8)


#: Measured peak RSS per grid node for a 4-connected BK graph (250-330 B/node
#: across 0.26-9.4 Mpx; see the PR body). Used only for the pre-flight check.
BYTES_PER_NODE = 260


def estimate_memory_gb(n_nodes):
    """Peak RAM the graph will need, in GB."""
    return n_nodes * BYTES_PER_NODE / 1e9


def _available_memory_gb():
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9


def check_memory(shape, *, available_gb=None, headroom=0.8):
    """Fail early, and actionably, if the graph cannot fit in RAM.

    Unlike LBP -- which streams messages through a 32 B/pixel memmap and so has a
    small resident set but pathological scattered I/O -- a graph cut is resident.
    That is the deliberate trade: no disk traffic, but the whole graph in memory.
    At ~260 B/node a full-resolution 913 Mpx grid would need ~240 GB, so this is
    checked rather than discovered through the OOM killer.
    """
    n = int(np.prod(shape))
    need = estimate_memory_gb(n)
    avail = _available_memory_gb() if available_gb is None else available_gb
    if need > headroom * avail:
        raise MemoryError(
            f"graph cut needs ~{need:.1f} GB for a {shape[0]}x{shape[1]} grid "
            f"({n:,} nodes at ~{BYTES_PER_NODE} B/node) but only {avail:.1f} GB is "
            f"available. Either lower --resolution (each level is 4x fewer pixels: "
            f"scale3 would need ~{need/16:.1f} GB), or pass --mrf_solver lbp, which "
            f"streams through a temp file instead of holding the graph in RAM."
        )
    return need


def solve_binary_mrf(prob_map, beta, *, connectivity=4):
    """Globally minimise the Potts energy above. Returns an int8 label array.

    Parameters
    ----------
    prob_map : 2D float array
        Per-pixel probability of the "good" class, as produced by the pixel or
        cell probability images.
    beta : float
        Pairwise smoothness penalty, the same parameter LBP takes.
    connectivity : {4, 8}
        Neighbourhood used for the pairwise term. 4 matches the LBP kernel.
    """
    import maxflow  # optional dependency; imported lazily so import spoqc is cheap

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    p = np.asarray(prob_map, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"prob_map must be 2D, got shape {p.shape}")

    check_memory(p.shape)

    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes(p.shape)

    structure = None
    if connectivity == 8:
        structure = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    if structure is None:
        graph.add_grid_edges(nodes, beta)
    else:
        graph.add_grid_edges(nodes, beta, structure=structure, symmetric=True)

    cost_bad = -np.log(1.0 - p + EPS)   # cost of assigning label 0
    cost_good = -np.log(p + EPS)        # cost of assigning label 1
    # calibrated order: (source=cost_good, sink=cost_bad) yields 0=bad, 1=good
    graph.add_grid_tedges(nodes, cost_good, cost_bad)

    graph.maxflow()
    return graph.get_grid_segments(nodes).astype(np.int8)


def soft_scores(prob_map, beta, *, weight=None):
    """Continuous companion to the labels, for the `*_beliefs` channels.

    Total-variation denoising is the convex relaxation of the same Potts energy
    (Chambolle-Pock), so this is the principled continuous analogue rather than
    an unrelated blur. It also returns a genuine value in [0, 1], unlike the LBP
    "beliefs", which are a normalised *cost ratio* (see the audit note in
    markov_random_field_zarr_parallel) and are not a probability.
    """
    from skimage.restoration import denoise_tv_chambolle

    p = np.asarray(prob_map, dtype=np.float32)
    if weight is None:
        # larger beta == stronger smoothing in the discrete energy; TV's weight
        # plays the same role, scaled into the range TV expects
        weight = float(np.clip(beta / 15.0, 0.01, 1.0))
    return denoise_tv_chambolle(p, weight=weight).astype(np.float32)


def first_version_loopy_belief_propagation_parallel(
    prob_map_np,
    spoqc_tmp_folder,
    modality,
    beta=1.0,
    alpha=0.3,
    max_iter=20,
    normalize="min",
    tolerance=1e-8,
    flip_tolerance=1e-6,
    flip_check=10,
):
    """Drop-in replacement with the LBP signature, so call sites need no change.

    alpha/max_iter/normalize/tolerance/flip_* are accepted and ignored: the graph
    cut is exact, so there is nothing to damp, iterate, or test for convergence.
    Nothing is written to spoqc_tmp_folder -- the 32 bytes/pixel message memmap
    that LBP needs does not exist here.
    """
    print("[NOTE] Solving the binary MRF exactly via graph cut (s-t min cut)")
    labels = solve_binary_mrf(prob_map_np, beta)
    beliefs = soft_scores(prob_map_np, beta)
    return beliefs, labels
