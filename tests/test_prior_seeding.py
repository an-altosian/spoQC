"""The GaussianMixture priors are seeded from the run's seed, independent of numpy's global RNG.

origin/dev fitted them with random_state=None, i.e. on numpy's global random state, so each fit
depended on everything that drew from it before (step order, running a step alone). Seeded, a
fit neither reads nor advances the global state: its result is the same whatever ran before it,
and it leaves the state for later consumers untouched.
"""

import numpy as np
import pandas as pd
import pytest

from spoqc import helperfuncs
from spoqc.priors.hqcr import negative_probe_counts
from spoqc.priors.hqpr import pixel_score

SEED = 123  # cli.py's run seed
PRIOR_DRAWS = [0, 1, 7, 1000]  # np.random.rand() calls made before the fit


@pytest.fixture(autouse=True)
def no_histograms(monkeypatch):
    monkeypatch.setattr(
        helperfuncs, "plot_histogram_for_array", lambda *args, **kwargs: None
    )


def pixel_scores():
    """100 cluster scores (production fits one score per pixel cluster, 100 clusters) in three
    overlapping groups, so the 3-component fit's k-means initialisation decides the optimum."""
    rng = np.random.default_rng(3)
    return np.concatenate(
        [rng.gamma(2.0, 1.0, 60), rng.normal(4.0, 1.5, 30), rng.normal(7.0, 2.0, 10)]
    )


def negative_probe_frame():
    counts = np.random.default_rng(5).poisson(0.8, 2_000)
    return pd.DataFrame({"control_probe_counts": counts})


def after_draws(draws, fit):
    np.random.seed(0)
    np.random.rand(draws)
    return fit()


PRIORS = {
    "pixel_score": lambda: pixel_score.calc_probs_pixel_score(
        pixel_scores(), None, 3, 6, seed=SEED
    ),
    "negative_probes": lambda: negative_probe_counts.calc_probs(
        negative_probe_frame(), None, seed=SEED
    ),
    # t=None: the fitted mean is used, so the result depends on the fit
    "negative_probes_fitted_mean": lambda: negative_probe_counts.calc_probs(
        negative_probe_frame(), None, 3, 1, None, None, seed=SEED
    ),
}


@pytest.mark.parametrize("name", PRIORS)
def test_prior_does_not_depend_on_earlier_global_draws(name):
    expected = after_draws(0, PRIORS[name])
    for draws in PRIOR_DRAWS[1:]:
        got = after_draws(draws, PRIORS[name])
        assert np.array_equal(got, expected, equal_nan=True), (
            f"{name} changed after {draws} global draws"
        )


@pytest.mark.parametrize("name", PRIORS)
def test_prior_leaves_the_global_state_untouched(name):
    np.random.seed(0)
    np.random.rand(7)
    before = np.random.get_state()
    PRIORS[name]()
    after = np.random.get_state()
    assert (
        after[0] == before[0]
        and np.array_equal(after[1], before[1])
        and after[2:] == before[2:]
    ), f"{name} advanced numpy's global random state"


def test_prior_follows_the_seed():
    """The seed is used: another seed gives another 3-component fit on these scores."""
    scores = pixel_scores()
    same = pixel_score.calc_probs_pixel_score(scores, None, 3, 6, seed=SEED)
    assert np.array_equal(
        pixel_score.calc_probs_pixel_score(scores, None, 3, 6, seed=SEED), same
    )
    others = [
        pixel_score.calc_probs_pixel_score(scores, None, 3, 6, seed=s)
        for s in range(20)
    ]
    assert any(not np.array_equal(o, same) for o in others), (
        "no seed changes the fit: the test data is not init-sensitive"
    )
