"""Differential test: KD-tree transcript/doublet proximity vs the original serial loop."""

import numpy as np
import pandas as pd
import pytest

from spoqc.metrics.segmentation.doublet_score import flag_transcripts_near_doublets

DISTANCE_THRESH = 10  # value qc_doublets.py passes to calc_doublet_score
THREADS = 4
ORIGIN = 20_000.0  # Xenium-scale coordinates, so float32 rounding is live
N_RANDOM_TRANSCRIPTS = 200_000
N_DOUBLETS = 60


def original_loop(transcript_coordinates_df, corrected_doublet_df, distance_thresh):
    """Verbatim copy of the loop replaced in calc_doublet_score (spoQC db00d98)."""
    transcript_doublet = np.array([False] * len(transcript_coordinates_df))
    transcript_wdoublet = np.array([0] * len(transcript_coordinates_df))
    for i, doublet in corrected_doublet_df.iterrows():
        x1, y1 = doublet["x"], doublet["y"]
        distances = np.sqrt(
            (transcript_coordinates_df["x"] - x1) ** 2
            + (transcript_coordinates_df["y"] - y1) ** 2
        )
        transcript_doublet[distances <= distance_thresh] = True
        transcript_wdoublet[distances <= distance_thresh] = 1
    return transcript_doublet, transcript_wdoublet


def ovrlpy_style_doublets(rng, n, min_x, min_y, extent):
    """Mimic ovrlpy.detect_doublets().to_pandas() + the coordinate correction in calc_doublet_score."""
    doublet_df = (
        pd.DataFrame(
            {
                "x": rng.integers(0, extent, n).astype(np.int64),
                "y": rng.integers(0, extent, n).astype(np.int64),
                "integrity": rng.random(n).astype(np.float32),
                "signal": (rng.random(n) * 10).astype(np.float32),
            }
        )
        .sort_values("integrity")
        .reset_index(drop=True)
    )
    corrected = doublet_df.copy()
    corrected["x"] = doublet_df["x"] + min_x
    corrected["y"] = doublet_df["y"] + min_y
    return corrected


def boundary_points(corrected_doublet_df, distance_thresh, rng, n_angles):
    """float32 points at, and 1-2 float32 ULPs either side of, distance_thresh from each doublet."""
    xs, ys = [], []
    for _, doublet in corrected_doublet_df.iterrows():
        x1, y1 = np.float32(doublet["x"]), np.float32(doublet["y"])
        # axis-aligned: exact float32 distance equals the threshold
        base = [
            (x1 + np.float32(distance_thresh), y1),
            (x1, y1 - np.float32(distance_thresh)),
        ]
        # 3-4-5 diagonal and random angles
        base.append(
            (
                x1 + np.float32(0.6 * distance_thresh),
                y1 + np.float32(0.8 * distance_thresh),
            )
        )
        for theta in rng.uniform(0, 2 * np.pi, n_angles):
            base.append(
                (
                    np.float32(x1 + distance_thresh * np.cos(theta)),
                    np.float32(y1 + distance_thresh * np.sin(theta)),
                )
            )
        for bx, by in base:
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-1, 0, 1):
                    px, py = bx, by
                    for _ in range(abs(dx)):
                        px = np.nextafter(px, np.float32(np.inf * dx), dtype=np.float32)
                    for _ in range(abs(dy)):
                        py = np.nextafter(py, np.float32(np.inf * dy), dtype=np.float32)
                    xs.append(px)
                    ys.append(py)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def transcripts_frame(rng, x, y):
    """Transcript frame shaped like sdata.points[...].compute(): float32 x/y, non-range index."""
    n = len(x)
    df = pd.DataFrame(
        {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "z": rng.random(n).astype(np.float32),
            "feature_name": pd.Categorical(rng.choice(["A", "B", "C"], n)),
        }
    )
    df.index = rng.permutation(n) + 10_000
    return df


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def case(rng):
    extent = 400
    random_x = ORIGIN + rng.random(N_RANDOM_TRANSCRIPTS) * extent
    random_y = ORIGIN + rng.random(N_RANDOM_TRANSCRIPTS) * extent
    min_x = min(
        pd.Series(random_x.astype(np.float32))
    )  # as calc_doublet_score: builtin min -> Python float
    min_y = min(pd.Series(random_y.astype(np.float32)))
    doublets = ovrlpy_style_doublets(rng, N_DOUBLETS, min_x, min_y, extent)
    # two overlapping doublets (a transcript near both) and one isolated doublet with no transcripts near it
    extra = pd.DataFrame(
        {
            "x": [doublets["x"][0] + 3, ORIGIN + 10 * extent],
            "y": [doublets["y"][0], ORIGIN + 10 * extent],
            "integrity": np.float32([0.5, 0.5]),
            "signal": np.float32([5, 5]),
        }
    )
    doublets = pd.concat([doublets, extra], ignore_index=True)
    bx, by = boundary_points(doublets, DISTANCE_THRESH, rng, n_angles=8)
    x = np.concatenate([random_x.astype(np.float32), bx])
    y = np.concatenate([random_y.astype(np.float32), by])
    return transcripts_frame(rng, x, y), doublets, len(bx)


class TestFlagTranscriptsNearDoublets:
    def test_fixture_mimics_ovrlpy_dtypes(self, case):
        transcripts, doublets, _ = case
        assert transcripts["x"].dtype == np.float32
        assert (
            doublets["x"].dtype == np.float64
            and doublets["integrity"].dtype == np.float32
        )
        _, row = next(doublets.iterrows())
        assert type(row["x"]) is np.float64

    def test_boundary_points_straddle_threshold(self, case):
        transcripts, doublets, n_boundary = case
        ref, _ = original_loop(transcripts, doublets, DISTANCE_THRESH)
        boundary = ref[-n_boundary:]
        assert boundary.any() and not boundary.all(), (
            "adversarial points must fall on both sides"
        )

    def test_matches_original_loop_exactly(self, case):
        transcripts, doublets, _ = case
        ref_doublet, ref_wdoublet = original_loop(
            transcripts, doublets, DISTANCE_THRESH
        )
        new_doublet, new_wdoublet = flag_transcripts_near_doublets(
            transcripts, doublets, DISTANCE_THRESH, THREADS
        )
        assert new_doublet.dtype == ref_doublet.dtype
        assert new_wdoublet.dtype == ref_wdoublet.dtype
        assert np.array_equal(new_doublet, ref_doublet)
        assert np.array_equal(new_wdoublet, ref_wdoublet)
        assert ref_doublet.any()

    def test_single_thread_matches_multi_thread(self, case):
        transcripts, doublets, _ = case
        one = flag_transcripts_near_doublets(transcripts, doublets, DISTANCE_THRESH, 1)
        many = flag_transcripts_near_doublets(
            transcripts, doublets, DISTANCE_THRESH, THREADS
        )
        assert np.array_equal(one[0], many[0]) and np.array_equal(one[1], many[1])

    def test_no_doublets_flags_nothing(self, case):
        transcripts, doublets, _ = case
        empty = doublets.iloc[:0]
        ref = original_loop(transcripts, empty, DISTANCE_THRESH)
        new = flag_transcripts_near_doublets(
            transcripts, empty, DISTANCE_THRESH, THREADS
        )
        assert np.array_equal(new[0], ref[0]) and np.array_equal(new[1], ref[1])
        assert new[0].dtype == ref[0].dtype and new[1].dtype == ref[1].dtype
