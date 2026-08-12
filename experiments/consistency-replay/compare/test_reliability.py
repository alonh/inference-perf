"""Unit tests for the reliability Consistency (C) metrics (compare/reliability.py)."""

import math

import pytest

from compare import (
    compute_outcome_consistency,
    compute_trajectory_consistency,
    compute_resource_consistency,
    seq_levenshtein_similarity,
    weighted_r_con,
    session_success,
    summarize_session_run,
    SessionRun,
)
from compare.reliability import cv


def _ev(ok=True, finish_reason="stop", tool_calls=None, pt=100, ct=50, t=1.0):
    """A minimal parsed-response dict as reliability.py consumes."""
    return {
        "ok": ok, "finish_reason": finish_reason, "tool_calls": tool_calls or [],
        "prompt_tokens": pt, "completion_tokens": ct, "start_time": 0.0, "end_time": t,
    }


def _call(name):
    return {"function": {"name": name, "arguments": "{}"}}


class TestOutcomeConsistency:
    def test_all_identical_is_one(self):
        assert compute_outcome_consistency([True] * 5) == 1.0
        assert compute_outcome_consistency([False] * 5) == 1.0

    def test_single_flip_at_high_p_is_zero(self):
        # p=0.9, var(ddof=1)=0.1, p(1-p)=0.09 -> 1 - 0.1/0.09 < 0 -> clipped to 0.
        assert compute_outcome_consistency([True] * 9 + [False]) == 0.0

    def test_half_split(self):
        # p=0.5, var(ddof=1)=0.5*... -> equal split is maximally inconsistent -> clips to 0.
        assert compute_outcome_consistency([True, True, False, False]) == 0.0

    def test_under_two_runs_is_nan(self):
        assert math.isnan(compute_outcome_consistency([True]))


class TestSeqLevenshtein:
    def test_identical(self):
        assert seq_levenshtein_similarity(("a", "b", "c"), ("a", "b", "c")) == 1.0

    def test_both_empty_identical(self):
        assert seq_levenshtein_similarity((), ()) == 1.0

    def test_one_empty_is_zero(self):
        assert seq_levenshtein_similarity(("a",), ()) == 0.0

    def test_one_substitution(self):
        assert seq_levenshtein_similarity(("a", "b", "c"), ("a", "x", "c")) == pytest.approx(2 / 3)

    def test_order_sensitive(self):
        assert seq_levenshtein_similarity(("a", "b"), ("b", "a")) < 1.0


class TestTrajectoryConsistency:
    def test_identical_trajectories(self):
        d, s = compute_trajectory_consistency([("g", "g", "f"), ("g", "g", "f")])
        assert d == pytest.approx(1.0)
        assert s == pytest.approx(1.0)

    def test_all_empty_trajectories_are_consistent(self):
        # No tool calls in any run -> perfectly consistent (JSD of empty dists = 0).
        d, s = compute_trajectory_consistency([(), (), ()])
        assert d == pytest.approx(1.0)
        assert s == pytest.approx(1.0)

    def test_reorder_hits_sequence_not_distribution(self):
        # Same multiset, different order: distribution identical, sequence penalized.
        d, s = compute_trajectory_consistency([("a", "b"), ("b", "a")])
        assert d == pytest.approx(1.0)
        assert s < 1.0

    def test_under_two_is_nan(self):
        d, s = compute_trajectory_consistency([("a",)])
        assert math.isnan(d) and math.isnan(s)


class TestCV:
    def test_constant_is_zero(self):
        assert cv([3.0, 3.0, 3.0]) == 0.0

    def test_zero_mean_non_error_is_none(self):
        # A non-error channel that is identically zero carries no CV signal.
        assert cv([0.0, 0.0]) is None

    def test_all_zero_error_channel_is_zero_not_none(self):
        # For the error channel, "no errors in any run" is meaningful consistency (CV 0),
        # not a missing signal — this is the branch that separates errors_like from the rest.
        assert cv([0.0, 0.0, 0.0], errors_like=True) == 0.0
        assert cv([0.0, 0.0, 0.0]) is None


class TestResourceConsistency:
    def test_identical_runs_is_one(self):
        r = SessionRun(True, ("a",), cost=0.1, time=1.0, api_calls=2,
                       num_actions=1, num_errors=0, latency_mean=0.5)
        assert compute_resource_consistency([r, r, r]) == pytest.approx(1.0)

    def test_variation_below_one(self):
        r1 = SessionRun(True, ("a",), 0.1, 1.0, 2, 1, 0, 0.5)
        r2 = SessionRun(True, ("a",), 0.2, 3.0, 2, 1, 0, 1.5)
        assert 0.0 < compute_resource_consistency([r1, r2]) < 1.0


class TestSessionSuccessAndSummary:
    def test_success_requires_ok_and_no_length_truncation(self):
        assert session_success([_ev(), _ev(finish_reason="tool_calls")]) is True
        assert session_success([_ev(finish_reason="length")]) is False
        assert session_success([_ev(ok=False, finish_reason=None)]) is False

    def test_summarize_counts_actions_errors_and_trajectory(self):
        events = [
            _ev(tool_calls=[_call("search"), _call("book")]),
            _ev(ok=False, finish_reason=None, tool_calls=[]),
        ]
        sr = summarize_session_run(events)
        assert sr.trajectory == ("search", "book")
        assert sr.num_actions == 2
        assert sr.num_errors == 1
        assert sr.api_calls == 2
        assert sr.success is False  # an errored event fails the session


class TestWeightedRcon:
    def test_all_present_equal_weight(self):
        # mean(traj_d, traj_s) = 0.9; (1.0 + 0.9 + 0.8)/3 = 0.9.
        assert weighted_r_con(1.0, 1.0, 0.8, 0.8) == pytest.approx(0.9)

    def test_nan_masked_and_renormalized(self):
        # Trajectory NaN -> aggregate over outcome + resource only: (1.0 + 0.6)/2 = 0.8.
        assert weighted_r_con(1.0, math.nan, math.nan, 0.6) == pytest.approx(0.8)

    def test_all_nan_is_nan(self):
        assert math.isnan(weighted_r_con(math.nan, math.nan, math.nan, math.nan))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
