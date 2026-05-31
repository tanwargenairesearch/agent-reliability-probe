"""Worked-value tests for the reliability estimators.

These pin the math behind every reported number — the verifiable core of the
repo. Values are hand-checked against C(c,k)/C(n,k).
"""

from math import isclose, isnan

import pytest

from agent_reliability_probe.metrics import (
    TaskResult,
    aggregate_pass_hat_k,
    pass_at_k,
    pass_hat_k,
    reliability_curve,
    reliability_gap,
)


def test_pass_hat_k_all_success_is_one():
    assert pass_hat_k(8, 8, 8) == 1.0
    assert pass_hat_k(8, 8, 4) == 1.0


def test_pass_hat_k_insufficient_successes_is_zero():
    assert pass_hat_k(7, 8, 8) == 0.0
    assert pass_hat_k(0, 8, 1) == 0.0


def test_pass_hat_k_known_fractions():
    assert isclose(pass_hat_k(4, 8, 1), 0.5)          # 4/8
    assert isclose(pass_hat_k(6, 8, 2), 15 / 28)      # C(6,2)/C(8,2)
    assert isclose(pass_hat_k(2, 4, 2), 1 / 6)        # C(2,2)/C(4,2)


def test_pass_hat_k_requires_enough_trials():
    with pytest.raises(ValueError):
        pass_hat_k(2, 4, 5)


def test_pass_at_k_contrast_with_pass_hat_k():
    # 50% single-shot: at-least-one of 8 is near-certain; all-of-8 is tiny.
    assert pass_at_k(4, 8, 8) > 0.9
    assert pass_hat_k(4, 8, 8) == 0.0


def test_aggregate_and_gap():
    results = [
        TaskResult("a", successes=8, trials=8, domain="retail"),  # perfectly reliable
        TaskResult("b", successes=4, trials=8, domain="retail"),  # flaky
    ]
    # pass@1 = mean(1.0, 0.5) = 0.75
    assert isclose(aggregate_pass_hat_k(results, 1), 0.75)
    # pass^8 = mean(1.0, C(4,8)->0) = 0.5
    assert isclose(aggregate_pass_hat_k(results, 8), 0.5)
    assert isclose(reliability_gap(results, 8), 0.25)


def test_reliability_curve_is_monotone_non_increasing():
    results = [TaskResult("a", 6, 8), TaskResult("b", 7, 8)]
    curve = reliability_curve(results, max_k=8)
    vals = list(curve.values())
    assert all(earlier >= later for earlier, later in zip(vals, vals[1:]))


def test_aggregate_skips_when_no_task_has_k_trials():
    results = [TaskResult("a", 2, 4)]
    assert isnan(aggregate_pass_hat_k(results, 8))
