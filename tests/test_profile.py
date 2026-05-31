"""Tests for the reliability profile (offline, deterministic bootstrap via seed)."""

from math import isclose

from agent_reliability_probe.profile import ScenarioRun, reliability_profile


def _run(decisions, expected="APPROVE", safety=False, domain="d"):
    return ScenarioRun("t", domain, safety, expected, decisions)


def test_scenariorun_derived_counts():
    r = _run(["APPROVE", "APPROVE", "DENY", None], expected="APPROVE", safety=False)
    assert r.trials == 4
    assert r.successes == 2
    assert r.violations == 0  # not safety-critical
    assert isclose(r.modal_agreement(), 0.5)  # APPROVE appears 2/4


def test_violations_only_count_for_safety_critical_affirmative_errors():
    # must-not-do task: expected DENY; APPROVE is a violation, None (abstain) is not
    r = _run(["DENY", "APPROVE", "APPROVE", None], expected="DENY", safety=True)
    assert r.violations == 2
    assert r.successes == 1


def test_profile_separates_reliable_from_flaky():
    runs = [
        _run(["APPROVE"] * 8, expected="APPROVE"),            # perfectly reliable
        _run(["APPROVE", "DENY"] * 4, expected="APPROVE"),    # flaky: 4/8
    ]
    p = reliability_profile(runs, k=8, n_boot=200, seed=0)
    assert p["tasks"] == 2
    # pass@1 = mean(1.0, 0.5) = 0.75 ; pass^8 = mean(1.0, 0) = 0.5
    assert isclose(p["point"]["pass@1"], 0.75)
    assert isclose(p["point"]["pass^8"], 0.5)
    # consistency: reliable task = 1.0, flaky task = 0.5 -> mean 0.75
    assert isclose(p["point"]["decision_consistency"], 0.75)
    # no safety-critical tasks -> violate_ever is None
    assert p["point"]["violate_ever@8"] is None


def test_profile_violate_ever_on_safety_tasks():
    runs = [
        _run(["DENY"] * 8, expected="DENY", safety=True),          # never violates
        _run(["DENY"] * 6 + ["APPROVE"] * 2, expected="DENY", safety=True),  # violates
    ]
    p = reliability_profile(runs, k=8, n_boot=200, seed=0)
    # task1 violate_ever@8 = 0 ; task2 has 2 violations in 8 -> at least one in 8 = 1.0
    # mean = 0.5
    assert isclose(p["point"]["violate_ever@8"], 0.5)


def test_profile_reports_bootstrap_cis():
    runs = [_run(["APPROVE", "DENY"] * 4, expected="APPROVE") for _ in range(6)]
    p = reliability_profile(runs, k=8, n_boot=300, seed=1)
    lo, hi = p["ci"]["pass^8"]
    assert lo <= p["point"]["pass^8"] <= hi or lo == hi  # point within CI
