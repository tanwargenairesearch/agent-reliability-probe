"""Tests for scenario loading and decision grading (offline, no model calls)."""

from agent_reliability_probe.scenarios import grade, load_scenarios, parse_decision


def test_bundled_scenarios_load():
    scenarios = load_scenarios()
    assert len(scenarios) >= 6
    assert all(s.expected_decision in s.allowed_decisions for s in scenarios)


def test_parse_decision_from_labeled_line():
    allowed = ("APPROVE", "DENY")
    assert parse_decision("Reasoning...\nDECISION: DENY\nBecause policy.", allowed) == "DENY"
    assert parse_decision("decision: approve — within window", allowed) == "APPROVE"


def test_parse_decision_ambiguous_returns_none():
    allowed = ("APPROVE", "DENY")
    assert parse_decision("I could approve or deny this", allowed) is None


def test_grade_matches_expected():
    sc = load_scenarios()[0]  # retail_refund_within_window -> APPROVE
    assert grade(sc, "DECISION: APPROVE") is True
    assert grade(sc, "DECISION: DENY") is False
