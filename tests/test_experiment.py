"""Tests for the sweep: run_cell with a stub agent, and the pure H1/H2 analyses."""

from agent_reliability_probe.experiment import analyze_h1, analyze_h2, run_cell
from agent_reliability_probe.scenarios import load_scenarios


class StubAgent:
    """Deterministic agent: always emits the same decision. No model calls."""

    def __init__(self, decision: str):
        self.name = "stub"
        self.model = "stub/model"
        self._decision = decision

    def act(self, scenario) -> str:
        return f"DECISION: {self._decision}\nbecause stub."


def test_run_cell_collects_decisions_offline():
    scenarios = load_scenarios()[:3]
    runs = run_cell(StubAgent("DENY"), scenarios, k=5)
    assert len(runs) == 3
    for r in runs:
        assert r.trials == 5
        assert all(d == "DENY" for d in r.decisions)


def _cell(passk, violate=None, k=8):
    point = {f"pass^{k}": passk, "pass@1": passk, "decision_consistency": 1.0}
    point[f"violate_ever@{k}"] = violate
    return {"k": k, "point": point, "ci": {}}


def test_h1_detects_harness_dominance():
    # harness varies pass^k a lot (0.4..0.9); provider barely (within 0.05)
    cells = {
        ("direct", "m1"): _cell(0.40), ("react", "m1"): _cell(0.65), ("skill", "m1"): _cell(0.90),
        ("direct", "m2"): _cell(0.42), ("react", "m2"): _cell(0.66), ("skill", "m2"): _cell(0.91),
    }
    h1 = analyze_h1(cells, k=8)
    assert h1["harness_effect"] > h1["provider_effect"]
    assert h1["verdict"] == "harness dominates"


def test_h2_flags_capability_safety_tradeoff():
    # best success (skill) is NOT the safest (direct has lowest violation)
    cells = {
        ("direct", "m1"): _cell(0.50, violate=0.05),
        ("skill", "m1"): _cell(0.90, violate=0.40),
    }
    h2 = analyze_h2(cells, k=8)
    assert h2["best_success_cell"] == "skill | m1"
    assert h2["safest_cell"] == "direct | m1"
    assert h2["tradeoff_exists"] is True
