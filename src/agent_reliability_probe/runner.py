"""Run an agent over scenarios, k trials each, and compute reliability.

What is measured is the **agent** — its pattern (ReAct / skill-based / baseline)
on top of a swappable provider — not the model in isolation. Each scenario is run
`k` times, graded, and reduced to pass@1 vs pass^k, exposing how often an agent
that "looks right once" actually holds up across attempts on real business
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import TaskResult, aggregate_pass_hat_k, reliability_curve, reliability_gap
from .scenarios import Scenario, grade


@dataclass
class TrialLog:
    scenario_id: str
    domain: str
    successes: int
    trials: int
    raw_responses: list[str] = field(default_factory=list)


def evaluate(agent, scenarios: list[Scenario], k: int = 8, temperature: float = 0.7) -> dict:
    """Run every scenario k times against `agent`; return a report dict.

    `agent` is any object satisfying the Agent protocol (agents/base.py): it
    carries the harness pattern under test and the provider backend.
    """
    logs: list[TrialLog] = []
    for sc in scenarios:
        successes = 0
        responses: list[str] = []
        for _ in range(k):
            text = agent.act(sc)
            responses.append(text)
            if grade(sc, text):
                successes += 1
        logs.append(TrialLog(sc.id, sc.domain, successes, k, responses))

    results = [TaskResult(l.scenario_id, l.successes, l.trials, l.domain) for l in logs]
    return build_report(
        results,
        harness=getattr(agent, "name", "?"),
        model=getattr(agent, "model", "?"),
        k=k,
        temperature=temperature,
    )


def build_report(
    results: list[TaskResult], *, harness: str = "reproduced", model: str, k: int, temperature: float
) -> dict:
    """Aggregate TaskResults into a JSON-serializable reliability report."""
    curve = reliability_curve(results, max_k=k)
    by_domain: dict[str, dict] = {}
    domains = sorted({r.domain for r in results})
    for d in domains:
        subset = [r for r in results if r.domain == d]
        by_domain[d] = {
            "tasks": len(subset),
            "pass@1": round(aggregate_pass_hat_k(subset, 1), 4),
            f"pass^{k}": round(aggregate_pass_hat_k(subset, k), 4),
        }
    return {
        "harness": harness,
        "model": model,
        "k": k,
        "temperature": temperature,
        "tasks": len(results),
        "pass@1": round(aggregate_pass_hat_k(results, 1), 4),
        f"pass^{k}": round(aggregate_pass_hat_k(results, k), 4),
        "reliability_gap": round(reliability_gap(results, k), 4),
        "reliability_curve": {str(kk): round(v, 4) for kk, v in curve.items()},
        "by_domain": by_domain,
    }


def render_markdown(report: dict) -> str:
    """A compact, shareable markdown summary of a reliability report."""
    k = report["k"]
    lines = [
        f"# Reliability report — agent `{report.get('harness', '?')}` on `{report['model']}`",
        "",
        f"- Tasks: **{report['tasks']}** · trials/task (k): **{k}** · temp: {report['temperature']}",
        f"- **pass@1 = {report['pass@1']:.2f}** (accuracy)  vs  "
        f"**pass^{k} = {report[f'pass^{k}']:.2f}** (reliability)",
        f"- Reliability gap (pass@1 − pass^{k}): **{report['reliability_gap']:.2f}**",
        "",
        "## pass^k curve",
        "",
        "| k | pass^k |",
        "|---|--------|",
    ]
    for kk, v in report["reliability_curve"].items():
        lines.append(f"| {kk} | {v:.2f} |")
    lines += ["", "## By domain", "", f"| domain | tasks | pass@1 | pass^{k} |", "|---|---|---|---|"]
    for d, m in report["by_domain"].items():
        lines.append(f"| {d} | {m['tasks']} | {m['pass@1']:.2f} | {m[f'pass^{k}']:.2f} |")
    lines += ["", "_Methodology and citations: see SOURCES.md._"]
    return "\n".join(lines)
