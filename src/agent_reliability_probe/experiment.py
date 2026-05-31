"""The pre-registered experiment: harness × provider reliability sweep.

Runs the same scenarios through every (harness, provider) cell, k trials each,
and computes a reliability profile per cell. Two analyses fall out of the one
grid (see EXPERIMENT.md for the pre-registration):

  H1  Does the harness move reliability more than the provider?
  H2  Is the most *capable* configuration also the *safest* one?

Both `run_cell` (live) and the analysis functions (pure) are separable so the
analysis can be tested without any model calls.
"""

from __future__ import annotations

from .profile import ScenarioRun, reliability_profile
from .scenarios import Scenario, parse_decision


def run_cell(agent, scenarios: list[Scenario], k: int) -> list[ScenarioRun]:
    """Run every scenario k times through one agent; capture the parsed decisions."""
    runs: list[ScenarioRun] = []
    for sc in scenarios:
        decisions = [parse_decision(agent.act(sc), sc.allowed_decisions) for _ in range(k)]
        runs.append(ScenarioRun(sc.id, sc.domain, sc.safety_critical, sc.expected_decision, decisions))
    return runs


def run_sweep(
    harnesses,
    models,
    scenarios,
    k: int = 8,
    temperature: float = 0.7,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Run the full grid and return profiles + the H1/H2 analyses."""
    from .agents import build_agent

    cells: dict[tuple[str, str], dict] = {}
    for h in harnesses:
        for m in models:
            agent = build_agent(h, m, temperature)
            runs = run_cell(agent, scenarios, k)
            cells[(h, m)] = reliability_profile(runs, k, n_boot=n_boot, seed=seed)
    return {
        "k": k,
        "harnesses": list(harnesses),
        "models": list(models),
        "cells": {f"{h} | {m}": p for (h, m), p in cells.items()},
        "h1": analyze_h1(cells, k),
        "h2": analyze_h2(cells, k),
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def analyze_h1(cells: dict[tuple[str, str], dict], k: int) -> dict:
    """H1: compare reliability spread from varying harness vs varying provider."""
    key = f"pass^{k}"
    harnesses = sorted({h for h, _ in cells})
    models = sorted({m for _, m in cells})

    def val(h, m):
        return cells[(h, m)]["point"][key]

    harness_spreads = []
    for m in models:
        vals = [val(h, m) for h in harnesses if (h, m) in cells]
        if len(vals) > 1:
            harness_spreads.append(max(vals) - min(vals))
    provider_spreads = []
    for h in harnesses:
        vals = [val(h, m) for m in models if (h, m) in cells]
        if len(vals) > 1:
            provider_spreads.append(max(vals) - min(vals))

    he, pe = _mean(harness_spreads), _mean(provider_spreads)
    verdict = "harness dominates" if he > pe else "provider dominates" if pe > he else "comparable"
    return {
        "metric": key,
        "harness_effect": round(he, 4),
        "provider_effect": round(pe, 4),
        "ratio_harness_over_provider": (round(he / pe, 2) if pe else None),
        "verdict": verdict,
    }


def analyze_h2(cells: dict[tuple[str, str], dict], k: int) -> dict:
    """H2: is the highest-pass^k cell also the lowest-violation cell?"""
    pk, vk = f"pass^{k}", f"violate_ever@{k}"
    rows = [
        {"cell": f"{h} | {m}", "passk": p["point"][pk], "violate": p["point"].get(vk)}
        for (h, m), p in cells.items()
    ]
    with_safety = [r for r in rows if r["violate"] is not None]
    best = max(rows, key=lambda r: r["passk"])
    safest = min(with_safety, key=lambda r: r["violate"]) if with_safety else None
    tradeoff = safest is not None and best["cell"] != safest["cell"]
    return {
        "best_success_cell": best["cell"],
        "best_success_passk": round(best["passk"], 4),
        "best_success_violate_ever": (round(best["violate"], 4) if best["violate"] is not None else None),
        "safest_cell": safest["cell"] if safest else None,
        "tradeoff_exists": tradeoff,
        "interpretation": (
            "the most capable configuration is NOT the safest"
            if tradeoff
            else "capability and safety coincide (no trade-off observed)"
        ),
    }


def render_markdown(report: dict) -> str:
    k = report["k"]
    pk, vk = f"pass^{k}", f"violate_ever@{k}"
    lines = [
        "# Experiment — harness × provider reliability sweep",
        "",
        f"k = {k} trials/scenario. Pre-registration: EXPERIMENT.md. "
        "Not an official benchmark score.",
        "",
        f"| cell (harness \\| provider) | pass@1 | {pk} (95% CI) | consistency | {vk} |",
        "|---|---|---|---|---|",
    ]
    for cell, p in report["cells"].items():
        if not p.get("point"):
            continue
        pt, ci = p["point"], p["ci"]
        ci_pk = ci.get(pk, ("?", "?"))
        ve = pt.get(vk)
        lines.append(
            f"| {cell} | {pt['pass@1']:.2f} | {pt[pk]:.2f} [{ci_pk[0]:.2f}–{ci_pk[1]:.2f}] "
            f"| {pt['decision_consistency']:.2f} | {ve if ve is None else f'{ve:.2f}'} |"
        )
    h1, h2 = report["h1"], report["h2"]
    lines += [
        "",
        "## H1 — harness vs provider",
        f"- harness effect on {pk}: **{h1['harness_effect']:.2f}** · "
        f"provider effect: **{h1['provider_effect']:.2f}** · "
        f"ratio: **{h1['ratio_harness_over_provider']}×** → **{h1['verdict']}**",
        "",
        "## H2 — capability vs safety",
        f"- highest {pk}: **{h2['best_success_cell']}** "
        f"(violate_ever = {h2['best_success_violate_ever']}) · "
        f"safest: **{h2['safest_cell']}**",
        f"- **{h2['interpretation']}**",
        "",
        "_Scope: controlled, prompt-matched implementations of each pattern on "
        "policy-decision tasks. Measures these implementations, not the patterns in "
        "the abstract. See EXPERIMENT.md for limitations._",
    ]
    return "\n".join(lines)
