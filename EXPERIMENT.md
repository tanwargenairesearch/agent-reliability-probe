# Pre-registration — "Model or harness: what moves enterprise-agent reliability?"

> Pre-registered **before** results exist. Committing this first is the line
> between "I ran some evals" and "I ran an experiment." Hypotheses, metrics, and
> the analysis plan are fixed here; the writeup will report against them honestly,
> including null results.

## Hypotheses

- **H1 (headline):** On identical policy-decision tasks, the **agent harness**
  (`direct` → `react` → `skill`) moves reliability (`pass^k`) *more* than the
  **provider** (model) does.
- **H2 (the surprise):** The configuration with the **highest task reliability**
  (`pass^k`) is **not** the one with the **lowest safety-violation rate**
  (`violate_ever@k`) — capability and safety trade off.

Every outcome is reportable: H1 true → "fix your harness, not your model"; H1
false → "harness hype is overblown"; H2 true → "your most capable agent is your
least safe"; H2 false → "structure buys success *and* safety."

## Design

- **Factors:** `harness {direct, react, skill}` × `provider {3 same-tier models}`
  = up to 9 cells. ReAct/Skill are built on Google ADK; the provider is a
  swappable backend (ADK native Gemini / ADK `LiteLlm` for others).
- **Held constant** (this is what neutralizes the implementation confound): the
  scenarios, the policy text, the available tools, temperature, `k`, the grader,
  and the prompt — *only the control-flow scaffold differs* between harnesses.
- **Scenario classes:**
  - *must-do* (issue the valid refund, qualify the eligible lead) → metric `pass^k`
  - *must-not-do / safety-critical* (`safety_critical: true`: cross-account
    confidentiality, unauthorized discount, disclosure without identity) →
    metric `violate_ever@k`
- **Trials:** `k = 8` (subsample to `k = 5` and fewer tasks for the first pass).

## Metrics (the profile — `profile.py`)

| metric | meaning | source |
|---|---|---|
| `pass@1` | accuracy, right once | Codex (Chen 2021) |
| `pass^k` | reliability, right every time | τ-bench (Yao 2024) |
| `decision_consistency` | behavioral stability across runs | Self-Consistency (Wang 2022) |
| `violate_ever@k` | worst-case safety on must-not-do tasks | CRMArena confidentiality |

All reported with **percentile bootstrap 95% CIs** over the task set. Point
estimates without CIs will not be reported.

## Analysis plan

- **H1:** reliability spread from varying harness (mean over providers of
  max−min across harnesses) vs from varying provider (the reverse), on `pass^k`.
  Report both, their ratio, and the verdict. (`experiment.analyze_h1`)
- **H2:** rank cells by `pass^k` and by `violate_ever@k`; report whether the
  best-success cell is also the safest. (`experiment.analyze_h2`)

## Sequencing (do NOT build all datasets up front)

1. **Phase 1 — synthetic, end-to-end.** Run the sweep on the bundled scenarios to
   validate the whole pipeline (ADK agents + sweep + profile + CIs) cheaply.
   `arprobe experiment --harnesses direct react skill --models ... --k 8`
2. **Phase 1b — one real domain.** Wire τ-bench **retail** (`integrations/tau2.py`),
   re-run. If H1/H2 signal survives real tasks + a real grader, continue.
3. **Phase 2 — expand only on signal.** Add τ-bench airline + telecom (cheap; same
   adapter) → cross-domain robustness. Optionally BFCL for tool-use breadth.

Datasets: **τ-bench retail / airline / telecom** (run — MIT, real grader) are the
core. **BFCL** optional. **CRMArena / WorkArena** are *cited context, not run*
(CC-BY-NC + heavy environments).

## Limitations (state these in any writeup — they add credibility)

- One implementation of each pattern; a better-engineered ReAct could differ.
  We measure control-flow-held-minimal, **not** the Platonic pattern.
- Phase-1 scenarios are single-decision and synthetic; τ-bench grounding (Phase
  1b+) is multi-turn and externally graded.
- Numbers using τ-bench are an **ADK-agent reliability study using τ-bench
  tasks/grader — not an official τ-bench leaderboard score.**
- The implementation confound (is Skill just better-engineered than ReAct?) is
  mitigated by prompt-matching, not eliminated. Say so.

## Stop criteria

- If Phase 1 shows no separation between harnesses **and** the CIs are wide,
  fix scenarios/k before scaling — do not integrate more datasets.
- If a null result is robust (tight CIs, no effect), publish the null. It is a
  finding.
