# Result — Gemini-3.5-flash vs Kimi-K2.6 on τ-bench retail (n=100)

**Setup.** τ-bench retail, **100 tasks**, k=5 trials, DB end-state grading
(`EvaluationType.ENV` — deterministic, no LLM judge), user-simulator =
`gemini/gemini-3.5-flash`. Agents are τ-bench's *native* `llm_agent` on each
provider (not an ADK harness). Gemini ran with medium thinking; Kimi at
temperature 1 (its requirement). Reproduce: `scripts/tau2_resumable.py`.

> Not an official τ-bench leaderboard score: DB-state grading excludes τ-bench's
> communication/NL-assertion checks, and the user-simulator is not the gpt-4.1
> default. This is an agent-reliability study *using* τ-bench tasks + grader.

## Numbers (n=100)

| metric | gemini-3.5-flash (medium) | kimi-k2.6 |
|---|---|---|
| pass@1 (accuracy) | 0.842  (95% CI 0.78–0.90) | 0.848  (0.79–0.90) |
| pass^5 (reliability) | 0.700  (0.61–0.79) | 0.720  (0.63–0.80) |
| decision consistency | 0.936 | 0.936 |
| reliability gap (pass@1 − pass^5) | 0.142 | 0.128 |
| always-pass / always-fail / flaky | 70 / 7 / 23 | 72 / 5 / 23 |

## The findings

1. **Accuracy overstates reliability by ~14 points.** Both models are ~85% "right
   once" but only ~71–72% "right every time." That gap is the demo-to-production
   gap.
2. **The two models are statistically tied.** pass@1 0.846 vs 0.848, pass^5 0.710
   vs 0.720 — CIs almost fully overlap. There is no meaningful reliability
   difference between them on this task set.
3. **The unreliability is mostly flakiness.** ~22% of tasks are flaky (pass on
   some trials, fail on others), plus ~5–7% that fail every time. Intermittent
   failure is the hardest mode to catch pre-production.

## The small-sample reversal (a finding in itself)

At **n=20**, the picture was misleading in two ways:
- Kimi looked clearly better (pass@1 0.95 vs Gemini 0.90).
- Both showed almost **no** reliability gap (pass^5 ≈ pass@1 ≈ 0.90).

At **n=100**, both reversed: the ranking vanished (tied ~0.85) and a ~14-point
reliability gap emerged for both. Small evaluation sets can invert both the
ranking and the headline metric — measure reliability on enough tasks.

## Caveats

- DB-state grading only (excludes communication/NL checks); user-sim = Gemini;
  τ-bench **native** agent, not an ADK ReAct/Skill harness (the harness
  comparison is the next build).
- 100 tasks, k=5; CIs are still ~±0.09 on pass^5 — directionally solid, not
  precise to the percent.
