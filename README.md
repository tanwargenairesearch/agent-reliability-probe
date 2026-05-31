# agent-reliability-probe

**How reliably do enterprise AI agents solve real business problems — not just once, but every time?**

Enterprise agents (CRM, support, sales, operations) get judged on *accuracy* —
did it get it right once? Production needs *reliability* — does it get it right
**every** time, while respecting policy and confidentiality? This is a small,
local, provider-agnostic harness that measures that gap on real policy-bound
business decisions, with methodology you can audit and numbers you can reproduce.

[![ci](https://github.com/OWNER/agent-reliability-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/agent-reliability-probe/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> Personal, educational project. Opinions are my own. Every reported number is
> cited in [SOURCES.md](SOURCES.md) to a primary source.

---

## The one idea

A **90%-accurate** agent is **not** 90% reliable. `pass@1` measures one shot;
`pass^k` measures the chance that **all k** attempts succeed. This repo lets you
measure that gap yourself — on your own provider and your own scenarios.

This isn't one benchmark's opinion. Independent evaluations — closest to real
business problems first — reach the same conclusion: agents look acceptable on a
single shot and fall apart on realistic, multi-turn, or repeated tasks.

| Benchmark | Business domain | Headline result |
|---|---|---|
| [CRMArena-Pro](https://arxiv.org/abs/2505.18878) (Salesforce, 2025) | CRM: sales, service, CPQ (B2B/B2C) | **58%** single-turn → **35%** multi-turn; ~0 confidentiality awareness |
| [CRMArena](https://arxiv.org/abs/2411.02305) (Salesforce, 2024) | Professional CRM tasks | **<40%** (ReAct), **<55%** with tools |
| [WorkArena](https://arxiv.org/abs/2403.07718) (ServiceNow, ICLR'24) | Enterprise knowledge work | Considerable gap to automation |
| [τ-bench](https://arxiv.org/abs/2406.12045) (Sierra, 2024) | Retail / airline / telecom policies | **<50%**; `pass^8 < 25%` (retail) |
| [WebArena](https://arxiv.org/abs/2307.13854) (ICLR'24) | Web tasks, 4 domains | GPT-4 **14.4%** vs **78%** human |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) (ICML'25) | Function calling / tool use | Aces one-shot, **stumbles multi-step** |

Full citations and BibTeX: [SOURCES.md](SOURCES.md). The `pass^k` metric comes
from τ-bench; the unbiased estimator from the Codex paper.

```
accuracy   = pass@1   "right once"
reliability = pass^k   "right every time"   ← what production actually needs
```

## What we measure: the agent, not the model

This is **not** a model leaderboard. We measure the reliability of an **enterprise
agent** — built on [Google ADK](https://google.github.io/adk-docs/) — with the LLM
provider as a *swappable backend*. Three layers; only the middle one is under test:

```
PROBE      runs scenarios × k trials, computes the reliability profile
   │ act(scenario)
AGENT      ← under test: ReAct  vs  Skill-based   (built on Google ADK)
   │ uses
PROVIDER   swappable backend: openai/* · anthropic/* · gemini/* · perplexity/*
```

Two dominant enterprise agent patterns, compared head-to-head:

| Harness | Pattern | The bet |
|---|---|---|
| `react` | Reason → act loop, dynamic tool use | Flexibility handles open-ended tasks |
| `skill` | Composed, governed skill procedures | Structure buys reliability & policy discipline |
| `direct` | Single completion, no harness | *Baseline floor* — what the bare model does |

The headline question: **ReAct vs Skill-based — which is the more reliable enterprise
agent, and does it depend on the provider?**

> Scope note: the patterns differ most on **multi-step, tool-using** tasks. The
> bundled scenarios are single-decision (policy in the prompt), so they're a
> starting point — add tool-using, stateful scenarios to fully separate the
> patterns. This probe *measures* agents; it does not *optimize* them.

## What it does

- **Reproduce** the published reliability gap from any result file — **offline, no API keys.**
- **Evaluate** an **agent pattern** (`react` / `skill` / `direct`) on **real business
  decision scenarios** (CRM confidentiality, entitlements, discount approvals, refunds, identity).
- Swap the **provider** backend per run (OpenAI, Anthropic, Gemini, Perplexity, …).
- Reports **pass@1 vs pass^k**, a **reliability curve**, and the **reliability gap**, per domain.

## Install

```bash
pip install -e .                      # core (reproduce; offline)
pip install -e ".[providers]"         # direct baseline across providers (LiteLLM)
pip install -e ".[adk]"               # ReAct / skill agents on Google ADK
pip install -e ".[dev]"               # tests
```

## Quickstart

**1. Reproduce a cited number — offline, zero keys.**
Point it at per-trial outcomes (your own runs, or public τ-bench result dumps):

```bash
arprobe reproduce results.json --k 8
```
Accepted shapes are auto-detected (`{"trials":[{"task_id","success"}]}` or
τ-bench-style `{"simulations":[{"task_id","reward"}]}`). See
[reproduce.py](src/agent_reliability_probe/reproduce.py).

**2. Measure an agent pattern — pick the harness, pick the provider backend.**

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GEMINI_API_KEY / PERPLEXITYAI_API_KEY

# same provider, two agent patterns — the head-to-head
arprobe eval --harness react --model anthropic/claude-haiku-4-5 --k 8
arprobe eval --harness skill --model anthropic/claude-haiku-4-5 --k 8

# same pattern, swap the provider backend
arprobe eval --harness skill --model openai/gpt-4o-mini --k 8 --json > skill_openai.json

# the no-harness baseline (just the model, no ADK)
arprobe eval --harness direct --model gemini/gemini-2.5-flash --k 8
```

Sample output:

```
# Reliability report — agent `skill` on `anthropic/claude-haiku-4-5`
- Tasks: 11 · trials/task (k): 8 · temp: 0.7
- pass@1 = 0.93 (accuracy)  vs  pass^8 = 0.71 (reliability)
- Reliability gap (pass@1 − pass^8): 0.22
```
*(Illustrative shape; run it to get real figures for the agent/provider you pick.
`react`/`skill` need the `[adk]` extra; `direct` needs `[providers]`.)*

**3. Inspect the scenarios.**

```bash
arprobe scenarios
```

## How the metrics work

`pass^k` per task with `c` successes in `n` trials uses the unbiased estimator
`C(c, k) / C(n, k)`, averaged over tasks — the same combinatorial estimator the
Codex paper introduced for `pass@k`, applied to "all k succeed."
Implementation: [metrics.py](src/agent_reliability_probe/metrics.py).
Worked values are pinned in [tests/test_metrics.py](tests/test_metrics.py):

```bash
pytest -q
```

## Real business scenarios

Bundled scenarios are hand-authored, generic policy decisions across **CRM,
retail, airline, telecom, and banking** — each a case where an agent must apply a
rule *consistently*: a cross-account confidentiality gate, a CRM entitlement /
priority check, a CPQ discount-approval threshold, a refund window, a
non-changeable fare, an identity gate before disclosure. (These are original,
not derived from any licensed dataset.) Bring your own at the same schema:

```bash
arprobe eval --model anthropic/claude-haiku-4-5 --scenarios my_scenarios.json
```

Schema: [data/scenarios/sample_business.json](data/scenarios/sample_business.json).

## Why this framing matters

The benchmarks above agree that single-shot accuracy isn't the blocker —
consistency is. And it shows up in production: MIT's 2025 enterprise study found
~95% of GenAI pilots delivered no measurable return, with the core barrier being
systems that *"do not retain feedback, adapt to context, or improve over time."*
S&P Global reports the share of companies abandoning most AI initiatives rose
from 17% to 42% in a year. Measuring reliability honestly is the first step. All
figures cited to primary sources in [SOURCES.md](SOURCES.md).

## Citing

Methodology and every figure: [SOURCES.md](SOURCES.md) (with BibTeX).

## License

MIT — see [LICENSE](LICENSE).
