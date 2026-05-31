# Sources

Every number this project reports traces to a **primary source** — the paper,
the report, or raw benchmark data you can re-run yourself. Secondary coverage
(news) is linked only as a convenience pointer, never as the citation of record.
Source *type* is labeled, because an independent benchmark is not a vendor survey.

The framing — *single-shot accuracy overstates how dependable an enterprise agent
is* — is not one benchmark's opinion. It shows up across independent evaluations
of CRM agents, enterprise-workflow agents, web agents, and tool-use agents. This
repo measures that gap with a standard, citable metric.

## 1. Enterprise agents on real business tasks

The benchmarks closest to production business problems — CRM, sales, service,
operations — show the widest reliability gaps, especially multi-turn.

- **CRMArena: LLM Agents on Professional CRM Tasks.** Huang et al. (Salesforce
  AI Research), 2024. A simulated org with 16 interconnected CRM objects in a
  real Salesforce environment. SOTA agents succeed on **<40%** of tasks with
  ReAct, **<55%** even with hand-crafted function-calling tools.
  Type: *independent benchmark.* License: **CC-BY-NC-4.0** (reference only — do
  not vendor). https://arxiv.org/abs/2411.02305

- **CRMArena-Pro: Holistic Assessment Across Business Scenarios.** Huang et al.
  (Salesforce), 2025. 19 expert-validated tasks across sales, service, and CPQ,
  for B2B and B2C, with **multi-turn** interactions and **confidentiality**
  checks. Leading agents reach ~**58%** single-turn but drop to ~**35%**
  multi-turn; workflow execution ~83%; **near-zero confidentiality awareness**.
  Type: *independent benchmark.*
  https://arxiv.org/abs/2505.18878 · https://huggingface.co/datasets/Salesforce/CRMArenaPro

- **WorkArena: Web Agents on Common Knowledge-Work Tasks.** Drouin et al.
  (ServiceNow Research), ICLR 2024. 33 tasks / 19,912 instances on the
  ServiceNow enterprise platform (used by ~85% of the Fortune 500). Finds a
  **considerable gap to full task automation**.
  Type: *independent benchmark.* https://arxiv.org/abs/2403.07718

## 2. The gap is field-wide (web & tool-use agents)

Different teams, different tasks, same conclusion: agents look fine on one shot
and fall apart on realistic, multi-step, or repeated tasks.

- **WebArena.** Zhou et al., ICLR 2024. GPT-4 reaches just **14.4%** end-to-end
  success vs **78%** for humans across 812 real web tasks.
  https://arxiv.org/abs/2307.13854

- **AgentBench.** Liu et al., ICLR 2024. Across 8 environments, **long-horizon
  reasoning, decision-making, and instruction-following** are the main blockers.
  https://arxiv.org/abs/2308.03688

- **τ-bench / τ²-bench.** Yao et al. (Sierra Research), 2024. Tool-agent-user
  tasks under real policies: **<50%** success, `pass^8 < 25%` in retail.
  Introduces the **`pass^k`** metric this repo uses.
  https://arxiv.org/abs/2406.12045 · https://github.com/sierra-research/tau2-bench

- **Berkeley Function Calling Leaderboard (BFCL).** Patil et al., ICML 2025.
  Models **ace one-shot calls but stumble in stateful, multi-step** settings and
  at knowing when *not* to act. https://gorilla.cs.berkeley.edu/leaderboard.html

## 3. Why consistency — not single-shot accuracy — is the right lens

- **Self-Consistency Improves Chain-of-Thought Reasoning.** Wang et al., 2022.
  Sampling the *same* prompt yields **diverse paths and answers** — one run does
  not characterize behavior. https://arxiv.org/abs/2203.11171

- **Evaluating LLMs Trained on Code (Codex).** Chen et al., 2021. Source of the
  unbiased combinatorial **`pass@k`** estimator this repo adapts for `pass^k`.
  https://arxiv.org/abs/2107.03374

> **`pass^k`** = probability that **all** k i.i.d. attempts on a task succeed.
> Per task with `c` successes in `n` trials, the unbiased estimate is
> `C(c, k) / C(n, k)`, averaged over tasks. `pass@1` is accuracy; `pass^k` is
> reliability. Implementation: `metrics.py`; worked values: `tests/test_metrics.py`.

## 4. Why it matters for production (enterprise impact)

- **The GenAI Divide: State of AI in Business 2025.** MIT NANDA, Aug 2025.
  ~**95%** of enterprise GenAI pilots show no measurable P&L return; core barrier
  is systems that *"do not retain feedback, adapt to context, or improve over
  time."* https://fortune.com/2025/08/18/mit-report-95-percent-generative-ai-pilots-at-companies-failing-cfo/

- **Voice of the Enterprise: AI & ML.** S&P Global, 2025. Companies abandoning
  most AI initiatives rose from **17% to 42%** YoY.
  https://www.spglobal.com/market-intelligence/en/news-insights/research/2025/10/generative-ai-shows-rapid-growth-but-yields-mixed-results

- **Gartner, 25 Jun 2025.** **40%+** of agentic AI projects canceled by end of
  2027. https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027

## Citation hygiene used here

1. Cite the primary source; link secondary coverage only as a pointer.
2. Quote the exact figure, the date, and the sample size.
3. Label source type (benchmark / research study / survey / analyst forecast).
4. Prefer a number you can **reproduce** over one you can only quote.
5. Triangulate: lean on what multiple independent sources agree on, not one.
6. Respect dataset licenses (e.g., CRMArena is CC-BY-NC-4.0 — cite, don't vendor).

## BibTeX

```bibtex
@article{huang2024crmarena,
  title   = {CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments},
  author  = {Huang, Kung-Hsiang and others},
  journal = {arXiv preprint arXiv:2411.02305},
  year    = {2024}
}

@article{huang2025crmarenapro,
  title   = {CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions},
  author  = {Huang, Kung-Hsiang and others},
  journal = {arXiv preprint arXiv:2505.18878},
  year    = {2025}
}

@inproceedings{drouin2024workarena,
  title     = {WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?},
  author    = {Drouin, Alexandre and others},
  booktitle = {ICLR},
  year      = {2024},
  note      = {arXiv:2403.07718}
}

@inproceedings{zhou2024webarena,
  title     = {WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author    = {Zhou, Shuyan and others},
  booktitle = {ICLR}, year = {2024}, note = {arXiv:2307.13854}
}

@inproceedings{liu2024agentbench,
  title     = {AgentBench: Evaluating LLMs as Agents},
  author    = {Liu, Xiao and others},
  booktitle = {ICLR}, year = {2024}, note = {arXiv:2308.03688}
}

@article{yao2024taubench,
  title   = {{$\tau$-bench}: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
  author  = {Yao, Shunyu and others},
  journal = {arXiv preprint arXiv:2406.12045}, year = {2024}
}

@inproceedings{patil2025bfcl,
  title     = {The Berkeley Function Calling Leaderboard (BFCL)},
  author    = {Patil, Shishir G. and others},
  booktitle = {ICML}, year = {2025}
}

@inproceedings{wang2023selfconsistency,
  title     = {Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author    = {Wang, Xuezhi and others},
  booktitle = {ICLR}, year = {2023}, note = {arXiv:2203.11171}
}

@article{chen2021codex,
  title   = {Evaluating Large Language Models Trained on Code},
  author  = {Chen, Mark and others},
  journal = {arXiv preprint arXiv:2107.03374}, year = {2021}
}
```
