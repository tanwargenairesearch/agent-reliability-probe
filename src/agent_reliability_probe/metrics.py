"""Reliability metrics for LLM agents.

The headline distinction this repo exists to make:

    accuracy  = "did it succeed on one attempt?"   (pass@1)
    reliability = "does it succeed every time?"     (pass^k)

`pass^k` is the probability that **all** k independent attempts on a task
succeed. A 90%-accurate agent (pass@1 = 0.90) is far less than 90% reliable
once k > 1. The reliability gap is a field-wide finding (WebArena, AgentBench,
τ-bench, BFCL — see SOURCES.md); the metric and estimator below come from:

  - pass^k metric: τ-bench, Yao et al. 2024 (arXiv:2406.12045)
  - unbiased combinatorial estimator: Codex, Chen et al. 2021 (arXiv:2107.03374)
  - run-to-run variance motivating it: Self-Consistency, Wang et al. 2022 (arXiv:2203.11171)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class TaskResult:
    """Outcome of running one task `n` times."""

    task_id: str
    successes: int
    trials: int
    domain: str = "default"

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError(f"{self.task_id}: trials must be > 0")
        if not 0 <= self.successes <= self.trials:
            raise ValueError(f"{self.task_id}: successes out of range")


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimate that all k of k i.i.d. attempts succeed, for one task.

    Estimator: C(c, k) / C(n, k), where c = successes, n = trials.
    Returns 0.0 if there aren't k successes to draw from.

    Raises ValueError if k > trials (cannot estimate pass^k without k samples).
    """
    if k > trials:
        raise ValueError(f"k={k} exceeds trials={trials}; run more trials")
    if successes < k:
        return 0.0
    return comb(successes, k) / comb(trials, k)


def pass_at_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimate that at least one of k i.i.d. attempts succeeds (Codex).

    Estimator: 1 - C(n - c, k) / C(n, k). Provided for contrast with pass^k —
    pass@k flatters, pass^k is honest about consistency.
    """
    if k > trials:
        raise ValueError(f"k={k} exceeds trials={trials}; run more trials")
    failures = trials - successes
    if failures < k:
        return 1.0
    return 1.0 - comb(failures, k) / comb(trials, k)


def aggregate_pass_hat_k(results: list[TaskResult], k: int) -> float:
    """Mean pass^k across tasks. Tasks with fewer than k trials are skipped."""
    usable = [r for r in results if r.trials >= k]
    if not usable:
        return float("nan")
    return sum(pass_hat_k(r.successes, r.trials, k) for r in usable) / len(usable)


def reliability_curve(results: list[TaskResult], max_k: int | None = None) -> dict[int, float]:
    """pass^k for k = 1..max_k. Defaults max_k to the smallest trial count."""
    if not results:
        return {}
    cap = min(r.trials for r in results)
    top = cap if max_k is None else min(max_k, cap)
    return {k: aggregate_pass_hat_k(results, k) for k in range(1, top + 1)}


def reliability_gap(results: list[TaskResult], k: int) -> float:
    """pass@1 minus pass^k — how much consistency is lost across k attempts."""
    return aggregate_pass_hat_k(results, 1) - aggregate_pass_hat_k(results, k)
