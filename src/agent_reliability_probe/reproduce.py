"""Reproduce pass^k reliability from published benchmark result files — offline.

This needs no API keys. Point it at a JSON file of per-trial outcomes (e.g. raw
τ-bench result dumps, or your own runs) and it recomputes the pass@1 vs pass^k
gap that the τ-bench paper reports. The goal: make a cited number something you
can re-run, not just quote.

Accepted shapes (auto-detected):

  {"trials": [{"task_id": "t1", "success": true},  ...]}
  {"simulations": [{"task_id": "t1", "reward": 1.0}, ...]}   # τ-bench style
  [{"task_id": "t1", "reward": 1.0}, ...]

`reward >= --success-threshold` (default 1.0) counts as a success.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .metrics import TaskResult
from .runner import build_report


def _rows(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    for key in ("trials", "simulations", "results"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    raise ValueError("Could not find a list of trials in the file")


def load_results(path: str | Path, success_threshold: float = 1.0) -> list[TaskResult]:
    """Group per-trial rows by task and return TaskResults."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped: dict[str, list[bool]] = defaultdict(list)
    domains: dict[str, str] = {}
    for row in _rows(payload):
        task_id = str(row.get("task_id") or row.get("id"))
        if "success" in row:
            ok = bool(row["success"])
        elif "reward" in row:
            ok = float(row["reward"]) >= success_threshold
        else:
            raise ValueError(f"row for {task_id} has neither 'success' nor 'reward'")
        grouped[task_id].append(ok)
        domains.setdefault(task_id, str(row.get("domain", "default")))
    return [
        TaskResult(tid, successes=sum(flags), trials=len(flags), domain=domains[tid])
        for tid, flags in grouped.items()
    ]


def reproduce(path: str | Path, k: int = 8, success_threshold: float = 1.0) -> dict:
    results = load_results(path, success_threshold)
    if not results:
        raise ValueError("no tasks found")
    return build_report(results, model=f"reproduced:{Path(path).name}", k=k, temperature=0.0)
