"""Scaled τ-bench reliability run → reliability profile + report.

Requires the [tau2] path: Python 3.12/3.13, tau2-bench installed, TAU2_DATA_DIR
set, and provider keys in the environment. Writes a JSON report.

  python scripts/tau2_scale_run.py --agent-model gemini/gemini-3.5-flash \
    --user-model gemini/gemini-3.5-flash --reasoning-effort medium \
    --domain retail --num-tasks 20 --k 5 --concurrency 8 --out runs/retail_gemini.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_reliability_probe.integrations.tau2 import run_tau2_native_cell
from agent_reliability_probe.profile import reliability_profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-model", required=True)
    ap.add_argument("--user-model", default=None)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--num-tasks", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = run_tau2_native_cell(
        a.agent_model,
        domain=a.domain,
        num_tasks=a.num_tasks,
        k=a.k,
        user_model=a.user_model,
        max_steps=a.max_steps,
        max_concurrency=a.concurrency,
        reasoning_effort=a.reasoning_effort,
    )
    p = reliability_profile(runs, k=a.k, n_boot=1000)
    pk = f"pass^{a.k}"
    out = {
        "agent_model": a.agent_model,
        "user_model": a.user_model or a.agent_model,
        "reasoning_effort": a.reasoning_effort,
        "domain": a.domain,
        "num_tasks": len(runs),
        "k": a.k,
        "per_task": [{"task": r.task_id, "successes": r.successes, "trials": r.trials} for r in runs],
        "profile": p,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", a.out)
    print(
        f"{a.agent_model} on {a.domain}: tasks={len(runs)} "
        f"pass@1={p['point']['pass@1']:.3f} {pk}={p['point'][pk]:.3f} "
        f"CI={p['ci'][pk]} consistency={p['point']['decision_consistency']:.3f}"
    )


if __name__ == "__main__":
    main()
