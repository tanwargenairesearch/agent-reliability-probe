"""Resumable τ-bench reliability run: checkpoints each task to JSONL.

Each completed task is flushed to --checkpoint immediately, so a kill loses at
most the in-flight task. Re-run with the same checkpoint to resume; it skips
tasks already recorded and aggregates at the end.

  python scripts/tau2_resumable.py --agent-model gemini/gemini-3.5-flash \
    --user-model gemini/gemini-3.5-flash --reasoning-effort medium \
    --domain retail --num-tasks 20 --k 5 --concurrency 3 \
    --checkpoint runs/retail_gemini.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_reliability_probe.profile import ScenarioRun, reliability_profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-model", required=True)
    ap.add_argument("--user-model", default=None)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--num-tasks", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--agent-temperature", type=float, default=None,
                    help="Force agent llm temperature (e.g. 1.0 for kimi-k2.6).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--transcripts", default=None,
                    help="Dir to save per-trial transcripts (messages+reward) for failure-mode analysis.")
    ap.add_argument("--task-ids", default=None,
                    help="Comma-separated task ids to run (default: first --num-tasks).")
    a = ap.parse_args()

    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_single_task

    _agent_args = {}
    if a.reasoning_effort:
        _agent_args["reasoning_effort"] = a.reasoning_effort
    if a.agent_temperature is not None:
        _agent_args["temperature"] = a.agent_temperature
    extra = {"llm_args_agent": _agent_args} if _agent_args else {}
    cfg = TextRunConfig(
        domain=a.domain, task_set_name=a.domain, num_trials=1,
        llm_agent=a.agent_model, llm_user=a.user_model or a.agent_model,
        agent="llm_agent", user="user_simulator", max_steps=a.max_steps,
        log_level="ERROR", max_concurrency=1, hallucination_retries=0, **extra,
    )
    et = EvaluationType("env")
    if a.task_ids:
        ids = [t.strip() for t in a.task_ids.split(",") if t.strip()]
        tasks = get_tasks(a.domain, task_ids=ids)
    else:
        tasks = get_tasks(a.domain, num_tasks=a.num_tasks)

    tdir = Path(a.transcripts) if a.transcripts else None
    if tdir:
        tdir.mkdir(parents=True, exist_ok=True)

    def _save_transcript(sim, tid, trial, decision):
        if not tdir:
            return
        try:
            msgs = []
            for m in (getattr(sim, "messages", None) or []):
                msgs.append({
                    "role": getattr(m, "role", type(m).__name__),
                    "content": getattr(m, "content", None),
                    "tool_calls": [
                        {"name": getattr(tc, "name", None), "arguments": getattr(tc, "arguments", None)}
                        for tc in (getattr(m, "tool_calls", None) or [])
                    ] or None,
                })
            ri = getattr(sim, "reward_info", None)
            out = {
                "task_id": tid, "trial": trial, "domain": a.domain,
                "decision": decision,
                "reward": getattr(ri, "reward", None),
                "termination_reason": str(getattr(sim, "termination_reason", None)),
                "messages": msgs,
            }
            (tdir / f"{a.domain}_{tid}_trial{trial}.json").write_text(
                json.dumps(out, default=str, indent=2), encoding="utf-8")
        except Exception as exc:  # never let capture break a run
            print(f"[transcript-skip] {tid}/{trial}: {exc!r}", file=sys.stderr)

    cp = Path(a.checkpoint)
    cp.parent.mkdir(parents=True, exist_ok=True)

    # lockfile: prevent two processes writing the same checkpoint (causes
    # duplicate task rows). Released on exit.
    lock = cp.with_suffix(cp.suffix + ".lock")
    try:
        lock_fd = open(lock, "x")
    except FileExistsError:
        print(f"ERROR: {lock} exists — another run owns {cp}. "
              f"If stale, delete it and retry.", file=sys.stderr)
        sys.exit(2)
    import atexit
    atexit.register(lambda: (lock_fd.close(), lock.unlink(missing_ok=True)))

    done = set()
    if cp.exists():
        for line in cp.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["task_id"])

    def episode(task, trial):
        tid = str(getattr(task, "id", task))
        for attempt in range(5):
            try:
                sim = run_single_task(cfg, task, seed=trial, evaluation_type=et)
                ri = getattr(sim, "reward_info", None)
                rew = getattr(ri, "reward", None)
                decision = "PASS" if (rew is not None and rew >= 1.0) else "FAIL"
                _save_transcript(sim, tid, trial, decision)
                return decision
            except Exception as exc:
                m = (str(exc) + type(exc).__name__).lower()
                if "api_key" in m or "credential" in m or "authentication" in m or attempt == 4:
                    print(f"[drop] {exc!r}", file=sys.stderr)
                    return None
                time.sleep(2 * (attempt + 1))

    with cp.open("a") as fh:
        for task in tasks:
            tid = str(getattr(task, "id", task))
            if tid in done:
                continue
            with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                decisions = list(pool.map(lambda tr: episode(task, tr), range(a.k)))
            valid = [d for d in decisions if d is not None]
            fh.write(json.dumps({"task_id": tid, "domain": a.domain, "decisions": valid, "k": a.k}) + "\n")
            fh.flush()
            print(f"[done] task {tid}: {valid}", file=sys.stderr)

    recs = {}
    for line in cp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["task_id"]] = r
    runs = [
        ScenarioRun(r["task_id"], r.get("domain", "?"), False, "PASS", r["decisions"])
        for r in recs.values()
        if len(r["decisions"]) == a.k
    ]
    print(f"AGGREGATE: graded {len(runs)}/{a.num_tasks} tasks")
    if runs:
        p = reliability_profile(runs, k=a.k, n_boot=1000)
        pk = f"pass^{a.k}"
        print(f"pass@1={p['point']['pass@1']:.3f}  {pk}={p['point'][pk]:.3f}  CI={p['ci'][pk]}  consistency={p['point']['decision_consistency']:.3f}")


if __name__ == "__main__":
    main()
