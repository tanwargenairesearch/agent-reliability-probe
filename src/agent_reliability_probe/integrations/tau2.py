"""τ-bench / τ²-bench grounding (Sierra Research, MIT) — real multi-turn tasks.

Validated against tau2-bench installed from source on Python 3.13 (audioop-lts
shim) with TAU2_DATA_DIR set to the repo's data/ directory. τ-bench provides the
tasks, the simulated user, the tool environment, AND the DB-state grader — so the
probe never grades itself on real-task runs.

Two paths:

  run_tau2_native_cell(...)   — measure τ-bench's OWN llm_agent on a provider.
                                Real numbers today, no bridge. (implemented)

  run_tau2_adk_cell(...)      — measure our ADK ReAct/Skill agent inside τ-bench
                                by registering an agent factory. (TODO: bridge
                                τ-bench's HalfDuplexAgent tool-calling loop to ADK)

Both return the same `ScenarioRun` objects the reliability profile consumes:
success = τ-bench reward >= 1.0.

Honesty: numbers here are an agent-reliability study USING τ-bench tasks/grader —
NOT an official τ-bench leaderboard score. Using a non-gpt-4.1 user simulator
(e.g. Gemini) is a deviation from τ-bench's default and should be reported.
"""

from __future__ import annotations

from ..profile import ScenarioRun

TAU2_DOMAINS = ("retail", "airline", "telecom", "banking_knowledge")


def _require_tau2():
    try:
        import tau2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError(
            "τ-bench grounding needs Python 3.12/3.13 + the tau2-bench package and data:\n"
            '  pip install "git+https://github.com/sierra-research/tau2-bench.git" audioop-lts\n'
            "  git clone --depth 1 https://github.com/sierra-research/tau2-bench.git /tmp/tau2src\n"
            "  export TAU2_DATA_DIR=/tmp/tau2src/data"
        ) from exc


def _reward(sim) -> float | None:
    info = getattr(sim, "reward_info", None)
    if info is None:
        return None
    return getattr(info, "reward", None) if not isinstance(info, (int, float)) else float(info)


def run_tau2_native_cell(
    agent_model: str,
    domain: str = "retail",
    num_tasks: int = 5,
    k: int = 3,
    user_model: str | None = None,
    max_steps: int = 40,
    seed0: int = 0,
    max_concurrency: int = 1,
    reasoning_effort: str | None = None,
    evaluation_type: str = "env",
) -> list[ScenarioRun]:
    """Measure τ-bench's native llm_agent on `agent_model` over real tasks.

    Runs k independent trials per task (varying the seed) and reads τ-bench's
    DB-state grade. `user_model` defaults to `agent_model` (note the deviation
    from τ-bench's gpt-4.1 default if you don't have an OpenAI key). Episodes are
    I/O-bound API calls, so `max_concurrency` runs them in a thread pool.
    `reasoning_effort` is passed to the agent llm via τ-bench's llm_args_agent.
    """
    _require_tau2()
    if domain not in TAU2_DOMAINS:
        raise ValueError(f"domain must be one of {TAU2_DOMAINS}")
    from concurrent.futures import ThreadPoolExecutor

    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import get_tasks, run_single_task

    # ENV grading = deterministic DB end-state check, no LLM judge. Avoids the
    # OpenAI-default judge that τ-bench's ALL mode invokes on some tasks, and
    # removes LLM-judge variance from the reliability signal. (Excludes
    # communication/NL grading — a deliberate, documented scope.)
    eval_type = EvaluationType(evaluation_type)

    extra = {}
    if reasoning_effort:
        extra["llm_args_agent"] = {"reasoning_effort": reasoning_effort}
    cfg = TextRunConfig(
        domain=domain,
        task_set_name=domain,
        num_trials=1,  # we drive k ourselves, one independent run per seed
        llm_agent=agent_model,
        llm_user=user_model or agent_model,
        agent="llm_agent",
        user="user_simulator",
        max_steps=max_steps,
        log_level="ERROR",
        max_concurrency=1,
        hallucination_retries=0,
        **extra,
    )
    tasks = get_tasks(domain, num_tasks=num_tasks)

    import sys
    import time

    def _episode(args):
        task, trial = args
        tid = str(getattr(task, "id", task))
        for attempt in range(5):  # episodes are flaky upstream; retry before giving up
            try:
                sim = run_single_task(cfg, task, seed=seed0 + trial, evaluation_type=eval_type)
                reward = _reward(sim)
                return tid, ("PASS" if (reward is not None and reward >= 1.0) else "FAIL")
            except Exception as exc:
                msg = (str(exc) + type(exc).__name__).lower()
                permanent = "api_key" in msg or "credential" in msg or "authentication" in msg
                if permanent or attempt == 4:
                    print(f"[tau2] episode {tid}/trial{trial} dropped: {exc!r}", file=sys.stderr)
                    return tid, None
                time.sleep(2 * (attempt + 1))

    jobs = [(t, trial) for t in tasks for trial in range(k)]
    decisions: dict[str, list[str | None]] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for tid, dec in pool.map(_episode, jobs):
            decisions.setdefault(tid, []).append(dec)

    runs, dropped = [], []
    for tid, decs in decisions.items():
        valid = [d for d in decs if d is not None]
        if len(valid) == k:
            runs.append(ScenarioRun(task_id=tid, domain=domain, safety_critical=False, expected="PASS", decisions=valid))
        else:
            dropped.append(tid)
    if dropped:
        print(f"[tau2] dropped {len(dropped)} task(s) with unrecoverable episodes: {dropped}", file=sys.stderr)
    return runs


def run_tau2_adk_cell(agent, domain: str = "retail", num_tasks: int = 5, k: int = 3):
    """Measure our ADK ReAct/Skill agent inside τ-bench. TODO (the bridge).

    Plan: subclass τ-bench's HalfDuplexAgent so its tool-calling loop delegates
    to `agent` (the ADK harness); register it via
    `tau2.registry.register_agent_factory(...)`; set `agent=<name>` in the
    TextRunConfig; reuse the run/grade path above. This is the real ReAct-vs-Skill
    comparison and is the next build.
    """
    raise NotImplementedError(
        "ADK-in-τ-bench bridge not wired yet. Use run_tau2_native_cell for now; "
        "see docstring for the HalfDuplexAgent + register_agent_factory plan."
    )
