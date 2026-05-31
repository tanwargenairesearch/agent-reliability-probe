"""Command-line interface: `arprobe`.

  arprobe reproduce <results.json> [--k 8]              # offline, no keys: recompute pass^k
  arprobe eval --harness skill --model openai/gpt-4o-mini   # measure an agent pattern
  arprobe scenarios                                      # list bundled business scenarios
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _print_report(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        from .runner import render_markdown

        print(render_markdown(report))


def cmd_reproduce(args: argparse.Namespace) -> int:
    from .reproduce import reproduce

    report = reproduce(args.results, k=args.k, success_threshold=args.success_threshold)
    _print_report(report, args.json)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .agents import build_agent
    from .runner import evaluate
    from .scenarios import load_scenarios

    scenarios = load_scenarios(args.scenarios)
    agent = build_agent(args.harness, args.model, args.temperature)
    report = evaluate(agent, scenarios, k=args.k, temperature=args.temperature)
    _print_report(report, args.json)
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from .experiment import render_markdown, run_sweep
    from .scenarios import load_scenarios

    scenarios = load_scenarios(args.scenarios)
    report = run_sweep(
        args.harnesses,
        args.models,
        scenarios,
        k=args.k,
        temperature=args.temperature,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_markdown(report))
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    from .scenarios import load_scenarios

    for sc in load_scenarios(args.scenarios):
        print(f"[{sc.domain}] {sc.id}: expects {sc.expected_decision} — {sc.request[:70]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arprobe", description="Agent reliability probe.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reproduce", help="Recompute pass^k from a result file (offline).")
    r.add_argument("results", help="Path to a JSON file of per-trial outcomes.")
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--success-threshold", type=float, default=1.0)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_reproduce)

    e = sub.add_parser("eval", help="Measure an agent's reliability across k trials.")
    e.add_argument(
        "--harness",
        required=True,
        choices=["react", "skill", "direct"],
        help="Agent pattern under test: react / skill (ADK), or direct (no-harness baseline).",
    )
    e.add_argument(
        "--model",
        required=True,
        help="Provider backend id, e.g. openai/gpt-4o-mini, anthropic/claude-haiku-4-5, gemini/gemini-2.5-flash",
    )
    e.add_argument("--scenarios", default=None, help="Scenario JSON (defaults to bundled set).")
    e.add_argument("--k", type=int, default=8)
    e.add_argument("--temperature", type=float, default=0.7)
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_eval)

    x = sub.add_parser("experiment", help="Run the pre-registered harness × provider sweep.")
    x.add_argument("--harnesses", nargs="+", default=["direct", "react", "skill"])
    x.add_argument("--models", nargs="+", required=True, help="Provider backend ids.")
    x.add_argument("--scenarios", default=None)
    x.add_argument("--k", type=int, default=8)
    x.add_argument("--temperature", type=float, default=0.7)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--json", action="store_true")
    x.set_defaults(func=cmd_experiment)

    s = sub.add_parser("scenarios", help="List the loaded business scenarios.")
    s.add_argument("--scenarios", default=None)
    s.set_defaults(func=cmd_scenarios)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
