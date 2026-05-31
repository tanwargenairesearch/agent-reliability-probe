"""Enterprise agent patterns the probe can measure.

  react  — reason-then-act loop on Google ADK
  skill  — skill-based (governed procedure) on Google ADK
  direct — no-harness baseline (single completion; the floor, not an agent)

The provider is a swappable backend chosen via the model id (e.g.
`openai/gpt-4o-mini`, `anthropic/claude-haiku-4-5`, `gemini/gemini-2.5-flash`).
"""

from __future__ import annotations

HARNESSES = ("react", "skill", "direct")


def build_agent(harness: str, model: str, temperature: float = 0.7):
    if harness == "direct":
        from .direct import DirectAgent

        return DirectAgent(model, temperature)
    if harness == "react":
        from .react import ReActAgent

        return ReActAgent(model, temperature)
    if harness == "skill":
        from .skill import SkillAgent

        return SkillAgent(model, temperature)
    raise ValueError(f"unknown harness {harness!r}; choose from {HARNESSES}")
