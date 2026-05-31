"""The Agent boundary the probe measures.

The probe is deliberately ignorant of *how* an agent works. It hands an agent a
scenario and reads back the agent's final decision text. Everything below this
line — the control loop (ReAct, skill-based), the framework (ADK), and the LLM
provider underneath — is the agent's business and is the thing under test.

    probe ── act(scenario) ──▶ Agent ── (ADK: ReAct | Skill) ── (provider) ──▶ text

The LLM provider is a *swappable backend*, not the subject of measurement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..scenarios import Scenario


@runtime_checkable
class Agent(Protocol):
    """Anything the probe can measure: name it, ask it to act on a scenario."""

    name: str   # the pattern under test, e.g. "react" | "skill" | "direct"
    model: str  # provider-prefixed backend id, e.g. "openai/gpt-4o-mini"

    def act(self, scenario: Scenario) -> str:
        """Run the agent on one scenario; return its final response text.

        The probe parses the DECISION out of this text and grades it. Whether the
        agent reasoned in a loop, called tools, or invoked skills is opaque here —
        that is exactly the point.
        """
        ...
