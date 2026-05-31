"""ReActAgent — a reason-then-act agent on Google ADK.

The classic enterprise pattern: the model reasons about the request, may call
tools to gather facts, then decides. Here it is given a `lookup_policy` tool and
asked to reason explicitly before emitting DECISION.

This pattern shows its real character on **multi-step, tool-using** tasks; on the
bundled single-decision scenarios (facts baked into the request) it mostly
reasons. Add tool-using scenarios to differentiate it meaningfully from Skill.

Requires the [adk] extra. Validate the run path in `_adk.py` against your
google-adk version.
"""

from __future__ import annotations

from ..scenarios import Scenario
from .spec import parse_model_spec

_INSTRUCTION = (
    "You are an enterprise support agent using ReAct: think step by step, use "
    "tools when they help, then apply the company policy exactly. Reason briefly, "
    "then end with the decision on its own line as `DECISION: <LABEL>`."
)


def _lookup_policy(query: str) -> dict:
    """Stub tool: in a real deployment this retrieves the governing policy text.

    Kept trivial because the bundled scenarios embed the policy in the prompt.
    Replace with a real retrieval/CRM tool for tool-using scenarios.
    """
    return {"note": "Policy is provided in the system context for these scenarios."}


class ReActAgent:
    name = "react"

    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model  # full spec for display (may include :effort)
        self._model, self._effort = parse_model_spec(model)
        self.temperature = temperature
        # TODO(adk-thinking): apply self._effort via ADK's thinking_config when set.

    def act(self, scenario: Scenario) -> str:
        from google.adk.agents import LlmAgent

        from ._adk import adk_model, run_once

        agent = LlmAgent(
            name="react_support_agent",
            model=adk_model(self._model),
            instruction=_INSTRUCTION + "\n\n" + scenario.system_prompt(),
            tools=[_lookup_policy],
        )
        return run_once(agent, scenario.request)
