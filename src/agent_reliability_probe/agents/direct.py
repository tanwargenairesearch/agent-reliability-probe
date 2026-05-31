"""DirectAgent — the no-harness baseline.

A single model completion with the policy in the prompt. No loop, no tools, no
state. This is NOT an enterprise agent pattern; it is the *floor* — "what does
the bare model do?" — so the ReAct and Skill agents have something to beat.

Provider is swappable (any LiteLLM-supported backend). Needs the [providers]
extra and the matching API key.
"""

from __future__ import annotations

from ..scenarios import Scenario
from .spec import parse_model_spec


class DirectAgent:
    name = "direct"

    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model  # full spec, kept for display (may include :effort)
        self._model, self._effort = parse_model_spec(model)
        self.temperature = temperature

    def act(self, scenario: Scenario) -> str:
        from ..providers import complete

        return complete(
            self._model,
            [
                {"role": "system", "content": scenario.system_prompt()},
                {"role": "user", "content": scenario.request},
            ],
            temperature=self.temperature,
            reasoning_effort=self._effort,
        )
