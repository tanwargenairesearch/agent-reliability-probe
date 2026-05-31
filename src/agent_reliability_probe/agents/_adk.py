"""Shared Google ADK plumbing for the agent patterns.

Requires the [adk] extra (`pip install 'agent-reliability-probe[adk]'`).

NOTE: ADK's runner/session API surface has shifted across versions. The flow
below targets google-adk >= 1.x (InMemoryRunner + a synchronous run). If your
installed version differs, adjust `run_once` — the rest of the probe is unaffected
because it only depends on the Agent.act() -> str contract.
"""

from __future__ import annotations

APP_NAME = "arprobe"


def adk_model(model: str):
    """Map a provider-prefixed model id to an ADK model.

    Gemini runs natively in ADK; every other provider goes through ADK's LiteLlm
    wrapper, which keeps the provider a swappable backend.
    """
    if model.startswith(("gemini/", "gemini-")):
        return model.split("/", 1)[-1]  # native Gemini model name
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=model)


def run_once(agent, prompt: str) -> str:
    """Run an ADK agent on a single prompt and return its final text."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = runner.session_service.create_session_sync(
        app_name=APP_NAME, user_id="probe"
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    final = ""
    for event in runner.run(
        user_id="probe", session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final = part.text
    return final
