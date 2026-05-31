"""Provider-agnostic chat completion.

One interface over OpenAI, Anthropic, Google Gemini, Mistral, local Ollama, and
more — via LiteLLM. Models use LiteLLM naming so the provider is explicit:

    openai/gpt-4o-mini
    anthropic/claude-haiku-4-5
    gemini/gemini-2.5-flash
    ollama/llama3.1            (fully local, no API key)

API keys come from the usual environment variables (OPENAI_API_KEY,
ANTHROPIC_API_KEY, GEMINI_API_KEY, ...). Nothing is sent anywhere unless you
run a live evaluation with a real model.
"""

from __future__ import annotations


def complete(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 60,
    reasoning_effort: str | None = None,
) -> str:
    """Return the assistant text for a chat completion.

    temperature defaults to 0.7 on purpose: reliability is about consistency
    under realistic sampling, so the default surfaces run-to-run variance rather
    than hiding it at temperature 0.

    reasoning_effort ("low" | "medium" | "high") is LiteLLM's provider-agnostic
    thinking control — it maps to Gemini's thinking budget, Anthropic extended
    thinking, etc. Left None for models that don't take it.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Live evaluation needs the providers extra: "
            "pip install 'agent-reliability-probe[providers]'"
        ) from exc

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content or ""
