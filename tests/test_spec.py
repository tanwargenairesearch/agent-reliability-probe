"""Tests for model-spec parsing (model id + optional thinking level)."""

from agent_reliability_probe.agents import build_agent
from agent_reliability_probe.agents.spec import parse_model_spec


def test_plain_model_has_no_effort():
    assert parse_model_spec("moonshot/kimi-k2.6") == ("moonshot/kimi-k2.6", None)
    assert parse_model_spec("gemini/gemini-3.5-flash") == ("gemini/gemini-3.5-flash", None)


def test_thinking_suffix_is_parsed():
    assert parse_model_spec("gemini/gemini-3.5-flash:medium") == (
        "gemini/gemini-3.5-flash",
        "medium",
    )
    assert parse_model_spec("anthropic/claude-haiku-4-5:high") == (
        "anthropic/claude-haiku-4-5",
        "high",
    )


def test_unknown_suffix_is_not_treated_as_effort():
    # a stray colon that isn't a known level stays part of the model id
    assert parse_model_spec("vendor/model:v2") == ("vendor/model:v2", None)


def test_direct_agent_threads_effort_and_keeps_display_model():
    agent = build_agent("direct", "gemini/gemini-3.5-flash:medium")
    assert agent.model == "gemini/gemini-3.5-flash:medium"   # display
    assert agent._model == "gemini/gemini-3.5-flash"          # call target
    assert agent._effort == "medium"
