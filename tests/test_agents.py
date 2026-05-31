"""Offline tests for the agent layer (no provider/ADK calls)."""

import pytest

from agent_reliability_probe.agents import HARNESSES, build_agent
from agent_reliability_probe.agents.base import Agent


def test_all_harnesses_build_without_provider_or_adk():
    # Construction must not import providers/ADK — those load lazily in act().
    for harness in HARNESSES:
        agent = build_agent(harness, "openai/gpt-4o-mini")
        assert agent.name == harness
        assert agent.model == "openai/gpt-4o-mini"
        assert isinstance(agent, Agent)  # satisfies the protocol the probe needs


def test_unknown_harness_rejected():
    with pytest.raises(ValueError):
        build_agent("autogpt", "openai/gpt-4o-mini")
