"""Tests for agent system."""
from unittest.mock import MagicMock, patch
from src.agents.config import AgentConfig
from src.agents.base import Agent
from src.agents.prompts.defaults import PLANNER_PROMPT, RESEARCHER_PROMPT, AGENT_PROMPTS
from src.tools.registry import _registry, tool


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self):
        """AgentConfig should have sensible defaults."""
        config = AgentConfig(
            name="test",
            system_prompt="You are a test agent",
        )
        assert config.tools == []
        assert config.max_retries == 3
        assert config.temperature == 0.1

    def test_custom_tools(self):
        """AgentConfig should accept custom tool list."""
        config = AgentConfig(
            name="researcher",
            system_prompt="You research",
            tools=["web_search", "read_file"],
        )
        assert config.tools == ["web_search", "read_file"]

    def test_to_dict(self):
        """to_dict should serialize config."""
        config = AgentConfig(name="x", system_prompt="y", tools=["a"])
        d = config.to_dict()
        assert d["name"] == "x"
        assert d["tools"] == ["a"]


class TestAgent:
    """Tests for Agent execution."""

    @patch("src.agents.base.LLMProvider")
    def test_run_without_tools(self, MockProvider):
        """Agent without tools should return LLM response directly."""
        mock_llm = MagicMock()
        mock_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": "Here is my analysis"}}]
        }

        config = AgentConfig(
            name="planner",
            system_prompt="You plan tasks: {task}",
            tools=[],
        )
        agent = Agent(config, mock_llm)
        result = agent.run(task="Plan a trip")

        assert "Here is my analysis" in result
        mock_llm.chat_completion.assert_called_once()

    def test_run_with_tools(self):
        """Agent should execute tool calls and continue."""
        # Register a test tool
        @tool(name="echo", description="Echo back")
        def echo(message: str) -> str:
            return f"ECHO: {message}"

        mock_llm = MagicMock()
        # First call: LLM requests a tool call
        mock_llm.chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "echo",
                                "arguments": '{"message": "hello"}'
                            }
                        }]
                    }
                }]
            },
            # Second call: LLM returns final response
            {
                "choices": [{"message": {"content": "Done! The echo worked."}}]
            },
        ]

        config = AgentConfig(
            name="tester",
            system_prompt="Test agent: {task}",
            tools=["echo"],
        )
        agent = Agent(config, mock_llm)
        result = agent.run(task="Test the echo")

        assert "Done" in result
        assert mock_llm.chat_completion.call_count == 2

        # Clean up registry
        del _registry["echo"]

    def test_run_max_retries(self):
        """Agent should stop after max_retries even if LLM keeps calling tools."""
        mock_llm = MagicMock()
        # Always return tool calls (never stops)
        mock_llm.chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "echo", "arguments": '{"message":"x"}'}
                    }]
                }
            }]
        }

        @tool(name="echo", description="Echo")
        def echo(message: str) -> str:
            return message

        config = AgentConfig(
            name="looper",
            system_prompt="Loop: {task}",
            tools=["echo"],
            max_retries=2,
        )
        agent = Agent(config, mock_llm)
        result = agent.run(task="loop")

        assert mock_llm.chat_completion.call_count == 2

        del _registry["echo"]


class TestPrompts:
    """Tests for agent prompts."""

    def test_all_agents_have_prompts(self):
        """Every expected agent type should have a prompt."""
        for name in ["planner", "researcher", "writer", "reviewer"]:
            assert name in AGENT_PROMPTS, f"Missing prompt for {name}"

    def test_planner_prompt_format(self):
        """Planner prompt should accept task variable."""
        formatted = PLANNER_PROMPT.format(task="Research AI")
        assert "Research AI" in formatted

    def test_researcher_prompt_format(self):
        """Researcher prompt should accept task and context."""
        formatted = RESEARCHER_PROMPT.format(task="Find info", context="Previous: ...")
        assert "Find info" in formatted
        assert "Previous: ..." in formatted
