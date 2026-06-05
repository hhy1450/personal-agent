"""Tests for router and executor nodes."""
from unittest.mock import MagicMock, patch
from src.engine.nodes.router import RouterNode
from src.engine.nodes.executor import ExecutorNode


class TestRouterNode:
    """Tests for RouterNode."""

    def test_routes_research(self):
        """Should route 'research' subtask to 'researcher'."""
        router = RouterNode()
        state = {
            "task": "Test",
            "plan": [{"type": "research", "description": "Find info"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }
        assert router(state) == "researcher"

    def test_routes_write(self):
        """Should route 'write' subtask to 'writer'."""
        router = RouterNode()
        state = {
            "task": "Test",
            "plan": [{"type": "write", "description": "Write report"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }
        assert router(state) == "writer"

    def test_routes_review(self):
        """Should route 'review' subtask to 'reviewer'."""
        router = RouterNode()
        state = {
            "task": "Test",
            "plan": [{"type": "review", "description": "Check quality"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }
        assert router(state) == "reviewer"

    def test_plan_exhausted(self):
        """Should route to aggregator when plan is complete."""
        router = RouterNode()
        state = {
            "task": "Test",
            "plan": [{"type": "research", "description": "Find"}],
            "current_step": 1,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }
        assert router(state) == "aggregator"

    def test_unknown_type(self):
        """Should route to aggregator for unknown subtask types."""
        router = RouterNode()
        state = {
            "task": "Test",
            "plan": [{"type": "unknown_xyz", "description": "???"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }
        assert router(state) == "aggregator"

    def test_fatal_error(self):
        """Should route to aggregator when there's a fatal error."""
        router = RouterNode()
        state = {
            "task": "Test",
            "plan": [{"type": "research", "description": "Find"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [{"step": "planner", "type": "Fatal", "detail": "API down"}],
            "next_action": "continue",
        }
        assert router(state) == "aggregator"


class TestExecutorNode:
    """Tests for ExecutorNode."""

    @patch("src.engine.nodes.executor.Agent")
    def test_execute_research_subtask(self, MockAgent):
        """Should execute a research subtask and advance step."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Research findings: ..."
        MockAgent.return_value = mock_agent

        mock_llm = MagicMock()
        executor = ExecutorNode(mock_llm)

        state = {
            "task": "Research AI",
            "plan": [{"type": "research", "description": "Find AI trends"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }

        result = executor(state)

        assert result["next_action"] == "continue"
        assert result["current_step"] == 1
        assert str(0) in result["results"]
        mock_agent.run.assert_called_once_with(
            task="Find AI trends", context="None"
        )

    @patch("src.engine.nodes.executor.Agent")
    def test_execute_preserves_previous_results(self, MockAgent):
        """Should append new results without losing existing ones."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Step 1 result"
        MockAgent.return_value = mock_agent

        mock_llm = MagicMock()
        executor = ExecutorNode(mock_llm)

        state = {
            "task": "Test",
            "plan": [
                {"type": "research", "description": "Step 0"},
                {"type": "write", "description": "Step 1"},
            ],
            "current_step": 1,
            "results": {"0": "Previous result"},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }

        result = executor(state)

        assert "0" in result["results"]
        assert "1" in result["results"]
        assert result["current_step"] == 2

    @patch("src.engine.nodes.executor.Agent")
    def test_execute_handles_error(self, MockAgent):
        """Should catch agent errors and record them."""
        mock_agent = MagicMock()
        mock_agent.run.side_effect = Exception("Tool not available")
        MockAgent.return_value = mock_agent

        mock_llm = MagicMock()
        executor = ExecutorNode(mock_llm)

        state = {
            "task": "Test",
            "plan": [{"type": "research", "description": "Find"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }

        result = executor(state)

        assert result["next_action"] == "retry"
        assert len(result["errors"]) == 1
        assert "Tool not available" in result["errors"][0]["detail"]
