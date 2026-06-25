"""Tests for planner node."""
from unittest.mock import MagicMock
from src.engine.nodes.planner import PlannerNode


class TestPlannerNode:
    """Tests for PlannerNode."""

    def test_valid_plan_json(self):
        """Should parse valid JSON array from LLM response."""
        mock_llm = MagicMock()
        mock_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": '[{"type": "research", "description": "Search for info"}, {"type": "write", "description": "Write report"}]'}}]
        }

        planner = PlannerNode(mock_llm)
        result = planner({"task": "Research AI trends"})

        assert len(result["plan"]) == 2
        assert result["plan"][0]["type"] == "research"
        assert result["plan"][1]["type"] == "write"
        assert result["current_step"] == 0
        assert result["next_action"] == "continue"

    def test_plan_in_code_block(self):
        """Should extract JSON from markdown code blocks."""
        mock_llm = MagicMock()
        mock_llm.chat_completion.return_value = {
            "choices": [{"message": {"content": 'Here is the plan:\n```json\n[{"type": "research", "description": "Find data"}]\n```\nLet me know if this works.'}}]
        }

        planner = PlannerNode(mock_llm)
        result = planner({"task": "Test"})

        assert len(result["plan"]) == 1
        assert result["plan"][0]["type"] == "research"

    def test_empty_task(self):
        """Should handle empty task gracefully."""
        mock_llm = MagicMock()
        planner = PlannerNode(mock_llm)
        result = planner({"task": ""})

        assert result["plan"] == []
        assert len(result["errors"]) == 1
        assert result["next_action"] == "finish"

    def test_llm_error(self):
        """Should catch LLM errors and return error state."""
        mock_llm = MagicMock()
        mock_llm.chat_completion.side_effect = Exception("API error")

        planner = PlannerNode(mock_llm)
        result = planner({"task": "Test"})

        assert result["plan"] == []
        assert result["errors"][0]["type"] == "Fatal"
        assert "API error" in result["errors"][0]["detail"]

    def test_invalid_json(self):
        """_parse_plan should return empty steps for invalid JSON."""
        planner = PlannerNode(MagicMock())
        result = planner._parse_plan("This is not JSON at all")
        assert result == {"strategy": "sequential", "steps": []}

    def test_partial_valid_json(self):
        """_parse_plan should filter out items without type."""
        planner = PlannerNode(MagicMock())
        result = planner._parse_plan('[{"type": "research"}, {"other": "value"}]')
        # Both items missing type field
        assert result == {"strategy": "sequential", "steps": []}
