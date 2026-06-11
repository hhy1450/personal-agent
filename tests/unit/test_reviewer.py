"""Tests for reviewer node and aggregator."""
from unittest.mock import MagicMock, patch
from src.engine.nodes.reviewer import ReviewerNode, aggregator_node


class TestReviewerNode:
    """Tests for ReviewerNode."""

    @patch("src.engine.nodes.reviewer.Agent")
    def test_approves_good_result(self, MockAgent):
        """Should increment current_step and continue when reviewer approves."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "APPROVED — the output is good."
        MockAgent.return_value = mock_agent

        mock_llm = MagicMock()
        reviewer = ReviewerNode(mock_llm)

        state = {
            "task": "Research AI",
            "plan": [{"type": "research", "description": "Find trends"}],
            "current_step": 0,
            "results": {"0": "Great research findings about AI."},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
            "retry_count": 0,
        }

        result = reviewer(state)
        assert result["next_action"] == "finish"  # only 1 step, done after this
        assert result["current_step"] == 1

    @patch("src.engine.nodes.reviewer.Agent")
    def test_rejects_poor_result(self, MockAgent):
        """Should set next_action='retry' and keep current_step when reviewer rejects."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "REJECTED — missing key details about model architecture."
        MockAgent.return_value = mock_agent

        mock_llm = MagicMock()
        reviewer = ReviewerNode(mock_llm)

        state = {
            "task": "Research AI",
            "plan": [{"type": "research", "description": "Find trends"}],
            "current_step": 0,
            "results": {"0": "AI is cool."},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
            "retry_count": 0,
        }

        result = reviewer(state)
        assert result["next_action"] == "retry"
        assert result["current_step"] == 0  # stays on same step for retry

    def test_error_result_auto_reject(self):
        """Should auto-reject results that start with 'Error:'."""
        mock_llm = MagicMock()
        reviewer = ReviewerNode(mock_llm)

        state = {
            "task": "Test",
            "plan": [{"type": "research", "description": "Find"}],
            "current_step": 0,
            "results": {"0": "Error: API call failed"},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
            "retry_count": 0,
        }

        result = reviewer(state)
        assert result["next_action"] == "retry"
        assert result["current_step"] == 0  # stays on same step

    def test_no_results_skip_review(self):
        """Should return continue when no results to review."""
        mock_llm = MagicMock()
        reviewer = ReviewerNode(mock_llm)

        state = {
            "task": "Test",
            "plan": [{"type": "research", "description": "Find"}],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }

        result = reviewer(state)
        assert result["next_action"] == "continue"

    @patch("src.engine.nodes.reviewer.Agent")
    def test_reviewer_exception_is_lenient(self, MockAgent):
        """Should advance step if reviewer itself fails (lenient — don't block)."""
        mock_agent = MagicMock()
        mock_agent.run.side_effect = Exception("LLM timeout")
        MockAgent.return_value = mock_agent

        mock_llm = MagicMock()
        reviewer = ReviewerNode(mock_llm)

        state = {
            "task": "Test",
            "plan": [{"type": "research", "description": "Find"}],
            "current_step": 0,
            "results": {"0": "Research done."},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
            "retry_count": 0,
        }

        result = reviewer(state)
        assert result["next_action"] == "finish"  # only 1 step, so finish
        assert result["current_step"] == 1


class TestAggregator:
    """Tests for aggregator_node."""

    def test_aggregates_results(self):
        """Should combine all results into final_output."""
        state = {
            "task": "Research AI",
            "plan": [
                {"type": "research", "description": "Find trends"},
                {"type": "write", "description": "Write report"},
            ],
            "current_step": 2,  # both steps done
            "results": {"0": "AI trends found.", "1": "Report written."},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
            "retry_count": 0,
        }

        result = aggregator_node(state)
        assert "Research AI" in result["final_output"]
        assert "AI trends found" in result["final_output"]
        assert "Report written" in result["final_output"]
        assert result["next_action"] == "finish"

    def test_empty_results(self):
        """Should handle no results gracefully."""
        state = {
            "task": "Test",
            "plan": [],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }

        result = aggregator_node(state)
        assert "No results" in result["final_output"]
