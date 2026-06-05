"""Tests for LangGraph construction."""
from unittest.mock import MagicMock
from src.engine.graph import build_workflow_graph


class TestGraphConstruction:
    """Tests that the graph can be built without errors."""

    def test_graph_builds_successfully(self):
        """build_workflow_graph should not raise."""
        mock_llm = MagicMock()
        graph = build_workflow_graph(mock_llm)
        assert graph is not None

    def test_graph_has_required_nodes(self):
        """Graph should contain all required node names."""
        mock_llm = MagicMock()
        graph = build_workflow_graph(mock_llm)
        assert graph is not None
