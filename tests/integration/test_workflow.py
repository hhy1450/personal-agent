"""Integration test: full workflow with mocked LLM (MySQL)."""
import os
from unittest.mock import MagicMock

from src.engine.graph import run_workflow
from src.storage.database import (
    init_db, get_connection,
    create_task, update_task_status, get_task,
    create_workflow_run, update_workflow_run,
)
from src.storage.models import TaskStatus
from src.tools.sandbox import Sandbox, set_sandbox


def _use_test_db():
    """Switch to test database and clean tables."""
    os.environ["MYSQL_DATABASE"] = "personal_agent_test"
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM workflow_runs")
    conn.execute("DELETE FROM agent_configs")
    conn.execute("DELETE FROM tasks")
    conn.commit()
    conn.close()
    return conn  # unused, but consistent


def _make_response(content=None, tool_calls=None):
    """Helper to create a mock chat_completion response dict."""
    return {
        "choices": [{
            "message": {
                "content": content,
                "tool_calls": tool_calls or [],
            }
        }]
    }


class TestFullWorkflow:
    """End-to-end workflow tests with mocked LLM."""

    def test_workflow_with_mocked_llm(self, tmp_path):
        """Full workflow from task to result with mocked LLM responses."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"

        mock_llm.chat_completion.side_effect = [
            # 1. Planner
            _make_response(content='[{"type": "research", "description": "Search for AI info"}, {"type": "write", "description": "Write summary"}]'),
            # 2. Researcher tool call
            _make_response(content=None, tool_calls=[{
                "id": "call_1",
                "function": {"name": "web_search", "arguments": '{"query":"AI trends","max_results":2}'}
            }]),
            # 3. Researcher final
            _make_response(content="AI is advancing rapidly with LLMs and agents."),
            # 4. Reviewer after researcher
            _make_response(content="APPROVED"),
            # 5. Writer tool call
            _make_response(content=None, tool_calls=[{
                "id": "call_2",
                "function": {"name": "read_file", "arguments": '{"path":"results_0.md"}'}
            }]),
            # 6. Writer final
            _make_response(content="# AI Report\n\nAI is advancing rapidly with LLMs and agents."),
            # 7. Reviewer after writer
            _make_response(content="APPROVED"),
        ]

        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)

        try:
            result = run_workflow(mock_llm, "Tell me about AI")

            assert result is not None
            assert "final_output" in result
            assert "plan" in result
            assert len(result.get("plan", [])) == 2
            assert len(result.get("results", {})) == 2
            assert result["final_output"] != ""
            assert "AI" in result["final_output"]
        finally:
            set_sandbox(None)

    def test_workflow_empty_task(self):
        """Workflow with empty task should produce error output."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"

        result = run_workflow(mock_llm, "")

        assert result is not None
        assert "final_output" in result or "errors" in result
        assert len(result.get("plan", [])) == 0 or len(result.get("errors", [])) > 0

    def test_workflow_single_step_plan(self, tmp_path):
        """Workflow with a single-step plan should still complete."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"

        mock_llm.chat_completion.side_effect = [
            # Planner: single research task
            _make_response(content='[{"type": "research", "description": "Find info"}]'),
            # Researcher tool call
            _make_response(content=None, tool_calls=[{
                "id": "call_1",
                "function": {"name": "web_search", "arguments": '{"query":"test"}'}
            }]),
            # Researcher final
            _make_response(content="Here is the information."),
            # Reviewer
            _make_response(content="APPROVED"),
        ]

        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)

        try:
            result = run_workflow(mock_llm, "Find info about test")

            assert result is not None
            assert len(result.get("plan", [])) == 1
            assert len(result.get("results", {})) == 1
            assert result["final_output"] != ""
        finally:
            set_sandbox(None)

    def test_workflow_with_review_step(self, tmp_path):
        """Plan with a review-type step should not loop infinitely.

        Regression test: the graph used to route review-type steps to the
        quality gate instead of an executor, causing an infinite loop
        (recursion limit). Now review steps go through a dedicated
        review agent executor, then the quality gate.
        """
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"

        mock_llm.chat_completion.side_effect = [
            # Planner: research + review
            _make_response(content='[{"type": "research", "description": "Find info"}, {"type": "review", "description": "Review findings"}]'),
            # Researcher tool call
            _make_response(content=None, tool_calls=[{
                "id": "call_1",
                "function": {"name": "web_search", "arguments": '{"query":"test"}'}
            }]),
            # Researcher final
            _make_response(content="Research results here."),
            # Quality gate after researcher
            _make_response(content="APPROVED"),
            # Reviewer agent final (review-type step)
            _make_response(content="Review: the research looks good and complete."),
            # Quality gate after reviewer agent
            _make_response(content="APPROVED"),
        ]

        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)

        try:
            result = run_workflow(mock_llm, "Find and review info about test")

            assert result is not None
            assert len(result.get("plan", [])) == 2
            assert len(result.get("results", {})) == 2
            assert result["final_output"] != ""
        finally:
            set_sandbox(None)


class TestStorageIntegration:
    """Integration tests for storage layer (MySQL)."""

    def test_task_lifecycle(self):
        """Full task lifecycle: create -> update to running -> complete."""
        _use_test_db()

        task = create_task("Integration test", "Testing lifecycle")
        assert task.id is not None
        assert task.status == TaskStatus.PENDING

        # Update to running
        update_task_status(task.id, TaskStatus.RUNNING)
        t = get_task(task.id)
        assert t is not None
        assert t.status == TaskStatus.RUNNING

        # Complete
        update_task_status(task.id, TaskStatus.COMPLETED)
        t = get_task(task.id)
        assert t.status == TaskStatus.COMPLETED

    def test_workflow_run_record(self):
        """Workflow run records should be creatable and updatable."""
        _use_test_db()

        task = create_task("Workflow task", "Testing runs")
        run = create_workflow_run(task.id)
        assert run.id is not None
        assert run.task_id == task.id
        assert run.status == TaskStatus.RUNNING

        update_workflow_run(
            run.id,
            '{"plan": [{"type": "research", "description": "test"}]}',
            TaskStatus.COMPLETED,
        )

        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM workflow_runs WHERE id = %s", (run.id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["status"] == "completed"
        assert "plan" in row["state_json"]
