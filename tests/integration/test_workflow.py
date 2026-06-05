"""Integration test: full workflow with mocked LLM."""
from unittest.mock import MagicMock

from src.engine.graph import run_workflow
from src.storage.database import init_db, create_task, update_task_status, get_task
from src.storage.models import TaskStatus
from src.tools.sandbox import Sandbox, set_sandbox


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
        """Full workflow from task to result with mocked LLM responses.

        Verifies the complete graph executes: planner -> router -> researcher
        -> reviewer -> router -> writer -> reviewer -> aggregator.
        """
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"

        # The graph calls chat_completion in this order for a 2-step plan
        # (research + write), with reviewer after each step:
        #   1. Planner: decompose task into JSON plan
        #   2. Researcher Agent: request tool call (web_search)
        #   3. Researcher Agent: final text response
        #   4. Reviewer Agent: review researcher output -> APPROVED
        #   5. Writer Agent: request tool call (read_file)
        #   6. Writer Agent: final text response
        #   7. Reviewer Agent: review writer output -> APPROVED
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

        # Set up isolated sandbox under tmp_path
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
            # Reset global sandbox to avoid leaking into other tests
            set_sandbox(None)

    def test_workflow_empty_task(self):
        """Workflow with empty task should produce error output."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"

        result = run_workflow(mock_llm, "")

        assert result is not None
        assert "final_output" in result or "errors" in result
        # Should either have errors or an empty plan
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

        # Set up isolated sandbox
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


class TestStorageIntegration:
    """Integration tests for storage layer."""

    def test_task_lifecycle(self, tmp_path):
        """Full task lifecycle: create -> update to running -> complete."""
        import src.storage.database as db_module

        db_path = tmp_path / "test.db"
        original = db_module.DB_PATH
        db_module.DB_PATH = db_path
        init_db(db_path)

        try:
            # Create
            task = create_task("Integration test", "Testing lifecycle", db_path=db_path)
            assert task.id is not None
            assert task.status == TaskStatus.PENDING

            # Update to running
            update_task_status(task.id, TaskStatus.RUNNING, db_path=db_path)
            t = get_task(task.id, db_path=db_path)
            assert t is not None
            assert t.status == TaskStatus.RUNNING

            # Complete
            update_task_status(task.id, TaskStatus.COMPLETED, db_path=db_path)
            t = get_task(task.id, db_path=db_path)
            assert t.status == TaskStatus.COMPLETED
        finally:
            db_module.DB_PATH = original

    def test_workflow_run_record(self, tmp_path):
        """Workflow run records should be creatable and updatable."""
        import src.storage.database as db_module
        from src.storage.database import create_workflow_run, update_workflow_run

        db_path = tmp_path / "test.db"
        original = db_module.DB_PATH
        db_module.DB_PATH = db_path
        init_db(db_path)

        try:
            task = create_task("Workflow task", "Testing runs", db_path=db_path)
            run = create_workflow_run(task.id, db_path=db_path)
            assert run.id is not None
            assert run.task_id == task.id
            assert run.status == TaskStatus.RUNNING

            update_workflow_run(
                run.id,
                '{"plan": [{"type": "research", "description": "test"}]}',
                TaskStatus.COMPLETED,
                db_path=db_path,
            )

            # Verify via direct query
            from src.storage.database import get_connection
            conn = get_connection(db_path)
            row = conn.execute(
                "SELECT * FROM workflow_runs WHERE id = ?", (run.id,)
            ).fetchone()
            conn.close()
            assert row is not None
            assert row["status"] == "completed"
            assert "plan" in row["state_json"]
        finally:
            db_module.DB_PATH = original
