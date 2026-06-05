"""Tests for storage layer."""
import pytest
from pathlib import Path

from src.storage.database import (
    init_db, get_connection,
    create_task, get_task, list_tasks, update_task_status,
    create_workflow_run, update_workflow_run, seed_agent_configs,
)
from src.storage.models import TaskStatus


@pytest.fixture
def test_db(tmp_path):
    """Create a test database."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return db_path


class TestTaskCRUD:
    """Tests for task CRUD operations."""

    def test_create_task(self, test_db):
        """create_task should persist a task and return it with ID."""
        task = create_task("Test task", "A description", db_path=test_db)
        assert task.id is not None
        assert task.title == "Test task"
        assert task.status == TaskStatus.PENDING

    def test_get_task(self, test_db):
        """get_task should retrieve by ID."""
        created = create_task("Find me", "desc", db_path=test_db)
        retrieved = get_task(created.id, db_path=test_db)
        assert retrieved is not None
        assert retrieved.title == "Find me"
        assert retrieved.description == "desc"

    def test_get_task_not_found(self, test_db):
        """get_task should return None for missing ID."""
        assert get_task(9999, db_path=test_db) is None

    def test_list_tasks(self, test_db):
        """list_tasks should return tasks ordered by created_at DESC."""
        create_task("Task A", db_path=test_db)
        create_task("Task B", db_path=test_db)
        tasks = list_tasks(db_path=test_db)
        assert len(tasks) == 2
        # Most recent first
        assert tasks[0].title >= tasks[1].title

    def test_update_task_status(self, test_db):
        """update_task_status should change status."""
        task = create_task("Test", db_path=test_db)
        update_task_status(task.id, TaskStatus.COMPLETED, db_path=test_db)
        updated = get_task(task.id, db_path=test_db)
        assert updated.status == TaskStatus.COMPLETED


class TestWorkflowRun:
    """Tests for workflow run operations."""

    def test_create_workflow_run(self, test_db):
        """create_workflow_run should create a run record."""
        task = create_task("Run me", db_path=test_db)
        run = create_workflow_run(task.id, db_path=test_db)
        assert run.id is not None
        assert run.task_id == task.id
        assert run.status == TaskStatus.RUNNING

    def test_update_workflow_run(self, test_db):
        """update_workflow_run should update state and status."""
        task = create_task("Run me", db_path=test_db)
        run = create_workflow_run(task.id, db_path=test_db)
        update_workflow_run(run.id, '{"key": "value"}', TaskStatus.COMPLETED, db_path=test_db)

        # Verify via direct query
        conn = get_connection(test_db)
        row = conn.execute("SELECT * FROM workflow_runs WHERE id = ?", (run.id,)).fetchone()
        conn.close()
        assert row["status"] == "completed"
        assert row["state_json"] == '{"key": "value"}'


class TestSeedData:
    """Tests for seed_agent_configs."""

    def test_seed_agent_configs(self, test_db):
        """seed_agent_configs should insert default configs once."""
        seed_agent_configs(db_path=test_db)

        conn = get_connection(test_db)
        rows = conn.execute("SELECT * FROM agent_configs").fetchall()
        conn.close()
        names = [r["name"] for r in rows]
        assert "planner" in names
        assert "researcher" in names
        assert "writer" in names
        assert "reviewer" in names

    def test_seed_is_idempotent(self, test_db):
        """seed_agent_configs should not duplicate on second call."""
        seed_agent_configs(db_path=test_db)
        seed_agent_configs(db_path=test_db)
        conn = get_connection(test_db)
        count = conn.execute("SELECT COUNT(*) FROM agent_configs").fetchone()[0]
        conn.close()
        assert count == 4
