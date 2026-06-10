"""Tests for storage layer (MySQL)."""
import os
import pytest

from src.storage.database import (
    init_db, get_connection,
    create_task, get_task, list_tasks, update_task_status,
    create_workflow_run, update_workflow_run, seed_agent_configs,
)
from src.storage.models import TaskStatus


@pytest.fixture
def test_db():
    """Switch to test database and ensure clean tables before each test."""
    original_db = os.environ.get("MYSQL_DATABASE", "personal_agent")
    os.environ["MYSQL_DATABASE"] = "personal_agent_test"
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM workflow_runs")
    conn.execute("DELETE FROM agent_configs")
    conn.execute("DELETE FROM tasks")
    conn.commit()
    conn.close()
    yield
    # Cleanup and restore
    conn = get_connection()
    conn.execute("DELETE FROM workflow_runs")
    conn.execute("DELETE FROM agent_configs")
    conn.execute("DELETE FROM tasks")
    conn.commit()
    conn.close()
    os.environ["MYSQL_DATABASE"] = original_db


class TestTaskCRUD:
    """Tests for task CRUD operations."""

    def test_create_task(self, test_db):
        """create_task should persist a task and return it with ID."""
        task = create_task("Test task", "A description")
        assert task.id is not None
        assert task.title == "Test task"
        assert task.status == TaskStatus.PENDING

    def test_get_task(self, test_db):
        """get_task should retrieve by ID."""
        created = create_task("Find me", "desc")
        retrieved = get_task(created.id)
        assert retrieved is not None
        assert retrieved.title == "Find me"
        assert retrieved.description == "desc"

    def test_get_task_not_found(self, test_db):
        """get_task should return None for missing ID."""
        assert get_task(9999) is None

    def test_list_tasks(self, test_db):
        """list_tasks should return tasks ordered by created_at DESC."""
        create_task("Task A")
        create_task("Task B")
        tasks = list_tasks()
        assert len(tasks) == 2
        titles = {t.title for t in tasks}
        assert "Task A" in titles
        assert "Task B" in titles

    def test_update_task_status(self, test_db):
        """update_task_status should change status."""
        task = create_task("Test")
        update_task_status(task.id, TaskStatus.COMPLETED)
        updated = get_task(task.id)
        assert updated.status == TaskStatus.COMPLETED


class TestWorkflowRun:
    """Tests for workflow run operations."""

    def test_create_workflow_run(self, test_db):
        """create_workflow_run should create a run record."""
        task = create_task("Run me")
        run = create_workflow_run(task.id)
        assert run.id is not None
        assert run.task_id == task.id
        assert run.status == TaskStatus.RUNNING

    def test_update_workflow_run(self, test_db):
        """update_workflow_run should update state and status."""
        task = create_task("Run me")
        run = create_workflow_run(task.id)
        update_workflow_run(run.id, '{"key": "value"}', TaskStatus.COMPLETED)

        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM workflow_runs WHERE id = %s", (run.id,)
        ).fetchone()
        conn.close()
        assert row["status"] == "completed"
        assert row["state_json"] == '{"key": "value"}'


class TestSeedData:
    """Tests for seed_agent_configs."""

    def test_seed_agent_configs(self, test_db):
        """seed_agent_configs should insert default configs once."""
        seed_agent_configs()

        conn = get_connection()
        rows = conn.execute("SELECT * FROM agent_configs").fetchall()
        conn.close()
        names = [r["name"] for r in rows]
        assert "planner" in names
        assert "researcher" in names
        assert "writer" in names
        assert "reviewer" in names

    def test_seed_is_idempotent(self, test_db):
        """seed_agent_configs should not duplicate on second call."""
        seed_agent_configs()
        seed_agent_configs()
        conn = get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM agent_configs").fetchone()
        conn.close()
        assert row["cnt"] == 4
