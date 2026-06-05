"""SQLite database operations."""
import sqlite3
import json
from pathlib import Path
from typing import Optional

from src.storage.models import Task, TaskStatus, WorkflowRun


DB_PATH = Path(__file__).parent.parent.parent / "personal_agent.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Initialize database schema."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workflow_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            state_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS agent_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            system_prompt TEXT NOT NULL,
            tools_json TEXT NOT NULL DEFAULT '[]',
            model TEXT NOT NULL DEFAULT 'deepseek-chat'
        );

        CREATE TABLE IF NOT EXISTS triggers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            cron_expr TEXT NOT NULL,
            task_template TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            last_run_at TEXT
        );
    """)

    conn.commit()
    conn.close()


# ---- Task CRUD ----

def create_task(title: str, description: str = "", db_path: Path | None = None) -> Task:
    """Create a new task and return it with ID."""
    conn = get_connection(db_path)
    now = Task().created_at  # Use current time
    cursor = conn.execute(
        "INSERT INTO tasks (title, description, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (title, description, TaskStatus.PENDING.value, now, now),
    )
    conn.commit()
    task = Task(
        id=cursor.lastrowid,
        title=title,
        description=description,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    conn.close()
    return task


def get_task(task_id: int, db_path: Path | None = None) -> Optional[Task]:
    """Get a task by ID."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return Task.from_row(tuple(row))


def list_tasks(limit: int = 20, offset: int = 0, db_path: Path | None = None) -> list[Task]:
    """List recent tasks."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [Task.from_row(tuple(r)) for r in rows]


def update_task_status(task_id: int, status: TaskStatus, db_path: Path | None = None) -> None:
    """Update task status."""
    conn = get_connection(db_path)
    now = Task().created_at
    conn.execute(
        "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
        (status.value, now, task_id),
    )
    conn.commit()
    conn.close()


# ---- WorkflowRun CRUD ----

def create_workflow_run(task_id: int, db_path: Path | None = None) -> WorkflowRun:
    """Create a workflow run record."""
    conn = get_connection(db_path)
    now = WorkflowRun().started_at
    cursor = conn.execute(
        "INSERT INTO workflow_runs (task_id, state_json, started_at, status) VALUES (?, ?, ?, ?)",
        (task_id, "{}", now, TaskStatus.RUNNING.value),
    )
    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return WorkflowRun(id=run_id, task_id=task_id, started_at=now, status=TaskStatus.RUNNING)


def update_workflow_run(run_id: int, state_json: str, status: TaskStatus, db_path: Path | None = None) -> None:
    """Update workflow run with current state and status."""
    conn = get_connection(db_path)
    now = Task().created_at
    conn.execute(
        "UPDATE workflow_runs SET state_json = ?, status = ?, finished_at = ? WHERE id = ?",
        (state_json, status.value, now, run_id),
    )
    conn.commit()
    conn.close()


# ---- Seed Data ----

def seed_agent_configs(db_path: Path | None = None) -> None:
    """Insert default agent configurations if not present."""
    conn = get_connection(db_path)
    existing = conn.execute("SELECT COUNT(*) FROM agent_configs").fetchone()[0]
    if existing > 0:
        conn.close()
        return

    configs = [
        ("planner", "You are a task planning expert. Break down the user's request into a sequence of clear, executable subtasks. Each subtask should have a type (research/write/review) and a description. Output ONLY valid JSON: [{\"type\": \"...\", \"description\": \"...\"}]"),
        ("researcher", "You are a research agent. Search for information and return accurate, well-organized findings. Cite your sources."),
        ("writer", "You are a writing agent. Create well-structured documents based on research findings. Use clear language and proper formatting."),
        ("reviewer", "You are a quality reviewer. Check if the output meets the user's requirements. Be critical but constructive. Reply with 'APPROVED' if the work is good, or explain what needs improvement."),
    ]

    for name, prompt in configs:
        conn.execute(
            "INSERT INTO agent_configs (name, system_prompt, tools_json) VALUES (?, ?, ?)",
            (name, prompt, "[]"),
        )
    conn.commit()
    conn.close()
