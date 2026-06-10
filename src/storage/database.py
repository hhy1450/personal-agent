"""MySQL database operations."""
import pymysql

from src.config.settings import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER,
    MYSQL_PASSWORD, MYSQL_DATABASE,
)
from src.storage.models import Task, TaskStatus, WorkflowRun


def _current_database():
    """Allow tests to override the database name via env var."""
    import os
    return os.getenv("MYSQL_DATABASE", MYSQL_DATABASE)


class _Connection:
    """Thin wrapper around pymysql.Connection that exposes sqlite3-style .execute()."""

    def __init__(self, raw):
        self._raw = raw

    def execute(self, sql, params=()):
        cursor = self._raw.cursor()
        cursor.execute(sql, params)
        cursor._conn = self._raw  # keep ref so caller can commit
        return cursor

    def commit(self):
        self._raw.commit()

    def close(self):
        self._raw.close()


def get_connection() -> _Connection:
    """Get a MySQL connection wrapper with .execute() / .commit() / .close()."""
    raw = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=_current_database(),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    return _Connection(raw)


# ---- Schema ----

SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        title VARCHAR(500) NOT NULL,
        description TEXT NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    CREATE TABLE IF NOT EXISTS workflow_runs (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        task_id INTEGER NOT NULL,
        state_json LONGTEXT NOT NULL,
        started_at DATETIME NOT NULL,
        finished_at DATETIME,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    CREATE TABLE IF NOT EXISTS agent_configs (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100) NOT NULL UNIQUE,
        system_prompt TEXT NOT NULL,
        tools_json TEXT NOT NULL,
        model VARCHAR(50) NOT NULL DEFAULT 'deepseek-chat'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    CREATE TABLE IF NOT EXISTS triggers (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(200) NOT NULL,
        cron_expr VARCHAR(100) NOT NULL,
        task_template TEXT NOT NULL,
        enabled TINYINT NOT NULL DEFAULT 1,
        last_run_at DATETIME
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def init_db() -> None:
    """Initialise database schema (idempotent — safe to run every startup)."""
    conn = get_connection()
    for stmt in SCHEMA_SQL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    conn.close()


# ---- Task CRUD ----

def create_task(title: str, description: str = "") -> Task:
    conn = get_connection()
    now = Task().created_at
    cursor = conn.execute(
        "INSERT INTO tasks (title, description, status, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)",
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


def get_task(task_id: int) -> Task | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM tasks WHERE id = %s", (task_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return Task.from_row(tuple(row.values()))


def list_tasks(limit: int = 20, offset: int = 0) -> list[Task]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM tasks ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [Task.from_row(tuple(r.values())) for r in rows]


def update_task_status(task_id: int, status: TaskStatus) -> None:
    conn = get_connection()
    now = Task().created_at
    conn.execute(
        "UPDATE tasks SET status = %s, updated_at = %s WHERE id = %s",
        (status.value, now, task_id),
    )
    conn.commit()
    conn.close()


# ---- WorkflowRun CRUD ----

def create_workflow_run(task_id: int) -> WorkflowRun:
    conn = get_connection()
    now = WorkflowRun().started_at
    cursor = conn.execute(
        "INSERT INTO workflow_runs (task_id, state_json, started_at, status) VALUES (%s, %s, %s, %s)",
        (task_id, "{}", now, TaskStatus.RUNNING.value),
    )
    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return WorkflowRun(id=run_id, task_id=task_id, started_at=now, status=TaskStatus.RUNNING)


def update_workflow_run(run_id: int, state_json: str, status: TaskStatus) -> None:
    conn = get_connection()
    now = Task().created_at
    conn.execute(
        "UPDATE workflow_runs SET state_json = %s, status = %s, finished_at = %s WHERE id = %s",
        (state_json, status.value, now, run_id),
    )
    conn.commit()
    conn.close()


# ---- Seed Data ----

def seed_agent_configs() -> None:
    """Insert default agent configurations if not present (idempotent)."""
    conn = get_connection()
    row = conn.execute("SELECT COUNT(*) as cnt FROM agent_configs").fetchone()
    if row["cnt"] > 0:
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
            "INSERT INTO agent_configs (name, system_prompt, tools_json) VALUES (%s, %s, %s)",
            (name, prompt, "[]"),
        )
    conn.commit()
    conn.close()
