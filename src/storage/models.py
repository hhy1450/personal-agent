"""Data models for storage layer."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a user-submitted task."""
    id: int | None = None
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "Task":
        return cls(
            id=row[0],
            title=row[1],
            description=row[2],
            status=TaskStatus(row[3]),
            created_at=row[4],
            updated_at=row[5],
        )


@dataclass
class WorkflowRun:
    """Records a single workflow execution."""
    id: int | None = None
    task_id: int = 0
    state_json: str = "{}"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str | None = None
    status: TaskStatus = TaskStatus.PENDING
