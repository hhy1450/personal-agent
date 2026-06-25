"""Event models for the workflow event system."""
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field


class EventType(str, Enum):
    """Types of events emitted during workflow execution."""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_DONE = "workflow_done"
    STEP_START = "step_start"
    STEP_DONE = "step_done"
    STEP_REVIEW = "step_review"
    STEP_RETRY = "step_retry"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


@dataclass
class WorkflowEvent:
    """An event emitted during workflow execution.

    This is the message format sent over WebSocket to clients.
    """
    type: str                        # EventType value
    task_id: int | None = None       # DB task ID (if available)
    step: int | None = None          # Current step index
    agent: str | None = None         # Agent name (researcher/writer/reviewer)
    description: str | None = None   # Human-readable description
    data: dict | None = None         # Additional payload
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict for WebSocket transmission."""
        d = {
            "type": self.type,
            "timestamp": self.timestamp,
        }
        if self.task_id is not None:
            d["task_id"] = self.task_id
        if self.step is not None:
            d["step"] = self.step
        if self.agent is not None:
            d["agent"] = self.agent
        if self.description is not None:
            d["description"] = self.description
        if self.data is not None:
            d["data"] = self.data
        return d
