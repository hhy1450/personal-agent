"""WorkflowState — the data that flows through LangGraph nodes."""
from typing import TypedDict


class SubTask(TypedDict):
    """A single subtask in the execution plan."""
    type: str          # "research" | "write" | "review"
    description: str   # What to do


class WorkflowState(TypedDict):
    """State that flows between nodes in the LangGraph workflow."""
    task: str                       # User's original request
    plan: list[SubTask]             # Planner output: ordered subtasks
    current_step: int               # Index into plan (which subtask we're on)
    results: dict                   # Map subtask index -> result string
    final_output: str               # Aggregated final result
    errors: list[dict]              # Error log: [{step, type, detail}]
    next_action: str                # Router decision: "continue" | "finish" | "retry"
    retry_count: int                # Current step retry counter (max 3)
    strategy: str                   # Orchestration strategy: "sequential" | "parallel" | "loop"
    context_summary: str            # Compressed summary from ContextManager (for long tasks)
    history: list[dict]             # Structured execution history: [{step, agent, desc, result, approved}]
