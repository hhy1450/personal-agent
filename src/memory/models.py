"""Data models for the memory and context management subsystem."""
from dataclasses import dataclass, field


@dataclass
class HistoryStep:
    """A single step in the execution history.

    Stores the result and metadata for context construction
    and history trimming.
    """
    step_index: int
    agent_name: str
    description: str
    result: str
    approved: bool = True
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None or self.result.startswith("Error:")

    @property
    def summary_line(self) -> str:
        """One-line summary for context display."""
        status = "  " if self.approved else "  "
        err = f" (error: {self.error})" if self.error else ""
        return (
            f"[{status}] Step {self.step_index} [{self.agent_name}]: "
            f"{self.description[:80]}{err}"
        )


@dataclass
class ContextConfig:
    """Configuration for the dynamic context manager.

    Controls the two-layer compression pipeline:
    1. Token budget coarse trim (rule-based)
    2. LLM summary fine compression (semantic)
    """
    # Total token budget for the context window
    max_tokens: int = 8000

    # Number of recent steps to always preserve in full
    keep_recent_steps: int = 5

    # Token threshold above which LLM summarization is triggered
    summary_threshold: int = 6000

    # Target token count after summarization (leaves room for response)
    summary_target: int = 3000

    # Maximum characters to keep per old step after trimming
    max_chars_per_old_step: int = 200
