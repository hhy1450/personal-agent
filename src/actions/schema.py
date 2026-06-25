"""Structured action definitions — JSON Schema-constrained LLM outputs."""
from enum import Enum
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Well-known action types that agents can emit.

    These map LLM decisions to concrete, executable operations.
    """
    SEARCH = "search"                # Web search
    READ_FILE = "read_file"          # Read file from workspace
    WRITE_FILE = "write_file"        # Write file to workspace
    ANALYZE_IMAGE = "analyze_image"  # Analyze an image (requires vision)
    SUMMARIZE = "summarize"          # Summarize text content
    GENERATE = "generate"            # Generate new content
    REVIEW = "review"                # Review/check existing content


class Action(BaseModel):
    """A structured action emitted by an agent.

    Each action has a well-defined type and parameters, making
    agent outputs predictable and auditable.

    Example:
        Action(
            action=ActionType.SEARCH,
            params={"query": "DeepSeek V4 features", "max_results": 5},
            reason="Need to find latest information about the topic",
        )
    """
    action: ActionType = Field(description="The type of action to execute")
    params: dict = Field(
        default_factory=dict,
        description="Parameters specific to this action type",
    )
    reason: str = Field(
        default="",
        description="Why this action is being taken (for audit trail)",
    )

    def to_tool_call(self) -> tuple[str, dict]:
        """Convert to a tool function name + arguments tuple.

        Returns:
            (tool_name, args_dict) suitable for tool execution.
        """
        tool_map = {
            ActionType.SEARCH: "web_search",
            ActionType.READ_FILE: "read_file",
            ActionType.WRITE_FILE: "write_file",
            ActionType.ANALYZE_IMAGE: "analyze_image",
            ActionType.SUMMARIZE: "summarize",
            ActionType.GENERATE: "generate",
            ActionType.REVIEW: "review",
        }
        tool_name = tool_map.get(self.action, self.action.value)
        return tool_name, self.params
