"""Router node: determines which agent should handle the current subtask."""
from src.engine.state import WorkflowState

# Maps subtask type to agent name
TYPE_TO_AGENT = {
    "research": "researcher",
    "write": "writer",
    "review": "reviewer",
}


class RouterNode:
    """LangGraph conditional edge function.

    Reads the current subtask type and routes to the appropriate agent.
    Returns the next node name as a string.
    """

    def __call__(self, state: WorkflowState) -> str:
        """Determine next agent based on current subtask type.

        Args:
            state: Current workflow state.

        Returns:
            Node name string: "researcher", "writer", "reviewer",
            or "aggregator" if plan is exhausted or errored.
        """
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        # Check if we're done
        if current_step >= len(plan):
            return "aggregator"

        # Check for fatal errors
        errors = state.get("errors", [])
        if any(e.get("type") == "Fatal" for e in errors):
            return "aggregator"

        subtask = plan[current_step]
        subtask_type = subtask.get("type", "").lower()
        return TYPE_TO_AGENT.get(subtask_type, "aggregator")
