"""Router node: determines execution path based on strategy and step type."""
import logging

from src.engine.state import WorkflowState

logger = logging.getLogger(__name__)

# Maps subtask type to agent name
TYPE_TO_AGENT = {
    "research": "researcher",
    "write": "writer",
    "review": "reviewer",
}


class RouterNode:
    """LangGraph conditional edge function.

    Routes based on:
    1. Strategy: "sequential" | "parallel" | "loop"
    2. For parallel: delegate to FanOut if batch steps exist
    3. For sequential/loop: route by subtask type
    """

    def __call__(self, state: WorkflowState) -> str:
        """Determine next node based on current state.

        Returns:
            Node name: "fanout", "researcher", "writer", "reviewer",
            or "aggregator" if plan exhausted or fatal error.
        """
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        strategy = state.get("strategy", "sequential")

        # Check if we're done
        if current_step >= len(plan):
            logger.debug("Router: plan exhausted → aggregator")
            return "aggregator"

        # Check for fatal errors
        errors = state.get("errors", [])
        if any(e.get("type") == "Fatal" for e in errors):
            logger.debug("Router: fatal error → aggregator")
            return "aggregator"

        # Parallel strategy: route to fanout for group coordination
        if strategy == "parallel":
            subtask = plan[current_step]
            if subtask.get("group"):
                logger.debug("Router: parallel strategy, group=%s → fanout", subtask.get("group"))
                return "fanout"
            # Fall through to normal type-based routing for non-grouped steps

        # Normal type-based routing
        subtask = plan[current_step]
        subtask_type = subtask.get("type", "").lower()
        target = TYPE_TO_AGENT.get(subtask_type, "aggregator")
        logger.debug("Router: step %d type=%s → %s", current_step, subtask_type, target)
        return target


class StrategyRouterNode:
    """Initial router after Planner: chooses the top-level strategy path.

    This runs once at the start of execution to set up the graph topology
    for the chosen orchestration strategy.

    Returns a node name for the first execution step.
    """

    def __call__(self, state: WorkflowState) -> str:
        """Route to the appropriate first node based on strategy.

        For all strategies, go through the main Router which handles
        type-based routing. The strategy affects how quality gate results
        are interpreted (loop → retry until condition met).
        """
        strategy = state.get("strategy", "sequential")
        logger.info("StrategyRouter: strategy=%s", strategy)
        # All strategies start through the main router
        return "router"
