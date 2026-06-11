"""LangGraph StateGraph: wires nodes into the full workflow."""
import logging

from langgraph.graph import StateGraph, END

from src.engine.state import WorkflowState
from src.engine.nodes.planner import PlannerNode
from src.engine.nodes.router import RouterNode
from src.engine.nodes.executor import ExecutorNode
from src.engine.nodes.reviewer import ReviewerNode, aggregator_node
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def build_workflow_graph(llm_provider: LLMProvider):
    """Build the LangGraph StateGraph for the multi-agent workflow.

    Graph structure:
        START -> planner -> router -> [agent nodes] -> reviewer
                       ^                                  |
                       |            retry                  |
                       +----------------------------------+
                       |            continue               |
                       +-> aggregator -> END (when plan done/fatal)

    The reviewer owns step progression:
    - APPROVED  → current_step + 1, go back to router (next subtask or finish)
    - REJECTED  → current_step unchanged, go back to router (retry same step)
    - FINISH    → aggregator → END

    Args:
        llm_provider: The LLM backend to use.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    planner = PlannerNode(llm_provider)
    router = RouterNode()
    executor = ExecutorNode(llm_provider)
    reviewer = ReviewerNode(llm_provider)

    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("planner", planner)
    graph.add_node("router", lambda state: {})  # pure routing junction
    graph.add_node("researcher", executor)
    graph.add_node("writer", executor)
    graph.add_node("reviewer", executor)         # plan step executor for review tasks
    graph.add_node("quality_gate", reviewer)    # quality check after every step
    graph.add_node("aggregator", aggregator_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Planner -> Router
    graph.add_edge("planner", "router")

    # Conditional edges from router to agent nodes or aggregator
    graph.add_conditional_edges(
        "router",
        router,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",       # review-type plan step → executor
            "aggregator": "aggregator",
        },
    )

    # After each agent → quality gate
    graph.add_edge("researcher", "quality_gate")
    graph.add_edge("writer", "quality_gate")
    graph.add_edge("reviewer", "quality_gate")

    # Quality gate conditional: retry, next step, or finish
    def quality_gate_router(state: WorkflowState) -> str:
        """After review, decide: retry same step, continue to next, or finish."""
        next_action = state.get("next_action", "continue")
        if next_action == "finish":
            return "aggregator"
        # Both "retry" and "continue" go back to the router;
        # the router reads current_step to pick the correct agent.
        return "router"

    graph.add_conditional_edges(
        "quality_gate",
        quality_gate_router,
        {
            "router": "router",
            "aggregator": "aggregator",
        },
    )

    # Aggregator -> END
    graph.add_edge("aggregator", END)

    logger.info("LangGraph workflow graph compiled successfully")
    return graph.compile()


def run_workflow(llm_provider: LLMProvider, task: str, plan: list[dict] | None = None) -> dict:
    """Run a complete workflow for a given task using the compiled LangGraph.

    Args:
        llm_provider: The LLM backend to use.
        task: The user's natural language task description.
        plan: Optional pre-computed plan. When provided the planner node is
            still invoked (it may refine) but the plan is seeded as a starting
            point. This avoids redundant LLM calls when the caller has already
            generated a plan for display purposes.

    Returns:
        The final WorkflowState dict after graph execution.
    """
    logger.info("Starting workflow for task: %s", task[:80])

    graph = build_workflow_graph(llm_provider)

    initial_state: dict = {
        "task": task,
        "plan": plan or [],
        "current_step": 0,
        "results": {},
        "final_output": "",
        "errors": [],
        "next_action": "continue",
        "retry_count": 0,
    }

    result = graph.invoke(initial_state)

    plan_count = len(result.get("plan", []))
    result_count = len(result.get("results", {}))
    errors = result.get("errors", [])
    logger.info(
        "Workflow finished — %d/%d steps completed, %d error(s)",
        result_count, plan_count, len(errors),
    )

    return result
