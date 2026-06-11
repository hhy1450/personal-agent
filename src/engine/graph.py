"""LangGraph StateGraph: wires nodes into the full workflow."""
from langgraph.graph import StateGraph, END

from src.engine.state import WorkflowState
from src.engine.nodes.planner import PlannerNode
from src.engine.nodes.router import RouterNode
from src.engine.nodes.executor import ExecutorNode
from src.engine.nodes.reviewer import ReviewerNode, aggregator_node
from src.llm.base import LLMProvider


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
    graph.add_node("reviewer_node", reviewer)
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
            "reviewer": "reviewer_node",
            "aggregator": "aggregator",
        },
    )

    # After each agent -> reviewer
    graph.add_edge("researcher", "reviewer_node")
    graph.add_edge("writer", "reviewer_node")

    # Reviewer conditional: retry, next step, or finish
    def reviewer_router(state: WorkflowState) -> str:
        """After review, decide: retry same step, continue to next, or finish."""
        next_action = state.get("next_action", "continue")
        if next_action == "finish":
            return "aggregator"
        # Both "retry" and "continue" go back to the router;
        # the router reads current_step to pick the correct agent.
        return "router"

    graph.add_conditional_edges(
        "reviewer_node",
        reviewer_router,
        {
            "router": "router",
            "aggregator": "aggregator",
        },
    )

    # Aggregator -> END
    graph.add_edge("aggregator", END)

    return graph.compile()


def run_workflow(llm_provider: LLMProvider, task: str) -> dict:
    """Run a complete workflow for a given task using the compiled LangGraph.

    Args:
        llm_provider: The LLM backend to use.
        task: The user's natural language task description.

    Returns:
        The final WorkflowState dict after graph execution.
    """
    graph = build_workflow_graph(llm_provider)

    initial_state: dict = {
        "task": task,
        "plan": [],
        "current_step": 0,
        "results": {},
        "final_output": "",
        "errors": [],
        "next_action": "continue",
        "retry_count": 0,
    }

    result = graph.invoke(initial_state)
    return result
