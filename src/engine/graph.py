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
    graph.add_node("router_conditional", lambda state: {})
    graph.add_node("researcher", executor)
    graph.add_node("writer", executor)
    graph.add_node("reviewer_node", reviewer)
    graph.add_node("aggregator", aggregator_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Planner -> Router
    graph.add_edge("planner", "router_conditional")

    # Conditional edges from router
    graph.add_conditional_edges(
        "router_conditional",
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

    # Reviewer conditional: retry or continue
    def reviewer_router(state: WorkflowState) -> str:
        """After review, decide: retry or move forward."""
        next_action = state.get("next_action", "continue")
        if next_action == "retry":
            return "router_conditional"
        else:
            return "router_conditional"

    graph.add_conditional_edges(
        "reviewer_node",
        reviewer_router,
        {
            "router_conditional": "router_conditional",
        },
    )

    # Aggregator -> END
    graph.add_edge("aggregator", END)

    return graph.compile()


def run_workflow(llm_provider: LLMProvider, task: str) -> dict:
    """Run a complete workflow for a given task.

    Args:
        llm_provider: The LLM backend to use.
        task: The user's task description.

    Returns:
        The final WorkflowState as a dict.
    """
    graph = build_workflow_graph(llm_provider)

    initial_state: WorkflowState = {
        "task": task,
        "plan": [],
        "current_step": 0,
        "results": {},
        "final_output": "",
        "errors": [],
        "next_action": "continue",
    }

    final_state = graph.invoke(initial_state)
    return final_state
