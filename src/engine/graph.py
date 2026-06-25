"""LangGraph StateGraph: wires nodes into the full workflow.

Supports three orchestration strategies:
- sequential: steps execute one after another (v0.1 behavior)
- parallel: independent steps in same group run consecutively,
  then pass through quality gate as a batch
- loop: step repeats until stop condition or max iterations
"""
import logging

from langgraph.graph import StateGraph, END

from src.engine.state import WorkflowState
from src.engine.nodes.planner import PlannerNode
from src.engine.nodes.router import RouterNode, StrategyRouterNode
from src.engine.nodes.executor import ExecutorNode
from src.engine.nodes.reviewer import ReviewerNode, aggregator_node
from src.engine.nodes.fanout import FanOutNode, FanInNode
from src.llm.base import LLMProvider
from src.llm.factory import LLMFactory

logger = logging.getLogger(__name__)


def build_workflow_graph(llm_provider: LLMProvider | LLMFactory):
    """Build the LangGraph StateGraph with multi-strategy support.

    Graph structure (v0.2):
        START → planner → strategy_router
                              ↓
                     ┌─── router ────────────────────────┐
                     │   ├─ fanout (parallel batches)     │
                     │   ├─ researcher / writer / reviewer│
                     │   └─ aggregator (plan exhausted)   │
                     └──────────────┬────────────────────┘
                                    ↓
                              quality_gate
                           ├─ retry → router
                           ├─ continue → router
                           └─ finish → aggregator → END

    Args:
        llm_provider: LLM backend or LLMFactory for multi-provider support.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    # Resolve providers
    if isinstance(llm_provider, LLMFactory):
        text_provider = llm_provider.text_provider
    else:
        text_provider = llm_provider

    planner = PlannerNode(text_provider)
    strategy_router = StrategyRouterNode()
    router = RouterNode()
    executor = ExecutorNode(llm_provider)
    reviewer = ReviewerNode(text_provider)
    fanout = FanOutNode()
    fanin = FanInNode()

    graph = StateGraph(WorkflowState)

    # ---- Nodes ----
    graph.add_node("planner", planner)
    graph.add_node("strategy_router", lambda state: {})  # pure junction
    graph.add_node("router", lambda state: {})
    graph.add_node("fanout", fanout)                    # determine parallel batch
    graph.add_node("fanin", fanin)                      # collect parallel results
    graph.add_node("researcher", executor)
    graph.add_node("writer", executor)
    graph.add_node("reviewer", executor)
    graph.add_node("quality_gate", reviewer)
    graph.add_node("aggregator", aggregator_node)

    # ---- Entry ----
    graph.set_entry_point("planner")

    # ---- Edges ----
    graph.add_edge("planner", "strategy_router")
    graph.add_edge("strategy_router", "router")

    # Router: conditional dispatch
    graph.add_conditional_edges(
        "router",
        router,
        {
            "fanout": "fanout",
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "aggregator": "aggregator",
        },
    )

    # FanOut → route to correct executor based on step type
    def fanout_to_executor(state: WorkflowState) -> str:
        """After FanOut, route to the correct agent for the current step."""
        plan = state.get("plan", [])
        step = state.get("current_step", 0)
        if step >= len(plan):
            return "aggregator"
        agent_type = TYPE_TO_AGENT.get(plan[step].get("type", "").lower(), "aggregator")
        return agent_type

    graph.add_conditional_edges(
        "fanout",
        fanout_to_executor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "aggregator": "aggregator",
        },
    )

    # After each agent → quality gate
    graph.add_edge("researcher", "quality_gate")
    graph.add_edge("writer", "quality_gate")
    graph.add_edge("reviewer", "quality_gate")

    # ---- Quality Gate: strategy-aware routing ----
    def quality_gate_router(state: WorkflowState) -> str:
        """After review, decide: retry, next step, finish, or loop continuation.

        Strategy-aware:
        - sequential: retry/continue/finish
        - parallel: batch progression (fanin) or finish
        - loop: retry until stop condition or max iterations
        """
        next_action = state.get("next_action", "continue")
        strategy = state.get("strategy", "sequential")

        if next_action == "finish":
            return "aggregator"

        # Loop strategy: retry → go back to executor for same step
        if strategy == "loop" and next_action == "retry":
            loop_count = state.get("retry_count", 0)
            plan = state.get("plan", [])
            step = state.get("current_step", 0)
            if step < len(plan):
                max_iter = plan[step].get("max_iterations", 5)
                if loop_count >= max_iter:
                    logger.info("Loop max iterations (%d) reached, advancing", max_iter)
                    return "fanin"  # Advance past loop step
                logger.info("Loop iteration %d/%d", loop_count + 1, max_iter)
            return "router"

        # Parallel strategy: after quality gate, check if more batch steps
        if strategy == "parallel":
            batch_steps = state.get("_batch_steps", [])
            batch_pos = state.get("_batch_position", 0)
            if batch_steps and batch_pos + 1 < len(batch_steps):
                # More steps in this batch → fanin advances to next
                return "fanin"
            # Batch complete or single step → router for next plan step
            return "fanin"

        # Default: back to router
        return "router"

    graph.add_conditional_edges(
        "quality_gate",
        quality_gate_router,
        {
            "router": "router",
            "fanin": "fanin",
            "aggregator": "aggregator",
        },
    )

    # FanIn → router (continue with next step after batch/loop)
    graph.add_edge("fanin", "router")

    # Aggregator → END
    graph.add_edge("aggregator", END)

    logger.info("LangGraph workflow graph compiled (v0.2 with 3 strategies)")
    return graph.compile()


# Re-export for graph internal use
TYPE_TO_AGENT = {
    "research": "researcher",
    "write": "writer",
    "review": "reviewer",
}


def run_workflow(
    llm_provider: LLMProvider | LLMFactory,
    task: str,
    plan: list[dict] | None = None,
    strategy: str = "sequential",
) -> dict:
    """Run a complete workflow for a given task using the compiled LangGraph.

    Args:
        llm_provider: LLM backend or LLMFactory for multi-provider support.
        task: The user's natural language task description.
        plan: Optional pre-computed plan. Seeds the planner node.
        strategy: Orchestration strategy hint (if plan is provided).

    Returns:
        The final WorkflowState dict after graph execution.
    """
    logger.info("Starting workflow for task: %s (strategy=%s)", task[:80], strategy)

    graph = build_workflow_graph(llm_provider)

    initial_state: dict = {
        "task": task,
        "plan": plan or [],
        "strategy": strategy,
        "current_step": 0,
        "results": {},
        "final_output": "",
        "errors": [],
        "next_action": "continue",
        "retry_count": 0,
        "context_summary": "",
        "history": [],
        "_batch_steps": [],
        "_batch_position": 0,
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
