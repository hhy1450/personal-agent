"""Executor node: runs an agent on the current subtask."""
import logging

from src.engine.state import WorkflowState
from src.agents.base import Agent
from src.agents.config import AgentConfig
from src.llm.base import LLMProvider
from src.llm.factory import LLMFactory
from src.memory.context_manager import ContextManager
from src.memory.models import HistoryStep
from src.actions.mapper import ActionMapper
from src.agents.prompts.defaults import AGENT_PROMPTS

logger = logging.getLogger(__name__)

# Tool assignments per agent
AGENT_TOOLS = {
    "researcher": ["web_search", "write_file"],
    "writer": ["read_file", "write_file"],
    "reviewer": ["read_file", "web_search"],
}

# Vision requirements per agent (agents that may need image understanding)
AGENT_VISION_REQUIREMENTS = {
    "researcher": False,
    "writer": False,
    "reviewer": False,
    # Vision-aware agents can be added here later
}


class ExecutorNode:
    """LangGraph node that executes a single subtask via an agent.

    Creates an Agent with the appropriate config and tools,
    runs it on the current subtask, and updates the state.

    Uses LLMFactory to automatically select the right provider:
    vision-capable agents get QwenVLProvider, others get DeepSeekProvider.
    """

    def __init__(self, llm_provider: LLMProvider | LLMFactory):
        if isinstance(llm_provider, LLMFactory):
            self._factory = llm_provider
        else:
            # Backward-compatible: wrap single provider in a simple adapter
            self._factory = _SingleProviderFactory(llm_provider)
        self._agent_cache: dict[str, Agent] = {}

    # Maps planner subtask type to agent config name.
    _TYPE_TO_AGENT: dict[str, str] = {
        "research": "researcher",
        "write": "writer",
        "review": "reviewer",
    }

    def __call__(self, state: WorkflowState) -> dict:
        """Execute the current subtask.

        Does NOT advance current_step — the ReviewerNode owns step
        progression so that retries stay on the same step naturally.

        Args:
            state: Current workflow state.

        Returns:
            Partial state update with results, errors, and next_action.
        """
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        if current_step >= len(plan):
            return {"next_action": "finish"}

        subtask = plan[current_step]
        subtask_type = subtask.get("type", "").lower()
        subtask_desc = subtask.get("description", "")

        agent_name = self._TYPE_TO_AGENT.get(
            subtask_type,
            self._TYPE_TO_AGENT.get("write", "writer"),
        )

        try:
            agent = self._get_or_create_agent(agent_name)
            context = self._build_context(state)
            logger.info("Step %d [%s]: %s", current_step, agent_name, subtask_desc[:60])
            result = agent.run(task=subtask_desc, context=context)

            # Map agent output to structured Action
            action_result = self._map_to_action(result, agent_name)

            # Update results dict — keep current_step unchanged
            results = dict(state.get("results", {}))
            results[str(current_step)] = result

            # Store structured action info alongside result for reviewer
            action_key = f"{current_step}_action"
            results[action_key] = action_result

            logger.debug("Step %d completed — %d chars output", current_step, len(result))
            return {
                "results": results,
                "next_action": "continue",
            }
        except Exception as e:
            logger.error("Step %d failed: %s", current_step, e)
            errors = list(state.get("errors", []))
            errors.append({
                "step": current_step,
                "type": "Retryable",
                "detail": str(e),
            })
            # Also write an error result so the reviewer can see it,
            # auto-reject, and properly track retry_count.
            # Without this the reviewer sees no result → returns
            # "continue" → infinite loop in the graph.
            error_result = f"Error: {str(e)}"
            results = dict(state.get("results", {}))
            results[str(current_step)] = error_result
            return {
                "results": results,
                "errors": errors,
            }

    def _get_or_create_agent(self, agent_name: str) -> Agent:
        """Get or create an agent instance for the given type.

        Uses LLMFactory to select the right provider based on the
        agent's vision requirements.
        """
        if agent_name in self._agent_cache:
            return self._agent_cache[agent_name]

        prompt = AGENT_PROMPTS.get(agent_name, "You are a helpful assistant: {task}")
        tools = AGENT_TOOLS.get(agent_name, [])
        requires_vision = AGENT_VISION_REQUIREMENTS.get(agent_name, False)

        provider = self._factory.get_provider(requires_vision=requires_vision)

        config = AgentConfig(
            name=agent_name,
            system_prompt=prompt,
            tools=tools,
            temperature=0.1,
            max_retries=2,
            requires_vision=requires_vision,
        )
        agent = Agent(config, provider)
        self._agent_cache[agent_name] = agent
        return agent

    def _map_to_action(self, result: str, agent_name: str) -> dict | None:
        """Map agent output to a structured Action (best-effort).

        Returns a dict with action info, or None if mapping fails.
        The result is stored alongside the raw output for the reviewer.
        """
        try:
            text_provider = self._factory.get_provider(requires_vision=False)
            mapper = ActionMapper(text_provider)
            action = mapper.map(result)
            if action:
                logger.debug(
                    "ActionMapper: %s → %s(%s)",
                    agent_name, action.action.value, action.reason[:60],
                )
                return {
                    "action_type": action.action.value,
                    "params": action.params,
                    "reason": action.reason,
                    "mapped": True,
                }
        except Exception as e:
            logger.debug("ActionMapper skipped for %s: %s", agent_name, e)

        return {"mapped": False, "action_type": "unknown"}

    def _build_context(self, state: WorkflowState) -> str:
        """Build context string using ContextManager for token-aware compression.

        Converts raw state.results into HistoryStep objects, then delegates
        to ContextManager for two-layer compression (coarse trim + LLM summary).
        Falls back to simple concatenation if ContextManager is unavailable.
        """
        task = state.get("task", "")
        results = state.get("results", {})
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        if not results:
            return "None"

        # Build HistoryStep list from results (skip non-numeric keys like '0_action')
        history: list[HistoryStep] = []
        numeric_results = {
            k: v for k, v in results.items()
            if k.isdigit()
        }
        for step_idx_str, result in sorted(
            numeric_results.items(), key=lambda x: int(x[0])
        ):
            step_idx = int(step_idx_str)
            step_desc = ""
            agent_name = "unknown"
            if step_idx < len(plan):
                step_desc = plan[step_idx].get("description", "")
                agent_name = plan[step_idx].get("type", "unknown")

            history.append(HistoryStep(
                step_index=step_idx,
                agent_name=agent_name,
                description=step_desc,
                result=result,
                approved=not result.startswith("Error:"),
            ))

        # Use ContextManager if we have an LLM
        if hasattr(self, '_factory') and self._factory is not None:
            try:
                text_provider = self._factory.get_provider(requires_vision=False)
                ctx_mgr = ContextManager(llm=text_provider)
                return ctx_mgr.build_context(task=task, history=history, current_step=current_step)
            except Exception as e:
                logger.warning("ContextManager failed, falling back to simple context: %s", e)

        # Simple fallback: use last 5 numeric result entries
        parts = []
        numeric_results = {k: v for k, v in results.items() if k.isdigit()}
        sorted_results = sorted(numeric_results.items(), key=lambda x: int(x[0]))
        for step_idx_str, result in sorted_results[-5:]:
            step_idx = int(step_idx_str)
            step_desc = ""
            agent_name = "unknown"
            if step_idx < len(plan):
                step_desc = plan[step_idx].get("description", "")
                agent_name = plan[step_idx].get("type", "unknown")
            parts.append(
                f"Step {step_idx} [{agent_name}]: {step_desc}\n"
                f"Result: {result[:500]}"
            )
        return "\n\n".join(parts)


class _SingleProviderFactory:
    """Adapter: wraps a single LLMProvider to match LLMFactory interface.

    Used when code was originally written to pass a single provider
    (backward compatibility).
    """

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def get_provider(self, requires_vision: bool = False) -> LLMProvider:
        return self._provider
