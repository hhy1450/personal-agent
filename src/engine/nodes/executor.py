"""Executor node: runs an agent on the current subtask."""
import logging

from src.engine.state import WorkflowState
from src.agents.base import Agent
from src.agents.config import AgentConfig
from src.llm.base import LLMProvider
from src.agents.prompts.defaults import AGENT_PROMPTS

logger = logging.getLogger(__name__)

# Tool assignments per agent
AGENT_TOOLS = {
    "researcher": ["web_search", "write_file"],
    "writer": ["read_file", "write_file"],
    "reviewer": ["read_file", "web_search"],
}


class ExecutorNode:
    """LangGraph node that executes a single subtask via an agent.

    Creates an Agent with the appropriate config and tools,
    runs it on the current subtask, and updates the state.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
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

            # Update results dict — keep current_step unchanged
            results = dict(state.get("results", {}))
            results[str(current_step)] = result

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
            return {
                "errors": errors,
                "next_action": "retry",
            }

    def _get_or_create_agent(self, agent_name: str) -> Agent:
        """Get or create an agent instance for the given type."""
        if agent_name in self._agent_cache:
            return self._agent_cache[agent_name]

        prompt = AGENT_PROMPTS.get(agent_name, "You are a helpful assistant: {task}")
        tools = AGENT_TOOLS.get(agent_name, [])

        config = AgentConfig(
            name=agent_name,
            system_prompt=prompt,
            tools=tools,
            temperature=0.1,
            max_retries=2,
        )
        agent = Agent(config, self.llm)
        self._agent_cache[agent_name] = agent
        return agent

    def _build_context(self, state: WorkflowState) -> str:
        """Build context string from previous results."""
        results = state.get("results", {})
        if not results:
            return "None"

        parts = []
        for step_idx, result in sorted(results.items()):
            parts.append(f"Step {step_idx} result:\n{result[:500]}")
        return "\n\n".join(parts)
