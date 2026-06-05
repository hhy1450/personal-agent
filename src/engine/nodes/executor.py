"""Executor node: runs an agent on the current subtask."""
from src.engine.state import WorkflowState
from src.agents.base import Agent
from src.agents.config import AgentConfig
from src.llm.base import LLMProvider
from src.agents.prompts.defaults import AGENT_PROMPTS

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

    def __call__(self, state: WorkflowState) -> dict:
        """Execute the current subtask.

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
        subtask_type = subtask.get("type", "")
        subtask_desc = subtask.get("description", "")

        agent_name = subtask_type.lower()
        if agent_name.endswith("e"):
            agent_name = agent_name + "r"
        if agent_name == "review":
            agent_name = "reviewer"
        if agent_name == "research":
            agent_name = "researcher"
        if agent_name == "write":
            agent_name = "writer"

        try:
            agent = self._get_or_create_agent(agent_name)
            context = self._build_context(state)
            result = agent.run(task=subtask_desc, context=context)

            # Update results dict
            results = dict(state.get("results", {}))
            results[str(current_step)] = result

            return {
                "results": results,
                "current_step": current_step + 1,
                "next_action": "continue",
            }
        except Exception as e:
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
