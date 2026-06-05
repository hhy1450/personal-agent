"""Reviewer node: quality check after each subtask execution."""
from src.engine.state import WorkflowState
from src.agents.base import Agent
from src.agents.config import AgentConfig
from src.llm.base import LLMProvider
from src.agents.prompts.defaults import REVIEWER_PROMPT


class ReviewerNode:
    """LangGraph node that reviews the latest subtask result.

    Decides whether the result is good enough or needs a retry.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self._agent: Agent | None = None

    def __call__(self, state: WorkflowState) -> dict:
        """Review the latest subtask result.

        Returns partial state with updated next_action:
        - "continue" if approved
        - "retry" if needs another attempt
        """
        results = state.get("results", {})
        current_step = state.get("current_step", 0)
        last_step = current_step - 1

        # If no results yet, skip review
        if not results or str(last_step) not in results:
            return {"next_action": "continue"}

        last_result = results.get(str(last_step), "")
        task = state.get("task", "")

        # Quick heuristic: if result starts with error, it failed
        if last_result.startswith("Error:"):
            return {"next_action": "retry"}

        try:
            agent = self._get_agent()
            review_prompt = (
                f"Original task: {task}\n\n"
                f"Output to review:\n{last_result[:1000]}\n\n"
                f"Reply with 'APPROVED' if this meets requirements, "
                f"or explain what needs improvement."
            )
            verdict = agent.run(task=review_prompt, context="")

            if "APPROVED" in verdict.upper():
                return {"next_action": "continue"}
            else:
                return {"next_action": "retry"}
        except Exception:
            # If reviewer fails, be lenient — continue anyway
            return {"next_action": "continue"}

    def _get_agent(self) -> Agent:
        """Get or create the reviewer agent."""
        if self._agent is None:
            config = AgentConfig(
                name="reviewer",
                system_prompt=REVIEWER_PROMPT,
                tools=["read_file"],
                temperature=0.0,
            )
            self._agent = Agent(config, self.llm)
        return self._agent


def aggregator_node(state: WorkflowState) -> dict:
    """Aggregate all subtask results into a final output.

    This is a stateless function (not a class) because it doesn't
    need an LLM — it just concatenates results.
    """
    results = state.get("results", {})
    plan = state.get("plan", [])
    task = state.get("task", "")

    if not results:
        return {
            "final_output": f"No results produced for task: {task}",
            "next_action": "finish",
        }

    parts = [f"# Results for: {task}\n"]
    for i, subtask in enumerate(plan):
        step_key = str(i)
        if step_key in results:
            parts.append(f"## Step {i + 1}: {subtask.get('description', 'Unknown')}")
            parts.append(results[step_key])
            parts.append("")

    return {
        "final_output": "\n".join(parts),
        "next_action": "finish",
    }
