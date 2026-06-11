"""Reviewer node: quality check after each subtask execution."""
import logging

from src.engine.state import WorkflowState
from src.agents.base import Agent
from src.agents.config import AgentConfig
from src.llm.base import LLMProvider
from src.agents.prompts.defaults import REVIEWER_PROMPT

logger = logging.getLogger(__name__)


class ReviewerNode:
    """LangGraph node that reviews the latest subtask result.

    Decides whether the result is good enough or needs a retry.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self._agent: Agent | None = None

    def __call__(self, state: WorkflowState) -> dict:
        """Review the current subtask result and control step progression.

        The reviewer is the sole owner of current_step advancement:
        - APPROVED  → increment current_step (or finish if plan exhausted)
        - NOT APPROVED → keep current_step unchanged so the graph re-runs
          the same step (retry)

        Returns partial state with updated next_action and current_step.
        """
        plan = state.get("plan", [])
        results = state.get("results", {})
        current_step = state.get("current_step", 0)

        # If no results yet for this step, skip review
        if not results or str(current_step) not in results:
            return {"next_action": "continue"}

        last_result = results.get(str(current_step), "")
        task = state.get("task", "")
        retry_count = state.get("retry_count", 0)
        max_retries = 3

        # Quick heuristic: if result starts with error, it failed
        if last_result.startswith("Error:"):
            if retry_count >= max_retries:
                return self._advance_step(current_step, plan, retry_count=0)
            return self._retry(current_step, retry_count)

        # Use reviewer LLM to judge quality
        approved = "APPROVED" in last_result.upper()
        if not approved:
            try:
                agent = self._get_agent()
                review_prompt = (
                    f"Original task: {task}\n\n"
                    f"Output to review:\n{last_result[:1000]}\n\n"
                    f"Reply with APPROVED if this meets requirements, "
                    f"or explain what needs improvement."
                )
                verdict = agent.run(task=review_prompt, context="")
                approved = "APPROVED" in verdict.upper()
            except Exception:
                approved = True  # reviewer failed, continue to avoid blocking

        if approved:
            logger.info("Step %d APPROVED", current_step)
            return self._advance_step(current_step, plan, retry_count=0)
        elif retry_count >= max_retries:
            logger.warning("Step %d max retries (%d) reached — advancing anyway", current_step, max_retries)
            return self._advance_step(current_step, plan, retry_count=0)
        else:
            logger.info("Step %d REJECTED (retry %d/%d)", current_step, retry_count + 1, max_retries)
            return self._retry(current_step, retry_count)

    @staticmethod
    def _advance_step(current_step: int, plan: list, retry_count: int) -> dict:
        """Move to the next plan step, or signal finish if exhausted."""
        next_step = current_step + 1
        if next_step >= len(plan):
            logger.info("All %d step(s) complete — finishing workflow", len(plan))
            return {
                "current_step": next_step,
                "next_action": "finish",
                "retry_count": retry_count,
            }
        return {
            "current_step": next_step,
            "next_action": "continue",
            "retry_count": retry_count,
        }

    @staticmethod
    def _retry(current_step: int, retry_count: int) -> dict:
        """Stay on the same step for another attempt."""
        return {
            "next_action": "retry",
            "retry_count": retry_count + 1,
            "current_step": current_step,
        }

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
