"""Reviewer node: quality check after each subtask execution.

When structured action info is available (from ActionMapper), the reviewer
also validates that the action type matches the expected step type.
"""
import logging

from src.engine.state import WorkflowState
from src.agents.base import Agent
from src.agents.config import AgentConfig
from src.llm.base import LLMProvider
from src.agents.prompts.defaults import REVIEWER_PROMPT

logger = logging.getLogger(__name__)

# Expected action types per plan step type
STEP_ACTION_MAP = {
    "research": {"search"},
    "write": {"write_file", "generate"},
    "review": {"review", "read_file"},
}


class ReviewerNode:
    """LangGraph node that reviews the latest subtask result.

    Decides whether the result is good enough or needs a retry.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self._agent: Agent | None = None

    def __call__(self, state: WorkflowState) -> dict:
        """Review the current subtask result and control step progression.

        Strategy-aware behavior:
        - sequential: APPROVED → advance, REJECTED → retry (max 3)
        - parallel: same as sequential, but batch steps skip review
        - loop: check stop_condition; if met → advance, else → retry

        Returns partial state with updated next_action and current_step.
        """
        plan = state.get("plan", [])
        results = state.get("results", {})
        current_step = state.get("current_step", 0)
        strategy = state.get("strategy", "sequential")

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

        # ---- Structured action validation (if available) ----
        action_key = f"{current_step}_action"
        action_info = results.get(action_key, {})
        if action_info.get("mapped"):
            action_type = action_info.get("action_type", "")
            step_type = plan[current_step].get("type", "") if current_step < len(plan) else ""
            expected = STEP_ACTION_MAP.get(step_type, set())
            if expected and action_type not in expected:
                logger.warning(
                    "Step %d action mismatch: step_type=%s but action=%s (expected %s)",
                    current_step, step_type, action_type, expected,
                )
                # Don't auto-reject — just flag and let LLM review decide

        # ---- Loop strategy: check stop_condition ----
        if strategy == "loop":
            subtask = plan[current_step] if current_step < len(plan) else {}
            stop_condition = subtask.get("stop_condition", "")
            max_iterations = subtask.get("max_iterations", 5)

            if retry_count >= max_iterations:
                logger.info("Loop step %d: max iterations (%d) reached", current_step, max_iterations)
                return self._advance_step(current_step, plan, retry_count=0)

            # Ask reviewer to check if stop condition is met
            loop_met = self._check_loop_condition(
                task=task,
                result=last_result,
                stop_condition=stop_condition,
            )
            if loop_met:
                logger.info("Loop step %d: stop condition MET → advancing", current_step)
                return self._advance_step(current_step, plan, retry_count=0)
            else:
                logger.info("Loop step %d: stop condition NOT met (iteration %d/%d)",
                            current_step, retry_count + 1, max_iterations)
                return self._retry(current_step, retry_count)

        # ---- Sequential/Parallel: standard quality review ----
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
            logger.warning("Step %d max retries (%d) reached — advancing anyway",
                           current_step, max_retries)
            return self._advance_step(current_step, plan, retry_count=0)
        else:
            logger.info("Step %d REJECTED (retry %d/%d)",
                        current_step, retry_count + 1, max_retries)
            return self._retry(current_step, retry_count)

    def _check_loop_condition(
        self, task: str, result: str, stop_condition: str,
    ) -> bool:
        """Check if a loop step's stop condition has been met.

        Uses a lightweight LLM call to evaluate whether the result
        satisfies the stop condition.
        """
        if not stop_condition:
            return True  # No condition specified → one shot

        try:
            agent = self._get_agent()
            check_prompt = (
                f"Original task: {task}\n\n"
                f"Stop condition: {stop_condition}\n\n"
                f"Latest execution result:\n{result[:1500]}\n\n"
                f"Has the stop condition been met? "
                f"Reply with exactly YES or NO, then a brief explanation."
            )
            verdict = agent.run(task=check_prompt, context="")
            return "YES" in verdict.upper() and "NO" not in verdict.upper().split("YES")[0]
        except Exception:
            # Reviewer failed → assume condition met to avoid infinite loop
            return True

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
            parts.append(str(results[step_key]))
            parts.append("")

    return {
        "final_output": "\n".join(parts),
        "next_action": "finish",
    }
