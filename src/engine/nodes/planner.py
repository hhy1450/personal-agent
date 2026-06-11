"""Planner node: breaks down user task into subtasks."""
import json
import logging
import re

from src.engine.state import WorkflowState
from src.llm.base import LLMProvider
from src.agents.prompts.defaults import PLANNER_PROMPT

logger = logging.getLogger(__name__)


class PlannerNode:
    """LangGraph node that decomposes a user task into subtasks.

    Uses the LLM to generate a structured plan: a JSON array of
    {type, description} objects. Includes JSON extraction and
    fallback logic for robustness.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def __call__(self, state: WorkflowState) -> dict:
        """Execute the planner node.

        Args:
            state: Current workflow state with at least 'task' set.

        Returns:
            Partial state update with 'plan' and 'current_step'.
        """
        task = state.get("task", "")
        if not task:
            return {
                "plan": [],
                "current_step": 0,
                "errors": [{"step": "planner", "type": "Fatal", "detail": "No task provided"}],
                "next_action": "finish",
            }

        prompt = PLANNER_PROMPT.format(task=task)

        try:
            response = self.llm.chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": task},
                ],
                temperature=0.1,
            )
            content = response["choices"][0]["message"]["content"]
            plan = self._parse_plan(content)
            logger.info("Planner generated %d subtask(s)", len(plan))
            return {
                "plan": plan,
                "current_step": 0,
                "next_action": "continue" if plan else "finish",
            }
        except Exception as e:
            return {
                "plan": [],
                "current_step": 0,
                "errors": [{"step": "planner", "type": "Fatal", "detail": str(e)}],
                "next_action": "finish",
            }

    def _parse_plan(self, content: str) -> list[dict]:
        """Extract JSON plan from LLM response.

        Handles cases where LLM wraps JSON in markdown code blocks
        or adds explanatory text before/after.
        """
        # Try to find JSON in code block first
        code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if code_match:
            json_str = code_match.group(1)
        else:
            # Try to find raw JSON array
            array_match = re.search(r"\[.*?\]", content, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
            else:
                # Last resort: treat whole content as JSON
                json_str = content

        try:
            plan = json.loads(json_str)
            # Validate structure
            if isinstance(plan, list):
                return [
                    item for item in plan
                    if isinstance(item, dict) and "type" in item and "description" in item
                ]
            return []
        except json.JSONDecodeError:
            return []
