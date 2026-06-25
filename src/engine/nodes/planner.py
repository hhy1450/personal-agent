"""Planner node: breaks down user task into subtasks."""
import json
import logging
import re

from src.engine.state import WorkflowState
from src.llm.base import LLMProvider
from src.agents.prompts.defaults import PLANNER_PROMPT

logger = logging.getLogger(__name__)

# Default strategy when not specified
DEFAULT_STRATEGY = "sequential"


class PlannerNode:
    """LangGraph node that decomposes a user task into subtasks.

    Supports three orchestration strategies:
    - sequential: steps execute one after another
    - parallel: independent steps (same group) run concurrently
    - loop: step repeats until stop condition or max iterations

    Output format (v0.2):
        {"strategy": "sequential", "steps": [{...}, ...]}

    Also backward-compatible with v0.1 plain array format:
        [{"type": "...", "description": "..."}]
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def __call__(self, state: WorkflowState) -> dict:
        """Execute the planner node.

        If a non-empty plan is already present in state (e.g. seeded by
        the caller), skip the LLM call to avoid redundant work.

        Args:
            state: Current workflow state with at least 'task' set.

        Returns:
            Partial state update with 'plan', 'strategy', and 'current_step'.
        """
        # Reuse pre-seeded plan to avoid redundant LLM calls
        existing_plan = state.get("plan", [])
        if existing_plan:
            existing_strategy = state.get("strategy", DEFAULT_STRATEGY)
            logger.info(
                "Using pre-seeded plan with %d step(s), strategy=%s",
                len(existing_plan), existing_strategy,
            )
            return {
                "current_step": 0,
                "next_action": "continue",
                "strategy": existing_strategy,
            }

        task = state.get("task", "")
        if not task:
            return {
                "plan": [],
                "strategy": DEFAULT_STRATEGY,
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
            plan_data = self._parse_plan(content)
            plan = plan_data.get("steps", [])
            strategy = plan_data.get("strategy", DEFAULT_STRATEGY)
            logger.info(
                "Planner generated %d subtask(s), strategy=%s",
                len(plan), strategy,
            )
            return {
                "plan": plan,
                "strategy": strategy,
                "current_step": 0,
                "next_action": "continue" if plan else "finish",
            }
        except Exception as e:
            return {
                "plan": [],
                "strategy": DEFAULT_STRATEGY,
                "current_step": 0,
                "errors": [{"step": "planner", "type": "Fatal", "detail": str(e)}],
                "next_action": "finish",
            }

    def _parse_plan(self, content: str) -> dict:
        """Extract JSON plan from LLM response.

        Supports v0.2 format: {"strategy": "...", "steps": [...]}
        Also backward-compatible with v0.1 array format: [...]

        Handles markdown code blocks and surrounding text.
        """
        json_str = self._extract_json(content)

        try:
            parsed = json.loads(json_str)

            # v0.2 format: {"strategy": "...", "steps": [...]}
            if isinstance(parsed, dict) and "steps" in parsed:
                steps = [
                    item for item in parsed["steps"]
                    if isinstance(item, dict) and "type" in item
                ]
                return {
                    "strategy": parsed.get("strategy", DEFAULT_STRATEGY),
                    "steps": steps,
                }

            # v0.1 backward-compatible: plain array
            if isinstance(parsed, list):
                return {
                    "strategy": DEFAULT_STRATEGY,
                    "steps": [
                        item for item in parsed
                        if isinstance(item, dict) and "type" in item and "description" in item
                    ],
                }

            return {"strategy": DEFAULT_STRATEGY, "steps": []}

        except json.JSONDecodeError:
            return {"strategy": DEFAULT_STRATEGY, "steps": []}

    def _extract_json(self, content: str) -> str:
        """Extract JSON string from LLM response content.

        Handles: code blocks (```json ... ```), bare JSON objects/arrays,
        and text with embedded JSON. Tries multiple patterns and uses the
        first that parses successfully.
        """
        # Candidates in priority order — try array before object to avoid
        # the object regex greedily capturing a single {}-pair from an array.
        candidates = []

        # 1. Code block (both object and array)
        code_match = re.search(
            r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
            content, re.DOTALL,
        )
        if code_match:
            candidates.append(code_match.group(1))

        # 2. Balanced JSON array (try BEFORE object — more specific)
        arr_match = re.search(r"\[.*\]", content, re.DOTALL)
        if arr_match:
            candidates.append(arr_match.group(0))

        # 3. Balanced JSON object
        obj_match = re.search(r"\{.*\}", content, re.DOTALL)
        if obj_match:
            candidates.append(obj_match.group(0))

        # 4. Fallback: entire content
        candidates.append(content)

        # Try each candidate, return the first that parses as valid JSON
        for candidate in candidates:
            candidate = candidate.strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

        # Nothing worked — return raw content for caller to handle
        return content
