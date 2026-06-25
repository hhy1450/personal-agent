"""ActionMapper — converts free-text agent output into structured Actions."""
import json
import logging

from src.actions.schema import Action, ActionType
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

MAPPER_PROMPT = """You are an action parser. Given an agent's natural language output, extract the primary action as structured JSON.

Available actions: {available_actions}

Output a single JSON object with:
- "action": one of the available action types
- "params": a dict of relevant parameters from the output
- "reason": a brief explanation of why this action was chosen

If the output describes multiple actions, pick the PRIMARY one.
If no clear action is found, use "generate" as the default.

Output ONLY valid JSON. No markdown, no extra text.

Agent output:
{agent_output}"""


class ActionMapper:
    """Maps free-text agent output to structured Action objects.

    Uses a lightweight LLM call (temperature=0) to extract structured
    actions from natural language. Falls back gracefully to a default
    GENERATE action when parsing fails.

    Usage:
        mapper = ActionMapper(llm)
        action = mapper.map("I searched and found 5 results about AI")
        # Action(action=ActionType.SEARCH, params={...}, reason="...")
    """

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def map(self, agent_output: str, context: dict | None = None) -> Action | None:
        """Parse agent output into a structured Action.

        Args:
            agent_output: The agent's raw text output.
            context: Optional context dict with task info (unused currently).

        Returns:
            An Action object, or None if parsing fails completely.
        """
        if not agent_output or not agent_output.strip():
            return None

        # Heuristic short-circuit: if output is very short, skip LLM parsing
        if len(agent_output) < 20:
            return self._heuristic_parse(agent_output)

        prompt = MAPPER_PROMPT.format(
            available_actions=", ".join(a.value for a in ActionType),
            agent_output=agent_output[:3000],
        )

        try:
            response = self._llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            content = response["choices"][0]["message"]["content"]
            return self._parse_response(content)

        except Exception as e:
            logger.warning("ActionMapper LLM call failed: %s — falling back to heuristic", e)
            return self._heuristic_parse(agent_output)

    def _parse_response(self, content: str) -> Action | None:
        """Parse the LLM's JSON response into an Action.

        Handles markdown code blocks and bare JSON.
        """
        import re

        # Try code block
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        json_str = code_match.group(1) if code_match else content

        # Try to find JSON object
        obj_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if obj_match:
            json_str = obj_match.group(0)

        try:
            data = json.loads(json_str)
            # Validate and normalize
            action_value = data.get("action", "generate")
            try:
                action_type = ActionType(action_value)
            except ValueError:
                action_type = ActionType.GENERATE

            return Action(
                action=action_type,
                params=data.get("params", {}),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug("ActionMapper JSON parse failed: %s", e)
            return None

    def _heuristic_parse(self, output: str) -> Action:
        """Simple heuristic action detection without LLM call.

        Fast path for short or error-case outputs.
        """
        output_lower = output.lower()

        if "search" in output_lower or "found" in output_lower:
            return Action(
                action=ActionType.SEARCH,
                params={"query": ""},
                reason="Heuristic: output mentions search",
            )
        if "write" in output_lower or "saved" in output_lower or "created" in output_lower:
            return Action(
                action=ActionType.WRITE_FILE,
                params={"path": "", "content": ""},
                reason="Heuristic: output mentions file write",
            )
        if "review" in output_lower or "check" in output_lower:
            return Action(
                action=ActionType.REVIEW,
                reason="Heuristic: output mentions review",
            )

        # Default
        return Action(
            action=ActionType.GENERATE,
            reason="Default action for unstructured output",
        )
