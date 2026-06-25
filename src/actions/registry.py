"""Action registry — maps Action objects to executable tool calls."""
import logging
from typing import Any, Callable

from src.actions.schema import Action, ActionType
from src.tools.registry import execute_tool

logger = logging.getLogger(__name__)


class ActionRegistry:
    """Registry for executing structured Actions via the tool system.

    Each ActionType is mapped to a handler that:
    1. Validates required params
    2. Calls the corresponding tool via the tool registry
    3. Returns the tool's result

    Usage:
        registry = ActionRegistry()
        result = registry.execute(some_action)
    """

    def __init__(self):
        self._handlers: dict[ActionType, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default handlers for standard action types."""
        self._handlers[ActionType.SEARCH] = self._handle_search
        self._handlers[ActionType.READ_FILE] = self._handle_read_file
        self._handlers[ActionType.WRITE_FILE] = self._handle_write_file
        self._handlers[ActionType.GENERATE] = self._handle_generate
        self._handlers[ActionType.SUMMARIZE] = self._handle_summarize
        self._handlers[ActionType.REVIEW] = self._handle_review
        self._handlers[ActionType.ANALYZE_IMAGE] = self._handle_analyze_image

    def register(self, action_type: ActionType, handler: Callable):
        """Register a custom action handler."""
        self._handlers[action_type] = handler

    def execute(self, action: Action) -> dict:
        """Execute a structured Action and return the result.

        Args:
            action: The Action to execute.

        Returns:
            Dict with {"success": bool, "result": Any, "error": str|None}
        """
        handler = self._handlers.get(action.action)
        if handler is None:
            return {
                "success": False,
                "result": None,
                "error": f"No handler registered for action: {action.action}",
            }

        try:
            result = handler(action)
            return {"success": True, "result": result, "error": None}
        except Exception as e:
            logger.error("Action execution failed: %s → %s", action.action, e)
            return {"success": False, "result": None, "error": str(e)}

    # ---- Default handlers ----

    @staticmethod
    def _handle_search(action: Action) -> Any:
        query = action.params.get("query", "")
        max_results = action.params.get("max_results", 5)
        return execute_tool("web_search", {"query": query, "max_results": max_results})

    @staticmethod
    def _handle_read_file(action: Action) -> Any:
        path = action.params.get("path", "")
        return execute_tool("read_file", {"path": path})

    @staticmethod
    def _handle_write_file(action: Action) -> Any:
        path = action.params.get("path", "output.txt")
        content = action.params.get("content", "")
        return execute_tool("write_file", {"path": path, "content": content})

    @staticmethod
    def _handle_generate(action: Action) -> str:
        """Generate is a no-op — the LLM already generated the content."""
        return f"Content generated. Reason: {action.reason}"

    @staticmethod
    def _handle_summarize(action: Action) -> str:
        """Summarize requires LLM — delegate back to caller."""
        return f"Summarization requested. Params: {action.params}"

    @staticmethod
    def _handle_review(action: Action) -> str:
        """Review requires LLM — delegate back to caller."""
        return f"Review requested. Reason: {action.reason}"

    @staticmethod
    def _handle_analyze_image(action: Action) -> str:
        """Image analysis requires vision-capable LLM."""
        return f"Image analysis requested. URL: {action.params.get('image_url', 'N/A')}"


# Global singleton
_registry: ActionRegistry | None = None


def get_action_registry() -> ActionRegistry:
    """Get the global ActionRegistry (creates on first call)."""
    global _registry
    if _registry is None:
        _registry = ActionRegistry()
    return _registry
