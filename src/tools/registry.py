"""Tool registration system.

Tools are Python functions decorated with @tool. The registry auto-generates
OpenAI Function Calling JSON Schemas from type annotations and docstrings.
"""
import inspect
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_registry: dict[str, "Tool"] = {}


class Tool:
    """A registered tool callable by LLM agents."""

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
    ):
        self.func = func
        self.name = name
        self.description = description

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def to_openai_schema(self) -> dict:
        """Generate OpenAI Function Calling schema for this tool."""
        sig = inspect.signature(self.func)
        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation is int:
                param_type = "integer"
            elif param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"
            elif param.annotation is list:
                param_type = "array"

            properties[param_name] = {
                "type": param_type,
                "description": f"The {param_name} parameter",
            }

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


def tool(name: str, description: str):
    """Decorator to register a function as a tool.

    Usage:
        @tool(name="web_search", description="Search the web")
        def web_search(query: str, max_results: int = 10) -> list[dict]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        t = Tool(func=func, name=name, description=description)
        _registry[name] = t
        return func
    return decorator


def get_tool(name: str) -> Tool | None:
    """Get a registered tool by name."""
    return _registry.get(name)


def get_all_tools() -> dict[str, Tool]:
    """Get all registered tools."""
    return dict(_registry)


def get_tools_as_openai_schemas(tool_names: list[str] | None = None) -> list[dict]:
    """Get tool schemas in OpenAI format.

    Args:
        tool_names: Specific tool names, or None for all registered tools.

    Returns:
        List of OpenAI Function Calling tool definitions.
    """
    tools = _registry.values()
    if tool_names:
        tools = [t for t in tools if t.name in tool_names]
    return [t.to_openai_schema() for t in tools]


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Execute a tool by name with given arguments.

    Args:
        name: Tool name.
        arguments: Keyword arguments for the tool function.

    Returns:
        Tool execution result.

    Raises:
        ValueError: If tool not found.
    """
    tool = get_tool(name)
    if tool is None:
        logger.error("Tool '%s' not found in registry (available: %s)", name, list(_registry.keys()))
        raise ValueError(f"Tool '{name}' not found in registry")
    logger.debug("Executing tool '%s' with args: %s", name, arguments)
    return tool(**arguments)
