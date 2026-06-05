"""Tests for tool registry."""
import pytest
from src.tools.registry import tool, get_tool, get_all_tools, get_tools_as_openai_schemas, execute_tool, _registry


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    _registry.clear()
    yield
    _registry.clear()


def test_register_tool():
    """@tool decorator should register in the global registry."""

    @tool(name="test_tool", description="A test tool")
    def test_tool(query: str, max_results: int = 10) -> list[dict]:
        return [{"result": query}]

    t = get_tool("test_tool")
    assert t is not None
    assert t.name == "test_tool"
    assert t.description == "A test tool"


def test_get_all_tools():
    """get_all_tools should return all registered tools."""

    @tool(name="tool1", description="First")
    def tool1():
        pass

    @tool(name="tool2", description="Second")
    def tool2():
        pass

    tools = get_all_tools()
    assert len(tools) == 2
    assert "tool1" in tools
    assert "tool2" in tools


def test_to_openai_schema():
    """Tool should generate valid OpenAI function schema."""

    @tool(name="search", description="Search the internet")
    def search(query: str, max_results: int = 10) -> list[dict]:
        return []

    t = get_tool("search")
    schema = t.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "Search the internet"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "max_results" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]


def test_get_tools_as_openai_schemas_filtered():
    """get_tools_as_openai_schemas should filter by tool name."""

    @tool(name="a", description="A")
    def a():
        pass

    @tool(name="b", description="B")
    def b():
        pass

    schemas = get_tools_as_openai_schemas(tool_names=["a"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "a"


def test_execute_tool():
    """execute_tool should call the tool function with arguments."""

    @tool(name="add", description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    result = execute_tool("add", {"a": 3, "b": 4})
    assert result == 7


def test_execute_tool_not_found():
    """execute_tool should raise ValueError for unknown tools."""
    with pytest.raises(ValueError, match="not found"):
        execute_tool("nonexistent", {})
