"""File operation tools: read_file, write_file."""
from src.tools.registry import tool
from src.tools.sandbox import get_sandbox


@tool(
    name="read_file",
    description="Read the contents of a file from the workspace. Returns the file content as a string.",
)
def read_file(path: str) -> str:
    """Read a file from the workspace.

    Args:
        path: Relative path within workspace (e.g., "reports/output.md").

    Returns:
        File content as a string, or an error message.
    """
    sandbox = get_sandbox()
    try:
        file_path = sandbox.resolve(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


@tool(
    name="write_file",
    description="Write content to a file in the workspace. Creates parent directories as needed.",
)
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace.

    Args:
        path: Relative path within workspace (e.g., "reports/report.md").
        content: The text content to write.

    Returns:
        Success message or error.
    """
    sandbox = get_sandbox()
    try:
        file_path = sandbox.resolve(path)
        sandbox.ensure_dir(file_path)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'"
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"
