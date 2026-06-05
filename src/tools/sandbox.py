"""Sandbox utilities for safe tool execution."""
from pathlib import Path
from typing import Optional


class Sandbox:
    """Restrict file operations to a safe workspace directory.

    All paths are resolved relative to the workspace root. Paths that
    try to escape the workspace (via '..' or absolute paths) are rejected.
    """

    def __init__(self, workspace_dir: Path):
        self._workspace = workspace_dir.resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

    @property
    def workspace(self) -> Path:
        return self._workspace

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path, ensuring it stays within workspace.

        Args:
            path: Relative path within workspace.

        Returns:
            Resolved absolute path.

        Raises:
            ValueError: If path tries to escape workspace.
        """
        raw = Path(path)
        # Reject absolute paths (platform-specific) and paths starting with /
        # or \ which are drive-relative roots on Windows and absolute on Unix
        if raw.is_absolute() or str(path).startswith(("/", "\\")):
            raise ValueError(f"Absolute paths are not allowed: {path}")

        resolved = (self._workspace / raw).resolve()

        # Check that resolved path is still within workspace
        try:
            resolved.relative_to(self._workspace)
        except ValueError:
            raise ValueError(f"Path escapes workspace: {path} -> {resolved}")

        return resolved

    def ensure_dir(self, path: Path) -> None:
        """Create parent directories if they don't exist."""
        path.parent.mkdir(parents=True, exist_ok=True)


# Global sandbox instance (initialized in config)
_sandbox: Optional[Sandbox] = None


def get_sandbox() -> Sandbox:
    """Get the global sandbox instance. Creates one if not set."""
    global _sandbox
    if _sandbox is None:
        from src.config.settings import WORKSPACE_DIR
        _sandbox = Sandbox(WORKSPACE_DIR)
    return _sandbox


def set_sandbox(sandbox: Sandbox) -> None:
    """Set the global sandbox instance (useful for testing)."""
    global _sandbox
    _sandbox = sandbox
