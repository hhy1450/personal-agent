"""Tests for sandbox."""
import pytest
from pathlib import Path
from src.tools.sandbox import Sandbox, set_sandbox, get_sandbox


class TestSandbox:
    """Tests for Sandbox class."""

    def test_resolve_within_workspace(self, tmp_path):
        """resolve should return absolute path within workspace."""
        sandbox = Sandbox(tmp_path)
        result = sandbox.resolve("reports/output.md")
        expected = tmp_path / "reports" / "output.md"
        assert result == expected

    def test_resolve_rejects_absolute_path(self, tmp_path):
        """resolve should reject absolute paths."""
        sandbox = Sandbox(tmp_path)
        with pytest.raises(ValueError, match="Absolute paths"):
            sandbox.resolve("/etc/passwd")

    def test_resolve_rejects_escape(self, tmp_path):
        """resolve should reject paths that escape via '..'."""
        sandbox = Sandbox(tmp_path)
        with pytest.raises(ValueError, match="escapes workspace"):
            sandbox.resolve("../../../etc/passwd")

    def test_ensure_dir_creates_parents(self, tmp_path):
        """ensure_dir should create parent directories."""
        sandbox = Sandbox(tmp_path)
        file_path = sandbox.resolve("deep/nested/file.txt")
        sandbox.ensure_dir(file_path)
        assert file_path.parent.exists()

    def test_get_sandbox_returns_global(self, tmp_path):
        """get_sandbox should return the global instance."""
        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)
        assert get_sandbox() is sandbox
