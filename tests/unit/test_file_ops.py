"""Tests for file operation tools."""
from pathlib import Path
from src.tools.file_ops import read_file, write_file
from src.tools.sandbox import Sandbox, set_sandbox


class TestReadFile:
    """Tests for read_file tool."""

    def test_read_existing_file(self, tmp_path):
        """Should read and return file contents."""
        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)
        (tmp_path / "test.txt").write_text("Hello, World!")

        result = read_file(path="test.txt")
        assert result == "Hello, World!"

    def test_read_nonexistent_file(self, tmp_path):
        """Should return error message for missing files."""
        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)

        result = read_file(path="nope.txt")
        assert result.startswith("Error: File not found")


class TestWriteFile:
    """Tests for write_file tool."""

    def test_write_new_file(self, tmp_path):
        """Should write content and create parent dirs."""
        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)

        result = write_file(path="reports/test.md", content="# Hello")
        assert "Successfully wrote" in result
        assert (tmp_path / "reports" / "test.md").read_text() == "# Hello"

    def test_write_overwrites_existing(self, tmp_path):
        """Should overwrite existing file."""
        sandbox = Sandbox(tmp_path)
        set_sandbox(sandbox)
        (tmp_path / "notes.txt").write_text("old")

        result = write_file(path="notes.txt", content="new")
        assert "Successfully wrote" in result
        assert (tmp_path / "notes.txt").read_text() == "new"
