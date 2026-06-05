"""Tests for CLI commands."""
from click.testing import CliRunner
from src.cli.main import cli


class TestCLI:
    """Tests for CLI using Click's CliRunner."""

    def test_cli_help(self):
        """'pa --help' should show usage."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Personal Agent" in result.output

    def test_run_command_needs_task(self):
        """'pa run' without argument should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0

    def test_history_command(self):
        """'pa history' should work with empty DB."""
        runner = CliRunner()
        result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0

    def test_inspect_missing_task(self):
        """'pa inspect <nonexistent>' should error gracefully."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "99999"])
        assert "not found" in result.output.lower()
