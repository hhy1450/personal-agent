"""CLI entry point for Personal Agent."""
import sys
import click

from src.llm.deepseek import DeepSeekProvider
from src.engine.graph import run_workflow
from src.storage.database import (
    init_db, seed_agent_configs,
    create_task, get_task, list_tasks, update_task_status,
    create_workflow_run, update_workflow_run,
)
from src.storage.models import TaskStatus


@click.group()
@click.version_option(version="0.1.0", prog_name="personal-agent")
def cli():
    """Personal Agent — Multi-Agent Workflow Automation.

    Break down complex tasks, execute them with AI agents,
    and get structured results.
    """
    # Ensure database is ready
    init_db()
    seed_agent_configs()


@cli.command()
@click.argument("task", type=str)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress.")
def run(task: str, verbose: bool = False):
    """Run a task through the multi-agent workflow.

    Example:
        pa run "Research DeepSeek V3 features and write a report"
    """
    click.echo(f"\n  Task: {task}\n")
    click.echo("=" * 60)

    # Create task record
    db_task = create_task(title=task[:100], description=task)
    update_task_status(db_task.id, TaskStatus.RUNNING)
    run_record = create_workflow_run(db_task.id)

    try:
        provider = DeepSeekProvider()
        result = run_workflow(provider, task)

        if verbose:
            click.echo(f"\n  Plan: {len(result.get('plan', []))} steps")
            for i, step in enumerate(result.get("plan", [])):
                status_icon = "  " if str(i) in result.get("results", {}) else "  "
                click.echo(f"  [{status_icon}] Step {i + 1}: {step.get('description', '')}")

        # Output
        final = result.get("final_output", "No output produced.")
        click.echo("\n" + "=" * 60)
        click.echo(final)
        click.echo("=" * 60)

        # Check errors
        errors = result.get("errors", [])
        if errors:
            click.echo(f"\n  Warnings: {len(errors)} error(s) encountered:")
            for e in errors:
                click.echo(f"  [!] {e.get('detail', str(e))}")

        # Update records
        update_task_status(db_task.id, TaskStatus.COMPLETED)
        update_workflow_run(run_record.id, str(result), TaskStatus.COMPLETED)

        click.echo(f"\n  Task completed. ID: {db_task.id}")

    except Exception as e:
        update_task_status(db_task.id, TaskStatus.FAILED)
        update_workflow_run(run_record.id, "{}", TaskStatus.FAILED)
        click.echo(f"\n  Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--limit", "-n", default=10, help="Number of tasks to show.")
def history(limit: int):
    """Show recent task history."""
    tasks = list_tasks(limit=limit)

    if not tasks:
        click.echo("No tasks yet. Run 'pa run <task>' to get started.")
        return

    click.echo(f"\n  Recent {len(tasks)} tasks:\n")
    click.echo(f"  {'ID':<6} {'Status':<12} {'Title':<60} {'Created':<20}")
    click.echo("  " + "-" * 98)

    for t in tasks:
        title = t.title[:57] + "..." if len(t.title) > 60 else t.title
        created = t.created_at[:19]
        click.echo(f"  {t.id:<6} {t.status.value:<12} {title:<60} {created:<20}")


@cli.command()
@click.argument("task_id", type=int)
def inspect(task_id: int):
    """Inspect a specific task's details."""
    task = get_task(task_id)

    if task is None:
        click.echo(f"Task {task_id} not found.")
        sys.exit(1)

    click.echo(f"\n  Task #{task.id}")
    click.echo(f"  Status: {task.status.value}")
    click.echo(f"  Title: {task.title}")
    click.echo(f"  Description: {task.description}")
    click.echo(f"  Created: {task.created_at}")
    click.echo(f"  Updated: {task.updated_at}")


@cli.command()
def web():
    """Launch the Streamlit Web UI."""
    import subprocess
    from pathlib import Path

    web_app = Path(__file__).parent.parent.parent / "web" / "app.py"
    subprocess.run(["streamlit", "run", str(web_app)])


if __name__ == "__main__":
    cli()
