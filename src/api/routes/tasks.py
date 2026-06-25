"""Task API routes — submit, list, inspect tasks."""
import asyncio
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.llm.factory import get_llm_factory
from src.engine.graph import run_workflow
from src.storage.database import (
    create_task, get_task, list_tasks, update_task_status,
    create_workflow_run, update_workflow_run,
)
from src.storage.models import TaskStatus
from src.events.bus import get_event_bus
from src.events.models import WorkflowEvent, EventType

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/tasks")
async def submit_task(request: dict, background_tasks: BackgroundTasks):
    """Submit a new task for execution.

    Body: {"task": "your task description", "strategy": "sequential"}

    Returns immediately with task_id. Execution runs in background.
    WebSocket clients can subscribe to /ws/tasks/{task_id} for live updates.
    """
    task_text = request.get("task", "").strip()
    if not task_text:
        raise HTTPException(status_code=400, detail="Task description is required")

    strategy = request.get("strategy", "sequential")

    # Create DB records
    db_task = create_task(title=task_text[:100], description=task_text)
    update_task_status(db_task.id, TaskStatus.RUNNING)
    run_record = create_workflow_run(db_task.id)

    # Run workflow in background
    background_tasks.add_task(
        _execute_workflow,
        task_id=db_task.id,
        run_id=run_record.id,
        task_text=task_text,
        strategy=strategy,
    )

    return {
        "task_id": db_task.id,
        "status": "running",
        "strategy": strategy,
        "ws_url": f"/ws/tasks/{db_task.id}",
    }


@router.get("/tasks/{task_id}")
async def get_task_info(task_id: int):
    """Get task details and latest execution result."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@router.get("/tasks")
async def list_tasks_endpoint(limit: int = 20, offset: int = 0):
    """List recent tasks."""
    tasks_list = list_tasks(limit=limit, offset=offset)
    return [t.to_dict() for t in tasks_list]


async def _execute_workflow(task_id: int, run_id: int, task_text: str, strategy: str):
    """Background task: run workflow and emit events."""
    bus = get_event_bus()
    factory = get_llm_factory()

    # Emit start event
    bus.emit(WorkflowEvent(
        type=EventType.WORKFLOW_START,
        task_id=task_id,
        description=task_text[:100],
        data={"strategy": strategy},
    ))

    try:
        result = run_workflow(factory, task_text, strategy=strategy)

        # Emit completion
        bus.emit(WorkflowEvent(
            type=EventType.WORKFLOW_DONE,
            task_id=task_id,
            data={
                "final_output": result.get("final_output", "")[:500],
                "steps_completed": len(result.get("results", {})),
                "errors": len(result.get("errors", [])),
            },
        ))

        update_task_status(task_id, TaskStatus.COMPLETED)
        update_workflow_run(run_id, str(result), TaskStatus.COMPLETED)

    except Exception as e:
        logger.error("Workflow failed for task %d: %s", task_id, e)
        bus.emit(WorkflowEvent(
            type=EventType.ERROR,
            task_id=task_id,
            description=str(e),
        ))
        update_task_status(task_id, TaskStatus.FAILED)
        update_workflow_run(run_id, "{}", TaskStatus.FAILED)

    # Small delay to let consumers receive last events
    await asyncio.sleep(0.5)
