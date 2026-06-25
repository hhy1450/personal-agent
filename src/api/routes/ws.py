"""WebSocket route — real-time workflow execution streaming."""
import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.events.bus import get_event_bus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def task_websocket(websocket: WebSocket, task_id: int):
    """WebSocket endpoint for real-time task execution updates.

    Clients connect here to receive a stream of WorkflowEvent messages
    as the workflow executes. Messages are JSON objects with at least
    a "type" field.

    Message types: workflow_start, step_start, step_done, step_review,
    step_retry, tool_call, tool_result, error, workflow_done.
    """
    await websocket.accept()
    logger.info("WebSocket connected for task %d", task_id)

    bus = get_event_bus()

    # Send initial connection confirmation
    await websocket.send_json({
        "type": "connected",
        "task_id": task_id,
        "message": "Waiting for workflow to start...",
    })

    try:
        async for event in bus.subscribe(task_id=task_id):
            try:
                await websocket.send_json(event.to_dict())
            except Exception as e:
                logger.warning("Failed to send event to WebSocket: %s", e)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for task %d", task_id)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("WebSocket error for task %d: %s", task_id, e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/global")
async def global_websocket(websocket: WebSocket):
    """WebSocket endpoint for all workflow events (debug/monitoring).

    Receives events from ALL active workflows. Useful for admin dashboards.
    """
    await websocket.accept()
    logger.info("Global WebSocket connected")

    bus = get_event_bus()

    await websocket.send_json({
        "type": "connected",
        "message": "Subscribed to all workflow events",
    })

    try:
        async for event in bus.subscribe(task_id=None):
            try:
                await websocket.send_json(event.to_dict())
            except Exception as e:
                logger.warning("Failed to send global event: %s", e)
                break
    except WebSocketDisconnect:
        logger.info("Global WebSocket disconnected")
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
