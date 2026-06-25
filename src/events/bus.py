"""EventBus — pub/sub system for workflow execution events.

A lightweight in-process event bus that connects LangGraph nodes
(producers) to WebSocket handlers (consumers) without tight coupling.

Usage (producer — in graph nodes):
    bus = get_event_bus()
    bus.emit(WorkflowEvent(type="step_done", step=0, agent="researcher"))

Usage (consumer — in WebSocket handler):
    bus = get_event_bus()
    async for event in bus.subscribe(task_id=42):
        await websocket.send_json(event.to_dict())
"""
import asyncio
import logging
from collections import defaultdict

from src.events.models import WorkflowEvent

logger = logging.getLogger(__name__)


class EventBus:
    """In-process pub/sub event bus for workflow events.

    Supports both synchronous emit (from graph nodes) and async
    subscription (from WebSocket handlers).

    Events are routed by task_id so multiple concurrent workflows
    don't interfere with each other.
    """

    def __init__(self):
        # task_id → list of asyncio.Queue
        self._queues: dict[int, list[asyncio.Queue]] = defaultdict(list)
        # Global subscribers (receive all events)
        self._global_queues: list[asyncio.Queue] = []

    def emit(self, event: WorkflowEvent):
        """Emit an event to all subscribers for its task_id.

        Non-blocking — queues overflow is handled by dropping oldest
        events if a consumer is too slow.
        """
        task_id = event.task_id
        queues = self._global_queues.copy()

        if task_id is not None and task_id in self._queues:
            queues.extend(self._queues[task_id])

        if not queues:
            return  # No subscribers — event is silently dropped

        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest to make room
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    async def subscribe(self, task_id: int | None = None):
        """Async generator that yields events for a specific task_id.

        If task_id is None, subscribes to all events (global).

        Usage:
            async for event in bus.subscribe(task_id=42):
                await websocket.send_json(event.to_dict())
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        if task_id is None:
            self._global_queues.append(queue)
        else:
            self._queues[task_id].append(queue)

        try:
            while True:
                event = await queue.get()
                yield event
        except asyncio.CancelledError:
            pass
        finally:
            # Cleanup
            if task_id is None:
                if queue in self._global_queues:
                    self._global_queues.remove(queue)
            else:
                if task_id in self._queues:
                    queues = self._queues[task_id]
                    if queue in queues:
                        queues.remove(queue)
                    if not queues:
                        del self._queues[task_id]


# Global singleton
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global EventBus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
