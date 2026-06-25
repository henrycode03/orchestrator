"""In-process orchestration event bus.

Provides sub-second push delivery for orchestration events to active WebSocket
subscribers.  JSONL journal remains the authoritative store; this bus is an
additional delivery path only.

Threading model
---------------
``publish`` may be called from any thread (including Celery worker threads that
run in the same process as FastAPI in development/test mode).  Subscribers live
in the asyncio event loop.  The loop reference is captured on the first
``subscribe`` call and used by ``publish`` to schedule safe cross-thread
delivery via ``loop.call_soon_threadsafe``.

If no asyncio event loop is running at subscription time, or if ``publish`` is
called from a separate process (e.g. a Celery worker process), delivery is
silently skipped.  JSONL polling in the stream service acts as fallback.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 256


class OrchestrationEventBus:
    def __init__(self, maxsize: int = _QUEUE_MAXSIZE) -> None:
        self._queues: Dict[int, List[asyncio.Queue]] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self, session_id: int) -> asyncio.Queue:
        """Register a new subscriber queue for *session_id*.

        Must be called from within a running asyncio event loop.
        Returns the queue; caller awaits items from it.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        with self._lock:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            if session_id not in self._queues:
                self._queues[session_id] = []
            self._queues[session_id].append(q)
        return q

    def unsubscribe(self, session_id: int, queue: asyncio.Queue) -> None:
        """Remove *queue* from the subscriber list for *session_id*."""
        with self._lock:
            subscribers = self._queues.get(session_id, [])
            try:
                subscribers.remove(queue)
            except ValueError:
                pass
            if not subscribers:
                self._queues.pop(session_id, None)

    def publish(self, event: dict) -> None:
        """Deliver *event* to all active subscribers for its session.

        Best-effort: full queues are silently dropped; exceptions are swallowed.
        Never raises.  Safe to call from any thread.
        """
        session_id = event.get("session_id")
        if not isinstance(session_id, int):
            return
        with self._lock:
            queues = list(self._queues.get(session_id, []))
            loop = self._loop
        if not queues or loop is None or not loop.is_running():
            return
        for q in queues:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

    def subscriber_count(self, session_id: int) -> int:
        """Return the number of active subscribers for *session_id*."""
        with self._lock:
            return len(self._queues.get(session_id, []))


orchestration_event_bus = OrchestrationEventBus()
