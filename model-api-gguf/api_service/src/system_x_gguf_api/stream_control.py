"""Request-scoped disconnect, cancellation, cleanup, and evidence control."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import logging
from typing import AsyncIterator, Protocol

from .operation_records import OperationRecorder
from .router_client import PrivateRouterStream
from .stream_types import (
    ActiveStreamState,
    CanonicalStreamEvent,
    StreamInvariantError,
    StreamState,
)


LOGGER = logging.getLogger("uvicorn.error")
ACTIVE_STATES = frozenset({StreamState.BACKEND_OPENING, StreamState.STREAMING})


class DisconnectProbe(Protocol):
    async def is_disconnected(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class CancellationEvidence:
    request_id: str
    endpoint: str
    detection_source: str
    backend_response_closed: bool
    producer_cancelled: bool
    stream_state: str
    cleanup_complete: bool

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "detection_source": self.detection_source,
            "backend_response_closed": self.backend_response_closed,
            "producer_cancelled": self.producer_cancelled,
            "stream_state": self.stream_state,
            "cleanup_complete": self.cleanup_complete,
        }


class ActiveStreamRegistry:
    """Own bounded controls and cancellation evidence for this service lifespan."""

    def __init__(
        self,
        operations: OperationRecorder,
        *,
        maximum_evidence: int = 256,
    ) -> None:
        if type(maximum_evidence) is not int or not 1 <= maximum_evidence <= 4096:
            raise ValueError("stream evidence bound is invalid")
        self.operations = operations
        self._active: dict[str, StreamControl] = {}
        self._evidence: deque[CancellationEvidence] = deque(maxlen=maximum_evidence)
        self._lock = asyncio.Lock()
        self._shutting_down = False

    async def register(self, state: ActiveStreamState) -> "StreamControl":
        async with self._lock:
            if self._shutting_down:
                raise RuntimeError("stream registry is shutting down")
            if state.request_id in self._active:
                raise StreamInvariantError("stream request ID is already active")
            control = StreamControl(state, self)
            self._active[state.request_id] = control
            return control

    async def _remove(self, control: "StreamControl") -> None:
        async with self._lock:
            current = self._active.get(control.state.request_id)
            if current is not control:
                raise StreamInvariantError("active stream ownership changed")
            del self._active[control.state.request_id]

    async def _record(self, evidence: CancellationEvidence) -> None:
        async with self._lock:
            self._evidence.append(evidence)
        LOGGER.info(
            "system_x_stream_cancellation %s",
            json.dumps(evidence.as_dict(), sort_keys=True, separators=(",", ":")),
        )

    async def active_request_ids(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(sorted(self._active))

    async def evidence_for(
        self, request_id: str
    ) -> CancellationEvidence | None:
        async with self._lock:
            for evidence in reversed(self._evidence):
                if evidence.request_id == request_id:
                    return evidence
        return None

    async def shutdown(self) -> None:
        async with self._lock:
            self._shutting_down = True
            controls = tuple(self._active.values())
        await asyncio.gather(
            *(
                control.cancel("service_shutdown", cancel_producer=True)
                for control in controls
            ),
            return_exceptions=False,
        )
        tasks = [
            control.producer_task
            for control in controls
            if control.producer_task is not None
            and control.producer_task is not asyncio.current_task()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(
                control.finalize("service_shutdown")
                for control in controls
            ),
            return_exceptions=False,
        )
        if await self.active_request_ids():
            raise StreamInvariantError("stream controls remained after shutdown")


class StreamControl:
    """Coordinate one public stream and its one private response."""

    def __init__(
        self,
        state: ActiveStreamState,
        registry: ActiveStreamRegistry,
    ) -> None:
        self.state = state
        self.registry = registry
        self.producer_task: asyncio.Task[object] | None = None
        self.disconnect_task: asyncio.Task[None] | None = None
        self.upstream: PrivateRouterStream | None = None
        self.cancel_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._upstream_close_claimed = False
        self._finalize_claimed = False
        self._producer_cancelled_observed = False

    async def bind_producer(self) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("stream producer task is unavailable")
        async with self._lock:
            if self.producer_task is not None and self.producer_task is not task:
                raise StreamInvariantError("stream producer task changed")
            self.producer_task = task

    async def attach_upstream(self, upstream: PrivateRouterStream) -> None:
        async with self._lock:
            if self.upstream is not None:
                raise StreamInvariantError("stream already has an upstream response")
            self.state.attach_upstream()
            self.upstream = upstream

    async def _close_upstream_once(self) -> None:
        async with self._lock:
            if self.upstream is None or self._upstream_close_claimed:
                return
            self._upstream_close_claimed = True
            upstream = self.upstream
        await upstream.aclose()
        async with self._lock:
            self.state.mark_upstream_closed()

    async def _mark_producer_cancelled(self) -> None:
        async with self._lock:
            self._producer_cancelled_observed = True

    async def cancel(
        self,
        source: str,
        *,
        cancel_producer: bool,
    ) -> None:
        async with self._lock:
            if self.state.state in ACTIVE_STATES:
                self.state.cancel(source)
                self.cancel_event.set()
            producer = self.producer_task
        await self._close_upstream_once()
        if (
            cancel_producer
            and producer is not None
            and producer is not asyncio.current_task()
            and not producer.done()
        ):
            producer.cancel()

    async def _monitor_disconnect(
        self,
        request: DisconnectProbe,
        poll_interval_seconds: float,
    ) -> None:
        try:
            while not self.cancel_event.is_set():
                if await request.is_disconnected():
                    await self.cancel(
                        "asgi_disconnect",
                        cancel_producer=True,
                    )
                    return
                await asyncio.sleep(poll_interval_seconds)
        except asyncio.CancelledError:
            raise

    async def finalize(self, default_source: str) -> None:
        async with self._lock:
            if self._finalize_claimed:
                return
            self._finalize_claimed = True
        monitor = self.disconnect_task
        if (
            monitor is not None
            and monitor is not asyncio.current_task()
            and not monitor.done()
        ):
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        if self.state.state in ACTIVE_STATES:
            await self.cancel(default_source, cancel_producer=False)
        await self._close_upstream_once()
        cancellation_source = self.state.cancellation_source
        terminal_state = self.state.state
        self.state.close()
        await self.registry._remove(self)
        if terminal_state is StreamState.CANCELLED:
            self.registry.operations.note_cancelled_if_active(
                self.state.request_id
            )
        if cancellation_source is not None:
            evidence = CancellationEvidence(
                request_id=self.state.request_id,
                endpoint=self.state.endpoint,
                detection_source=cancellation_source,
                backend_response_closed=(
                    not self.state.upstream_attached
                    or self.state.upstream_closed
                ),
                producer_cancelled=self._producer_cancelled_observed,
                stream_state=terminal_state.value,
                cleanup_complete=self.state.cleanup_complete,
            )
            await self.registry._record(evidence)

    async def managed_events(
        self,
        request: DisconnectProbe,
        events: AsyncIterator[CanonicalStreamEvent],
        *,
        poll_interval_seconds: float = 0.05,
    ) -> AsyncIterator[CanonicalStreamEvent]:
        if not 0.005 <= poll_interval_seconds <= 1.0:
            raise ValueError("disconnect poll interval is out of bounds")
        await self.bind_producer()
        self.disconnect_task = asyncio.create_task(
            self._monitor_disconnect(request, poll_interval_seconds),
            name=f"system-x-disconnect-{self.state.request_id}",
        )
        default_source = "generator_close"
        try:
            async for event in events:
                if await request.is_disconnected():
                    default_source = "before_next_event"
                    await self._mark_producer_cancelled()
                    await self.cancel(default_source, cancel_producer=False)
                    return
                yield event
            default_source = "upstream_ended_without_terminal"
        except asyncio.CancelledError:
            default_source = "task_cancelled"
            await self._mark_producer_cancelled()
            await asyncio.shield(
                self.cancel(default_source, cancel_producer=False)
            )
            raise
        except GeneratorExit:
            default_source = "generator_close"
            await asyncio.shield(
                self.cancel(default_source, cancel_producer=False)
            )
            raise
        except (BrokenPipeError, ConnectionError, OSError):
            default_source = "send_failure"
            await self._mark_producer_cancelled()
            await asyncio.shield(
                self.cancel(default_source, cancel_producer=False)
            )
            raise
        finally:
            await asyncio.shield(self.finalize(default_source))
