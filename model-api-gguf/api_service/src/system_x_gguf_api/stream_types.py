"""Canonical protocol-neutral stream events and request-state invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any, Iterable, Literal


REQUEST_ID_PATTERN = re.compile(r"^sx_req_[0-9a-f]{32}$")
TOOL_CALL_ID_PATTERN = re.compile(r"^sx_call_[0-9a-f]{32}$")
StreamOperation = Literal["generate", "chat", "responses"]


class StreamInvariantError(RuntimeError):
    """A canonical stream violated an internal ordering or lifecycle law."""


class StreamState(StrEnum):
    CREATED = "CREATED"
    BACKEND_OPENING = "BACKEND_OPENING"
    STREAMING = "STREAMING"
    REQUIRES_ACTION = "REQUIRES_ACTION"
    COMPLETED = "COMPLETED"
    INCOMPLETE = "INCOMPLETE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


class CanonicalStreamEventType(StrEnum):
    STARTED = "response.started"
    REASONING_DELTA = "response.reasoning.delta"
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    TOOL_CALL_ADDED = "response.tool_call.added"
    TOOL_CALL_ARGUMENTS_DELTA = "response.tool_call.arguments.delta"
    TOOL_CALL_DONE = "response.tool_call.done"
    USAGE = "response.usage"
    REQUIRES_ACTION = "response.requires_action"
    COMPLETED = "response.completed"
    INCOMPLETE = "response.incomplete"
    FAILED = "response.failed"


TERMINAL_EVENT_STATES: dict[CanonicalStreamEventType, StreamState] = {
    CanonicalStreamEventType.REQUIRES_ACTION: StreamState.REQUIRES_ACTION,
    CanonicalStreamEventType.COMPLETED: StreamState.COMPLETED,
    CanonicalStreamEventType.INCOMPLETE: StreamState.INCOMPLETE,
    CanonicalStreamEventType.FAILED: StreamState.FAILED,
}
TERMINAL_EVENT_TYPES = frozenset(TERMINAL_EVENT_STATES)


ALLOWED_STATE_TRANSITIONS: dict[StreamState, frozenset[StreamState]] = {
    StreamState.CREATED: frozenset({StreamState.BACKEND_OPENING}),
    StreamState.BACKEND_OPENING: frozenset(
        {
            StreamState.STREAMING,
            StreamState.FAILED,
            StreamState.CANCELLED,
        }
    ),
    StreamState.STREAMING: frozenset(
        {
            StreamState.REQUIRES_ACTION,
            StreamState.COMPLETED,
            StreamState.INCOMPLETE,
            StreamState.FAILED,
            StreamState.CANCELLED,
        }
    ),
    StreamState.REQUIRES_ACTION: frozenset({StreamState.CLOSED}),
    StreamState.COMPLETED: frozenset({StreamState.CLOSED}),
    StreamState.INCOMPLETE: frozenset({StreamState.CLOSED}),
    StreamState.FAILED: frozenset({StreamState.CLOSED}),
    StreamState.CANCELLED: frozenset({StreamState.CLOSED}),
    StreamState.CLOSED: frozenset(),
}


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StreamInvariantError(f"{field_name} must be non-empty text")
    return value


def _require_nonnegative_index(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise StreamInvariantError("tool-call index must be a non-negative integer")
    return value


def _validate_tool_identity(value: Any, *, complete: bool) -> None:
    if not isinstance(value, dict):
        raise StreamInvariantError("tool_call must be an object")
    required = {"id", "type", "name"}
    if complete:
        required.add("arguments")
    if set(value) != required:
        raise StreamInvariantError("tool_call has an invalid canonical shape")
    if (
        not isinstance(value.get("id"), str)
        or TOOL_CALL_ID_PATTERN.fullmatch(value["id"]) is None
        or value.get("type") != "function"
        or not isinstance(value.get("name"), str)
        or not value["name"]
    ):
        raise StreamInvariantError("tool_call identity is invalid")
    if complete and not isinstance(value.get("arguments"), dict):
        raise StreamInvariantError("completed tool arguments must be an object")


def _validate_usage(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }:
        raise StreamInvariantError("usage has an invalid canonical shape")
    if any(type(value[key]) is not int or value[key] < 0 for key in value):
        raise StreamInvariantError("usage token counts must be non-negative integers")
    if value["total_tokens"] != value["input_tokens"] + value["output_tokens"]:
        raise StreamInvariantError("usage total does not equal input plus output")


def _validate_payload(
    event_type: CanonicalStreamEventType,
    payload: dict[str, Any],
) -> None:
    if event_type is CanonicalStreamEventType.STARTED:
        if set(payload) != {"operation", "status"}:
            raise StreamInvariantError("started event has an invalid payload")
        if payload["operation"] not in {"generate", "chat", "responses"}:
            raise StreamInvariantError("started event operation is invalid")
        if payload["status"] != "in_progress":
            raise StreamInvariantError("started event status is invalid")
        return
    if event_type in {
        CanonicalStreamEventType.REASONING_DELTA,
        CanonicalStreamEventType.OUTPUT_TEXT_DELTA,
    }:
        if set(payload) != {"delta"}:
            raise StreamInvariantError("delta event has an invalid payload")
        _require_nonempty_text(payload["delta"], "delta")
        return
    if event_type is CanonicalStreamEventType.TOOL_CALL_ADDED:
        if set(payload) != {"index", "tool_call"}:
            raise StreamInvariantError("tool-call added event has an invalid payload")
        _require_nonnegative_index(payload["index"])
        _validate_tool_identity(payload["tool_call"], complete=False)
        return
    if event_type is CanonicalStreamEventType.TOOL_CALL_ARGUMENTS_DELTA:
        if set(payload) != {"index", "tool_call_id", "delta"}:
            raise StreamInvariantError(
                "tool-call argument delta event has an invalid payload"
            )
        _require_nonnegative_index(payload["index"])
        if (
            not isinstance(payload["tool_call_id"], str)
            or TOOL_CALL_ID_PATTERN.fullmatch(payload["tool_call_id"]) is None
        ):
            raise StreamInvariantError("tool-call argument identity is invalid")
        _require_nonempty_text(payload["delta"], "delta")
        return
    if event_type is CanonicalStreamEventType.TOOL_CALL_DONE:
        if set(payload) != {"index", "tool_call"}:
            raise StreamInvariantError("tool-call done event has an invalid payload")
        _require_nonnegative_index(payload["index"])
        _validate_tool_identity(payload["tool_call"], complete=True)
        return
    if event_type is CanonicalStreamEventType.USAGE:
        if set(payload) != {"usage"}:
            raise StreamInvariantError("usage event has an invalid payload")
        _validate_usage(payload["usage"])
        return
    expected_status = {
        CanonicalStreamEventType.REQUIRES_ACTION: "requires_action",
        CanonicalStreamEventType.COMPLETED: "completed",
        CanonicalStreamEventType.INCOMPLETE: "incomplete",
        CanonicalStreamEventType.FAILED: "failed",
    }.get(event_type)
    required = {"status", "finish_reason"}
    if event_type is CanonicalStreamEventType.FAILED:
        required.add("error")
    if expected_status is None or set(payload) != required:
        raise StreamInvariantError("terminal event has an invalid payload")
    if payload["status"] != expected_status:
        raise StreamInvariantError("terminal event status is invalid")
    _require_nonempty_text(payload["finish_reason"], "finish_reason")
    if event_type is CanonicalStreamEventType.FAILED and not isinstance(
        payload["error"], dict
    ):
        raise StreamInvariantError("failed event error must be an object")


@dataclass(frozen=True, slots=True)
class CanonicalStreamEvent:
    """One validated event consumed by every public protocol encoder."""

    type: CanonicalStreamEventType
    sequence: int
    request_id: str
    model: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise StreamInvariantError("event sequence must be non-negative")
        if REQUEST_ID_PATTERN.fullmatch(self.request_id) is None:
            raise StreamInvariantError("event request ID is invalid")
        _require_nonempty_text(self.model, "model")
        if not isinstance(self.payload, dict):
            raise StreamInvariantError("event payload must be an object")
        _validate_payload(self.type, self.payload)

    @property
    def terminal(self) -> bool:
        return self.type in TERMINAL_EVENT_TYPES

    def system_x_data(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "sequence": self.sequence,
            "request_id": self.request_id,
            "model": self.model,
            **self.payload,
        }


@dataclass(slots=True)
class ActiveStreamState:
    """Bounded per-request lifecycle state; never stores prompt or output text."""

    request_id: str
    endpoint: str
    model: str
    state: StreamState = StreamState.CREATED
    next_sequence: int = 0
    terminal_event: CanonicalStreamEventType | None = None
    upstream_attached: bool = False
    upstream_closed: bool = False
    cancellation_source: str | None = None
    cleanup_complete: bool = False
    _event_types: list[CanonicalStreamEventType] = field(default_factory=list)

    def __post_init__(self) -> None:
        if REQUEST_ID_PATTERN.fullmatch(self.request_id) is None:
            raise StreamInvariantError("active stream request ID is invalid")
        _require_nonempty_text(self.endpoint, "endpoint")
        _require_nonempty_text(self.model, "model")

    @property
    def event_count(self) -> int:
        return len(self._event_types)

    def transition(self, target: StreamState) -> None:
        if target not in ALLOWED_STATE_TRANSITIONS[self.state]:
            raise StreamInvariantError(
                f"illegal stream transition {self.state.value}->{target.value}"
            )
        self.state = target

    def begin_backend(self) -> None:
        self.transition(StreamState.BACKEND_OPENING)

    def attach_upstream(self) -> None:
        if self.state is not StreamState.BACKEND_OPENING:
            raise StreamInvariantError("upstream may only attach while opening")
        if self.upstream_attached:
            raise StreamInvariantError("one stream may have only one upstream response")
        self.upstream_attached = True

    def start(self, operation: StreamOperation) -> CanonicalStreamEvent:
        if self.state is not StreamState.BACKEND_OPENING:
            raise StreamInvariantError("stream may only start after backend opening")
        self.transition(StreamState.STREAMING)
        return self.emit(
            CanonicalStreamEventType.STARTED,
            operation=operation,
            status="in_progress",
        )

    def emit(
        self,
        event_type: CanonicalStreamEventType,
        **payload: Any,
    ) -> CanonicalStreamEvent:
        if self.state is not StreamState.STREAMING:
            raise StreamInvariantError("events may only be emitted while streaming")
        if self.terminal_event is not None:
            raise StreamInvariantError("no event may follow a terminal event")
        if not self._event_types and event_type is not CanonicalStreamEventType.STARTED:
            raise StreamInvariantError("response.started must be the first event")
        if self._event_types and event_type is CanonicalStreamEventType.STARTED:
            raise StreamInvariantError("response.started may be emitted only once")
        event = CanonicalStreamEvent(
            type=event_type,
            sequence=self.next_sequence,
            request_id=self.request_id,
            model=self.model,
            payload=dict(payload),
        )
        self.next_sequence += 1
        self._event_types.append(event_type)
        terminal_state = TERMINAL_EVENT_STATES.get(event_type)
        if terminal_state is not None:
            self.terminal_event = event_type
            self.transition(terminal_state)
        return event

    def fail_before_start(self) -> None:
        if self.state is not StreamState.BACKEND_OPENING:
            raise StreamInvariantError("pre-stream failure requires backend opening")
        self.transition(StreamState.FAILED)

    def cancel(self, source: str) -> None:
        if self.state not in {StreamState.BACKEND_OPENING, StreamState.STREAMING}:
            raise StreamInvariantError("only an active stream may be cancelled")
        self.cancellation_source = _require_nonempty_text(
            source, "cancellation_source"
        )
        self.transition(StreamState.CANCELLED)

    def mark_upstream_closed(self) -> None:
        if not self.upstream_attached:
            raise StreamInvariantError("no upstream response is attached")
        if self.upstream_closed:
            raise StreamInvariantError("upstream response was already closed")
        self.upstream_closed = True

    def close(self) -> None:
        if self.upstream_attached and not self.upstream_closed:
            raise StreamInvariantError("upstream response must close before stream state")
        if self.state in {
            StreamState.REQUIRES_ACTION,
            StreamState.COMPLETED,
            StreamState.INCOMPLETE,
            StreamState.FAILED,
        } and self.state is not StreamState.FAILED and self.terminal_event is None:
            raise StreamInvariantError("normal stream is missing its terminal event")
        self.transition(StreamState.CLOSED)
        self.cleanup_complete = True
        self._event_types.clear()


def validate_canonical_event_sequence(
    events: Iterable[CanonicalStreamEvent],
    *,
    require_terminal: bool = True,
) -> None:
    """Validate a public-normal sequence or a disconnected partial sequence."""

    materialized = list(events)
    if not materialized:
        raise StreamInvariantError("event sequence is empty")
    first = materialized[0]
    if first.type is not CanonicalStreamEventType.STARTED:
        raise StreamInvariantError("response.started must be first")
    request_id = first.request_id
    model = first.model
    terminal_seen = False
    for expected_sequence, event in enumerate(materialized):
        if event.sequence != expected_sequence:
            raise StreamInvariantError("event sequence is not strictly contiguous")
        if event.request_id != request_id or event.model != model:
            raise StreamInvariantError("event identity changed within one stream")
        if terminal_seen:
            raise StreamInvariantError("event appeared after terminal")
        terminal_seen = event.terminal
    if require_terminal and not terminal_seen:
        raise StreamInvariantError("normal stream is missing a terminal event")
