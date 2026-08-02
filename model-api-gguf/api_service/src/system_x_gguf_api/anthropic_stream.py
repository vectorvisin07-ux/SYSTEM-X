"""Messages-compatible SSE encoding over canonical stream events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from typing import Any

from fastapi import Request
from starlette.responses import StreamingResponse

from .anthropic_contract import STREAMING_VERSION, anthropic_message_id
from .anthropic_errors import compatibility_headers
from .stream_types import CanonicalStreamEvent, CanonicalStreamEventType
from .streaming_inference import CanonicalStreamSession
from .tool_contract import anthropic_call_id


ANTHROPIC_STREAMING_CONTRACT = STREAMING_VERSION


@dataclass(frozen=True, slots=True)
class MessagesStreamConfiguration:
    request_id: str
    model: str
    input_tokens: int


def _event(event_type: str, value: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"event: {event_type}\ndata: {encoded}\n\n".encode("utf-8")


def messages_ping_event() -> bytes:
    """Build the only Messages keepalive event exposed by this contract."""

    return _event("ping", {"type": "ping"})


class MessagesEventEncoder:
    """Translate canonical reasoning, text and tool channels separately."""

    def __init__(self, configuration: MessagesStreamConfiguration) -> None:
        if configuration.input_tokens < 1:
            raise ValueError("Messages stream input usage is invalid")
        self.configuration = configuration
        self.message_id = anthropic_message_id(configuration.request_id)
        self.next_block_index = 0
        self.text_index: int | None = None
        self.text_open = False
        self.reasoning_index: int | None = None
        self.reasoning_open = False
        self.tool_index: int | None = None
        self.tool_id: str | None = None
        self.tool_name: str | None = None
        self.tool_arguments: list[str] = []
        self.tool_open = False
        self.tool_done = False
        self.usage: dict[str, int] | None = None
        self.terminal = False

    def _start_reasoning(self) -> bytes:
        if self.reasoning_index is not None:
            raise RuntimeError("Messages thinking block already started")
        self.reasoning_index = self.next_block_index
        self.next_block_index += 1
        self.reasoning_open = True
        return _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.reasoning_index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )

    def _stop_reasoning(self) -> bytes | None:
        if not self.reasoning_open:
            return None
        self.reasoning_open = False
        return _event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": self.reasoning_index,
            },
        )

    def _start_text(self) -> bytes:
        if self.text_index is not None:
            raise RuntimeError("Messages text block already started")
        self.text_index = self.next_block_index
        self.next_block_index += 1
        self.text_open = True
        return _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.text_index,
                "content_block": {"type": "text", "text": ""},
            },
        )

    def _stop_text(self) -> bytes | None:
        if not self.text_open:
            return None
        self.text_open = False
        return _event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": self.text_index,
            },
        )

    def _start_tool(self, event: CanonicalStreamEvent) -> list[bytes]:
        if self.tool_index is not None:
            raise RuntimeError("parallel Messages tool use is unsupported")
        frames: list[bytes] = []
        stopped_reasoning = self._stop_reasoning()
        if stopped_reasoning is not None:
            frames.append(stopped_reasoning)
        stopped = self._stop_text()
        if stopped is not None:
            frames.append(stopped)
        tool = event.payload["tool_call"]
        self.tool_index = self.next_block_index
        self.next_block_index += 1
        self.tool_id = anthropic_call_id(tool["id"])
        self.tool_name = tool["name"]
        self.tool_open = True
        frames.append(
            _event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.tool_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": self.tool_id,
                        "name": self.tool_name,
                        "input": {},
                    },
                },
            )
        )
        return frames

    def _finish_tool(self, event: CanonicalStreamEvent) -> bytes:
        if (
            not self.tool_open
            or self.tool_index is None
            or self.tool_done
        ):
            raise RuntimeError("Messages tool lifecycle is invalid")
        accumulated = "".join(self.tool_arguments)
        try:
            if json.loads(accumulated) != event.payload["tool_call"]["arguments"]:
                raise RuntimeError("Messages tool arguments changed at completion")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Messages tool arguments are invalid JSON") from exc
        self.tool_open = False
        self.tool_done = True
        return _event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": self.tool_index,
            },
        )

    def _store_usage(self, event: CanonicalStreamEvent) -> None:
        raw = event.payload.get("usage")
        if not isinstance(raw, dict):
            raise RuntimeError("canonical Messages usage is invalid")
        if raw["input_tokens"] != self.configuration.input_tokens:
            raise RuntimeError("Messages input usage changed during generation")
        self.usage = {
            "input_tokens": raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
        }

    @staticmethod
    def _stop_reason(event: CanonicalStreamEvent) -> str:
        if event.type is CanonicalStreamEventType.REQUIRES_ACTION:
            return "tool_use"
        if event.type is CanonicalStreamEventType.INCOMPLETE:
            return (
                "model_context_window_exceeded"
                if event.payload["finish_reason"] == "context_limit"
                else "max_tokens"
            )
        if event.type is CanonicalStreamEventType.COMPLETED:
            if event.payload["finish_reason"] == "stop_sequence":
                raise RuntimeError("exact matched stop sequence is unavailable")
            return "end_turn"
        raise RuntimeError("canonical Messages terminal is invalid")

    @staticmethod
    def _error(event: CanonicalStreamEvent) -> bytes:
        error = event.payload["error"]
        return _event(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error["message"],
                },
            },
        )

    def accept(self, event: CanonicalStreamEvent) -> list[bytes]:
        if self.terminal:
            raise RuntimeError("canonical event appeared after terminal")
        if event.type is CanonicalStreamEventType.STARTED:
            return [
                _event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": self.message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.configuration.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": self.configuration.input_tokens,
                                "output_tokens": 0,
                            },
                        },
                    },
                )
            ]
        if event.type is CanonicalStreamEventType.REASONING_DELTA:
            if self.text_index is not None or self.tool_index is not None:
                raise RuntimeError(
                    "Messages reasoning cannot follow final output"
                )
            frames: list[bytes] = []
            if self.reasoning_index is None:
                frames.append(self._start_reasoning())
            frames.append(
                _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.reasoning_index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": event.payload["delta"],
                        },
                    },
                )
            )
            return frames
        if event.type is CanonicalStreamEventType.OUTPUT_TEXT_DELTA:
            if self.tool_index is not None:
                raise RuntimeError("text cannot follow a Messages tool block")
            frames: list[bytes] = []
            stopped_reasoning = self._stop_reasoning()
            if stopped_reasoning is not None:
                frames.append(stopped_reasoning)
            if self.text_index is None:
                frames.append(self._start_text())
            frames.append(
                _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.text_index,
                        "delta": {
                            "type": "text_delta",
                            "text": event.payload["delta"],
                        },
                    },
                )
            )
            return frames
        if event.type is CanonicalStreamEventType.TOOL_CALL_ADDED:
            return self._start_tool(event)
        if event.type is CanonicalStreamEventType.TOOL_CALL_ARGUMENTS_DELTA:
            if not self.tool_open or self.tool_index is None:
                raise RuntimeError("Messages tool argument preceded its block")
            delta = event.payload["delta"]
            self.tool_arguments.append(delta)
            return [
                _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.tool_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": delta,
                        },
                    },
                )
            ]
        if event.type is CanonicalStreamEventType.TOOL_CALL_DONE:
            return [self._finish_tool(event)]
        if event.type is CanonicalStreamEventType.USAGE:
            self._store_usage(event)
            return []
        if event.type is CanonicalStreamEventType.FAILED:
            self.terminal = True
            return [self._error(event)]
        if event.terminal:
            if self.usage is None:
                raise RuntimeError("canonical Messages stream omitted usage")
            if self.tool_index is not None and not self.tool_done:
                raise RuntimeError("Messages tool block ended before validation")
            frames = []
            stopped_reasoning = self._stop_reasoning()
            if stopped_reasoning is not None:
                frames.append(stopped_reasoning)
            stopped = self._stop_text()
            if stopped is not None:
                frames.append(stopped)
            try:
                stop_reason = self._stop_reason(event)
            except RuntimeError:
                self.terminal = True
                frames.append(
                    _event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": (
                                    "Exact matched stop sequence is unavailable"
                                ),
                            },
                        },
                    )
                )
                return frames
            frames.extend(
                [
                    _event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason,
                                "stop_sequence": None,
                            },
                            "usage": {
                                "output_tokens": self.usage["output_tokens"],
                            },
                        },
                    ),
                    _event("message_stop", {"type": "message_stop"}),
                ]
            )
            self.terminal = True
            return frames
        raise RuntimeError("unsupported canonical Messages event")

    def clear(self) -> None:
        self.tool_arguments.clear()
        self.usage = None


async def encode_messages_events(
    events: AsyncIterator[CanonicalStreamEvent],
    configuration: MessagesStreamConfiguration,
) -> AsyncIterator[bytes]:
    encoder = MessagesEventEncoder(configuration)
    try:
        async for canonical_event in events:
            for frame in encoder.accept(canonical_event):
                yield frame
        if not encoder.terminal:
            raise RuntimeError("canonical Messages stream ended without terminal")
    finally:
        encoder.clear()


def messages_stream_response(
    request: Request,
    session: CanonicalStreamSession,
) -> StreamingResponse:
    if session.input_tokens is None:
        raise RuntimeError("Messages input usage preflight is unavailable")
    request_id = session.control.state.request_id
    configuration = MessagesStreamConfiguration(
        request_id=request_id,
        model=session.control.state.model,
        input_tokens=session.input_tokens,
    )
    managed = session.control.managed_events(request, session.events())
    content = encode_messages_events(managed, configuration)
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers={
            **compatibility_headers(request_id),
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-System-X-Anthropic-Streaming": ANTHROPIC_STREAMING_CONTRACT,
        },
    )
