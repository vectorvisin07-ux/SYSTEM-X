"""Typed OpenAI Responses SSE events over the canonical stream."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import time
from typing import Any

from fastapi import Request
from starlette.responses import StreamingResponse

from .openai_errors import compatibility_headers
from .openai_stream import OPENAI_STREAMING_CONTRACT
from .stream_types import CanonicalStreamEvent, CanonicalStreamEventType
from .streaming_inference import CanonicalStreamSession
from .tool_contract import openai_call_id
from .tool_schema import canonical_json


@dataclass(frozen=True, slots=True)
class ResponsesStreamConfiguration:
    request_id: str
    model: str
    max_output_tokens: int
    temperature: float | None
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any]


def _event(value: dict[str, Any]) -> bytes:
    event_type = value["type"]
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"event: {event_type}\ndata: {encoded}\n\n".encode("utf-8")


class ResponsesEventEncoder:
    """Expand canonical channels into the official typed Responses lifecycle."""

    def __init__(self, configuration: ResponsesStreamConfiguration) -> None:
        self.configuration = configuration
        suffix = configuration.request_id.removeprefix("sx_req_")
        self.response_id = f"resp_sx_{suffix}"
        self.created_at = float(int(time.time()))
        self.sequence = 0
        self.output_items: dict[int, dict[str, Any]] = {}
        self.reasoning_index: int | None = None
        self.reasoning_item_id = f"rs_sx_{suffix}"
        self.reasoning_parts: list[str] = []
        self.text_index: int | None = None
        self.text_item_id = f"msg_sx_{suffix}"
        self.text_parts: list[str] = []
        self.tool_index: int | None = None
        self.tool_item_id: str | None = None
        self.tool_call_id: str | None = None
        self.tool_name: str | None = None
        self.tool_argument_parts: list[str] = []
        self.tool_done = False
        self.usage: dict[str, int] | None = None
        self.terminal = False

    def _next(self, event_type: str, **fields: Any) -> bytes:
        value = {
            "type": event_type,
            "sequence_number": self.sequence,
            **fields,
        }
        self.sequence += 1
        return _event(value)

    def _next_output_index(self) -> int:
        return len(self.output_items)

    def _response(
        self,
        *,
        status: str,
        usage: dict[str, int] | None,
        incomplete_details: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.configuration.model,
            "output": [
                self.output_items[index]
                for index in sorted(self.output_items)
            ],
            "parallel_tool_calls": False,
            "tool_choice": self.configuration.tool_choice,
            "tools": self.configuration.tools,
            "max_output_tokens": self.configuration.max_output_tokens,
            "temperature": self.configuration.temperature,
            "usage": usage,
            "incomplete_details": incomplete_details,
            "error": None,
        }
        if status in {"completed", "incomplete"}:
            value["completed_at"] = float(int(time.time()))
        return value

    def _start_reasoning(self) -> bytes:
        if self.reasoning_index is not None:
            raise RuntimeError("reasoning item already started")
        index = self._next_output_index()
        self.reasoning_index = index
        item = {
            "id": self.reasoning_item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
            "content": [],
            "encrypted_content": "",
        }
        self.output_items[index] = item
        return self._next(
            "response.output_item.added",
            output_index=index,
            item=item,
        )

    def _start_text(self) -> list[bytes]:
        if self.text_index is not None:
            raise RuntimeError("text item already started")
        index = self._next_output_index()
        self.text_index = index
        item = {
            "id": self.text_item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        self.output_items[index] = item
        return [
            self._next(
                "response.output_item.added",
                output_index=index,
                item=item,
            ),
            self._next(
                "response.content_part.added",
                output_index=index,
                content_index=0,
                item_id=self.text_item_id,
                part={
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                    "logprobs": [],
                },
            ),
        ]

    def _start_tool(self, event: CanonicalStreamEvent) -> bytes:
        if self.tool_index is not None:
            raise RuntimeError("parallel function calls are unsupported")
        tool = event.payload["tool_call"]
        index = self._next_output_index()
        self.tool_index = index
        self.tool_call_id = openai_call_id(tool["id"])
        self.tool_item_id = f"fc_sx_{tool['id'].removeprefix('sx_call_')}"
        self.tool_name = tool["name"]
        item = {
            "id": self.tool_item_id,
            "type": "function_call",
            "status": "in_progress",
            "call_id": self.tool_call_id,
            "name": self.tool_name,
            "arguments": "",
        }
        self.output_items[index] = item
        return self._next(
            "response.output_item.added",
            output_index=index,
            item=item,
        )

    def _finish_reasoning(self, status: str) -> list[bytes]:
        if self.reasoning_index is None:
            return []
        text = "".join(self.reasoning_parts)
        item = {
            "id": self.reasoning_item_id,
            "type": "reasoning",
            "status": status,
            "summary": [],
            "content": [{"type": "reasoning_text", "text": text}],
            "encrypted_content": "",
        }
        self.output_items[self.reasoning_index] = item
        return [
            self._next(
                "response.reasoning_text.done",
                output_index=self.reasoning_index,
                content_index=0,
                item_id=self.reasoning_item_id,
                text=text,
            ),
            self._next(
                "response.output_item.done",
                output_index=self.reasoning_index,
                item=item,
            ),
        ]

    def _finish_text(self, status: str) -> list[bytes]:
        if self.text_index is None:
            return []
        text = "".join(self.text_parts)
        part = {
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        }
        item = {
            "id": self.text_item_id,
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [part],
        }
        self.output_items[self.text_index] = item
        return [
            self._next(
                "response.output_text.done",
                output_index=self.text_index,
                content_index=0,
                item_id=self.text_item_id,
                text=text,
                logprobs=[],
            ),
            self._next(
                "response.content_part.done",
                output_index=self.text_index,
                content_index=0,
                item_id=self.text_item_id,
                part=part,
            ),
            self._next(
                "response.output_item.done",
                output_index=self.text_index,
                item=item,
            ),
        ]

    def _finish_tool(self, event: CanonicalStreamEvent) -> list[bytes]:
        if (
            self.tool_index is None
            or self.tool_item_id is None
            or self.tool_call_id is None
            or self.tool_name is None
            or self.tool_done
        ):
            raise RuntimeError("function call lifecycle is invalid")
        tool = event.payload["tool_call"]
        arguments = canonical_json(tool["arguments"])
        accumulated = "".join(self.tool_argument_parts)
        try:
            if json.loads(accumulated) != tool["arguments"]:
                raise RuntimeError("function argument deltas changed at completion")
        except json.JSONDecodeError as exc:
            raise RuntimeError("function argument deltas are invalid JSON") from exc
        item = {
            "id": self.tool_item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": self.tool_call_id,
            "name": self.tool_name,
            "arguments": arguments,
        }
        self.output_items[self.tool_index] = item
        self.tool_done = True
        return [
            self._next(
                "response.function_call_arguments.done",
                output_index=self.tool_index,
                item_id=self.tool_item_id,
                name=self.tool_name,
                arguments=arguments,
            ),
            self._next(
                "response.output_item.done",
                output_index=self.tool_index,
                item=item,
            ),
        ]

    def _usage(self, event: CanonicalStreamEvent) -> None:
        raw = event.payload.get("usage")
        if not isinstance(raw, dict):
            raise RuntimeError("canonical usage is invalid")
        self.usage = {
            "input_tokens": raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
            "total_tokens": raw["total_tokens"],
        }

    def _successful_terminal(
        self, event: CanonicalStreamEvent
    ) -> list[bytes]:
        if self.usage is None:
            raise RuntimeError("canonical Responses stream omitted usage")
        if self.tool_index is not None and not self.tool_done:
            raise RuntimeError("function call ended before argument validation")
        if event.type is CanonicalStreamEventType.INCOMPLETE:
            status = "incomplete"
            detail = {"reason": "max_output_tokens"}
            terminal_type = "response.incomplete"
        else:
            status = "completed"
            detail = None
            terminal_type = "response.completed"
        frames = [
            *self._finish_reasoning(status),
            *self._finish_text(status),
        ]
        frames.append(
            self._next(
                terminal_type,
                response=self._response(
                    status=status,
                    usage=self.usage,
                    incomplete_details=detail,
                ),
            )
        )
        self.terminal = True
        return frames

    def accept(self, event: CanonicalStreamEvent) -> list[bytes]:
        if self.terminal:
            raise RuntimeError("canonical event appeared after terminal")
        if event.type is CanonicalStreamEventType.STARTED:
            return [
                self._next(
                    "response.created",
                    response=self._response(
                        status="in_progress",
                        usage=None,
                    ),
                )
            ]
        if event.type is CanonicalStreamEventType.REASONING_DELTA:
            frames = []
            if self.reasoning_index is None:
                frames.append(self._start_reasoning())
            delta = event.payload["delta"]
            self.reasoning_parts.append(delta)
            frames.append(
                self._next(
                    "response.reasoning_text.delta",
                    output_index=self.reasoning_index,
                    content_index=0,
                    item_id=self.reasoning_item_id,
                    delta=delta,
                )
            )
            return frames
        if event.type is CanonicalStreamEventType.OUTPUT_TEXT_DELTA:
            frames = []
            if self.text_index is None:
                frames.extend(self._start_text())
            delta = event.payload["delta"]
            self.text_parts.append(delta)
            frames.append(
                self._next(
                    "response.output_text.delta",
                    output_index=self.text_index,
                    content_index=0,
                    item_id=self.text_item_id,
                    delta=delta,
                    logprobs=[],
                )
            )
            return frames
        if event.type is CanonicalStreamEventType.TOOL_CALL_ADDED:
            return [self._start_tool(event)]
        if event.type is CanonicalStreamEventType.TOOL_CALL_ARGUMENTS_DELTA:
            if self.tool_index is None or self.tool_item_id is None:
                raise RuntimeError("function argument delta preceded its item")
            delta = event.payload["delta"]
            self.tool_argument_parts.append(delta)
            return [
                self._next(
                    "response.function_call_arguments.delta",
                    output_index=self.tool_index,
                    item_id=self.tool_item_id,
                    delta=delta,
                )
            ]
        if event.type is CanonicalStreamEventType.TOOL_CALL_DONE:
            return self._finish_tool(event)
        if event.type is CanonicalStreamEventType.USAGE:
            self._usage(event)
            return []
        if event.type is CanonicalStreamEventType.FAILED:
            raw = event.payload["error"]
            self.terminal = True
            return [
                self._next(
                    "error",
                    code=raw["code"],
                    message=raw["message"],
                    param=None,
                )
            ]
        if event.terminal:
            return self._successful_terminal(event)
        raise RuntimeError("unsupported canonical Responses event")

    def clear(self) -> None:
        self.reasoning_parts.clear()
        self.text_parts.clear()
        self.tool_argument_parts.clear()
        self.output_items.clear()
        self.usage = None


async def encode_responses_events(
    events: AsyncIterator[CanonicalStreamEvent],
    configuration: ResponsesStreamConfiguration,
) -> AsyncIterator[bytes]:
    encoder = ResponsesEventEncoder(configuration)
    try:
        async for canonical_event in events:
            for frame in encoder.accept(canonical_event):
                yield frame
        if not encoder.terminal:
            raise RuntimeError("canonical Responses stream ended without terminal")
    finally:
        encoder.clear()


def responses_stream_response(
    request: Request,
    session: CanonicalStreamSession,
    configuration: ResponsesStreamConfiguration,
) -> StreamingResponse:
    managed = session.control.managed_events(request, session.events())
    content = encode_responses_events(managed, configuration)
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers={
            **compatibility_headers(configuration.request_id),
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-System-X-OpenAI-Streaming": OPENAI_STREAMING_CONTRACT,
        },
    )
