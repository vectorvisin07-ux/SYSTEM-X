"""OpenAI Completion and Chat SSE encoders over canonical events."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
import time
from typing import Any

from fastapi import Request
from starlette.responses import StreamingResponse

from .openai_contract import STREAMING_VERSION
from .openai_errors import compatibility_headers
from .stream_types import CanonicalStreamEvent, CanonicalStreamEventType
from .streaming_inference import CanonicalStreamSession
from .tool_contract import openai_call_id


OPENAI_STREAMING_CONTRACT = STREAMING_VERSION
REASONING_STREAM_HEADER = "X-System-X-Reasoning-Stream"


def _data(value: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"data: {encoded}\n\n".encode("utf-8")


def _done() -> bytes:
    return b"data: [DONE]\n\n"


def _usage(value: dict[str, Any]) -> dict[str, int]:
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("canonical usage payload is invalid")
    return {
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
    }


def _error(event: CanonicalStreamEvent) -> dict[str, Any]:
    raw = event.payload.get("error")
    if not isinstance(raw, dict):
        raise RuntimeError("canonical failure payload is invalid")
    code = raw.get("code")
    message = raw.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        raise RuntimeError("canonical failure payload is invalid")
    return {
        "error": {
            "message": message,
            "type": "server_error",
            "param": None,
            "code": code,
        }
    }


def _finish_reason(event: CanonicalStreamEvent, *, chat: bool) -> str:
    if event.type is CanonicalStreamEventType.REQUIRES_ACTION:
        if not chat:
            raise RuntimeError("Completion stream cannot require tool action")
        return "tool_calls"
    if event.type is CanonicalStreamEventType.INCOMPLETE:
        return "length"
    if event.type is CanonicalStreamEventType.COMPLETED:
        return "stop"
    raise RuntimeError("canonical event is not a successful terminal")


async def encode_completion_events(
    events: AsyncIterator[CanonicalStreamEvent],
    *,
    completion_id: str,
    created: int,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    model: str | None = None
    usage: dict[str, int] | None = None
    terminal_seen = False
    async for event in events:
        if terminal_seen:
            raise RuntimeError("canonical event appeared after terminal")
        model = event.model
        if event.type is CanonicalStreamEventType.STARTED:
            continue
        if event.type is CanonicalStreamEventType.OUTPUT_TEXT_DELTA:
            yield _data(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": event.payload["delta"],
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                }
            )
            continue
        if event.type is CanonicalStreamEventType.USAGE:
            usage = _usage(event.payload)
            continue
        if event.type is CanonicalStreamEventType.FAILED:
            terminal_seen = True
            yield _data(_error(event))
            yield _done()
            continue
        if event.terminal:
            terminal_seen = True
            yield _data(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": _finish_reason(event, chat=False),
                        }
                    ],
                    "usage": None,
                }
            )
            if include_usage:
                if usage is None:
                    raise RuntimeError("canonical stream omitted usage")
                yield _data(
                    {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                )
            yield _done()
            continue
        raise RuntimeError("unsupported canonical Completion event")
    if not terminal_seen:
        raise RuntimeError("canonical Completion stream ended without terminal")


async def encode_chat_events(
    events: AsyncIterator[CanonicalStreamEvent],
    *,
    completion_id: str,
    created: int,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    model: str | None = None
    usage: dict[str, int] | None = None
    terminal_seen = False

    def chunk(delta: dict[str, Any], finish_reason: str | None) -> bytes:
        return _data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": None,
            }
        )

    async for event in events:
        if terminal_seen:
            raise RuntimeError("canonical event appeared after terminal")
        model = event.model
        if event.type is CanonicalStreamEventType.STARTED:
            yield chunk({"role": "assistant", "content": ""}, None)
            continue
        if event.type is CanonicalStreamEventType.REASONING_DELTA:
            yield chunk(
                {"reasoning_content": event.payload["delta"]},
                None,
            )
            continue
        if event.type is CanonicalStreamEventType.OUTPUT_TEXT_DELTA:
            yield chunk({"content": event.payload["delta"]}, None)
            continue
        if event.type is CanonicalStreamEventType.TOOL_CALL_ADDED:
            tool = event.payload["tool_call"]
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": event.payload["index"],
                            "id": openai_call_id(tool["id"]),
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "arguments": "",
                            },
                        }
                    ]
                },
                None,
            )
            continue
        if event.type is CanonicalStreamEventType.TOOL_CALL_ARGUMENTS_DELTA:
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": event.payload["index"],
                            "function": {
                                "arguments": event.payload["delta"],
                            },
                        }
                    ]
                },
                None,
            )
            continue
        if event.type is CanonicalStreamEventType.TOOL_CALL_DONE:
            continue
        if event.type is CanonicalStreamEventType.USAGE:
            usage = _usage(event.payload)
            continue
        if event.type is CanonicalStreamEventType.FAILED:
            terminal_seen = True
            yield _data(_error(event))
            yield _done()
            continue
        if event.terminal:
            terminal_seen = True
            yield chunk({}, _finish_reason(event, chat=True))
            if include_usage:
                if usage is None:
                    raise RuntimeError("canonical stream omitted usage")
                yield _data(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                )
            yield _done()
            continue
        raise RuntimeError("unsupported canonical Chat event")
    if not terminal_seen:
        raise RuntimeError("canonical Chat stream ended without terminal")


def _stream_headers(request_id: str) -> dict[str, str]:
    return {
        **compatibility_headers(request_id),
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "X-System-X-OpenAI-Streaming": OPENAI_STREAMING_CONTRACT,
    }


def completion_stream_response(
    request: Request,
    session: CanonicalStreamSession,
    *,
    include_usage: bool,
) -> StreamingResponse:
    request_id = session.control.state.request_id
    suffix = request_id.removeprefix("sx_req_")
    managed = session.control.managed_events(request, session.events())
    content = encode_completion_events(
        managed,
        completion_id=f"cmpl-sx-{suffix}",
        created=int(time.time()),
        include_usage=include_usage,
    )
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers=_stream_headers(request_id),
    )


def chat_stream_response(
    request: Request,
    session: CanonicalStreamSession,
    *,
    include_usage: bool,
) -> StreamingResponse:
    request_id = session.control.state.request_id
    suffix = request_id.removeprefix("sx_req_")
    managed = session.control.managed_events(request, session.events())
    content = encode_chat_events(
        managed,
        completion_id=f"chatcmpl-sx-{suffix}",
        created=int(time.time()),
        include_usage=include_usage,
    )
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers={
            **_stream_headers(request_id),
            REASONING_STREAM_HEADER: "separate-v1",
        },
    )
