"""System X native SSE framing over canonical stream events."""

from __future__ import annotations

import json

from fastapi import Request
from starlette.responses import StreamingResponse

from .stream_types import CanonicalStreamEvent
from .streaming_inference import CanonicalStreamSession


SYSTEM_X_STREAMING_CONTRACT = "system-x.streaming.v1"


def encode_system_x_event(event: CanonicalStreamEvent) -> bytes:
    data = json.dumps(
        event.system_x_data(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return (
        f"event: {event.type.value}\n"
        f"id: {event.request_id}:{event.sequence}\n"
        f"data: {data}\n\n"
    ).encode("utf-8")


def system_x_stream_response(
    request: Request,
    session: CanonicalStreamSession,
) -> StreamingResponse:
    async def content():
        async for event in session.control.managed_events(
            request,
            session.events(),
        ):
            yield encode_system_x_event(event)

    request_id = session.control.state.request_id
    return StreamingResponse(
        content(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-System-X-Request-ID": request_id,
            "X-System-X-Streaming": SYSTEM_X_STREAMING_CONTRACT,
        },
    )
