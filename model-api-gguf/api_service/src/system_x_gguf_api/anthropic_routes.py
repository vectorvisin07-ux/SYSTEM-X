"""Thin local Messages routes over the in-process compatibility adapter."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request
from starlette.responses import StreamingResponse

from .anthropic_adapter import AnthropicCompatibilityAdapter
from .anthropic_contract import ACCEPTED_ANTHROPIC_VERSION
from .anthropic_errors import ANTHROPIC_ERROR_RESPONSES
from .anthropic_stream import (
    ANTHROPIC_STREAMING_CONTRACT,
    messages_stream_response,
)
from .anthropic_schemas import (
    AnthropicCountTokensRequest,
    AnthropicMessage,
    AnthropicMessageRequest,
    AnthropicMessageTokensCount,
)
from .errors import SystemXError
from .request_context import request_id_for
from .streaming_inference import StreamingInferenceService


def _validate_headers(version: str, beta: str | None) -> None:
    if version != ACCEPTED_ANTHROPIC_VERSION:
        raise SystemXError(
            400,
            "system_x_validation_error",
            "Unsupported anthropic-version header",
        )
    if beta is not None:
        raise SystemXError(
            400,
            "system_x_validation_error",
            "anthropic-beta is unsupported",
        )


def build_anthropic_router(
    adapter: AnthropicCompatibilityAdapter,
    streaming: StreamingInferenceService,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["messages-compatibility"])

    @router.post(
        "/messages",
        response_model=AnthropicMessage,
        response_model_exclude_none=True,
        responses=ANTHROPIC_ERROR_RESPONSES,
    )
    async def messages(
        request: Request,
        body: AnthropicMessageRequest,
        anthropic_version: str = Header(alias="anthropic-version"),
        anthropic_beta: str | None = Header(
            default=None, alias="anthropic-beta"
        ),
    ) -> AnthropicMessage | StreamingResponse:
        _validate_headers(anthropic_version, anthropic_beta)
        if body.stream:
            session = await streaming.open_chat(
                request_id_for(request),
                "/v1/messages",
                adapter.canonical_message_request(body),
                require_input_tokens=True,
                protocol_surface=ANTHROPIC_STREAMING_CONTRACT,
            )
            return messages_stream_response(request, session)
        return await adapter.message(request_id_for(request), body)

    @router.post(
        "/messages/count_tokens",
        response_model=AnthropicMessageTokensCount,
        responses=ANTHROPIC_ERROR_RESPONSES,
    )
    async def count_tokens(
        request: Request,
        body: AnthropicCountTokensRequest,
        anthropic_version: str = Header(alias="anthropic-version"),
        anthropic_beta: str | None = Header(
            default=None, alias="anthropic-beta"
        ),
    ) -> AnthropicMessageTokensCount:
        _validate_headers(anthropic_version, anthropic_beta)
        return await adapter.count_tokens(request_id_for(request), body)

    return router
