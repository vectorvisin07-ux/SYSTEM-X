"""Thin local /v1 routes over the in-process compatibility adapter."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from .openai_adapter import OpenAICompatibilityAdapter
from .openai_errors import OPENAI_ERROR_RESPONSES
from .openai_stream import (
    OPENAI_STREAMING_CONTRACT,
    chat_stream_response,
    completion_stream_response,
)
from .openai_responses_stream import (
    ResponsesStreamConfiguration,
    responses_stream_response,
)
from .openai_schemas import (
    OpenAIChatCompletion,
    OpenAIChatCompletionRequest,
    OpenAICompletion,
    OpenAICompletionRequest,
    OpenAIResponse,
    OpenAIResponsesRequest,
)
from .request_context import request_id_for
from .streaming_inference import StreamingInferenceService


def build_openai_router(
    adapter: OpenAICompatibilityAdapter,
    streaming: StreamingInferenceService,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["compatibility"])

    @router.post(
        "/completions",
        response_model=OpenAICompletion,
        responses=OPENAI_ERROR_RESPONSES,
    )
    async def completions(
        request: Request,
        body: OpenAICompletionRequest,
    ) -> OpenAICompletion | StreamingResponse:
        if body.stream:
            session = await streaming.open_generate(
                request_id_for(request),
                "/v1/completions",
                adapter.canonical_completion_request(body),
                protocol_surface=OPENAI_STREAMING_CONTRACT,
            )
            return completion_stream_response(
                request,
                session,
                include_usage=bool(
                    body.stream_options
                    and body.stream_options.include_usage
                ),
            )
        return await adapter.completion(request_id_for(request), body)

    @router.post(
        "/chat/completions",
        response_model=OpenAIChatCompletion,
        responses=OPENAI_ERROR_RESPONSES,
    )
    async def chat_completions(
        request: Request,
        body: OpenAIChatCompletionRequest,
    ) -> OpenAIChatCompletion | StreamingResponse:
        if body.stream:
            session = await streaming.open_chat(
                request_id_for(request),
                "/v1/chat/completions",
                adapter.canonical_chat_request(body),
                protocol_surface=OPENAI_STREAMING_CONTRACT,
            )
            return chat_stream_response(
                request,
                session,
                include_usage=bool(
                    body.stream_options
                    and body.stream_options.include_usage
                ),
            )
        return await adapter.chat(request_id_for(request), body)

    @router.post(
        "/responses",
        response_model=OpenAIResponse,
        responses=OPENAI_ERROR_RESPONSES,
    )
    async def responses(
        request: Request,
        body: OpenAIResponsesRequest,
    ) -> OpenAIResponse | StreamingResponse:
        if body.stream:
            request_id = request_id_for(request)
            session = await streaming.open_responses(
                request_id,
                "/v1/responses",
                adapter.canonical_responses_request(body),
                protocol_surface=OPENAI_STREAMING_CONTRACT,
            )
            if body.tool_choice is None:
                tool_choice = "auto" if body.tools else "none"
            elif isinstance(body.tool_choice, str):
                tool_choice = body.tool_choice
            else:
                tool_choice = body.tool_choice.model_dump(mode="json")
            return responses_stream_response(
                request,
                session,
                ResponsesStreamConfiguration(
                    request_id=request_id,
                    model=session.control.state.model,
                    max_output_tokens=body.max_output_tokens,
                    temperature=body.temperature,
                    tools=[
                        tool.model_dump(mode="json")
                        for tool in body.tools
                    ],
                    tool_choice=tool_choice,
                ),
            )
        return await adapter.response(request_id_for(request), body)

    return router
