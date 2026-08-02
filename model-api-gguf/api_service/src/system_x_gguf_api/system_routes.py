"""Thin HTTP routes for the authoritative System X native contract."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from .errors import SYSTEM_X_ERROR_RESPONSES
from .inference_service import InferenceService
from .model_catalogue import ModelCatalogue
from .request_context import request_id_for
from .schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerationResponse,
    ModelDetailResponse,
    ModelListResponse,
    ResponsesRequest,
    ResponsesResponse,
    TokenCountRequest,
    TokenCountResponse,
)
from .streaming_inference import StreamingInferenceService
from .system_stream import SYSTEM_X_STREAMING_CONTRACT, system_x_stream_response


def build_system_router(
    catalogue: ModelCatalogue,
    inference: InferenceService,
    streaming: StreamingInferenceService,
) -> APIRouter:
    router = APIRouter(prefix="/system/v1", tags=["system"])

    @router.get(
        "/models",
        response_model=ModelListResponse,
        response_model_exclude_none=True,
        responses=SYSTEM_X_ERROR_RESPONSES,
    )
    async def list_models(request: Request) -> ModelListResponse:
        return await catalogue.list_models(request_id_for(request))

    @router.get(
        "/models/{model_id}",
        response_model=ModelDetailResponse,
        response_model_exclude_none=False,
        responses=SYSTEM_X_ERROR_RESPONSES,
    )
    async def model_detail(
        request: Request, model_id: str
    ) -> ModelDetailResponse:
        return await catalogue.model_detail(request_id_for(request), model_id)

    @router.post(
        "/generate",
        response_model=GenerationResponse,
        responses=SYSTEM_X_ERROR_RESPONSES,
    )
    async def generate(
        request: Request, body: GenerateRequest
    ) -> GenerationResponse | StreamingResponse:
        if body.stream:
            session = await streaming.open_generate(
                request_id_for(request),
                "/system/v1/generate",
                body,
                protocol_surface=SYSTEM_X_STREAMING_CONTRACT,
            )
            return system_x_stream_response(request, session)
        return await inference.generate(request_id_for(request), body)

    @router.post(
        "/chat",
        response_model=ChatResponse,
        responses=SYSTEM_X_ERROR_RESPONSES,
    )
    async def chat(
        request: Request, body: ChatRequest
    ) -> ChatResponse | StreamingResponse:
        if body.stream:
            session = await streaming.open_chat(
                request_id_for(request),
                "/system/v1/chat",
                body,
                protocol_surface=SYSTEM_X_STREAMING_CONTRACT,
            )
            return system_x_stream_response(request, session)
        return await inference.chat(request_id_for(request), body)

    @router.post(
        "/responses",
        response_model=ResponsesResponse,
        responses=SYSTEM_X_ERROR_RESPONSES,
    )
    async def responses(
        request: Request, body: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        if body.stream:
            session = await streaming.open_responses(
                request_id_for(request),
                "/system/v1/responses",
                body,
                protocol_surface=SYSTEM_X_STREAMING_CONTRACT,
            )
            return system_x_stream_response(request, session)
        return await inference.responses(request_id_for(request), body)

    @router.post(
        "/tokens/count",
        response_model=TokenCountResponse,
        response_model_exclude_none=True,
        responses=SYSTEM_X_ERROR_RESPONSES,
    )
    async def count_tokens(
        request: Request, body: TokenCountRequest
    ) -> TokenCountResponse:
        return await inference.count_tokens(request_id_for(request), body)

    return router
