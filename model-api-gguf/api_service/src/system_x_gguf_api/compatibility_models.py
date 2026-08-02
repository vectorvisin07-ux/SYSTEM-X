"""One explicit dispatcher for the shared OpenAI/Messages model-list path."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fastapi import APIRouter, Request

from .anthropic_adapter import AnthropicCompatibilityAdapter
from .anthropic_routes import _validate_headers
from .openai_adapter import OpenAICompatibilityAdapter
from .request_context import request_id_for


def _field(item: object, name: str) -> Any:
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)


def compatibility_model_references(
    snapshots: Iterable[object],
) -> list[tuple[str, str]]:
    """Return default first and each immutable READY identity exactly once."""

    normalized = list(snapshots)
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for snapshot in normalized:
        aliases = _field(snapshot, "aliases")
        if "default" in aliases and "default" not in seen:
            result.append(("default", str(_field(snapshot, "created_utc"))))
            seen.add("default")
    for snapshot in normalized:
        model_id = str(_field(snapshot, "public_model_id"))
        if model_id not in seen:
            result.append((model_id, str(_field(snapshot, "created_utc"))))
            seen.add(model_id)
    return result


def build_compatibility_models_router(
    openai: OpenAICompatibilityAdapter,
    anthropic: AnthropicCompatibilityAdapter,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["compatibility"])

    @router.get("/models", response_model=None)
    async def models(request: Request) -> object:
        if "anthropic-version" in request.headers:
            _validate_headers(
                request.headers.get("anthropic-version", ""),
                request.headers.get("anthropic-beta"),
            )
            return await anthropic.models(request_id_for(request))
        return await openai.models(request_id_for(request))

    return router
