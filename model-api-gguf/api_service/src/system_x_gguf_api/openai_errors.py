"""Deterministic, private-data-free error serialization for local /v1."""

from __future__ import annotations

from typing import Any

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .openai_contract import (
    COMPATIBILITY_HEADER,
    COMPATIBILITY_VERSION,
    OPENAI_REQUEST_ID_HEADER,
)
from .openai_schemas import OpenAIErrorDetail, OpenAIErrorResponse
from .request_context import REQUEST_ID_HEADER


OPENAI_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": OpenAIErrorResponse, "description": "invalid request"},
    404: {"model": OpenAIErrorResponse, "description": "not found"},
    405: {"model": OpenAIErrorResponse, "description": "method not allowed"},
    409: {"model": OpenAIErrorResponse, "description": "model conflict"},
    500: {"model": OpenAIErrorResponse, "description": "internal error"},
    502: {"model": OpenAIErrorResponse, "description": "backend protocol error"},
    503: {"model": OpenAIErrorResponse, "description": "service unavailable"},
    504: {"model": OpenAIErrorResponse, "description": "inference timeout"},
}

UNSUPPORTED_VALUE_FIELDS = {
    "background",
    "best_of",
    "echo",
    "logprobs",
    "n",
    "parallel_tool_calls",
    "store",
    "suffix",
    "top_logprobs",
}

TOOL_INPUT_ERROR_CODES = {
    "system_x_tool_schema_invalid",
    "system_x_tool_choice_invalid",
    "system_x_tool_call_invalid",
    "system_x_tool_arguments_invalid",
    "system_x_tool_result_mismatch",
    "system_x_tool_result_duplicate",
    "system_x_tool_result_missing",
    "system_x_structured_output_schema_invalid",
    "system_x_structured_output_invalid",
    "system_x_tool_and_output_format_conflict",
}

SYSTEM_ERROR_MAP: dict[str, tuple[int, str, str, str | None]] = {
    "system_x_validation_error": (
        400,
        "invalid_request_error",
        "invalid_request",
        "max_output_tokens",
    ),
    "system_x_model_not_found": (
        404,
        "invalid_request_error",
        "model_not_found",
        "model",
    ),
    "system_x_no_ready_model": (
        503,
        "server_error",
        "no_ready_model",
        "model",
    ),
    "system_x_model_unavailable": (
        503,
        "server_error",
        "model_unavailable",
        "model",
    ),
    "system_x_capability_unavailable": (
        503,
        "server_error",
        "model_unavailable",
        "model",
    ),
    "system_x_tool_capability_unavailable": (
        503,
        "server_error",
        "system_x_tool_capability_unavailable",
        "model",
    ),
    "system_x_tool_call_invalid": (
        502,
        "server_error",
        "system_x_tool_call_invalid",
        None,
    ),
    "system_x_tool_arguments_invalid": (
        502,
        "server_error",
        "system_x_tool_arguments_invalid",
        None,
    ),
    "system_x_structured_output_invalid": (
        502,
        "server_error",
        "system_x_structured_output_invalid",
        None,
    ),
    "system_x_streaming_structured_output_unsupported": (
        400,
        "invalid_request_error",
        "unsupported_parameter",
        "response_format",
    ),
    "system_x_tool_and_output_format_conflict": (
        502,
        "server_error",
        "system_x_tool_and_output_format_conflict",
        None,
    ),
    "system_x_model_conflict": (
        409,
        "conflict_error",
        "model_conflict",
        "model",
    ),
    "system_x_backend_unavailable": (
        503,
        "server_error",
        "backend_unavailable",
        None,
    ),
    "system_x_backend_timeout": (
        504,
        "server_error",
        "inference_timeout",
        None,
    ),
    "system_x_backend_response_invalid": (
        502,
        "server_error",
        "backend_protocol_error",
        None,
    ),
    "system_x_output_invalid": (
        502,
        "server_error",
        "backend_protocol_error",
        None,
    ),
    "system_x_internal_error": (
        500,
        "server_error",
        "internal_error",
        None,
    ),
}


def compatibility_headers(request_id: str) -> dict[str, str]:
    return {
        OPENAI_REQUEST_ID_HEADER: request_id,
        REQUEST_ID_HEADER: request_id,
        COMPATIBILITY_HEADER: COMPATIBILITY_VERSION,
    }


def openai_error_response(
    request_id: str,
    status_code: int,
    error_type: str,
    code: str,
    message: str,
    param: str | None,
) -> JSONResponse:
    body = OpenAIErrorResponse(
        error=OpenAIErrorDetail(
            message=message[:240],
            type=error_type,
            param=param,
            code=code,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=compatibility_headers(request_id),
    )


def system_error_response(
    request_id: str,
    code: str,
    message: str,
    system_status_code: int | None = None,
) -> JSONResponse:
    if (
        system_status_code is not None
        and system_status_code < 500
        and code in TOOL_INPUT_ERROR_CODES
    ):
        return openai_error_response(
            request_id,
            400,
            "invalid_request_error",
            code,
            message,
            (
                "response_format"
                if "structured" in code or "output_format" in code
                else "tools"
                if "schema" in code or "choice" in code
                else "messages"
            ),
        )
    mapping = SYSTEM_ERROR_MAP.get(
        code,
        (500, "server_error", "internal_error", None),
    )
    return openai_error_response(
        request_id,
        mapping[0],
        mapping[1],
        mapping[2],
        message,
        mapping[3],
    )


def _top_level_param(error: dict[str, Any]) -> str | None:
    context = error.get("ctx")
    if isinstance(context, dict):
        context_param = context.get("param")
        if isinstance(context_param, str):
            return context_param[:64]
    location = error.get("loc")
    if not isinstance(location, (list, tuple)):
        return None
    components = list(location)
    if components and components[0] == "body":
        components.pop(0)
    for component in components:
        if isinstance(component, str):
            return component[:64]
    return None


def validation_error_response(
    request_id: str,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    kind = str(first.get("type", "invalid_request"))
    param = _top_level_param(first)
    if kind == "json_invalid":
        return openai_error_response(
            request_id,
            400,
            "invalid_request_error",
            "invalid_json",
            "Request body is not valid JSON",
            None,
        )
    if kind in TOOL_INPUT_ERROR_CODES:
        return openai_error_response(
            request_id,
            400,
            "invalid_request_error",
            kind,
            "OpenAI tool or structured-output request is invalid",
            param,
        )
    if (
        kind in {"unsupported_parameter", "extra_forbidden"}
        or param in UNSUPPORTED_VALUE_FIELDS
    ):
        return openai_error_response(
            request_id,
            400,
            "invalid_request_error",
            "unsupported_parameter",
            (
                f"Parameter '{param}' is unsupported"
                if param is not None
                else "Request contains an unsupported parameter"
            ),
            param,
        )
    return openai_error_response(
        request_id,
        400,
        "invalid_request_error",
        "invalid_request",
        (
            f"Invalid value for '{param}'"
            if param is not None
            else "Request validation failed"
        ),
        param,
    )
