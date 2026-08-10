"""Private-data-free errors and headers for local Messages compatibility."""

from __future__ import annotations

from typing import Any

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .anthropic_contract import (
    ANTHROPIC_REQUEST_ID_HEADER,
    COMPATIBILITY_HEADER,
    COMPATIBILITY_VERSION,
    anthropic_request_id,
)
from .anthropic_schemas import AnthropicErrorDetail, AnthropicErrorResponse
from .request_context import REQUEST_ID_HEADER


ANTHROPIC_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": AnthropicErrorResponse, "description": "invalid request"},
    404: {"model": AnthropicErrorResponse, "description": "not found"},
    405: {"model": AnthropicErrorResponse, "description": "method not allowed"},
    409: {"model": AnthropicErrorResponse, "description": "conflict"},
    413: {"model": AnthropicErrorResponse, "description": "request too large"},
    422: {"model": AnthropicErrorResponse, "description": "token budget exceeded"},
    429: {"model": AnthropicErrorResponse, "description": "rate or concurrency limit"},
    500: {"model": AnthropicErrorResponse, "description": "API error"},
    502: {"model": AnthropicErrorResponse, "description": "backend protocol error"},
    503: {"model": AnthropicErrorResponse, "description": "service unavailable"},
    504: {"model": AnthropicErrorResponse, "description": "timeout"},
    529: {"model": AnthropicErrorResponse, "description": "overloaded"},
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

SYSTEM_ERROR_MAP: dict[str, tuple[int, str]] = {
    "system_x_validation_error": (400, "invalid_request_error"),
    "system_x_request_too_large": (413, "invalid_request_error"),
    "system_x_token_budget_exceeded": (422, "invalid_request_error"),
    "system_x_concurrency_limit_exceeded": (429, "rate_limit_error"),
    "system_x_rate_limit_exceeded": (429, "rate_limit_error"),
    "system_x_request_deadline_exceeded": (504, "api_error"),
    "system_x_model_not_found": (404, "not_found_error"),
    "system_x_no_ready_model": (503, "api_error"),
    "system_x_model_unavailable": (529, "overloaded_error"),
    "system_x_capability_unavailable": (529, "overloaded_error"),
    "system_x_tool_capability_unavailable": (529, "overloaded_error"),
    "system_x_model_conflict": (409, "conflict_error"),
    "system_x_backend_unavailable": (529, "overloaded_error"),
    "system_x_backend_timeout": (504, "timeout_error"),
    "system_x_backend_response_invalid": (502, "api_error"),
    "system_x_output_invalid": (502, "api_error"),
    "system_x_tool_call_invalid": (502, "api_error"),
    "system_x_tool_arguments_invalid": (502, "api_error"),
    "system_x_structured_output_invalid": (502, "api_error"),
    "system_x_streaming_structured_output_unsupported": (
        400,
        "invalid_request_error",
    ),
    "system_x_tool_and_output_format_conflict": (502, "api_error"),
    "system_x_internal_error": (500, "api_error"),
}


def compatibility_headers(system_request_id: str) -> dict[str, str]:
    return {
        ANTHROPIC_REQUEST_ID_HEADER: anthropic_request_id(system_request_id),
        REQUEST_ID_HEADER: system_request_id,
        COMPATIBILITY_HEADER: COMPATIBILITY_VERSION,
    }


def anthropic_error_response(
    system_request_id: str,
    status_code: int,
    error_type: str,
    message: str,
) -> JSONResponse:
    request_id = anthropic_request_id(system_request_id)
    body = AnthropicErrorResponse(
        error=AnthropicErrorDetail(type=error_type, message=message[:240]),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=compatibility_headers(system_request_id),
    )


def system_error_response(
    system_request_id: str,
    code: str,
    message: str,
    system_status_code: int | None = None,
) -> JSONResponse:
    if (
        system_status_code is not None
        and system_status_code < 500
        and code in TOOL_INPUT_ERROR_CODES
    ):
        return anthropic_error_response(
            system_request_id,
            400,
            "invalid_request_error",
            message,
        )
    status, error_type = SYSTEM_ERROR_MAP.get(code, (500, "api_error"))
    return anthropic_error_response(
        system_request_id, status, error_type, message
    )


def _param(error: dict[str, Any]) -> str | None:
    context = error.get("ctx")
    if isinstance(context, dict) and isinstance(context.get("param"), str):
        return context["param"][:64]
    location = list(error.get("loc", ()))
    if location and location[0] in {"body", "header"}:
        location.pop(0)
    return next(
        (str(item)[:64] for item in location if isinstance(item, str)),
        None,
    )


def validation_error_response(
    system_request_id: str, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    param = _param(first)
    kind = str(first.get("type", "invalid_request"))
    if kind == "json_invalid":
        message = "Request body is not valid JSON"
    elif kind in TOOL_INPUT_ERROR_CODES:
        message = f"Messages request is invalid: {kind}"
    elif param:
        message = f"Invalid or unsupported value for '{param}'"
    else:
        message = "Request validation failed"
    return anthropic_error_response(
        system_request_id, 400, "invalid_request_error", message
    )
