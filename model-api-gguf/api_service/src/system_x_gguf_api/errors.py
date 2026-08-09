"""Bounded public error normalization for the System X route namespace."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .anthropic_contract import ANTHROPIC_PATH_PREFIX
from .anthropic_errors import (
    anthropic_error_response,
    compatibility_headers as anthropic_compatibility_headers,
    system_error_response as anthropic_system_error_response,
    validation_error_response as anthropic_validation_error_response,
)
from .authentication import AuthenticationManager, protected_route_family
from .openai_contract import (
    COMPATIBILITY_HEADER,
    COMPATIBILITY_VERSION,
    OPENAI_PATH_PREFIX,
    OPENAI_REQUEST_ID_HEADER,
)
from .openai_errors import (
    openai_error_response,
    system_error_response,
    validation_error_response,
)
from .operation_records import (
    OperationRecordInvariantError,
    OperationRecorder,
    operation_route_for,
)
from .request_context import (
    REQUEST_ID_HEADER,
    authentication_context_for,
    new_request_id,
    request_id_for,
)
from .schemas import ErrorField, SystemXErrorCode, SystemXErrorDetail, SystemXErrorResponse
from .tool_contract import ToolContractError


LOGGER = logging.getLogger("system_x_gguf_api")
SYSTEM_PATH_PREFIX = "/system/v1/"
SYSTEM_PATH_ROOT = SYSTEM_PATH_PREFIX.removesuffix("/")
MAX_PUBLIC_MESSAGE = 240
SYSTEM_X_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {
        "model": SystemXErrorResponse,
        "description": (
            "system_x_route_not_found or system_x_model_not_found"
        ),
    },
    405: {
        "model": SystemXErrorResponse,
        "description": "system_x_method_not_allowed",
    },
    409: {
        "model": SystemXErrorResponse,
        "description": "system_x_model_conflict",
    },
    422: {
        "model": SystemXErrorResponse,
        "description": (
            "system_x_validation_error or a bounded System X tool/structured "
            "request-contract error"
        ),
    },
    500: {
        "model": SystemXErrorResponse,
        "description": "system_x_internal_error",
    },
    502: {
        "model": SystemXErrorResponse,
        "description": (
            "system_x_backend_response_invalid or system_x_output_invalid"
        ),
    },
    503: {
        "model": SystemXErrorResponse,
        "description": (
            "system_x_no_ready_model, system_x_model_unavailable, "
            "system_x_capability_unavailable, or system_x_backend_unavailable"
        ),
    },
    504: {
        "model": SystemXErrorResponse,
        "description": "system_x_backend_timeout",
    },
}

TOOL_REQUEST_ERROR_CODES = {
    "system_x_tool_schema_invalid",
    "system_x_tool_choice_invalid",
    "system_x_tool_result_mismatch",
    "system_x_tool_result_duplicate",
    "system_x_tool_result_missing",
    "system_x_tool_and_output_format_conflict",
    "system_x_structured_output_schema_invalid",
}


def _bounded_message(value: str) -> str:
    normalized = "".join(
        character if ord(character) >= 32 and ord(character) != 127 else " "
        for character in value
    ).strip()
    return (normalized or "System X request failed")[:MAX_PUBLIC_MESSAGE]


class SystemXError(RuntimeError):
    """One stable public failure classification without private evidence."""

    def __init__(
        self,
        status_code: int,
        code: SystemXErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.public_message = _bounded_message(message)
        self.retryable = retryable


def _is_system_request(request: Request) -> bool:
    path = request.url.path
    return path == SYSTEM_PATH_ROOT or path.startswith(SYSTEM_PATH_PREFIX)


def _response_validation_summary(
    exc: ResponseValidationError,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for error in exc.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:16]:
        location = error.get("loc")
        result.append(
            {
                "location": (
                    [str(item)[:80] for item in location]
                    if isinstance(location, tuple)
                    else []
                ),
                "issue": str(error.get("type", "invalid"))[:80],
            }
        )
    return result


def _is_openai_request(request: Request) -> bool:
    path = request.url.path
    return (
        path == OPENAI_PATH_PREFIX
        or path.startswith(f"{OPENAI_PATH_PREFIX}/")
    ) and not _is_anthropic_request(request)


def _is_anthropic_request(request: Request) -> bool:
    path = request.url.path
    return (
        path == ANTHROPIC_PATH_PREFIX
        or path.startswith(f"{ANTHROPIC_PATH_PREFIX}/")
        or (
            path == "/v1/models"
            and "anthropic-version" in request.headers
        )
    )


def _is_managed_request(request: Request) -> bool:
    return (
        _is_system_request(request)
        or _is_openai_request(request)
        or _is_anthropic_request(request)
    )


def _request_id(request: Request) -> str:
    try:
        return request_id_for(request)
    except RuntimeError:
        request_id = new_request_id()
        request.state.system_x_request_id = request_id
        return request_id


def _error_response(
    request: Request,
    status_code: int,
    code: SystemXErrorCode,
    message: str,
    *,
    retryable: bool = False,
    fields: list[ErrorField] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    body = SystemXErrorResponse(
        request_id=request_id,
        error=SystemXErrorDetail(
            code=code,
            message=_bounded_message(message),
            retryable=retryable,
            fields=fields,
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", exclude_none=True),
        headers={REQUEST_ID_HEADER: request_id},
    )


def _validation_fields(exc: RequestValidationError) -> list[ErrorField]:
    fields: list[ErrorField] = []
    for error in exc.errors()[:16]:
        public_location: list[str | int] = []
        for component in error.get("loc", ())[:16]:
            if isinstance(component, int):
                public_location.append(component)
            else:
                public_location.append(str(component)[:64])
        if not public_location:
            public_location = ["request"]
        issue = str(error.get("type", "invalid"))[:64] or "invalid"
        fields.append(ErrorField(location=public_location, issue=issue))
    return fields


def _tool_request_validation_code(
    exc: RequestValidationError,
) -> SystemXErrorCode | None:
    for error in exc.errors():
        kind = str(error.get("type", ""))
        if kind in TOOL_REQUEST_ERROR_CODES:
            return kind  # type: ignore[return-value]
        context = error.get("ctx")
        cause = context.get("error") if isinstance(context, dict) else None
        if (
            isinstance(cause, ToolContractError)
            and cause.code in TOOL_REQUEST_ERROR_CODES
        ):
            return cause.code  # type: ignore[return-value]
        message = str(error.get("msg", ""))
        if "tools and output_format cannot be combined" in message:
            return "system_x_tool_and_output_format_conflict"
    return None


def _note_operation_error(
    request: Request,
    code: SystemXErrorCode,
) -> None:
    recorder = getattr(request.app.state, "operations", None)
    if not isinstance(recorder, OperationRecorder):
        return
    try:
        recorder.note_error_if_active(_request_id(request), code)
    except OperationRecordInvariantError:
        return


def install_system_error_handling(
    application: FastAPI,
    authentication: AuthenticationManager | None = None,
) -> None:
    """Install identity middleware and namespace-specific exception handlers."""

    @application.middleware("http")
    async def system_request_identity(request: Request, call_next: Any):
        if not _is_managed_request(request):
            return await call_next(request)
        request.state.system_x_request_id = new_request_id()
        family = protected_route_family(
            request.method,
            request.url.path,
            anthropic_version_present=(
                "anthropic-version" in request.headers
            ),
        )
        response = (
            authentication.authenticate_request(request, family)
            if authentication is not None and family is not None
            else None
        )
        request_id = request_id_for(request)
        recorder = getattr(application.state, "operations", None)
        operation_route = operation_route_for(
            request.method,
            request.url.path,
            anthropic_version_present=(
                "anthropic-version" in request.headers
            ),
        )
        operation_started = False
        if (
            response is None
            and operation_route is not None
            and isinstance(recorder, OperationRecorder)
        ):
            key_id: str | None = None
            if authentication is None or authentication.enabled:
                context = authentication_context_for(request)
                key_id = context.key_id
            recorder.begin(
                operation_route,
                request_id=request_id,
                key_id=key_id,
            )
            operation_started = True
        if response is None:
            try:
                response = await call_next(request)
            except BaseException:
                if operation_started:
                    try:
                        recorder.note_error(
                            request_id, "system_x_internal_error"
                        )
                        recorder.finalize_if_active(
                            request_id, http_status=500
                        )
                    except OperationRecordInvariantError:
                        pass
                raise
        if operation_started:
            original_iterator = getattr(
                response, "body_iterator", None
            )
            status_code = int(response.status_code)
            if original_iterator is None:
                recorder.finalize_if_active(
                    request_id, http_status=status_code
                )
            else:

                async def operation_tracked_body():
                    try:
                        async for chunk in original_iterator:
                            yield chunk
                    except (
                        asyncio.CancelledError,
                        GeneratorExit,
                        BrokenPipeError,
                        ConnectionError,
                        OSError,
                    ):
                        try:
                            recorder.note_cancelled(request_id)
                        except OperationRecordInvariantError:
                            pass
                        raise
                    finally:
                        recorder.finalize_if_active(
                            request_id, http_status=status_code
                        )

                response.body_iterator = operation_tracked_body()
        response.headers[REQUEST_ID_HEADER] = request_id
        if _is_openai_request(request):
            response.headers[OPENAI_REQUEST_ID_HEADER] = request_id
            response.headers[COMPATIBILITY_HEADER] = COMPATIBILITY_VERSION
        elif _is_anthropic_request(request):
            response.headers.update(anthropic_compatibility_headers(request_id))
        return response

    @application.exception_handler(SystemXError)
    async def system_x_error_handler(
        request: Request, exc: SystemXError
    ) -> JSONResponse:
        _note_operation_error(request, exc.code)
        if _is_anthropic_request(request):
            return anthropic_system_error_response(
                _request_id(request),
                exc.code,
                exc.public_message,
                exc.status_code,
            )
        if _is_openai_request(request):
            return system_error_response(
                _request_id(request),
                exc.code,
                exc.public_message,
                exc.status_code,
            )
        return _error_response(
            request,
            exc.status_code,
            exc.code,
            exc.public_message,
            retryable=exc.retryable,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ):
        contract_code = _tool_request_validation_code(exc)
        operation_code = (
            contract_code
            if contract_code is not None
            else "system_x_validation_error"
        )
        _note_operation_error(request, operation_code)
        if _is_anthropic_request(request):
            return anthropic_validation_error_response(
                _request_id(request), exc
            )
        if _is_openai_request(request):
            return validation_error_response(_request_id(request), exc)
        if not _is_system_request(request):
            raise exc
        if contract_code is not None:
            return _error_response(
                request,
                422,
                contract_code,
                "System X tool or structured-output request is invalid",
                fields=_validation_fields(exc),
            )
        return _error_response(
            request,
            422,
            "system_x_validation_error",
            "Request validation failed",
            fields=_validation_fields(exc),
        )

    @application.exception_handler(StarletteHTTPException)
    async def route_error_handler(
        request: Request, exc: StarletteHTTPException
    ):
        if exc.status_code == 404:
            operation_code = "system_x_route_not_found"
        elif exc.status_code == 405:
            operation_code = "system_x_method_not_allowed"
        else:
            operation_code = "system_x_internal_error"
        _note_operation_error(request, operation_code)
        if _is_anthropic_request(request):
            if exc.status_code == 404:
                return anthropic_error_response(
                    _request_id(request),
                    404,
                    "not_found_error",
                    "Messages compatibility route was not found",
                )
            if exc.status_code == 405:
                return anthropic_error_response(
                    _request_id(request),
                    405,
                    "invalid_request_error",
                    "Method is not allowed for this Messages route",
                )
            return anthropic_error_response(
                _request_id(request),
                500,
                "api_error",
                "Messages compatibility request failed",
            )
        if _is_openai_request(request):
            if exc.status_code == 404:
                return openai_error_response(
                    _request_id(request),
                    404,
                    "invalid_request_error",
                    "route_not_found",
                    "Compatibility route was not found",
                    None,
                )
            if exc.status_code == 405:
                return openai_error_response(
                    _request_id(request),
                    405,
                    "invalid_request_error",
                    "method_not_allowed",
                    "Method is not allowed for this compatibility route",
                    None,
                )
            return openai_error_response(
                _request_id(request),
                500,
                "server_error",
                "internal_error",
                "Compatibility request failed",
                None,
            )
        if not _is_system_request(request):
            return await http_exception_handler(request, exc)
        if exc.status_code == 404:
            return _error_response(
                request,
                404,
                "system_x_route_not_found",
                "System X route was not found",
            )
        if exc.status_code == 405:
            return _error_response(
                request,
                405,
                "system_x_method_not_allowed",
                "Method is not allowed for this System X route",
            )
        return _error_response(
            request,
            500,
            "system_x_internal_error",
            "System X request failed",
        )

    @application.exception_handler(ResponseValidationError)
    async def response_validation_error_handler(
        request: Request, exc: ResponseValidationError
    ):
        _note_operation_error(request, "system_x_internal_error")
        if _is_anthropic_request(request):
            LOGGER.error(
                "Messages response validation failed request_id=%s",
                _request_id(request),
            )
            return anthropic_error_response(
                _request_id(request),
                500,
                "api_error",
                "Messages compatibility response validation failed",
            )
        if _is_openai_request(request):
            LOGGER.error(
                "compatibility response validation failed "
                "request_id=%s errors=%r",
                _request_id(request),
                _response_validation_summary(exc),
            )
            return openai_error_response(
                _request_id(request),
                500,
                "server_error",
                "internal_error",
                "Compatibility response validation failed",
                None,
            )
        if not _is_system_request(request):
            raise exc
        LOGGER.error(
            "response validation failed request_id=%s errors=%r",
            _request_id(request),
            _response_validation_summary(exc),
        )
        return _error_response(
            request,
            500,
            "system_x_internal_error",
            "System X response validation failed",
        )

    @application.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        _note_operation_error(request, "system_x_internal_error")
        if _is_anthropic_request(request):
            LOGGER.exception(
                "unhandled Messages request failure request_id=%s error_type=%s",
                _request_id(request),
                type(exc).__name__,
            )
            return anthropic_error_response(
                _request_id(request),
                500,
                "api_error",
                "Messages compatibility request failed",
            )
        if _is_openai_request(request):
            LOGGER.exception(
                "unhandled compatibility request failure "
                "request_id=%s error_type=%s",
                _request_id(request),
                type(exc).__name__,
            )
            return openai_error_response(
                _request_id(request),
                500,
                "server_error",
                "internal_error",
                "Compatibility request failed",
                None,
            )
        if not _is_system_request(request):
            raise exc
        LOGGER.exception(
            "unhandled System X request failure request_id=%s error_type=%s",
            _request_id(request),
            type(exc).__name__,
        )
        return _error_response(
            request,
            500,
            "system_x_internal_error",
            "System X request failed",
        )
