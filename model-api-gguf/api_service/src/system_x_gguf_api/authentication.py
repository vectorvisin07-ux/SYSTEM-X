"""Canonical private API authentication for every public protocol family."""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
import logging
import re
from typing import Literal

from fastapi import Request
from fastapi.responses import JSONResponse

from .anthropic_contract import (
    ANTHROPIC_REQUEST_ID_HEADER,
    COMPATIBILITY_HEADER as ANTHROPIC_COMPATIBILITY_HEADER,
    COMPATIBILITY_VERSION as ANTHROPIC_COMPATIBILITY_VERSION,
    anthropic_request_id,
)
from .credential_store import CredentialStore
from .credential_types import AuthenticationContext, CredentialScheme
from .openai_contract import (
    COMPATIBILITY_HEADER as OPENAI_COMPATIBILITY_HEADER,
    COMPATIBILITY_VERSION as OPENAI_COMPATIBILITY_VERSION,
    OPENAI_REQUEST_ID_HEADER,
)
from .request_context import REQUEST_ID_HEADER, request_id_for
from .secret_redaction import redact_text


AUTHENTICATION_MESSAGE = "Authentication credentials are missing or invalid."
CONFLICT_MESSAGE = "Conflicting authentication credentials."
CHALLENGE = 'Bearer realm="system-x"'
LOGGER = logging.getLogger("uvicorn.error")
RouteFamily = Literal["system", "openai", "anthropic"]
SYSTEM_MODEL_DETAIL = re.compile(r"^/system/v1/models/[^/]+$")
PROTECTED_OPERATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/system/v1/version"),
        ("GET", "/system/v1/metrics"),
        ("GET", "/system/v1/models"),
        ("POST", "/system/v1/generate"),
        ("POST", "/system/v1/chat"),
        ("POST", "/system/v1/responses"),
        ("POST", "/system/v1/tokens/count"),
        ("GET", "/v1/models"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
    }
)


@dataclass(frozen=True, slots=True)
class ExtractedCredential:
    raw_key: str
    scheme: CredentialScheme


@dataclass(frozen=True, slots=True)
class ExtractionFailure:
    reason: str
    conflicting: bool = False


def protected_route_family(
    method: str,
    path: str,
    *,
    anthropic_version_present: bool = False,
) -> RouteFamily | None:
    """Return the protocol family for an exact protected operation."""

    normalized_method = method.upper()
    if (
        normalized_method == "GET"
        and SYSTEM_MODEL_DETAIL.fullmatch(path) is not None
    ):
        return "system"
    if (normalized_method, path) not in PROTECTED_OPERATIONS:
        return None
    if path in {"/v1/messages", "/v1/messages/count_tokens"} or (
        path == "/v1/models" and anthropic_version_present
    ):
        return "anthropic"
    if path.startswith("/v1/"):
        return "openai"
    return "system"


def _header_values(request: Request, name: bytes) -> list[str]:
    return [
        value.decode("latin-1")
        for key, value in request.scope.get("headers", ())
        if key.lower() == name
    ]


def extract_credential(
    request: Request,
) -> ExtractedCredential | ExtractionFailure:
    """Extract exactly one Bearer or x-api-key credential."""

    authorization_values = _header_values(request, b"authorization")
    api_key_values = _header_values(request, b"x-api-key")
    if len(authorization_values) > 1:
        return ExtractionFailure("multiple_authorization_headers")
    if len(api_key_values) > 1:
        return ExtractionFailure("multiple_x_api_key_headers")

    bearer_token: str | None = None
    if authorization_values:
        components = authorization_values[0].strip().split()
        if (
            len(components) != 2
            or components[0].lower() != "bearer"
            or not components[1]
        ):
            return ExtractionFailure("unsupported_or_malformed_authorization")
        bearer_token = components[1].strip()
        if not bearer_token:
            return ExtractionFailure("empty_bearer")

    api_key_token: str | None = None
    if api_key_values:
        api_key_token = api_key_values[0].strip()
        if not api_key_token:
            return ExtractionFailure("empty_x_api_key")

    if bearer_token is None and api_key_token is None:
        return ExtractionFailure("missing")
    if bearer_token is not None and api_key_token is not None:
        identical = hmac.compare_digest(
            bearer_token.encode("utf-8"),
            api_key_token.encode("utf-8"),
        )
        if not identical:
            return ExtractionFailure("conflicting", conflicting=True)
        return ExtractedCredential(bearer_token, "dual")
    if bearer_token is not None:
        return ExtractedCredential(bearer_token, "bearer")
    if api_key_token is None:
        raise RuntimeError("credential extraction invariant failed")
    return ExtractedCredential(api_key_token, "x-api-key")


def _response_headers(
    family: RouteFamily,
    request_id: str,
    *,
    status_code: int,
) -> dict[str, str]:
    headers = {REQUEST_ID_HEADER: request_id}
    if family == "openai":
        headers[OPENAI_REQUEST_ID_HEADER] = request_id
        headers[OPENAI_COMPATIBILITY_HEADER] = OPENAI_COMPATIBILITY_VERSION
    elif family == "anthropic":
        headers[ANTHROPIC_REQUEST_ID_HEADER] = anthropic_request_id(request_id)
        headers[ANTHROPIC_COMPATIBILITY_HEADER] = (
            ANTHROPIC_COMPATIBILITY_VERSION
        )
    if status_code == 401:
        headers["WWW-Authenticate"] = CHALLENGE
    return headers


def authentication_error_response(
    family: RouteFamily,
    request_id: str,
    *,
    conflicting: bool = False,
) -> JSONResponse:
    """Render one family-shaped public authentication rejection."""

    status_code = 400 if conflicting else 401
    message = CONFLICT_MESSAGE if conflicting else AUTHENTICATION_MESSAGE
    if family == "system":
        content = {
            "request_id": request_id,
            "status": "error",
            "error": {
                "code": "system_x_authentication_error",
                "message": message,
                "retryable": False,
                "details": {},
            },
        }
    elif family == "openai":
        content = {
            "error": {
                "message": message,
                "type": (
                    "invalid_request_error" if conflicting else "authentication_error"
                ),
                "param": None,
                "code": (
                    "conflicting_credentials" if conflicting else "invalid_api_key"
                ),
            }
        }
    else:
        content = {
            "type": "error",
            "error": {
                "type": (
                    "invalid_request_error" if conflicting else "authentication_error"
                ),
                "message": message,
            },
            "request_id": anthropic_request_id(request_id),
        }
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=_response_headers(
            family,
            request_id,
            status_code=status_code,
        ),
    )


class AuthenticationManager:
    """Authenticate a request against one canonical credential store."""

    def __init__(self, store: CredentialStore, *, enabled: bool) -> None:
        self.store = store
        self.enabled = enabled

    def validate_startup(self) -> dict[str, object]:
        if not self.enabled:
            return {"authentication_enabled": False}
        return {
            "authentication_enabled": True,
            **self.store.inspect(require_active=True),
        }

    @staticmethod
    def _log_event(
        *,
        request_id: str,
        outcome: str,
        reason: str,
        scheme: CredentialScheme | None = None,
        key_id: str | None = None,
        label: str | None = None,
    ) -> None:
        event = {
            "event": "system_x_authentication",
            "request_id": request_id,
            "outcome": outcome,
            "reason": reason,
            "credential_scheme": scheme,
            "key_id": key_id,
            "label": redact_text(label)[:128] if label is not None else None,
        }
        LOGGER.info(
            "system_x_authentication %s",
            json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def authenticate_request(
        self,
        request: Request,
        family: RouteFamily,
    ) -> JSONResponse | None:
        if not self.enabled:
            return None
        request_id = request_id_for(request)
        extracted = extract_credential(request)
        if isinstance(extracted, ExtractionFailure):
            request.state.system_x_authentication_reason = extracted.reason
            self._log_event(
                request_id=request_id,
                outcome="rejected",
                reason=extracted.reason,
                scheme="dual" if extracted.conflicting else None,
            )
            return authentication_error_response(
                family,
                request_id,
                conflicting=extracted.conflicting,
            )
        verified = self.store.verify(extracted.raw_key)
        if (
            not verified.accepted
            or verified.key_id is None
            or verified.label is None
        ):
            request.state.system_x_authentication_reason = verified.reason
            self._log_event(
                request_id=request_id,
                outcome="rejected",
                reason=verified.reason,
                scheme=extracted.scheme,
            )
            return authentication_error_response(family, request_id)
        context = AuthenticationContext(
            request_id=request_id,
            key_id=verified.key_id,
            label=verified.label,
            credential_scheme=extracted.scheme,
        )
        request.state.system_x_authentication = context
        request.state.system_x_authentication_reason = "accepted"
        self._log_event(
            request_id=request_id,
            outcome="accepted",
            reason="accepted",
            scheme=context.credential_scheme,
            key_id=context.key_id,
            label=context.label,
        )
        return None
