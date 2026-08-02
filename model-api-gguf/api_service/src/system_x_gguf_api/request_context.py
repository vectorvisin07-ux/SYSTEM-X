"""Server-owned request identity for every public System X request."""

from __future__ import annotations

import re
import secrets

from fastapi import Request

from .credential_types import AuthenticationContext


REQUEST_ID_HEADER = "X-System-X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^sx_req_[0-9a-f]{32}$")


def new_request_id() -> str:
    request_id = f"sx_req_{secrets.token_hex(16)}"
    if REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise RuntimeError("generated request identity is invalid")
    return request_id


def request_id_for(request: Request) -> str:
    request_id = getattr(request.state, "system_x_request_id", None)
    if not isinstance(request_id, str) or REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise RuntimeError("System X request identity is unavailable")
    return request_id


def authentication_context_for(request: Request) -> AuthenticationContext:
    context = getattr(request.state, "system_x_authentication", None)
    if not isinstance(context, AuthenticationContext) or not context.authenticated:
        raise RuntimeError("System X authentication context is unavailable")
    if context.request_id != request_id_for(request):
        raise RuntimeError("System X authentication context identity changed")
    return context
