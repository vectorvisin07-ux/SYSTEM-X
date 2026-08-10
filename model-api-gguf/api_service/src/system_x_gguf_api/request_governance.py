"""Bounded, content-free admission controls for protected API requests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any, Callable

from fastapi import Request


REQUEST_GOVERNANCE_CONTRACT = "system-x.request-governance.v1"


class GovernanceRejection(RuntimeError):
    """A bounded public rejection with no private request payload."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.public_message = message[:240]
        self.retry_after = retry_after


def _header_values(request: Request, name: bytes) -> list[bytes]:
    return [
        value
        for key, value in request.scope.get("headers", ())
        if key.lower() == name
    ]


async def read_body_and_replay(request: Request, maximum: int) -> int:
    """Read at most the configured body bound and replay accepted bytes once."""

    if request.method.upper() in {"GET", "HEAD"}:
        return 0
    declared_values = _header_values(request, b"content-length")
    declared: int | None = None
    if declared_values:
        parsed: list[int] = []
        for raw in declared_values:
            try:
                text = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise GovernanceRejection(
                    400,
                    "system_x_validation_error",
                    "Content-Length is malformed",
                ) from exc
            if not text.isdecimal():
                raise GovernanceRejection(
                    400,
                    "system_x_validation_error",
                    "Content-Length is malformed",
                )
            parsed.append(int(text))
        if len(set(parsed)) != 1:
            raise GovernanceRejection(
                400,
                "system_x_validation_error",
                "Content-Length values conflict",
            )
        declared = parsed[0]
        if declared > maximum:
            raise GovernanceRejection(
                413,
                "system_x_request_too_large",
                "Request body exceeds the configured byte limit",
            )

    receive = request._receive  # type: ignore[attr-defined]
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        message_type = message.get("type")
        if message_type == "http.disconnect":
            raise GovernanceRejection(
                400,
                "system_x_validation_error",
                "Request body disconnected before completion",
            )
        if message_type != "http.request":
            continue
        chunk = message.get("body", b"")
        if not isinstance(chunk, bytes):
            raise GovernanceRejection(
                400,
                "system_x_validation_error",
                "Request body bytes are invalid",
            )
        total += len(chunk)
        if total > maximum:
            raise GovernanceRejection(
                413,
                "system_x_request_too_large",
                "Request body exceeds the configured byte limit",
            )
        chunks.append(chunk)
        if not message.get("more_body", False):
            break
    if declared is not None and declared != total:
        raise GovernanceRejection(
            400,
            "system_x_validation_error",
            "Content-Length does not match the received body",
        )
    body = b"".join(chunks)
    # Starlette's BaseHTTPMiddleware replays a cached _body to downstream
    # requests.  Replacing _receive here would make its disconnect watcher
    # observe a second http.request after the body was consumed.
    request._body = body  # type: ignore[attr-defined]
    return total


class SlidingRateWindow:
    """Per-key monotonic sliding-window admission with bounded state."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._timestamps: dict[str | None, deque[float]] = {}
        self._lock = threading.Lock()

    def admit(self, key_id: str | None, allowance: int, window: float) -> int | None:
        now = float(self._clock())
        with self._lock:
            bucket = self._timestamps.setdefault(key_id, deque())
            while bucket and now - bucket[0] >= window:
                bucket.popleft()
            if len(bucket) >= allowance:
                remaining = max(0.0, window - (now - bucket[0]))
                return max(1, int(math.ceil(remaining)))
            bucket.append(now)
            return None

    def snapshot(self) -> dict[str | None, tuple[float, ...]]:
        with self._lock:
            return {key: tuple(values) for key, values in self._timestamps.items()}


@dataclass(slots=True)
class ConcurrencyLease:
    _owner: "RequestGovernance"
    _key_id: str | None
    _released: bool = False

    def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        self._owner._release(self._key_id)
        return True

    @property
    def released(self) -> bool:
        return self._released


class RequestGovernance:
    """One application-local manager for body, rate, concurrency and deadlines."""

    def __init__(
        self,
        settings: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_body_bytes = int(settings.request_max_body_bytes)
        self.max_total_tokens = int(settings.request_max_total_tokens)
        self.timeout_seconds = float(settings.request_timeout_seconds)
        self.concurrency_limit_per_key = int(
            settings.request_concurrency_limit_per_key
        )
        self.rate_limit_requests_per_key = int(
            settings.request_rate_limit_requests_per_key
        )
        self.rate_limit_window_seconds = float(
            settings.request_rate_limit_window_seconds
        )
        self._clock = clock
        self._rate = SlidingRateWindow(clock)
        self._active: dict[str | None, int] = {}
        self._lock = threading.Lock()

    def new_deadline(self) -> float:
        return float(self._clock()) + self.timeout_seconds

    def admit(self, key_id: str | None) -> ConcurrencyLease:
        retry_after = self._rate.admit(
            key_id,
            self.rate_limit_requests_per_key,
            self.rate_limit_window_seconds,
        )
        if retry_after is not None:
            raise GovernanceRejection(
                429,
                "system_x_rate_limit_exceeded",
                "Request rate limit exceeded",
                retry_after=retry_after,
            )
        with self._lock:
            active = self._active.get(key_id, 0)
            if active >= self.concurrency_limit_per_key:
                raise GovernanceRejection(
                    429,
                    "system_x_concurrency_limit_exceeded",
                    "Request concurrency limit exceeded",
                )
            self._active[key_id] = active + 1
        return ConcurrencyLease(self, key_id)

    def _release(self, key_id: str | None) -> None:
        with self._lock:
            active = self._active.get(key_id, 0)
            if active <= 1:
                self._active.pop(key_id, None)
            else:
                self._active[key_id] = active - 1

    def active_snapshot(self) -> dict[str | None, int]:
        with self._lock:
            return dict(self._active)

    def enforce_total_token_budget(
        self,
        *,
        input_tokens: int,
        requested_output_tokens: int,
        model_context_tokens: int | None,
        model_maximum_output_tokens: int | None,
    ) -> int:
        if model_maximum_output_tokens is not None and (
            requested_output_tokens > model_maximum_output_tokens
        ):
            raise GovernanceRejection(
                422,
                "system_x_token_budget_exceeded",
                "Requested output exceeds the selected model output limit",
            )
        effective = self.max_total_tokens
        if model_context_tokens is not None:
            effective = min(effective, model_context_tokens)
        if input_tokens + requested_output_tokens > effective:
            raise GovernanceRejection(
                422,
                "system_x_token_budget_exceeded",
                "Request input and requested output exceed the token budget",
            )
        return effective


def deadline_event(family: str, request_id: str) -> bytes:
    """Return one bounded protocol-family-shaped post-header error event."""

    if family == "openai":
        payload = {
            "error": {
                "message": "Public request deadline exceeded",
                "type": "server_error",
                "param": None,
                "code": "system_x_request_deadline_exceeded",
            }
        }
        return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"
    if family == "anthropic":
        payload = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Public request deadline exceeded",
            },
        }
        return (
            b"event: error\ndata: "
            + json.dumps(payload, separators=(",", ":")).encode()
            + b"\n\n"
        )
    payload = {
        "request_id": request_id,
        "status": "error",
        "error": {
            "code": "system_x_request_deadline_exceeded",
            "message": "Public request deadline exceeded",
            "retryable": True,
            "details": {},
        },
    }
    return (
        b"event: error\ndata: "
        + json.dumps(payload, separators=(",", ":")).encode()
        + b"\n\n"
    )
