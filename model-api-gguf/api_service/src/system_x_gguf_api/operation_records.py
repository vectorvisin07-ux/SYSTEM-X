"""Metadata-only terminal operation records with strict privacy invariants."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
import logging
import os
import re
import threading
import time
from typing import Any, Final, Literal


OPERATION_RECORD_SCHEMA: Final = "system-x.operation-record.v1"
OperationRecordSchema = Literal[OPERATION_RECORD_SCHEMA]
OPERATION_EVENT_NAME: Final = "system_x_operation"
OPERATION_FAILURE_EVENT_NAME: Final = (
    "system_x_operation_record_failure"
)
SERVICE_TRANSACTION_ENV: Final = "SYSTEM_X_API_SERVICE_TRANSACTION_ID"
MAX_OPERATION_RECORD_BYTES: Final = 4096
MAX_TOKEN_COUNT: Final = 2_147_483_647
MAX_LATENCY_MS: Final = 86_400_000

REQUEST_ID_PATTERN = re.compile(r"^sx_req_[0-9a-f]{32}$")
KEY_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
PUBLIC_MODEL_ID_PATTERN = re.compile(
    r"^sx-gguf-[a-z0-9](?:[a-z0-9-]{0,118}[a-z0-9])?$"
)
BOUNDED_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)
TRANSACTION_ID_PATTERN = re.compile(
    r"^tx-[A-Za-z0-9][A-Za-z0-9._:-]{0,124}$"
)
ERROR_CODE_PATTERN = re.compile(r"^system_x_[a-z0-9_]{1,96}$")
UTC_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?Z$"
)
SYSTEM_MODEL_DETAIL_PATTERN = re.compile(
    r"^/system/v1/models/[^/]+$"
)

ProtocolFamily = Literal[
    "system_x",
    "openai_compatible",
    "messages_compatible",
]
OperationName = Literal[
    "model.list",
    "model.detail",
    "generate",
    "chat",
    "responses",
    "tokens.count",
]
FinishReason = Literal[
    "completed",
    "output_limit",
    "stop_sequence",
    "context_limit",
    "tool_call",
    "unknown",
]
OperationState = Literal[
    "completed",
    "incomplete",
    "requires_action",
    "cancelled",
    "failed",
]


class OperationRecordInvariantError(RuntimeError):
    """An operation record or span violated its private contract."""


@dataclass(frozen=True, slots=True)
class OperationRoute:
    protocol_family: ProtocolFamily
    endpoint: str
    operation: OperationName


OPERATION_ROUTES: Final[dict[tuple[str, str], OperationRoute]] = {
    ("GET", "/system/v1/models"): OperationRoute(
        "system_x", "/system/v1/models", "model.list"
    ),
    ("POST", "/system/v1/generate"): OperationRoute(
        "system_x", "/system/v1/generate", "generate"
    ),
    ("POST", "/system/v1/chat"): OperationRoute(
        "system_x", "/system/v1/chat", "chat"
    ),
    ("POST", "/system/v1/responses"): OperationRoute(
        "system_x", "/system/v1/responses", "responses"
    ),
    ("POST", "/system/v1/tokens/count"): OperationRoute(
        "system_x", "/system/v1/tokens/count", "tokens.count"
    ),
    ("GET", "/v1/models"): OperationRoute(
        "openai_compatible", "/v1/models", "model.list"
    ),
    ("POST", "/v1/completions"): OperationRoute(
        "openai_compatible", "/v1/completions", "generate"
    ),
    ("POST", "/v1/chat/completions"): OperationRoute(
        "openai_compatible", "/v1/chat/completions", "chat"
    ),
    ("POST", "/v1/responses"): OperationRoute(
        "openai_compatible", "/v1/responses", "responses"
    ),
    ("POST", "/v1/messages"): OperationRoute(
        "messages_compatible", "/v1/messages", "chat"
    ),
    ("POST", "/v1/messages/count_tokens"): OperationRoute(
        "messages_compatible",
        "/v1/messages/count_tokens",
        "tokens.count",
    ),
}
ALLOWED_PROTOCOL_FAMILIES: Final = frozenset(
    {"system_x", "openai_compatible", "messages_compatible"}
)
ALLOWED_OPERATIONS: Final = frozenset(
    {
        "model.list",
        "model.detail",
        "generate",
        "chat",
        "responses",
        "tokens.count",
    }
)
ALLOWED_ENDPOINTS: Final = frozenset(
    {
        route.endpoint for route in OPERATION_ROUTES.values()
    }
    | {"/system/v1/models/{model_id}"}
)
ALLOWED_FINISH_REASONS: Final = frozenset(
    {
        "completed",
        "output_limit",
        "stop_sequence",
        "context_limit",
        "tool_call",
        "unknown",
    }
)
ALLOWED_OPERATION_STATES: Final = frozenset(
    {
        "completed",
        "incomplete",
        "requires_action",
        "cancelled",
        "failed",
    }
)
OPERATION_RECORD_FIELDS: Final = (
    "schema",
    "request_id",
    "key_id",
    "protocol_family",
    "endpoint",
    "operation",
    "streamed",
    "public_model_id",
    "artifact_version_id",
    "api_service_transaction_id",
    "router_transaction_id",
    "started_utc",
    "completed_utc",
    "latency_ms",
    "http_status",
    "error_code",
    "finish_reason",
    "operation_state",
    "input_tokens",
    "output_tokens",
)


def operation_route_for(
    method: str,
    path: str,
    *,
    anthropic_version_present: bool = False,
) -> OperationRoute | None:
    """Map one public request to a content-free canonical route."""

    normalized_method = method.upper()
    if (
        normalized_method == "GET"
        and SYSTEM_MODEL_DETAIL_PATTERN.fullmatch(path) is not None
    ):
        return OperationRoute(
            "system_x",
            "/system/v1/models/{model_id}",
            "model.detail",
        )
    route = OPERATION_ROUTES.get((normalized_method, path))
    if (
        route is not None
        and normalized_method == "GET"
        and path == "/v1/models"
        and anthropic_version_present
    ):
        return OperationRoute(
            "messages_compatible", "/v1/models", "model.list"
        )
    return route


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _require_utc(value: str, name: str) -> None:
    if not isinstance(value, str) or UTC_PATTERN.fullmatch(value) is None:
        raise OperationRecordInvariantError(f"{name} is not bounded UTC")
    try:
        dt.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise OperationRecordInvariantError(
            f"{name} is not valid UTC"
        ) from exc


def _require_optional_token_count(value: int | None, name: str) -> None:
    if value is not None and (
        type(value) is not int
        or not 0 <= value <= MAX_TOKEN_COUNT
    ):
        raise OperationRecordInvariantError(
            f"{name} is not a physical token count"
        )


@dataclass(frozen=True, slots=True)
class OperationRecord:
    schema: str
    request_id: str
    key_id: str | None
    protocol_family: ProtocolFamily
    endpoint: str
    operation: OperationName
    streamed: bool
    public_model_id: str | None
    artifact_version_id: str | None
    api_service_transaction_id: str
    router_transaction_id: str | None
    started_utc: str
    completed_utc: str
    latency_ms: int
    http_status: int
    error_code: str | None
    finish_reason: FinishReason | None
    operation_state: OperationState
    input_tokens: int | None
    output_tokens: int | None

    def __post_init__(self) -> None:
        if self.schema != OPERATION_RECORD_SCHEMA:
            raise OperationRecordInvariantError(
                "operation schema identity is invalid"
            )
        if REQUEST_ID_PATTERN.fullmatch(self.request_id) is None:
            raise OperationRecordInvariantError("request ID is invalid")
        if self.key_id is not None and (
            KEY_ID_PATTERN.fullmatch(self.key_id) is None
        ):
            raise OperationRecordInvariantError("key ID is invalid")
        if self.protocol_family not in ALLOWED_PROTOCOL_FAMILIES:
            raise OperationRecordInvariantError(
                "protocol family is invalid"
            )
        if self.endpoint not in ALLOWED_ENDPOINTS:
            raise OperationRecordInvariantError(
                "endpoint template is invalid"
            )
        if self.operation not in ALLOWED_OPERATIONS:
            raise OperationRecordInvariantError("operation is invalid")
        if type(self.streamed) is not bool:
            raise OperationRecordInvariantError(
                "streamed must be Boolean"
            )
        if (self.public_model_id is None) != (
            self.artifact_version_id is None
        ):
            raise OperationRecordInvariantError(
                "model and artifact identities must be jointly nullable"
            )
        if self.public_model_id is not None and (
            PUBLIC_MODEL_ID_PATTERN.fullmatch(self.public_model_id) is None
        ):
            raise OperationRecordInvariantError(
                "public model ID is invalid"
            )
        if self.artifact_version_id is not None and (
            BOUNDED_ID_PATTERN.fullmatch(self.artifact_version_id) is None
        ):
            raise OperationRecordInvariantError(
                "artifact version ID is invalid"
            )
        if (
            TRANSACTION_ID_PATTERN.fullmatch(
                self.api_service_transaction_id
            )
            is None
        ):
            raise OperationRecordInvariantError(
                "API-service transaction ID is invalid"
            )
        if self.router_transaction_id is not None and (
            TRANSACTION_ID_PATTERN.fullmatch(
                self.router_transaction_id
            )
            is None
        ):
            raise OperationRecordInvariantError(
                "router transaction ID is invalid"
            )
        if (
            self.router_transaction_id is not None
            and self.public_model_id is None
        ):
            raise OperationRecordInvariantError(
                "router identity requires resolved model metadata"
            )
        _require_utc(self.started_utc, "started_utc")
        _require_utc(self.completed_utc, "completed_utc")
        if type(self.latency_ms) is not int or not (
            0 <= self.latency_ms <= MAX_LATENCY_MS
        ):
            raise OperationRecordInvariantError("latency is invalid")
        if type(self.http_status) is not int or not (
            100 <= self.http_status <= 599
        ):
            raise OperationRecordInvariantError("HTTP status is invalid")
        if self.error_code is not None and (
            ERROR_CODE_PATTERN.fullmatch(self.error_code) is None
        ):
            raise OperationRecordInvariantError("error code is invalid")
        if (
            self.finish_reason is not None
            and self.finish_reason not in ALLOWED_FINISH_REASONS
        ):
            raise OperationRecordInvariantError(
                "finish reason is invalid"
            )
        if self.operation_state not in ALLOWED_OPERATION_STATES:
            raise OperationRecordInvariantError(
                "operation state is invalid"
            )
        _require_optional_token_count(
            self.input_tokens, "input_tokens"
        )
        _require_optional_token_count(
            self.output_tokens, "output_tokens"
        )
        if self.http_status >= 400 and (
            self.operation_state != "failed"
            or self.error_code is None
        ):
            raise OperationRecordInvariantError(
                "public failures require failed state and error code"
            )
        if self.operation_state == "failed" and self.error_code is None:
            raise OperationRecordInvariantError(
                "failed operations require an error code"
            )
        if self.operation_state != "failed" and self.error_code is not None:
            raise OperationRecordInvariantError(
                "non-failed operations cannot carry an error code"
            )
        if self.operation_state == "cancelled" and (
            self.error_code is not None
        ):
            raise OperationRecordInvariantError(
                "cancelled operations cannot carry an error code"
            )
        if (
            self.operation_state == "requires_action"
            and self.finish_reason != "tool_call"
        ):
            raise OperationRecordInvariantError(
                "requires-action state must have tool-call finish"
            )

    def as_dict(self) -> dict[str, object]:
        values = {
            field: getattr(self, field)
            for field in OPERATION_RECORD_FIELDS
        }
        if tuple(values) != OPERATION_RECORD_FIELDS:
            raise OperationRecordInvariantError(
                "operation record field order changed"
            )
        return values

    def serialize(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if "\n" in encoded or "\r" in encoded or "\0" in encoded:
            raise OperationRecordInvariantError(
                "operation record is not one safe physical line"
            )
        if len(encoded.encode("utf-8")) > MAX_OPERATION_RECORD_BYTES:
            raise OperationRecordInvariantError(
                "operation record exceeds its serialization bound"
            )
        return encoded


@dataclass(slots=True)
class OperationSpan:
    """One active request; stores only allowlisted non-content metadata."""

    route: OperationRoute
    request_id: str
    key_id: str | None
    api_service_transaction_id: str
    started_utc: str
    started_monotonic_ns: int
    streamed: bool = False
    public_model_id: str | None = None
    artifact_version_id: str | None = None
    router_transaction_id: str | None = None
    error_code: str | None = None
    finish_reason: FinishReason | None = None
    operation_state: OperationState | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finalized: bool = False

    def __post_init__(self) -> None:
        if REQUEST_ID_PATTERN.fullmatch(self.request_id) is None:
            raise OperationRecordInvariantError("span request ID is invalid")
        if self.key_id is not None and KEY_ID_PATTERN.fullmatch(self.key_id) is None:
            raise OperationRecordInvariantError("span key ID is invalid")
        if (
            TRANSACTION_ID_PATTERN.fullmatch(
                self.api_service_transaction_id
            )
            is None
        ):
            raise OperationRecordInvariantError(
                "span API transaction ID is invalid"
            )
        _require_utc(self.started_utc, "started_utc")
        if (
            type(self.started_monotonic_ns) is not int
            or self.started_monotonic_ns < 0
        ):
            raise OperationRecordInvariantError(
                "span monotonic start is invalid"
            )

    def _require_active(self) -> None:
        if self.finalized:
            raise OperationRecordInvariantError(
                "operation span was already finalized"
            )

    def mark_streamed(self) -> None:
        self._require_active()
        self.streamed = True

    def note_model(
        self, public_model_id: str, artifact_version_id: str
    ) -> None:
        self._require_active()
        if PUBLIC_MODEL_ID_PATTERN.fullmatch(public_model_id) is None:
            raise OperationRecordInvariantError(
                "span public model ID is invalid"
            )
        if BOUNDED_ID_PATTERN.fullmatch(artifact_version_id) is None:
            raise OperationRecordInvariantError(
                "span artifact version ID is invalid"
            )
        if self.public_model_id not in {None, public_model_id}:
            raise OperationRecordInvariantError(
                "span public model identity changed"
            )
        if self.artifact_version_id not in {
            None,
            artifact_version_id,
        }:
            raise OperationRecordInvariantError(
                "span artifact version identity changed"
            )
        self.public_model_id = public_model_id
        self.artifact_version_id = artifact_version_id

    def note_router(self, transaction_id: str) -> None:
        self._require_active()
        if TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None:
            raise OperationRecordInvariantError(
                "span router transaction ID is invalid"
            )
        if self.public_model_id is None:
            raise OperationRecordInvariantError(
                "router correlation preceded model resolution"
            )
        if self.router_transaction_id not in {None, transaction_id}:
            raise OperationRecordInvariantError(
                "span router transaction identity changed"
            )
        self.router_transaction_id = transaction_id

    def note_terminal(
        self,
        *,
        state: OperationState,
        finish_reason: FinishReason | None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error_code: str | None = None,
    ) -> None:
        self._require_active()
        if self.operation_state is not None:
            raise OperationRecordInvariantError(
                "span terminal metadata was already supplied"
            )
        if state not in ALLOWED_OPERATION_STATES:
            raise OperationRecordInvariantError(
                "span terminal state is invalid"
            )
        if (
            finish_reason is not None
            and finish_reason not in ALLOWED_FINISH_REASONS
        ):
            raise OperationRecordInvariantError(
                "span finish reason is invalid"
            )
        _require_optional_token_count(input_tokens, "input_tokens")
        _require_optional_token_count(output_tokens, "output_tokens")
        if error_code is not None and (
            ERROR_CODE_PATTERN.fullmatch(error_code) is None
        ):
            raise OperationRecordInvariantError(
                "span error code is invalid"
            )
        if state == "failed" and error_code is None:
            raise OperationRecordInvariantError(
                "span failure requires an error code"
            )
        if state != "failed" and error_code is not None:
            raise OperationRecordInvariantError(
                "span non-failure cannot have an error code"
            )
        self.operation_state = state
        self.finish_reason = finish_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.error_code = error_code

    def note_cancelled(self) -> None:
        self._require_active()
        if self.operation_state is None:
            self.operation_state = "cancelled"
            self.finish_reason = None
            self.error_code = None
            self.input_tokens = None
            self.output_tokens = None

    def finalize(
        self,
        *,
        http_status: int,
        completed_utc: str,
        completed_monotonic_ns: int,
    ) -> OperationRecord:
        self._require_active()
        if (
            type(completed_monotonic_ns) is not int
            or completed_monotonic_ns < self.started_monotonic_ns
        ):
            raise OperationRecordInvariantError(
                "operation monotonic completion is invalid"
            )
        state = self.operation_state
        error_code = self.error_code
        if state is None:
            if http_status >= 400:
                raise OperationRecordInvariantError(
                    "public failure is missing canonical error metadata"
                )
            state = "completed"
        latency_ms = (
            completed_monotonic_ns - self.started_monotonic_ns
        ) // 1_000_000
        record = OperationRecord(
            schema=OPERATION_RECORD_SCHEMA,
            request_id=self.request_id,
            key_id=self.key_id,
            protocol_family=self.route.protocol_family,
            endpoint=self.route.endpoint,
            operation=self.route.operation,
            streamed=self.streamed,
            public_model_id=self.public_model_id,
            artifact_version_id=self.artifact_version_id,
            api_service_transaction_id=self.api_service_transaction_id,
            router_transaction_id=self.router_transaction_id,
            started_utc=self.started_utc,
            completed_utc=completed_utc,
            latency_ms=latency_ms,
            http_status=http_status,
            error_code=error_code,
            finish_reason=self.finish_reason,
            operation_state=state,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        record.serialize()
        self.finalized = True
        return record


class OperationRecorder:
    """Own only active spans and emit one terminal metadata record per request."""

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        failure_logger: logging.Logger | None = None,
        observer: Any | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger("uvicorn.error")
        self._failure_logger = failure_logger or logging.getLogger(
            "uvicorn.error"
        )
        self._active: dict[str, OperationSpan] = {}
        self._lock = threading.RLock()
        self._service_transaction_id: str | None = None
        self._running = False
        self._emission_failed = False
        self._emission_failure_count = 0
        self._observer = observer

    @staticmethod
    def _read_service_transaction_id() -> str:
        value = os.environ.get(SERVICE_TRANSACTION_ENV)
        if (
            not isinstance(value, str)
            or TRANSACTION_ID_PATTERN.fullmatch(value) is None
        ):
            raise OperationRecordInvariantError(
                "API-service transaction identity is unavailable"
            )
        return value

    def startup(
        self, *, service_transaction_id: str | None = None
    ) -> None:
        with self._lock:
            if self._running:
                raise OperationRecordInvariantError(
                    "operation recorder is already running"
                )
            if self._active:
                raise OperationRecordInvariantError(
                    "operation spans exist before startup"
                )
            identity = (
                service_transaction_id
                if service_transaction_id is not None
                else self._read_service_transaction_id()
            )
            if TRANSACTION_ID_PATTERN.fullmatch(identity) is None:
                raise OperationRecordInvariantError(
                    "API-service transaction identity is invalid"
                )
            self._service_transaction_id = identity
            self._running = True
            self._emission_failed = False

    def shutdown(self) -> None:
        with self._lock:
            if not self._running:
                raise OperationRecordInvariantError(
                    "operation recorder is not running"
                )
            if self._active:
                raise OperationRecordInvariantError(
                    "active operation spans remained at shutdown"
                )
            self._running = False
            self._service_transaction_id = None

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def emission_failure_observed(self) -> bool:
        with self._lock:
            return self._emission_failed

    @property
    def emission_failure_count(self) -> int:
        with self._lock:
            return self._emission_failure_count

    def begin(
        self,
        route: OperationRoute,
        *,
        request_id: str,
        key_id: str | None,
    ) -> OperationSpan:
        with self._lock:
            if (
                not self._running
                or self._service_transaction_id is None
            ):
                raise OperationRecordInvariantError(
                    "operation recorder is not running"
                )
            if request_id in self._active:
                raise OperationRecordInvariantError(
                    "operation request ID is already active"
                )
            span = OperationSpan(
                route=route,
                request_id=request_id,
                key_id=key_id,
                api_service_transaction_id=(
                    self._service_transaction_id
                ),
                started_utc=utc_now(),
                started_monotonic_ns=time.perf_counter_ns(),
            )
            self._active[request_id] = span
            return span

    def _span(self, request_id: str) -> OperationSpan:
        with self._lock:
            span = self._active.get(request_id)
            if span is None:
                raise OperationRecordInvariantError(
                    "operation request ID is not active"
                )
            return span

    def mark_streamed(self, request_id: str) -> None:
        with self._lock:
            self._span(request_id).mark_streamed()

    def note_model(
        self,
        request_id: str,
        public_model_id: str,
        artifact_version_id: str,
    ) -> None:
        with self._lock:
            self._span(request_id).note_model(
                public_model_id, artifact_version_id
            )

    def note_router(
        self, request_id: str, transaction_id: str
    ) -> None:
        with self._lock:
            self._span(request_id).note_router(transaction_id)

    def note_terminal(
        self,
        request_id: str,
        *,
        state: OperationState,
        finish_reason: FinishReason | None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            self._span(request_id).note_terminal(
                state=state,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error_code=error_code,
            )

    def note_error(self, request_id: str, error_code: str) -> None:
        if ERROR_CODE_PATTERN.fullmatch(error_code) is None:
            raise OperationRecordInvariantError(
                "operation error code is invalid"
            )
        with self._lock:
            span = self._span(request_id)
            span._require_active()
            span.operation_state = "failed"
            span.error_code = error_code

    def note_error_if_active(
        self, request_id: str, error_code: str
    ) -> bool:
        if ERROR_CODE_PATTERN.fullmatch(error_code) is None:
            raise OperationRecordInvariantError(
                "operation error code is invalid"
            )
        with self._lock:
            span = self._active.get(request_id)
            if span is None:
                return False
            span._require_active()
            span.operation_state = "failed"
            span.error_code = error_code
            return True

    def note_cancelled(self, request_id: str) -> None:
        with self._lock:
            self._span(request_id).note_cancelled()

    def note_cancelled_if_active(self, request_id: str) -> bool:
        with self._lock:
            span = self._active.get(request_id)
            if span is None:
                return False
            span.note_cancelled()
            return True

    def _record_emission_failure(
        self, request_id: str, error_type: str
    ) -> None:
        failure = {
            "event": OPERATION_FAILURE_EVENT_NAME,
            "request_id": request_id,
            "error_type": error_type[:96],
        }
        try:
            self._failure_logger.error(
                "%s %s",
                OPERATION_FAILURE_EVENT_NAME,
                json.dumps(
                    failure,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        except Exception:
            return

    def _notify_observer(self, method: str, *args: object) -> None:
        observer = self._observer
        if observer is None:
            return
        callback = getattr(observer, method, None)
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            if method == "observe_observer_failure":
                return
            failure_callback = getattr(
                observer, "observe_observer_failure", None
            )
            if failure_callback is not None:
                try:
                    failure_callback()
                except Exception:
                    return

    def finalize(
        self, request_id: str, *, http_status: int
    ) -> OperationRecord | None:
        with self._lock:
            span = self._active.get(request_id)
            if span is None:
                raise OperationRecordInvariantError(
                    "operation request ID is not active"
                )
            try:
                record = span.finalize(
                    http_status=http_status,
                    completed_utc=utc_now(),
                    completed_monotonic_ns=time.perf_counter_ns(),
                )
                serialized = record.serialize()
                self._logger.info(
                    "%s %s", OPERATION_EVENT_NAME, serialized
                )
            except Exception as exc:
                self._emission_failed = True
                self._emission_failure_count += 1
                self._record_emission_failure(
                    request_id, type(exc).__name__
                )
                self._notify_observer(
                    "observe_operation_record_emission_failure",
                    request_id,
                    type(exc).__name__,
                )
                record = None
            finally:
                del self._active[request_id]
            if record is not None:
                self._notify_observer("observe", record)
            return record

    def finalize_if_active(
        self, request_id: str, *, http_status: int
    ) -> OperationRecord | None:
        with self._lock:
            if request_id not in self._active:
                return None
        return self.finalize(request_id, http_status=http_status)
