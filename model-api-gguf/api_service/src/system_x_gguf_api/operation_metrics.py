"""Fixed-cardinality, process-local operation metrics."""

from __future__ import annotations

import json
import threading
from typing import Final

from .operation_records import OperationRecord, utc_now


METRICS_CONTRACT: Final = "system-x.operation-metrics.v1"
MAX_METRICS_RESPONSE_BYTES: Final = 16_384
USAGE_OPERATIONS: Final = frozenset(
    {"generate", "chat", "responses", "tokens.count"}
)
PROTOCOL_FAMILIES: Final = (
    "system_x",
    "openai_compatible",
    "messages_compatible",
)
OPERATIONS: Final = (
    "model.list",
    "model.detail",
    "generate",
    "chat",
    "responses",
    "tokens.count",
)
STATES: Final = (
    "completed",
    "incomplete",
    "requires_action",
    "cancelled",
    "failed",
)
HTTP_STATUS_CLASSES: Final = ("1xx", "2xx", "3xx", "4xx", "5xx")
FINISH_REASONS: Final = (
    "none",
    "completed",
    "output_limit",
    "stop_sequence",
    "context_limit",
    "tool_call",
    "unknown",
)
ERROR_CLASSES: Final = (
    "none",
    "authentication",
    "validation",
    "governance",
    "model",
    "backend",
    "tool",
    "structured_output",
    "internal",
    "other",
)
LATENCY_BUCKETS: Final = (
    "le_10",
    "le_50",
    "le_100",
    "le_250",
    "le_500",
    "le_1000",
    "le_5000",
    "gt_5000",
)
ERROR_CLASS_BY_CODE: Final[dict[str, str]] = {
    "system_x_authentication_error": "authentication",
    "system_x_validation_error": "validation",
    "system_x_request_too_large": "governance",
    "system_x_token_budget_exceeded": "governance",
    "system_x_concurrency_limit_exceeded": "governance",
    "system_x_rate_limit_exceeded": "governance",
    "system_x_request_deadline_exceeded": "governance",
    "system_x_route_not_found": "validation",
    "system_x_method_not_allowed": "validation",
    "system_x_model_not_found": "model",
    "system_x_no_ready_model": "model",
    "system_x_model_unavailable": "model",
    "system_x_model_conflict": "model",
    "system_x_capability_unavailable": "model",
    "system_x_backend_unavailable": "backend",
    "system_x_backend_timeout": "backend",
    "system_x_backend_response_invalid": "backend",
    "system_x_output_invalid": "backend",
    "system_x_tool_schema_invalid": "tool",
    "system_x_tool_choice_invalid": "tool",
    "system_x_tool_capability_unavailable": "tool",
    "system_x_tool_call_invalid": "tool",
    "system_x_tool_arguments_invalid": "tool",
    "system_x_tool_result_mismatch": "tool",
    "system_x_tool_result_duplicate": "tool",
    "system_x_tool_result_missing": "tool",
    "system_x_structured_output_schema_invalid": "structured_output",
    "system_x_structured_output_invalid": "structured_output",
    "system_x_streaming_structured_output_unsupported": "structured_output",
    "system_x_tool_and_output_format_conflict": "structured_output",
    "system_x_internal_error": "internal",
}


def error_class_for(error_code: str | None) -> str:
    if error_code is None:
        return "none"
    return ERROR_CLASS_BY_CODE.get(error_code, "other")


def latency_bucket(latency_ms: int) -> str:
    if latency_ms <= 10:
        return "le_10"
    if latency_ms <= 50:
        return "le_50"
    if latency_ms <= 100:
        return "le_100"
    if latency_ms <= 250:
        return "le_250"
    if latency_ms <= 500:
        return "le_500"
    if latency_ms <= 1_000:
        return "le_1000"
    if latency_ms <= 5_000:
        return "le_5000"
    return "gt_5000"


def http_status_class(status: int) -> str:
    return f"{status // 100}xx"


class OperationMetrics:
    """Count accepted terminal records without retaining request history."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._operations_total = 0
        self._streamed_operations_total = 0
        self._input_tokens_total = 0
        self._output_tokens_total = 0
        self._input_tokens_unknown_records = 0
        self._output_tokens_unknown_records = 0
        self._operation_record_emission_failures = 0
        self._operation_observer_failures = 0
        self._states = {key: 0 for key in STATES}
        self._protocol_families = {key: 0 for key in PROTOCOL_FAMILIES}
        self._operations = {key: 0 for key in OPERATIONS}
        self._http_status_classes = {key: 0 for key in HTTP_STATUS_CLASSES}
        self._finish_reasons = {key: 0 for key in FINISH_REASONS}
        self._error_classes = {key: 0 for key in ERROR_CLASSES}
        self._latency_ms = {key: 0 for key in LATENCY_BUCKETS}

    def observe(self, record: OperationRecord) -> None:
        with self._lock:
            self._operations_total += 1
            if record.streamed:
                self._streamed_operations_total += 1
            self._protocol_families[record.protocol_family] += 1
            self._operations[record.operation] += 1
            self._states[record.operation_state] += 1
            self._http_status_classes[http_status_class(record.http_status)] += 1
            finish = record.finish_reason or "none"
            self._finish_reasons[finish] += 1
            self._error_classes[error_class_for(record.error_code)] += 1
            self._latency_ms[latency_bucket(record.latency_ms)] += 1
            if record.operation not in USAGE_OPERATIONS:
                return
            if record.input_tokens is None:
                self._input_tokens_unknown_records += 1
            else:
                self._input_tokens_total += record.input_tokens
            if record.output_tokens is None:
                self._output_tokens_unknown_records += 1
            else:
                self._output_tokens_total += record.output_tokens

    def observe_operation_record_emission_failure(
        self, _request_id: str, _error_type: str
    ) -> None:
        with self._lock:
            self._operation_record_emission_failures += 1

    def observe_observer_failure(self) -> None:
        with self._lock:
            self._operation_observer_failures += 1

    def snapshot(
        self,
        *,
        request_id: str,
        active_operations: int,
    ) -> dict[str, object]:
        with self._lock:
            payload: dict[str, object] = {
                "contract": METRICS_CONTRACT,
                "request_id": request_id,
                "generated_utc": utc_now(),
                "active_operations": active_operations,
                "operation_record_emission_failures": self._operation_record_emission_failures,
                "operation_observer_failures": self._operation_observer_failures,
                "operations_total": self._operations_total,
                "streamed_operations_total": self._streamed_operations_total,
                "usage": {
                    "input_tokens_total": self._input_tokens_total,
                    "output_tokens_total": self._output_tokens_total,
                    "input_tokens_unknown_records": self._input_tokens_unknown_records,
                    "output_tokens_unknown_records": self._output_tokens_unknown_records,
                },
                "states": dict(self._states),
                "protocol_families": dict(self._protocol_families),
                "operations": dict(self._operations),
                "http_status_classes": dict(self._http_status_classes),
                "finish_reasons": dict(self._finish_reasons),
                "error_classes": dict(self._error_classes),
                "latency_ms": dict(self._latency_ms),
            }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        if "\n" in encoded or "\r" in encoded or "\x00" in encoded:
            raise RuntimeError("metrics snapshot is not one safe line")
        if len(encoded.encode("utf-8")) > MAX_METRICS_RESPONSE_BYTES:
            raise RuntimeError("metrics snapshot exceeds its serialization bound")
        return payload


class OperationRecordObserver:
    """Isolate metrics and privacy observers from recorder finalization."""

    def __init__(self, metrics: OperationMetrics, diagnostics: object) -> None:
        self.metrics = metrics
        self.diagnostics = diagnostics

    def observe(self, record: OperationRecord) -> None:
        try:
            self.metrics.observe(record)
        except Exception:
            self.metrics.observe_observer_failure()
            return
        try:
            self.diagnostics.observe(record)  # type: ignore[attr-defined]
        except Exception:
            self.metrics.observe_observer_failure()

    def observe_operation_record_emission_failure(
        self, request_id: str, error_type: str
    ) -> None:
        self.metrics.observe_operation_record_emission_failure(
            request_id, error_type
        )

    def observe_observer_failure(self) -> None:
        self.metrics.observe_observer_failure()
