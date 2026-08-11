"""Off-by-default, metadata-only privacy diagnostics."""

from __future__ import annotations

import json
import logging
from typing import Final, Literal

from .operation_metrics import error_class_for, latency_bucket
from .operation_records import OperationRecord


PRIVACY_DIAGNOSTIC_CONTRACT: Final = "system-x.privacy-diagnostic.v1"
PRIVACY_DIAGNOSTIC_EVENT_NAME: Final = "system_x_privacy_diagnostic"
MAX_PRIVACY_DIAGNOSTIC_BYTES: Final = 2_048
PrivacyDiagnosticMode = Literal["off", "metadata"]
PRIVACY_DIAGNOSTIC_FIELDS: Final = (
    "schema",
    "request_id",
    "protocol_family",
    "endpoint",
    "operation",
    "streamed",
    "model_resolved",
    "backend_reached",
    "usage_available",
    "operation_state",
    "http_status",
    "error_class",
    "finish_reason",
    "latency_bucket",
)


class PrivacyDiagnostics:
    """Emit at most one bounded allowlisted event per terminal record."""

    def __init__(
        self,
        mode: PrivacyDiagnosticMode,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        if mode not in {"off", "metadata"}:
            raise ValueError("privacy diagnostic mode is invalid")
        self.mode = mode
        self._logger = logger or logging.getLogger("system_x_gguf_api")

    def observe(self, record: OperationRecord) -> None:
        if self.mode == "off":
            return
        try:
            event = {
                "schema": PRIVACY_DIAGNOSTIC_CONTRACT,
                "request_id": record.request_id,
                "protocol_family": record.protocol_family,
                "endpoint": record.endpoint,
                "operation": record.operation,
                "streamed": record.streamed,
                "model_resolved": record.public_model_id is not None,
                "backend_reached": record.router_transaction_id is not None,
                "usage_available": (
                    record.input_tokens is not None
                    and record.output_tokens is not None
                ),
                "operation_state": record.operation_state,
                "http_status": record.http_status,
                "error_class": error_class_for(record.error_code),
                "finish_reason": record.finish_reason or "unknown",
                "latency_bucket": latency_bucket(record.latency_ms),
            }
            if tuple(event) != PRIVACY_DIAGNOSTIC_FIELDS:
                raise RuntimeError("privacy diagnostic field set changed")
            encoded = json.dumps(
                event,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            if "\n" in encoded or "\r" in encoded or "\x00" in encoded:
                raise RuntimeError("privacy diagnostic is not one safe line")
            if len(encoded.encode("utf-8")) > MAX_PRIVACY_DIAGNOSTIC_BYTES:
                raise RuntimeError("privacy diagnostic exceeds its bound")
            self._logger.info(
                "%s %s",
                PRIVACY_DIAGNOSTIC_EVENT_NAME,
                encoded,
            )
        except Exception:
            raise
