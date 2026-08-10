"""Focused bounded-metrics and metadata-only privacy tests."""

from __future__ import annotations

import json
import logging
import unittest

from system_x_gguf_api.application import create_application
from system_x_gguf_api.authentication import protected_route_family
from system_x_gguf_api.operation_metrics import (
    METRICS_CONTRACT,
    OperationMetrics,
    OperationRecordObserver,
)
from system_x_gguf_api.operation_records import (
    OPERATION_RECORD_FIELDS,
    OperationRecorder,
    OperationRoute,
    operation_route_for,
)
from system_x_gguf_api.privacy_diagnostics import (
    PRIVACY_DIAGNOSTIC_CONTRACT,
    PRIVACY_DIAGNOSTIC_EVENT_NAME,
    PRIVACY_DIAGNOSTIC_FIELDS,
    PrivacyDiagnostics,
)
from system_x_gguf_api.settings import ServiceSettings


def _record(
    *,
    operation: str = "generate",
    endpoint: str = "/system/v1/generate",
    status: int = 200,
    state: str = "completed",
    error_code: str | None = None,
    streamed: bool = False,
    usage: tuple[int, int] | None = (3, 4),
    resolved: bool = True,
):
    recorder = OperationRecorder()
    recorder.startup(service_transaction_id="tx-service")
    span = recorder.begin(
        OperationRoute("system_x", endpoint, operation),
        request_id="sx_req_" + "a" * 32,
        key_id="b" * 32,
    )
    if streamed:
        span.mark_streamed()
    if resolved:
        span.note_model("sx-gguf-test-model", "artifact-v1")
        span.note_router("tx-router")
    input_tokens = usage[0] if usage is not None else None
    output_tokens = usage[1] if usage is not None else None
    span.note_terminal(
        state=state,  # type: ignore[arg-type]
        finish_reason="completed" if state == "completed" else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error_code=error_code,
    )
    result = recorder.finalize(
        "sx_req_" + "a" * 32,
        http_status=status,
    )
    recorder.shutdown()
    assert result is not None
    return result


class _RaisingObserver:
    def __init__(self) -> None:
        self.calls = 0

    def observe(self, _record) -> None:
        self.calls += 1
        raise RuntimeError("observer fixture failure")


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class UsageMetricsPrivacyTests(unittest.TestCase):
    def test_record_contract_and_failing_observer_are_preserved(self) -> None:
        observer = _RaisingObserver()
        recorder = OperationRecorder(observer=observer)
        recorder.startup(service_transaction_id="tx-service")
        span = recorder.begin(
            OperationRoute("system_x", "/system/v1/generate", "generate"),
            request_id="sx_req_" + "c" * 32,
            key_id="d" * 32,
        )
        span.note_model("sx-gguf-test-model", "artifact-v1")
        span.note_router("tx-router")
        span.note_terminal(
            state="completed",
            finish_reason="completed",
            input_tokens=3,
            output_tokens=4,
        )
        record = recorder.finalize("sx_req_" + "c" * 32, http_status=200)
        self.assertIsNotNone(record)
        self.assertEqual(observer.calls, 1)
        self.assertEqual(recorder.active_count, 0)
        self.assertEqual(tuple(record.as_dict()), OPERATION_RECORD_FIELDS)
        self.assertEqual(len(record.as_dict()), 20)
        recorder.shutdown()

    def test_fixed_metrics_and_unknown_usage_rules(self) -> None:
        metrics = OperationMetrics()
        metrics.observe(
            _record(
                operation="model.list",
                endpoint="/system/v1/models",
                usage=None,
                resolved=False,
            )
        )
        metrics.observe(_record())
        metrics.observe(
            _record(
                status=413,
                state="failed",
                error_code="system_x_request_too_large",
                usage=None,
                resolved=False,
            )
        )
        snapshot = metrics.snapshot(
            request_id="sx_req_" + "e" * 32,
            active_operations=0,
        )
        self.assertEqual(snapshot["contract"], METRICS_CONTRACT)
        self.assertEqual(snapshot["operations_total"], 3)
        self.assertEqual(snapshot["operations"]["model.list"], 1)
        self.assertEqual(snapshot["operations"]["generate"], 2)
        self.assertEqual(snapshot["usage"]["input_tokens_total"], 3)
        self.assertEqual(snapshot["usage"]["output_tokens_total"], 4)
        self.assertEqual(snapshot["usage"]["input_tokens_unknown_records"], 1)
        self.assertEqual(snapshot["usage"]["output_tokens_unknown_records"], 1)
        self.assertEqual(snapshot["error_classes"]["governance"], 1)
        self.assertEqual(
            set(snapshot["states"]),
            {"completed", "incomplete", "requires_action", "cancelled", "failed"},
        )
        self.assertEqual(
            set(snapshot["protocol_families"]),
            {"system_x", "openai_compatible", "messages_compatible"},
        )
        self.assertLessEqual(
            len(
                json.dumps(
                    snapshot,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode()
            ),
            16_384,
        )

    def test_metadata_diagnostic_is_bounded_and_allowlisted(self) -> None:
        logger = logging.getLogger("usage-metrics-privacy-test")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        capture = _Capture()
        logger.addHandler(capture)
        record = _record()
        PrivacyDiagnostics("off", logger=logger).observe(record)
        self.assertEqual(capture.messages, [])
        PrivacyDiagnostics("metadata", logger=logger).observe(record)
        self.assertEqual(len(capture.messages), 1)
        message = capture.messages[0]
        self.assertTrue(message.startswith(PRIVACY_DIAGNOSTIC_EVENT_NAME + " "))
        event = json.loads(message.split(" ", 1)[1])
        self.assertEqual(set(event), set(PRIVACY_DIAGNOSTIC_FIELDS))
        self.assertEqual(event["schema"], PRIVACY_DIAGNOSTIC_CONTRACT)
        self.assertNotIn("CONTENT_SENTINEL", message)
        self.assertLessEqual(len(message.encode()), 2_048)
        logger.removeHandler(capture)

    def test_metrics_observer_failure_is_counted_without_record_failure(self) -> None:
        metrics = OperationMetrics()

        class FailingDiagnostics:
            def observe(self, _record) -> None:
                raise RuntimeError("diagnostic fixture failure")

        observer = OperationRecordObserver(metrics, FailingDiagnostics())
        observer.observe(_record())
        snapshot = metrics.snapshot(
            request_id="sx_req_" + "f" * 32,
            active_operations=0,
        )
        self.assertEqual(snapshot["operations_total"], 1)
        self.assertEqual(snapshot["operation_observer_failures"], 1)

    def test_authenticated_nonrecursive_metrics_route_is_published(self) -> None:
        settings = ServiceSettings(startup_model_policy="api_only")
        application = create_application(settings)
        schema = application.openapi()
        operation = schema["paths"]["/system/v1/metrics"]["get"]
        self.assertEqual(
            protected_route_family("GET", "/system/v1/metrics"),
            "system",
        )
        self.assertIsNone(operation_route_for("GET", "/system/v1/metrics"))
        self.assertEqual(
            operation["security"],
            [{"SystemXBearer": []}, {"SystemXApiKey": []}],
        )
        self.assertIn("401", operation["responses"])
        for status in ("413", "422", "429", "504"):
            self.assertNotIn(status, operation["responses"])
        self.assertIn("/system/v1/metrics", schema["paths"])


if __name__ == "__main__":
    unittest.main()

