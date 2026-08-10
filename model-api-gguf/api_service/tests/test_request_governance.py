import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from system_x_gguf_api.authentication import (
    AuthenticationManager,
    protected_route_family,
)
from system_x_gguf_api.authentication_openapi import (
    PROTECTED_OPENAPI_OPERATIONS,
    apply_authentication_openapi,
)
from system_x_gguf_api.credential_types import CredentialVerification
from system_x_gguf_api.request_governance import (
    GovernanceRejection,
    RequestGovernance,
    SlidingRateWindow,
    read_body_and_replay,
)
from system_x_gguf_api.errors import install_system_error_handling
from system_x_gguf_api.operation_records import OperationRecorder


def _settings(**overrides):
    values = {
        "request_max_body_bytes": 8,
        "request_max_total_tokens": 100,
        "request_timeout_seconds": 5.0,
        "request_concurrency_limit_per_key": 1,
        "request_rate_limit_requests_per_key": 2,
        "request_rate_limit_window_seconds": 10.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _request(*, method="POST", path="/system/v1/generate", headers=(), receive=None):
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": list(headers),
        "server": ("testserver", 80),
        "client": ("testclient", 123),
    }
    request = Request(scope, receive=receive)
    request.state.system_x_request_id = "sx_req_" + "0" * 32
    return request


def test_sliding_rate_window_evicts_only_expired_entries():
    now = [0.0]
    window = SlidingRateWindow(lambda: now[0])

    assert window.admit("key-a", 2, 10.0) is None
    now[0] = 0.5
    assert window.admit("key-a", 2, 10.0) is None
    now[0] = 1.0
    retry_after = window.admit("key-a", 2, 10.0)
    assert retry_after == 9
    assert window.admit("key-b", 2, 10.0) is None
    now[0] = 10.5
    assert window.admit("key-a", 2, 10.0) is None


def test_admission_lease_is_idempotent_and_separates_keys():
    governance = RequestGovernance(
        _settings(request_rate_limit_requests_per_key=100),
    )
    first = governance.admit("key-a")
    with pytest.raises(GovernanceRejection) as caught:
        governance.admit("key-a")
    assert caught.value.status_code == 429
    assert caught.value.code == "system_x_concurrency_limit_exceeded"
    other = governance.admit("key-b")
    assert governance.active_snapshot() == {"key-a": 1, "key-b": 1}
    assert first.release() is True
    assert first.release() is False
    assert other.release() is True
    assert governance.active_snapshot() == {}


def test_token_budget_uses_context_and_model_output_limits():
    governance = RequestGovernance(_settings())
    assert (
        governance.enforce_total_token_budget(
            input_tokens=30,
            requested_output_tokens=50,
            model_context_tokens=80,
            model_maximum_output_tokens=50,
        )
        == 80
    )
    for input_tokens, output_tokens, message in (
        (31, 50, "token budget"),
        (30, 51, "selected model output limit"),
        (90, 11, "token budget"),
    ):
        with pytest.raises(GovernanceRejection) as caught:
            governance.enforce_total_token_budget(
                input_tokens=input_tokens,
                requested_output_tokens=output_tokens,
                model_context_tokens=80,
                model_maximum_output_tokens=50,
            )
        assert caught.value.code == "system_x_token_budget_exceeded"
        assert message in caught.value.public_message


def test_body_bound_replays_exact_accepted_bytes():
    messages = iter(
        (
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"de", "more_body": False},
        )
    )

    async def receive():
        return next(messages)

    request = _request(
        headers=((b"content-length", b"5"),),
        receive=receive,
    )
    assert asyncio.run(read_body_and_replay(request, 8)) == 5
    assert asyncio.run(request.body()) == b"abcde"
    assert asyncio.run(request.body()) == b"abcde"


def test_body_bound_rejects_declared_and_received_overages():
    received = False

    async def should_not_receive():
        nonlocal received
        received = True
        return {"type": "http.request", "body": b"", "more_body": False}

    request = _request(
        headers=((b"content-length", b"9"),),
        receive=should_not_receive,
    )
    with pytest.raises(GovernanceRejection) as caught:
        asyncio.run(read_body_and_replay(request, 8))
    assert caught.value.status_code == 413
    assert caught.value.code == "system_x_request_too_large"
    assert received is False

    messages = iter(
        (
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": False},
        )
    )

    async def receive():
        return next(messages)

    request = _request(receive=receive)
    with pytest.raises(GovernanceRejection) as caught:
        asyncio.run(read_body_and_replay(request, 8))
    assert caught.value.status_code == 413


class _FakeStore:
    def verify(self, supplied):
        if supplied == "good-key":
            return CredentialVerification(True, "accepted", "a" * 32, "test")
        return CredentialVerification(False, "not_found")


def _isolated_application(settings, handler):
    application = FastAPI()
    recorder = OperationRecorder()
    recorder.startup(service_transaction_id="tx-governance-test")
    application.state.operations = recorder
    application.state.governance = RequestGovernance(settings)
    authentication = AuthenticationManager(_FakeStore(), enabled=True)
    install_system_error_handling(application, authentication)
    application.add_api_route(
        "/system/v1/generate", handler, methods=["POST"]
    )
    return application, recorder


def test_middleware_rejects_body_before_route_and_releases_lease():
    calls = []

    async def handler(_request: Request):
        calls.append("generation")
        return JSONResponse({"ok": True})

    application, recorder = _isolated_application(
        _settings(request_max_body_bytes=8), handler
    )
    try:
        with TestClient(application) as client:
            oversized = client.post(
                "/system/v1/generate",
                content=b"123456789",
                headers={"Authorization": "Bearer good-key"},
            )
            assert oversized.status_code == 413
            assert oversized.json()["error"]["code"] == "system_x_request_too_large"
            assert oversized.headers["x-system-x-request-id"]
            assert calls == []
            assert application.state.governance.active_snapshot() == {}
            assert recorder.active_count == 0

            exact = client.post(
                "/system/v1/generate",
                content=b"12345678",
                headers={"Authorization": "Bearer good-key"},
            )
            assert exact.status_code == 200
            assert calls == ["generation"]
            assert application.state.governance.active_snapshot() == {}
            assert recorder.active_count == 0
    finally:
        recorder.shutdown()


def test_middleware_deadline_covers_pre_response_and_stream_body():
    async def slow_handler(_request: Request):
        await asyncio.sleep(0.05)
        return JSONResponse({"ok": True})

    application, recorder = _isolated_application(
        _settings(request_timeout_seconds=0.01), slow_handler
    )
    try:
        with TestClient(application) as client:
            response = client.post(
                "/system/v1/generate",
                content=b"{}",
                headers={"Authorization": "Bearer good-key"},
            )
            assert response.status_code == 504
            assert response.json()["error"]["code"] == (
                "system_x_request_deadline_exceeded"
            )
            assert application.state.governance.active_snapshot() == {}
            assert recorder.active_count == 0
    finally:
        recorder.shutdown()

    async def stream_handler(_request: Request):
        async def body():
            yield b"data: first\n\n"
            await asyncio.sleep(0.05)
            yield b"data: second\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    application, recorder = _isolated_application(
        _settings(request_timeout_seconds=0.01), stream_handler
    )
    try:
        with TestClient(application) as client:
            response = client.post(
                "/system/v1/generate",
                content=b"{}",
                headers={"Authorization": "Bearer good-key"},
            )
            assert response.status_code == 200
            assert b"data: first" in response.content
            assert b"system_x_request_deadline_exceeded" in response.content
            assert application.state.governance.active_snapshot() == {}
            assert recorder.active_count == 0
    finally:
        recorder.shutdown()


def test_authentication_keeps_bearer_and_x_api_key_as_alternatives():
    manager = AuthenticationManager(_FakeStore(), enabled=True)
    bearer = _request(headers=((b"authorization", b"Bearer good-key"),))
    assert protected_route_family("POST", "/system/v1/generate") == "system"
    assert manager.authenticate_request(bearer, "system") is None
    assert bearer.state.system_x_authentication.key_id == "a" * 32
    assert bearer.state.system_x_authentication.credential_scheme == "bearer"

    dual = _request(
        headers=(
            (b"authorization", b"Bearer good-key"),
            (b"x-api-key", b"good-key"),
        )
    )
    assert manager.authenticate_request(dual, "system") is None
    assert dual.state.system_x_authentication.credential_scheme == "dual"

    conflicting = _request(
        headers=(
            (b"authorization", b"Bearer good-key"),
            (b"x-api-key", b"other-key"),
        )
    )
    response = manager.authenticate_request(conflicting, "openai")
    assert response is not None
    assert response.status_code == 400
    assert "www-authenticate" not in response.headers


def test_openapi_declares_governance_without_protecting_health():
    paths = {"/system/v1/health": {"get": {"responses": {}}}}
    for path, method, _family in PROTECTED_OPENAPI_OPERATIONS:
        paths.setdefault(path, {})[method] = {"responses": {}}
    schema = apply_authentication_openapi(
        {"paths": paths, "components": {}}, enabled=True
    )
    assert schema["x-system-x-request-governance-contract"] == (
        "system-x.request-governance.v1"
    )
    assert schema["paths"]["/system/v1/health"]["get"]["security"] == []
    generate = schema["paths"]["/system/v1/generate"]["post"]
    assert generate["security"] == [
        {"SystemXBearer": []},
        {"SystemXApiKey": []},
    ]
    assert {"413", "422", "429", "504"}.issubset(generate["responses"])
    version = schema["paths"]["/system/v1/version"]["get"]["responses"]
    assert "429" not in version
