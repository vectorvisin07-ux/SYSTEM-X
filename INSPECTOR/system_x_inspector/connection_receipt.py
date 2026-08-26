"""Sanitized API Connection Receipt and read-only connection verification."""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
from typing import Any, Callable, Iterable

from .constants import SCHEMA_IDENTITIES
from .errors import InspectorError
from .locking import TransactionLock
from .paths import InspectorPaths
from .records import (
    atomic_write_json,
    canonical_json_bytes,
    fsync_directory,
    read_json_record,
)
from .results import utc_now
from .runtime import (
    _status_value,
    _transaction_id,
    _write_status,
    _write_transaction,
)
from .service_publication import (
    LoopbackJsonClient,
    ServiceSnapshot,
    _operation_record_from_line,
    read_local_credential,
)


RECEIPT_ID_PATTERN = re.compile(
    r"connection-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
PUBLIC_MODEL_PATTERN = re.compile(r"sx-[a-z0-9-]{8,127}\Z")
ARTIFACT_VERSION_PATTERN = re.compile(r"bundle-[0-9a-f]{64}\Z")
REQUEST_ID_PATTERN = re.compile(r"sx_req_[0-9a-f]{32}\Z")
KEY_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
OPERATION_LOG_NAME_PATTERN = re.compile(
    r"tx-[A-Za-z0-9][A-Za-z0-9._:-]{0,124}\.log\Z"
)
OPERATION_RECORD_SCHEMA = "system-x.operation-record.v1"
OPERATION_RECORD_FIELDS = frozenset(
    {
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
    }
)
OPERATION_ROUTES = frozenset(
    {
        ("system_x", "/system/v1/generate", "generate"),
        ("system_x", "/system/v1/chat", "chat"),
        ("system_x", "/system/v1/responses", "responses"),
        ("openai_compatible", "/v1/completions", "generate"),
        ("openai_compatible", "/v1/chat/completions", "chat"),
        ("openai_compatible", "/v1/responses", "responses"),
        ("messages_compatible", "/v1/messages", "chat"),
    }
)
MAX_CONTROL_BYTES = 2 * 1024 * 1024
MESSAGES_COMPATIBILITY_VERSION = "2023-06-01"
BASE_URL_PATTERN = re.compile(
    r"(?P<scheme>http|https)://(?P<authority>[^/?#]+)"
    r"(?P<path>/[^?#]*)?\Z",
    re.ASCII,
)

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "receipt_identity",
        "generated_utc",
        "receipt_source",
        "deployment_id",
        "deployment_result_identity",
        "service",
        "connections",
        "model",
        "authentication",
        "capabilities",
        "proof",
        "lifecycle",
        "warnings",
    }
)
SERVICE_FIELDS = frozenset(
    {
        "public_origin",
        "service_available",
        "inference_ready",
        "service_readiness",
        "model_service_state",
        "desired_state",
        "always_on",
        "authentication_required",
    }
)
CONNECTION_COMMON_FIELDS = frozenset(
    {
        "protocol_family",
        "base_url",
        "endpoint_semantics",
        "model_reference",
        "authentication",
        "endpoints",
    }
)
MESSAGES_CONNECTION_FIELDS = CONNECTION_COMMON_FIELDS | frozenset(
    {"compatibility_version", "required_headers"}
)
LEGACY_CONNECTION_FIELDS = frozenset(
    {
        "protocol_family",
        "base_url",
        "model_reference",
        "authentication",
        "compatibility_version",
        "endpoints",
    }
)
MODEL_FIELDS = frozenset(
    {
        "recommended_reference",
        "default_alias",
        "resolved_immutable_model_id",
        "source_label",
        "physical_architecture",
        "physical_model_type",
        "official_checkpoint_attested",
        "artifact_sha256",
        "artifact_version_id",
        "capability_manifest_identity",
        "model_state",
        "warm",
        "context_window_tokens",
        "maximum_output_tokens",
    }
)
AUTHENTICATION_FIELDS = frozenset(
    {
        "required",
        "accepted_schemes",
        "non_secret_key_id",
        "raw_api_key_returned",
    }
)
CAPABILITY_FIELDS = frozenset(
    {
        "protocol_families",
        "streaming",
        "token_counting",
        "reasoning_output",
        "reasoning_control",
        "tool_calling",
        "structured_output",
        "context_window_tokens",
    }
)
PROOF_FIELDS = frozenset(
    {
        "health_http_status",
        "model_list_http_status",
        "model_detail_http_status",
        "proof_request_id",
        "proof_request_http_status",
        "response_model_matches",
        "artifact_version_matches",
        "final_content_nonempty",
        "operation_record_correlated",
        "openai_model_list_http_status",
        "openai_model_list_contains_recommended_model",
        "openai_model_list_contains_resolved_model",
        "messages_model_list_http_status",
        "messages_model_list_contains_recommended_model",
        "messages_model_list_contains_resolved_model",
        "messages_token_count_http_status",
        "messages_token_count_result_valid",
    }
)
LEGACY_PROOF_FIELDS = frozenset(
    {
        "health_http_status",
        "model_list_http_status",
        "model_detail_http_status",
        "proof_request_id",
        "proof_request_http_status",
        "response_model_matches",
        "artifact_version_matches",
        "final_content_nonempty",
        "operation_record_correlated",
    }
)
COMPATIBILITY_PROOF_FIELDS = frozenset(PROOF_FIELDS - LEGACY_PROOF_FIELDS)
TOKEN_COUNT_PROOF_FIELDS = frozenset(
    {
        "operation_exposed",
        "proof_performed",
        "authenticated",
        "http_status",
        "result_valid",
        "authoritative_unsupported",
    }
)
LIFECYCLE_FIELDS = frozenset(
    {
        "promotion_result",
        "rollback_result",
        "retirement_result",
        "service_left_running",
        "service_left_ready",
        "current_receipt_updated",
    }
)
FORBIDDEN_KEYS = frozenset(
    {
        "raw_api_key",
        "authorization_value",
        "x_api_key_value",
        "authorization_header",
        "x_api_key_header",
        "credential_verifier",
        "pepper",
        "private_router_url",
        "private_router_port",
        "model_child_pid",
        "model_child_port",
        "physical_gguf_path",
        "process_environment",
        "prompt_content",
        "answer_content",
        "reasoning_content",
        "tool_content",
        "prompt",
        "answer",
        "reasoning",
        "tool_arguments",
        "tool_results",
    }
)
CAPABILITY_STATES = frozenset(
    {
        True,
        False,
        None,
        "available",
        "unavailable",
        "unknown",
        "not_tested",
        "not_exposed",
        "gated_unavailable",
    }
)
TOKEN_COUNT_STATES = frozenset(
    {"available", "not_tested", "unavailable", "not_exposed"}
)
SYSTEM_X_ENDPOINTS = {
    "health": "/system/v1/health",
    "version": "/system/v1/version",
    "models": "/system/v1/models",
    "model_detail": "/system/v1/models/{model_id}",
    "generate": "/system/v1/generate",
    "chat": "/system/v1/chat",
    "responses": "/system/v1/responses",
    "count_tokens": "/system/v1/tokens/count",
}
OPENAI_ENDPOINTS = {
    "models": "/models",
    "completions": "/completions",
    "chat_completions": "/chat/completions",
    "responses": "/responses",
}
MESSAGES_ENDPOINTS = {
    "models": "/v1/models",
    "messages": "/v1/messages",
    "count_tokens": "/v1/messages/count_tokens",
}
MESSAGES_REQUIRED_HEADERS = {
    "anthropic-version": MESSAGES_COMPATIBILITY_VERSION
}
LEGACY_OPENAI_ENDPOINTS = {
    name: "/v1" + endpoint for name, endpoint in OPENAI_ENDPOINTS.items()
}
LEGACY_MESSAGES_ENDPOINTS = {
    "models": "/system/v1/models",
    "messages": "/v1/messages",
    "count_tokens": "/system/v1/tokens/count",
}


def _fail(
    reason_code: str,
    message: str,
    *,
    data: dict[str, Any] | None = None,
    internal: bool = False,
) -> InspectorError:
    return InspectorError(
        reason_code,
        message,
        data=data,
        exit_status=70 if internal else 2,
    )


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection value is not canonical JSON",
        ) from error


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _receipt_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"connection-{stamp}-{secrets.token_hex(8)}"


def receipt_identity(value: dict[str, Any]) -> str:
    if set(value) != TOP_LEVEL_FIELDS:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection receipt fields are not closed",
        )
    return _identity(
        {
            key: value[key]
            for key in sorted(value)
            if key != "receipt_identity"
        }
    )


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise _fail(
                    "CONNECTION_RECORD_INVALID",
                    f"connection receipt contains prohibited field: {key}",
                )
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _split_base_url(value: str) -> tuple[str, str, str, str, int]:
    matched = BASE_URL_PATTERN.fullmatch(value)
    if matched is None:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL is not one bounded absolute URL",
        )
    scheme = matched.group("scheme")
    authority = matched.group("authority")
    path = matched.group("path") or ""
    if "@" in authority:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL contains forbidden authority or suffix material",
        )
    if authority.startswith("["):
        closing = authority.find("]")
        if closing <= 1 or authority[closing + 1 :].count(":") != 1:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "base URL authority is malformed",
            )
        host = authority[1:closing]
        port_text = authority[closing + 2 :]
        if authority[closing + 1 : closing + 2] != ":":
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "base URL authority is malformed",
            )
    else:
        if authority.count(":") != 1:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "base URL authority is malformed",
            )
        host, port_text = authority.rsplit(":", 1)
    if (
        not host
        or not port_text
        or not port_text.isascii()
        or not port_text.isdecimal()
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL authority is malformed",
        )
    port = int(port_text)
    if str(port) != port_text:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL authority is not canonical",
        )
    return scheme, authority, path, host, port


def normalize_base_url(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL is not one bounded absolute URL",
        )
    scheme, authority, path, host_value, port = _split_base_url(value)
    try:
        address = ipaddress.ip_address(host_value)
    except ValueError as error:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL host is not a numeric loopback address",
        ) from error
    if (
        not address.is_loopback
        or str(address) != host_value
        or not 1 <= port <= 65535
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL is not one exact numeric-loopback origin",
        )
    host = f"[{address}]" if address.version == 6 else str(address)
    canonical_authority = f"{host}:{port}"
    if authority != canonical_authority:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL authority is not canonical",
        )
    if path == "/":
        path = ""
    elif path.endswith("/"):
        path = path[:-1]
        if path.endswith("/"):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "base URL has more than one trailing slash",
            )
    if (
        (path and not path.startswith("/"))
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "base URL path prefix is not bounded",
        )
    return f"{scheme}://{canonical_authority}{path}"


def _public_origin(value: object) -> str:
    normalized = normalize_base_url(value)
    _scheme, _authority, path, _host, _port = _split_base_url(normalized)
    if path:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "public origin contains a path prefix",
        )
    return normalized


def normalized_openai_base(public_origin: str) -> str:
    normalized = normalize_base_url(public_origin)
    scheme, authority, path, _host, _port = _split_base_url(normalized)
    if path not in {"", "/v1"}:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "OpenAI base URL has an unsupported path prefix",
        )
    origin = f"{scheme}://{authority}"
    return origin + "/v1"


def join_base_relative(base_url: object, endpoint: object) -> str:
    normalized = normalize_base_url(base_url)
    if (
        not isinstance(endpoint, str)
        or not endpoint.startswith("/")
        or endpoint.startswith("//")
        or "://" in endpoint
        or "\\" in endpoint
        or "?" in endpoint
        or "#" in endpoint
        or "//" in endpoint
        or any(part in {".", ".."} for part in endpoint.split("/"))
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "endpoint is not one base-URL-relative path",
        )
    return normalized + endpoint


def _exact(
    value: object, fields: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            f"{label} fields are not closed",
        )
    return value


def _http_status(value: object, label: str) -> int:
    if type(value) is not int or not 100 <= value <= 599:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            f"{label} is not an HTTP status",
        )
    return value


def _optional_positive(value: object, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            f"{label} is not a positive integer or null",
        )
    return value


def _optional_http_status(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _http_status(value, label)


def _optional_boolean(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            f"{label} is not a Boolean or null",
        )
    return value


def derive_token_counting(proof: object) -> str:
    if proof is None:
        return "not_tested"
    value = _exact(
        proof, TOKEN_COUNT_PROOF_FIELDS, "token-count proof"
    )
    for name in (
        "operation_exposed",
        "proof_performed",
        "authoritative_unsupported",
    ):
        if type(value[name]) is not bool:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                f"token-count proof {name} is not a Boolean",
            )
    authenticated = value["authenticated"]
    result_valid = value["result_valid"]
    status = value["http_status"]
    if authenticated is not None and type(authenticated) is not bool:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "token-count proof authentication is invalid",
        )
    _optional_boolean(result_valid, "token-count result validity")
    _optional_http_status(status, "token-count HTTP status")
    if value["operation_exposed"] is False:
        if (
            value["proof_performed"] is not False
            or authenticated is not None
            or status is not None
            or result_valid is not None
            or value["authoritative_unsupported"] is not False
        ):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "non-exposed token count contains fabricated proof",
            )
        return "not_exposed"
    if value["proof_performed"] is False:
        if (
            authenticated is not None
            or status is not None
            or result_valid is not None
            or value["authoritative_unsupported"] is not False
        ):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "untested token count contains fabricated proof",
            )
        return "not_tested"
    if authenticated is not True or status is None:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "performed token-count proof is not authenticated and bounded",
        )
    if value["authoritative_unsupported"] is True:
        if result_valid is not False:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "unsupported token-count proof fabricates a valid result",
            )
        return "unavailable"
    if 200 <= status <= 299 and result_valid is True:
        return "available"
    raise _fail(
        "CONNECTION_RECORD_INVALID",
        "token-count proof does not establish an allowed state",
    )


def _compatibility_proof_projection(
    value: object,
) -> dict[str, Any]:
    if value is None:
        return {name: None for name in COMPATIBILITY_PROOF_FIELDS}
    proof = _exact(
        value,
        COMPATIBILITY_PROOF_FIELDS,
        "compatibility proof",
    )
    for name in (
        "openai_model_list_http_status",
        "messages_model_list_http_status",
        "messages_token_count_http_status",
    ):
        _optional_http_status(proof[name], name)
    for name in COMPATIBILITY_PROOF_FIELDS - {
        "openai_model_list_http_status",
        "messages_model_list_http_status",
        "messages_token_count_http_status",
    }:
        _optional_boolean(proof[name], name)
    return dict(proof)


def _connection_map(
    origin: str, model_reference: str
) -> dict[str, dict[str, Any]]:
    return {
        "system_x_native": {
            "protocol_family": "system_x_native",
            "base_url": origin,
            "endpoint_semantics": "base_url_relative",
            "model_reference": model_reference,
            "authentication": [
                "x-api-key",
                "Authorization Bearer",
            ],
            "endpoints": dict(SYSTEM_X_ENDPOINTS),
        },
        "openai_compatible": {
            "protocol_family": "openai_compatible",
            "base_url": normalized_openai_base(origin),
            "endpoint_semantics": "base_url_relative",
            "model_reference": model_reference,
            "authentication": ["Authorization Bearer"],
            "endpoints": dict(OPENAI_ENDPOINTS),
        },
        "messages_compatible": {
            "protocol_family": "messages_compatible",
            "base_url": origin,
            "endpoint_semantics": "base_url_relative",
            "model_reference": model_reference,
            "authentication": ["x-api-key"],
            "compatibility_version": MESSAGES_COMPATIBILITY_VERSION,
            "required_headers": dict(MESSAGES_REQUIRED_HEADERS),
            "endpoints": dict(MESSAGES_ENDPOINTS),
        },
    }


def validate_receipt(value: object) -> dict[str, Any]:
    receipt = _exact(value, TOP_LEVEL_FIELDS, "connection receipt")
    list(_strings(receipt))
    if (
        receipt["schema_version"]
        != SCHEMA_IDENTITIES["api_connection_receipt"]
        or not isinstance(receipt["receipt_id"], str)
        or RECEIPT_ID_PATTERN.fullmatch(receipt["receipt_id"]) is None
        or not isinstance(receipt["generated_utc"], str)
        or not receipt["generated_utc"]
        or receipt["receipt_source"]
        not in {"DEPLOY_GGUF", "EXISTING_ACCEPTED_READY_BASELINE"}
        or not isinstance(receipt["deployment_id"], str)
        or not receipt["deployment_id"]
        or (
            receipt["deployment_result_identity"] is not None
            and (
                not isinstance(
                    receipt["deployment_result_identity"], str
                )
                or SHA256_PATTERN.fullmatch(
                    receipt["deployment_result_identity"]
                )
                is None
            )
        )
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection receipt identity fields are invalid",
        )
    service = _exact(
        receipt["service"], SERVICE_FIELDS, "connection service"
    )
    origin = _public_origin(service["public_origin"])
    if (
        service["public_origin"] != origin
        or type(service["service_available"]) is not bool
        or type(service["inference_ready"]) is not bool
        or service["service_readiness"]
        not in {"READY", "WAITING_FOR_MODEL", "DEGRADED", "STOPPED"}
        or service["model_service_state"]
        not in {"READY", "WAITING_FOR_MODEL", "DEGRADED", "STOPPED"}
        or service["desired_state"] not in {"RUNNING", "STOPPED"}
        or type(service["always_on"]) is not bool
        or type(service["authentication_required"]) is not bool
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection service projection is invalid",
        )
    connections = _exact(
        receipt["connections"],
        frozenset(
            {
                "system_x_native",
                "openai_compatible",
                "messages_compatible",
            }
        ),
        "connection families",
    )
    expected_bases = {
        "system_x_native": origin,
        "openai_compatible": normalized_openai_base(origin),
        "messages_compatible": origin,
    }
    expected_authentication = {
        "system_x_native": {
            "x-api-key",
            "Authorization Bearer",
        },
        "openai_compatible": {"Authorization Bearer"},
        "messages_compatible": {"x-api-key"},
    }
    expected_endpoints = {
        "system_x_native": SYSTEM_X_ENDPOINTS,
        "openai_compatible": OPENAI_ENDPOINTS,
        "messages_compatible": MESSAGES_ENDPOINTS,
    }
    model_reference: str | None = None
    for family, expected_base in expected_bases.items():
        expected_fields = (
            MESSAGES_CONNECTION_FIELDS
            if family == "messages_compatible"
            else CONNECTION_COMMON_FIELDS
        )
        item = _exact(
            connections[family],
            expected_fields,
            f"{family} connection",
        )
        if (
            item["protocol_family"] != family
            or item["base_url"] != expected_base
            or item["endpoint_semantics"] != "base_url_relative"
            or not isinstance(item["model_reference"], str)
            or not item["model_reference"]
            or not isinstance(item["authentication"], list)
            or set(item["authentication"])
            != expected_authentication[family]
            or len(item["authentication"])
            != len(expected_authentication[family])
            or (
                family == "messages_compatible"
                and (
                    item["compatibility_version"]
                    != MESSAGES_COMPATIBILITY_VERSION
                    or item["required_headers"]
                    != MESSAGES_REQUIRED_HEADERS
                )
            )
            or not isinstance(item["endpoints"], dict)
            or item["endpoints"] != expected_endpoints[family]
        ):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                f"{family} connection is invalid",
            )
        composed = {
            name: join_base_relative(item["base_url"], endpoint)
            for name, endpoint in item["endpoints"].items()
        }
        if (
            any("/v1/v1/" in url for url in composed.values())
            or (
                family == "openai_compatible"
                and any(
                    endpoint.startswith("/v1")
                    for endpoint in item["endpoints"].values()
                )
            )
            or (
                family == "messages_compatible"
                and any(
                    endpoint.startswith("/system/v1/")
                    for endpoint in item["endpoints"].values()
                )
            )
        ):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                f"{family} URL composition is invalid",
            )
        if model_reference is None:
            model_reference = item["model_reference"]
        elif item["model_reference"] != model_reference:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "connection model references disagree",
            )
    model = _exact(receipt["model"], MODEL_FIELDS, "connection model")
    if (
        model["recommended_reference"] != model_reference
        or not isinstance(model["default_alias"], str)
        or not model["default_alias"]
        or not isinstance(model["resolved_immutable_model_id"], str)
        or PUBLIC_MODEL_PATTERN.fullmatch(
            model["resolved_immutable_model_id"]
        )
        is None
        or (
            model["source_label"] is not None
            and (
                not isinstance(model["source_label"], str)
                or not model["source_label"]
                or len(model["source_label"]) > 128
                or Path(model["source_label"]).name
                != model["source_label"]
            )
        )
        or (
            model["physical_architecture"] is not None
            and not isinstance(model["physical_architecture"], str)
        )
        or (
            model["physical_model_type"] is not None
            and not isinstance(model["physical_model_type"], str)
        )
        or model["official_checkpoint_attested"] is not False
        or not isinstance(model["artifact_sha256"], str)
        or SHA256_PATTERN.fullmatch(model["artifact_sha256"]) is None
        or not isinstance(model["artifact_version_id"], str)
        or ARTIFACT_VERSION_PATTERN.fullmatch(
            model["artifact_version_id"]
        )
        is None
        or not isinstance(model["capability_manifest_identity"], str)
        or SHA256_PATTERN.fullmatch(
            model["capability_manifest_identity"]
        )
        is None
        or model["model_state"]
        not in {"ready", "registered", "probing", "unavailable"}
        or type(model["warm"]) is not bool
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection model projection is invalid",
        )
    _optional_positive(
        model["context_window_tokens"], "model context window"
    )
    _optional_positive(
        model["maximum_output_tokens"], "model maximum output"
    )
    authentication = _exact(
        receipt["authentication"],
        AUTHENTICATION_FIELDS,
        "connection authentication",
    )
    if (
        authentication["required"] is not True
        or not isinstance(authentication["accepted_schemes"], list)
        or set(authentication["accepted_schemes"])
        != {"x-api-key", "Authorization Bearer"}
        or len(authentication["accepted_schemes"]) != 2
        or not isinstance(authentication["non_secret_key_id"], str)
        or KEY_ID_PATTERN.fullmatch(
            authentication["non_secret_key_id"]
        )
        is None
        or authentication["raw_api_key_returned"] is not False
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection authentication projection is invalid",
        )
    capabilities = _exact(
        receipt["capabilities"],
        CAPABILITY_FIELDS,
        "connection capabilities",
    )
    if (
        not isinstance(capabilities["protocol_families"], list)
        or set(capabilities["protocol_families"])
        != {
            "system_x_native",
            "openai_compatible",
            "messages_compatible",
        }
        or len(capabilities["protocol_families"]) != 3
        or capabilities["token_counting"] not in TOKEN_COUNT_STATES
        or any(
            capabilities[name] not in CAPABILITY_STATES
            for name in (
                "streaming",
                "reasoning_output",
                "reasoning_control",
                "tool_calling",
                "structured_output",
            )
        )
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection capability projection is invalid",
        )
    _optional_positive(
        capabilities["context_window_tokens"],
        "capability context window",
    )
    proof = _exact(receipt["proof"], PROOF_FIELDS, "connection proof")
    for name in (
        "health_http_status",
        "model_list_http_status",
        "model_detail_http_status",
        "proof_request_http_status",
    ):
        _http_status(proof[name], name)
    if (
        not isinstance(proof["proof_request_id"], str)
        or REQUEST_ID_PATTERN.fullmatch(proof["proof_request_id"]) is None
        or any(
            type(proof[name]) is not bool
            for name in (
                "response_model_matches",
                "artifact_version_matches",
                "final_content_nonempty",
                "operation_record_correlated",
            )
        )
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection proof projection is invalid",
        )
    for name in (
        "openai_model_list_http_status",
        "messages_model_list_http_status",
        "messages_token_count_http_status",
    ):
        _optional_http_status(proof[name], name)
    for name in (
        "openai_model_list_contains_recommended_model",
        "openai_model_list_contains_resolved_model",
        "messages_model_list_contains_recommended_model",
        "messages_model_list_contains_resolved_model",
        "messages_token_count_result_valid",
    ):
        _optional_boolean(proof[name], name)
    if capabilities["token_counting"] == "available" and (
        proof["messages_token_count_http_status"] != 200
        or proof["messages_token_count_result_valid"] is not True
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "available token counting lacks physical Messages proof",
        )
    lifecycle = _exact(
        receipt["lifecycle"],
        LIFECYCLE_FIELDS,
        "connection lifecycle",
    )
    for name in (
        "promotion_result",
        "rollback_result",
        "retirement_result",
    ):
        if lifecycle[name] is not None and (
            not isinstance(lifecycle[name], str) or not lifecycle[name]
        ):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                f"connection lifecycle {name} is invalid",
            )
    if any(
        type(lifecycle[name]) is not bool
        for name in (
            "service_left_running",
            "service_left_ready",
            "current_receipt_updated",
        )
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection lifecycle Booleans are invalid",
        )
    if (
        not isinstance(receipt["warnings"], list)
        or any(
            not isinstance(item, str) or not item
            for item in receipt["warnings"]
        )
        or not isinstance(receipt["receipt_identity"], str)
        or SHA256_PATTERN.fullmatch(receipt["receipt_identity"]) is None
        or receipt["receipt_identity"] != receipt_identity(receipt)
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection receipt terminal identity is invalid",
        )
    return receipt


def _safe_json(path: Path, label: str) -> dict[str, Any]:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            f"{label} is absent",
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_size > MAX_CONTROL_BYTES
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            f"{label} has an unsafe physical type",
        )
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            f"{label} is invalid JSON",
        ) from error
    if not isinstance(value, dict):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            f"{label} is not an object",
        )
    return value


def _branch_root(paths: InspectorPaths) -> Path:
    branch = paths.inspector_root.parent / "model-api-gguf"
    try:
        details = branch.lstat()
        resolved = branch.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "GGUF branch is unavailable",
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or resolved != branch
        or branch.parent != paths.inspector_root.parent
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "GGUF branch containment is invalid",
        )
    return branch


def _registry_model(
    branch: Path, reference: str
) -> dict[str, Any]:
    database = (
        branch
        / "RUNTIME"
        / "api"
        / "database"
        / "model_registry.sqlite3"
    )
    try:
        details = database.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
        ):
            raise OSError("unsafe registry")
        database_open = sqlite3.connect
        connection = database_open(
            f"file:{database}?mode=ro",
            uri=True,
            isolation_level=None,
            timeout=5.0,
        )
    except (OSError, sqlite3.Error) as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "model registry is unavailable",
        ) from error
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        generation_row = connection.execute(
            "SELECT value FROM registry_metadata WHERE key=?",
            ("registry_generation",),
        ).fetchone()
        if generation_row is None:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "registry generation is absent",
            )
        default_row = connection.execute(
            """
            SELECT alias,model_version_id
            FROM aliases WHERE alias_kind=?
            """,
            ("default",),
        ).fetchall()
        if len(default_row) != 1:
            raise _fail(
                "CONNECTION_STALE",
                "default alias is not singular",
            )
        model_id = (
            str(default_row[0]["model_version_id"])
            if reference == str(default_row[0]["alias"])
            else reference
        )
        rows = connection.execute(
            """
            SELECT
              mv.model_version_id,
              mv.bundle_id,
              mv.state,
              ab.bundle_sha256,
              cm.manifest_sha256,
              cm.manifest_json
            FROM model_versions AS mv
            JOIN artifact_bundles AS ab
              ON ab.bundle_id=mv.bundle_id
            JOIN capability_manifests AS cm
              ON cm.model_version_id=mv.model_version_id
            JOIN model_version_locations AS mvl
              ON mvl.model_version_id=mv.model_version_id
            JOIN artifact_locations AS al
              ON al.relative_root=mvl.relative_root
            WHERE mv.model_version_id=?
              AND al.present=1
              AND al.current_bundle_id=mv.bundle_id
            """,
            (model_id,),
        ).fetchall()
        if len(rows) != 1:
            raise _fail(
                "CONNECTION_STALE",
                "model reference does not resolve to one present version",
            )
        row = rows[0]
        try:
            manifest = json.loads(str(row["manifest_json"]))
        except json.JSONDecodeError as error:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "capability manifest is invalid",
            ) from error
        if not isinstance(manifest, dict):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "capability manifest is not an object",
            )
        aliases = [
            str(item["alias"])
            for item in connection.execute(
                "SELECT alias FROM aliases WHERE model_version_id=? "
                "ORDER BY alias",
                (model_id,),
            )
        ]
        result = {
            "registry_generation": int(generation_row[0]),
            "default_alias": str(default_row[0]["alias"]),
            "default_target": str(default_row[0]["model_version_id"]),
            "model_version_id": str(row["model_version_id"]),
            "artifact_version_id": str(row["bundle_id"]),
            "artifact_sha256": "sha256:" + str(row["bundle_sha256"]),
            "capability_manifest_identity": (
                "sha256:" + str(row["manifest_sha256"])
            ),
            "state": str(row["state"]).lower(),
            "aliases": aliases,
            "manifest": manifest,
        }
        if connection.total_changes != 0:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "read-only registry observation changed the database",
                internal=True,
            )
        return result
    except sqlite3.Error as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "read-only registry query failed",
        ) from error
    finally:
        connection.close()


def _owned_operation_log_set(log_path: Path) -> tuple[Path, ...]:
    """Return the complete physical operation-log set owned by this service."""

    try:
        root_details = log_path.parent.lstat()
        active_details = log_path.lstat()
        root = log_path.parent.resolve(strict=True)
        active = log_path.resolve(strict=True)
    except OSError as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "operation record log root is unavailable",
        ) from error
    if (
        stat.S_ISLNK(root_details.st_mode)
        or not stat.S_ISDIR(root_details.st_mode)
        or stat.S_ISLNK(active_details.st_mode)
        or not stat.S_ISREG(active_details.st_mode)
        or active_details.st_nlink != 1
        or OPERATION_LOG_NAME_PATTERN.fullmatch(active.name) is None
        or active.parent != root
        or (root_details.st_uid, root_details.st_gid)
        != (active_details.st_uid, active_details.st_gid)
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "API service operation-log root or active log is unsafe",
        )
    owner = (active_details.st_uid, active_details.st_gid)
    candidates: list[Path] = []
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
        for candidate in entries:
            if OPERATION_LOG_NAME_PATTERN.fullmatch(candidate.name) is None:
                continue
            details = candidate.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
            ):
                raise _fail(
                    "CONNECTION_RECORD_INVALID",
                    "accepted operation log has an unsafe physical type",
                )
            if (details.st_uid, details.st_gid) != owner:
                continue
            if stat.S_IMODE(details.st_mode) & 0o022:
                raise _fail(
                    "CONNECTION_RECORD_INVALID",
                    "owned operation log is writable by a foreign principal",
                )
            resolved = candidate.resolve(strict=True)
            if resolved.parent != root or resolved != candidate:
                raise _fail(
                    "CONNECTION_RECORD_INVALID",
                    "operation log escapes its owned root",
                )
            candidates.append(resolved)
    except InspectorError:
        raise
    except OSError as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "durable operation record logs are unavailable",
        ) from error
    if active not in candidates:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "active operation log is absent from its owned log set",
        )
    return tuple(candidates)


def _operation_log_records(log_path: Path) -> tuple[dict[str, Any], ...]:
    """Read all complete operation records from one stable physical log."""

    try:
        before = log_path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "operation log is not a physical regular file",
            )
        records: list[dict[str, Any]] = []
        with log_path.open("rb") as handle:
            for line in handle:
                if b"system_x_operation " not in line:
                    continue
                record = _operation_record_from_line(line)
                if not isinstance(record, dict):
                    raise _fail(
                        "CONNECTION_RECORD_INVALID",
                        "operation log contains a malformed operation record",
                    )
                records.append(record)
        after = log_path.lstat()
    except InspectorError:
        raise
    except OSError as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "operation record log is unavailable",
        ) from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "operation record log changed during observation",
        )
    return tuple(records)


def _operation_record_matches(
    record: dict[str, Any],
    *,
    log_path: Path,
    model_id: str,
    artifact_version_id: str,
    key_id: str,
    request_id: str | None,
) -> bool:
    if set(record) != OPERATION_RECORD_FIELDS:
        return False
    if request_id is not None and record.get("request_id") != request_id:
        return False
    if (
        record.get("schema") != OPERATION_RECORD_SCHEMA
        or record.get("public_model_id") != model_id
        or record.get("artifact_version_id") != artifact_version_id
        or record.get("key_id") != key_id
        or record.get("api_service_transaction_id") != log_path.stem
        or record.get("http_status") != 200
        or record.get("operation_state") != "completed"
        or record.get("error_code") is not None
        or type(record.get("streamed")) is not bool
        or type(record.get("output_tokens")) is not int
        or int(record["output_tokens"]) < 1
        or not isinstance(record.get("finish_reason"), str)
        or not record["finish_reason"]
    ):
        return False
    route = (
        record.get("protocol_family"),
        record.get("endpoint"),
        record.get("operation"),
    )
    if route not in OPERATION_ROUTES:
        return False
    router_transaction_id = record.get("router_transaction_id")
    if router_transaction_id is not None and (
        not isinstance(router_transaction_id, str)
        or re.fullmatch(
            r"tx-[A-Za-z0-9][A-Za-z0-9._:-]{0,124}",
            router_transaction_id,
        )
        is None
    ):
        return False
    return True


def _operation_proof(
    log_path: Path,
    *,
    model_id: str,
    artifact_version_id: str,
    key_id: str,
    request_id: str | None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for candidate in _owned_operation_log_set(log_path):
        for record in _operation_log_records(candidate):
            if _operation_record_matches(
                record,
                log_path=candidate,
                model_id=model_id,
                artifact_version_id=artifact_version_id,
                key_id=key_id,
                request_id=request_id,
            ):
                matches.append(record)
    if request_id is not None and len(matches) != 1:
        raise _fail(
            "CONNECTION_STALE",
            "stored proof operation record is absent or ambiguous",
        )
    if request_id is None and len(matches) > 1:
        raise _fail(
            "CONNECTION_STALE",
            "accepted inference proof records are ambiguous",
        )
    if not matches:
        raise _fail(
            "CONNECTION_NOT_INITIALIZED",
            "no suitable accepted inference proof record exists",
        )
    record = matches[0]
    return {
        "request_id": record["request_id"],
        "http_status": record["http_status"],
        "response_model_matches": True,
        "artifact_version_matches": True,
        "final_content_nonempty": True,
        "operation_record_correlated": True,
    }


def observe_current_connection(
    paths: InspectorPaths,
    *,
    reference: str = "default",
    proof_request_id: str | None = None,
) -> dict[str, Any]:
    branch = _branch_root(paths)
    control = branch / "RUNTIME" / "service_control"
    profile = _safe_json(
        control / "operating-profile.json", "operating profile"
    )
    public = profile.get("public_endpoint")
    if not isinstance(public, dict):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "operating profile public endpoint is absent",
        )
    host = public.get("host")
    port = public.get("port")
    if not isinstance(host, str) or type(port) is not int:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "operating profile public endpoint is invalid",
        )
    origin = _public_origin(f"http://{host}:{port}")
    profile_identity = _identity(profile)
    desired = _safe_json(
        control / "desired-state.json", "desired state"
    )
    supervisor = _safe_json(
        control / "status" / "supervisor.json", "supervisor status"
    )
    recovery = supervisor.get("recovery_status")
    warm = supervisor.get("warm_model_identity")
    if (
        desired.get("desired_state") != "RUNNING"
        or desired.get("profile_identity") != profile_identity
        or supervisor.get("profile_identity") != profile_identity
        or supervisor.get("supervisor_state") != "RUNNING"
        or supervisor.get("service_readiness_state") != "READY"
        or not isinstance(recovery, dict)
        or recovery.get("recovery_state") != "IDLE"
        or recovery.get("fail_closed_latched") is not False
        or not isinstance(warm, dict)
    ):
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "System X is not coherently RUNNING and READY",
        )
    registry = _registry_model(branch, reference)
    service_status = _safe_json(
        branch / "RUNTIME" / "api" / "status" / "service.json",
        "API service status",
    )
    transaction_id = service_status.get("transaction_id")
    log_value = service_status.get("log_path")
    if (
        service_status.get("lifecycle_state") != "STARTED"
        or service_status.get("service_control_profile_identity")
        != profile_identity
        or service_status.get("host") != host
        or service_status.get("port") != port
        or not isinstance(transaction_id, str)
        or not isinstance(log_value, str)
    ):
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "API service activation does not match the operating profile",
        )
    log_path = Path(log_value)
    expected_log_root = (
        branch / "RUNTIME" / "api" / "logs"
    ).resolve(strict=True)
    try:
        log_details = log_path.lstat()
        resolved_log = log_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise _fail(
            "CONNECTION_SERVICE_UNAVAILABLE",
            "API service operation log is unavailable",
        ) from error
    if (
        stat.S_ISLNK(log_details.st_mode)
        or not stat.S_ISREG(log_details.st_mode)
        or log_details.st_nlink != 1
        or resolved_log.parent != expected_log_root
        or resolved_log.name != f"{transaction_id}.log"
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "API service operation log is unsafe",
        )
    service = ServiceSnapshot(
        profile_identity=profile_identity,
        host=host,
        port=port,
        base_url=origin,
        default_alias=registry["default_alias"],
        service_transaction_id=transaction_id,
        operation_log=resolved_log,
        readiness_state="READY",
    )
    credential = read_local_credential(branch)
    client = LoopbackJsonClient(service)
    health = client.request("GET", "/system/v1/health")
    catalogue = client.request(
        "GET", "/system/v1/models", credential=credential
    )
    detail = client.request(
        "GET",
        f"/system/v1/models/{registry['model_version_id']}",
        credential=credential,
    )
    model = detail.body.get("model")
    models = catalogue.body.get("models")
    if (
        health.status != 200
        or health.body.get("service_readiness_state") != "READY"
        or health.body.get("recovery_state") != "IDLE"
        or catalogue.status != 200
        or not isinstance(models, list)
        or detail.status != 200
        or not isinstance(model, dict)
        or model.get("resolved_model_id")
        != registry["model_version_id"]
        or model.get("artifact_version_id")
        != registry["artifact_version_id"]
        or model.get("state") != "ready"
        or not isinstance(model.get("capabilities"), dict)
    ):
        raise _fail(
            "CONNECTION_STALE",
            "public health or model metadata does not match registry state",
        )
    matching = [
        item
        for item in models
        if isinstance(item, dict)
        and item.get("id") == registry["model_version_id"]
    ]
    if len(matching) != 1:
        raise _fail(
            "CONNECTION_STALE",
            "public model catalogue does not contain the resolved model",
        )
    proof = _operation_proof(
        resolved_log,
        model_id=registry["model_version_id"],
        artifact_version_id=registry["artifact_version_id"],
        key_id=credential.key_id,
        request_id=proof_request_id,
    )
    capabilities = dict(model["capabilities"])
    token_count_proof = {
        "operation_exposed": True,
        "proof_performed": False,
        "authenticated": None,
        "http_status": None,
        "result_valid": None,
        "authoritative_unsupported": False,
    }
    warm_matches = (
        warm.get("resolved_public_model_id")
        == registry["model_version_id"]
        and warm.get("artifact_version_id")
        == registry["artifact_version_id"]
        and warm.get("health_state") == "ready"
    )
    return {
        "profile_identity": profile_identity,
        "public_origin": origin,
        "desired_state": "RUNNING",
        "service_readiness": "READY",
        "model_service_state": "READY",
        "service_available": True,
        "inference_ready": True,
        "always_on": profile.get("startup_model_policy")
        == "always_warm",
        "authentication_required": True,
        "default_alias": registry["default_alias"],
        "default_target": registry["default_target"],
        "resolved_immutable_model_id": registry["model_version_id"],
        "artifact_sha256": registry["artifact_sha256"],
        "artifact_version_id": registry["artifact_version_id"],
        "capability_manifest_identity": registry[
            "capability_manifest_identity"
        ],
        "model_state": "ready",
        "warm": warm_matches,
        "context_window_tokens": model.get("context_window_tokens"),
        "maximum_output_tokens": model.get("maximum_output_tokens"),
        "non_secret_key_id": credential.key_id,
        "capabilities": {
            "protocol_families": [
                "system_x_native",
                "openai_compatible",
                "messages_compatible",
            ],
            "streaming": capabilities.get("streaming", "not_exposed"),
            "token_counting": derive_token_counting(
                token_count_proof
            ),
            "reasoning_output": capabilities.get(
                "reasoning_output", "not_exposed"
            ),
            "reasoning_control": capabilities.get(
                "reasoning_control", "not_exposed"
            ),
            "tool_calling": capabilities.get(
                "tool_calling", "not_exposed"
            ),
            "structured_output": capabilities.get(
                "structured_output", "not_exposed"
            ),
            "context_window_tokens": model.get(
                "context_window_tokens"
            ),
        },
        "token_count_proof": token_count_proof,
        "compatibility_proof": None,
        "proof": {
            "health_http_status": health.status,
            "model_list_http_status": catalogue.status,
            "model_detail_http_status": detail.status,
            "proof_request_id": proof["request_id"],
            "proof_request_http_status": proof["http_status"],
            "response_model_matches": proof[
                "response_model_matches"
            ],
            "artifact_version_matches": proof[
                "artifact_version_matches"
            ],
            "final_content_nonempty": proof[
                "final_content_nonempty"
            ],
            "operation_record_correlated": proof[
                "operation_record_correlated"
            ],
        },
        "registry_generation": registry["registry_generation"],
        "recovery_state": "IDLE",
    }


def build_receipt(
    observation: dict[str, Any],
    *,
    receipt_source: str,
    deployment_id: str,
    deployment_result_identity: str | None,
    recommended_reference: str,
    promotion_result: str | None = None,
    rollback_result: str | None = None,
    retirement_result: str | None = None,
    current_receipt_updated: bool,
    receipt_id_factory: Callable[[], str] = _receipt_id,
) -> dict[str, Any]:
    if receipt_source not in {
        "DEPLOY_GGUF",
        "EXISTING_ACCEPTED_READY_BASELINE",
    }:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection receipt source is invalid",
        )
    origin = _public_origin(observation.get("public_origin"))
    receipt_id = receipt_id_factory()
    if RECEIPT_ID_PATTERN.fullmatch(receipt_id) is None:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection receipt ID factory returned an invalid ID",
        )
    connections = _connection_map(origin, recommended_reference)
    token_count_proof = observation.get("token_count_proof")
    token_counting = derive_token_counting(token_count_proof)
    compatibility_proof = _compatibility_proof_projection(
        observation.get("compatibility_proof")
    )
    if token_counting == "available" and token_count_proof is not None:
        compatibility_proof["messages_token_count_http_status"] = (
            token_count_proof["http_status"]
        )
        compatibility_proof["messages_token_count_result_valid"] = (
            token_count_proof["result_valid"]
        )
    capabilities = dict(observation["capabilities"])
    capabilities["token_counting"] = token_counting
    native_proof = {
        name: observation["proof"][name]
        for name in LEGACY_PROOF_FIELDS
    }
    basis = {
        "schema_version": SCHEMA_IDENTITIES[
            "api_connection_receipt"
        ],
        "receipt_id": receipt_id,
        "generated_utc": utc_now(),
        "receipt_source": receipt_source,
        "deployment_id": deployment_id,
        "deployment_result_identity": deployment_result_identity,
        "service": {
            name: observation[name]
            for name in (
                "public_origin",
                "service_available",
                "inference_ready",
                "service_readiness",
                "model_service_state",
                "desired_state",
                "always_on",
                "authentication_required",
            )
        },
        "connections": connections,
        "model": {
            "recommended_reference": recommended_reference,
            "default_alias": observation["default_alias"],
            "resolved_immutable_model_id": observation[
                "resolved_immutable_model_id"
            ],
            "source_label": observation.get("source_label"),
            "physical_architecture": observation.get(
                "physical_architecture"
            ),
            "physical_model_type": observation.get(
                "physical_model_type"
            ),
            "official_checkpoint_attested": False,
            "artifact_sha256": observation["artifact_sha256"],
            "artifact_version_id": observation[
                "artifact_version_id"
            ],
            "capability_manifest_identity": observation[
                "capability_manifest_identity"
            ],
            "model_state": observation["model_state"],
            "warm": observation["warm"],
            "context_window_tokens": observation[
                "context_window_tokens"
            ],
            "maximum_output_tokens": observation[
                "maximum_output_tokens"
            ],
        },
        "authentication": {
            "required": True,
            "accepted_schemes": [
                "x-api-key",
                "Authorization Bearer",
            ],
            "non_secret_key_id": observation["non_secret_key_id"],
            "raw_api_key_returned": False,
        },
        "capabilities": capabilities,
        "proof": {**native_proof, **compatibility_proof},
        "lifecycle": {
            "promotion_result": promotion_result,
            "rollback_result": rollback_result,
            "retirement_result": retirement_result,
            "service_left_running": (
                observation["desired_state"] == "RUNNING"
            ),
            "service_left_ready": (
                observation["service_readiness"] == "READY"
            ),
            "current_receipt_updated": current_receipt_updated,
        },
        "warnings": [],
    }
    receipt = {**basis, "receipt_identity": _identity(basis)}
    return validate_receipt(receipt)


def _validate_legacy_baseline_receipt(
    value: object,
) -> dict[str, Any]:
    legacy = _exact(
        value, TOP_LEVEL_FIELDS, "legacy connection receipt"
    )
    list(_strings(legacy))
    if (
        legacy["schema_version"]
        != SCHEMA_IDENTITIES["api_connection_receipt"]
        or legacy["receipt_source"]
        != "EXISTING_ACCEPTED_READY_BASELINE"
        or legacy["deployment_result_identity"] is not None
        or not isinstance(legacy["receipt_id"], str)
        or RECEIPT_ID_PATTERN.fullmatch(legacy["receipt_id"]) is None
        or not isinstance(legacy["receipt_identity"], str)
        or legacy["receipt_identity"] != receipt_identity(legacy)
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "legacy baseline identity is not eligible for repair",
        )
    service = _exact(
        legacy["service"], SERVICE_FIELDS, "legacy connection service"
    )
    origin = _public_origin(service["public_origin"])
    if (
        service["public_origin"] != origin
        or service["desired_state"] != "RUNNING"
        or service["service_readiness"] != "READY"
        or service["model_service_state"] != "READY"
        or service["service_available"] is not True
        or service["inference_ready"] is not True
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "legacy baseline service is not accepted READY state",
        )
    connections = _exact(
        legacy["connections"],
        frozenset(
            {
                "system_x_native",
                "openai_compatible",
                "messages_compatible",
            }
        ),
        "legacy connection families",
    )
    expected_bases = {
        "system_x_native": origin,
        "openai_compatible": normalized_openai_base(origin),
        "messages_compatible": origin,
    }
    expected_authentication = {
        "system_x_native": {"x-api-key", "Authorization Bearer"},
        "openai_compatible": {"Authorization Bearer"},
        "messages_compatible": {"x-api-key"},
    }
    expected_endpoints = {
        "system_x_native": SYSTEM_X_ENDPOINTS,
        "openai_compatible": LEGACY_OPENAI_ENDPOINTS,
        "messages_compatible": LEGACY_MESSAGES_ENDPOINTS,
    }
    model_reference: str | None = None
    for family in (
        "system_x_native",
        "openai_compatible",
        "messages_compatible",
    ):
        item = _exact(
            connections[family],
            LEGACY_CONNECTION_FIELDS,
            f"legacy {family} connection",
        )
        if (
            item["protocol_family"] != family
            or item["base_url"] != expected_bases[family]
            or not isinstance(item["model_reference"], str)
            or not item["model_reference"]
            or not isinstance(item["authentication"], list)
            or set(item["authentication"])
            != expected_authentication[family]
            or len(item["authentication"])
            != len(expected_authentication[family])
            or item["endpoints"] != expected_endpoints[family]
            or (
                family == "messages_compatible"
                and item["compatibility_version"]
                != MESSAGES_COMPATIBILITY_VERSION
            )
            or (
                family != "messages_compatible"
                and item["compatibility_version"] is not None
            )
        ):
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                f"legacy {family} defect shape is unknown",
            )
        if model_reference is None:
            model_reference = item["model_reference"]
        elif item["model_reference"] != model_reference:
            raise _fail(
                "CONNECTION_RECORD_INVALID",
                "legacy connection model references disagree",
            )
    capabilities = _exact(
        legacy["capabilities"],
        CAPABILITY_FIELDS,
        "legacy connection capabilities",
    )
    if capabilities["token_counting"] != "not_exposed":
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "legacy token-count defect shape is unknown",
        )
    _exact(legacy["proof"], LEGACY_PROOF_FIELDS, "legacy proof")
    lifecycle = _exact(
        legacy["lifecycle"], LIFECYCLE_FIELDS, "legacy lifecycle"
    )
    if (
        lifecycle["service_left_running"] is not True
        or lifecycle["service_left_ready"] is not True
        or lifecycle["current_receipt_updated"] is not True
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "legacy lifecycle is not the accepted baseline",
        )
    projection = json.loads(_canonical(legacy))
    projection["connections"] = _connection_map(
        origin, str(model_reference)
    )
    projection["capabilities"]["token_counting"] = "not_tested"
    projection["proof"].update(
        {name: None for name in COMPATIBILITY_PROOF_FIELDS}
    )
    projection["receipt_identity"] = receipt_identity(projection)
    validate_receipt(projection)
    return legacy


def _legacy_matches_repair_candidate(
    legacy_receipt: dict[str, Any],
    candidate_receipt: dict[str, Any],
) -> bool:
    legacy = _validate_legacy_baseline_receipt(legacy_receipt)
    candidate = validate_receipt(candidate_receipt)
    if (
        candidate["receipt_source"]
        != "EXISTING_ACCEPTED_READY_BASELINE"
        or candidate["connections"]
        != _connection_map(
            legacy["service"]["public_origin"],
            legacy["model"]["recommended_reference"],
        )
        or candidate["capabilities"]["token_counting"]
        not in {"not_tested", "available"}
    ):
        return False
    for name in (
        "schema_version",
        "receipt_source",
        "deployment_id",
        "deployment_result_identity",
        "service",
        "model",
        "authentication",
        "lifecycle",
        "warnings",
    ):
        if candidate[name] != legacy[name]:
            return False
    for name in CAPABILITY_FIELDS - {"token_counting"}:
        if candidate["capabilities"][name] != legacy["capabilities"][name]:
            return False
    for name in LEGACY_PROOF_FIELDS:
        if candidate["proof"][name] != legacy["proof"][name]:
            return False
    return True


def build_legacy_repair_candidate(
    legacy_receipt: dict[str, Any],
    observation: dict[str, Any],
    *,
    receipt_id_factory: Callable[[], str] = _receipt_id,
) -> dict[str, Any]:
    legacy = _validate_legacy_baseline_receipt(legacy_receipt)
    comparisons = {
        "public_origin": (
            legacy["service"]["public_origin"],
            observation.get("public_origin"),
        ),
        "default_alias": (
            legacy["model"]["default_alias"],
            observation.get("default_alias"),
        ),
        "resolved_immutable_model_id": (
            legacy["model"]["resolved_immutable_model_id"],
            observation.get("resolved_immutable_model_id"),
        ),
        "artifact_version_id": (
            legacy["model"]["artifact_version_id"],
            observation.get("artifact_version_id"),
        ),
        "capability_manifest_identity": (
            legacy["model"]["capability_manifest_identity"],
            observation.get("capability_manifest_identity"),
        ),
        "non_secret_key_id": (
            legacy["authentication"]["non_secret_key_id"],
            observation.get("non_secret_key_id"),
        ),
    }
    if (
        any(left != right for left, right in comparisons.values())
        or observation.get("desired_state") != "RUNNING"
        or observation.get("service_readiness") != "READY"
        or observation.get("model_service_state") != "READY"
        or observation.get("warm") is not True
    ):
        raise _fail(
            "CONNECTION_STALE",
            "legacy repair candidate does not match live service identity",
        )
    candidate_observation = dict(observation)
    candidate_observation["token_count_proof"] = {
        "operation_exposed": True,
        "proof_performed": False,
        "authenticated": None,
        "http_status": None,
        "result_valid": None,
        "authoritative_unsupported": False,
    }
    candidate_observation["compatibility_proof"] = None
    candidate = build_receipt(
        candidate_observation,
        receipt_source="EXISTING_ACCEPTED_READY_BASELINE",
        deployment_id=legacy["deployment_id"],
        deployment_result_identity=legacy[
            "deployment_result_identity"
        ],
        recommended_reference=legacy["model"][
            "recommended_reference"
        ],
        promotion_result=legacy["lifecycle"]["promotion_result"],
        rollback_result=legacy["lifecycle"]["rollback_result"],
        retirement_result=legacy["lifecycle"]["retirement_result"],
        current_receipt_updated=True,
        receipt_id_factory=receipt_id_factory,
    )
    if not _legacy_matches_repair_candidate(legacy, candidate):
        raise _fail(
            "CONNECTION_STALE",
            "legacy repair candidate changes accepted baseline identity",
        )
    return candidate


def complete_compatibility_proof(
    candidate: dict[str, Any],
    proof_outcomes: dict[str, Any],
    *,
    receipt_id_factory: Callable[[], str] = _receipt_id,
) -> dict[str, Any]:
    pending = validate_receipt(candidate)
    proof = _compatibility_proof_projection(proof_outcomes)
    if (
        proof["openai_model_list_http_status"] != 200
        or proof[
            "openai_model_list_contains_recommended_model"
        ]
        is not True
        or proof["openai_model_list_contains_resolved_model"]
        is not True
        or proof["messages_model_list_http_status"] != 200
        or proof[
            "messages_model_list_contains_recommended_model"
        ]
        is not True
        or proof["messages_model_list_contains_resolved_model"]
        is not True
        or proof["messages_token_count_http_status"] != 200
        or proof["messages_token_count_result_valid"] is not True
    ):
        raise _fail(
            "CONNECTION_STALE",
            "compatibility proof is incomplete or unsuccessful",
        )
    receipt_id = receipt_id_factory()
    if RECEIPT_ID_PATTERN.fullmatch(receipt_id) is None:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection receipt ID factory returned an invalid ID",
        )
    completed = json.loads(_canonical(pending))
    completed["receipt_id"] = receipt_id
    completed["generated_utc"] = utc_now()
    completed["proof"].update(proof)
    completed["capabilities"]["token_counting"] = "available"
    completed["receipt_identity"] = receipt_identity(completed)
    return validate_receipt(completed)


def _read_current_receipt_object(
    paths: InspectorPaths,
) -> dict[str, Any]:
    path = paths.current_connection_status
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _fail(
            "CONNECTION_NOT_INITIALIZED",
            "current API connection receipt is absent",
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or details.st_size > MAX_CONTROL_BYTES
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "current API connection receipt has an unsafe physical type",
        )
    value = read_json_record(path)
    if not isinstance(value, dict):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "current API connection receipt is not an object",
        )
    return value


def load_current_receipt(paths: InspectorPaths) -> dict[str, Any]:
    return validate_receipt(_read_current_receipt_object(paths))


def load_legacy_current_receipt_for_repair(
    paths: InspectorPaths,
    *,
    expected_identity: str,
) -> dict[str, Any]:
    legacy = _validate_legacy_baseline_receipt(
        _read_current_receipt_object(paths)
    )
    if legacy["receipt_identity"] != expected_identity:
        raise _fail(
            "CONNECTION_STATUS_CAS_CONFLICT",
            "legacy current receipt changed before repair",
        )
    return legacy


def _current_receipt_for_cas(
    paths: InspectorPaths,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    value = _read_current_receipt_object(paths)
    try:
        return validate_receipt(value), False
    except InspectorError as corrected_error:
        if corrected_error.reason_code != "CONNECTION_RECORD_INVALID":
            raise
    legacy = _validate_legacy_baseline_receipt(value)
    if not _legacy_matches_repair_candidate(legacy, candidate):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "legacy current receipt does not match repair candidate",
        )
    return legacy, True


def publish_current_receipt(
    paths: InspectorPaths,
    receipt: dict[str, Any],
    *,
    expected_previous_identity: str | None,
    replace: Callable[[str | bytes, str | bytes], None] = os.replace,
) -> str:
    validated = validate_receipt(receipt)
    path = paths.current_connection_status
    parent = path.parent
    details = parent.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or parent.resolve(strict=True) != paths.status
    ):
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection status root is unsafe",
        )
    current_identity: str | None
    if path.exists() or path.is_symlink():
        current, _legacy = _current_receipt_for_cas(
            paths, validated
        )
        if current == validated:
            return str(current["receipt_identity"])
        current_identity = str(current["receipt_identity"])
    else:
        current_identity = None
    if current_identity != expected_previous_identity:
        raise _fail(
            "CONNECTION_STATUS_CAS_CONFLICT",
            "current connection receipt changed before publication",
        )
    data = canonical_json_bytes(validated)
    temporary = parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short connection receipt write")
            offset += written
        os.fsync(descriptor)
        if path.exists() or path.is_symlink():
            observed, _legacy = _current_receipt_for_cas(
                paths, validated
            )
            if observed == validated:
                return str(observed["receipt_identity"])
            observed_previous = str(observed["receipt_identity"])
        else:
            observed_previous = None
        if observed_previous != expected_previous_identity:
            raise _fail(
                "CONNECTION_STATUS_CAS_CONFLICT",
                "current connection receipt changed during publication",
            )
        replace(temporary, path)
        fsync_directory(parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
            fsync_directory(parent)
    observed = load_current_receipt(paths)
    if observed != validated:
        raise _fail(
            "CONNECTION_RECORD_INVALID",
            "connection receipt did not round-trip atomically",
            internal=True,
        )
    return observed["receipt_identity"]


def show_connection(
    paths: InspectorPaths,
    *,
    observer: Callable[..., dict[str, Any]] = observe_current_connection,
) -> dict[str, Any]:
    try:
        stored = load_current_receipt(paths)
    except InspectorError as error:
        if error.reason_code == "CONNECTION_NOT_INITIALIZED":
            return {
                "result_class": "CONNECTION_NOT_INITIALIZED",
                "reason_code": error.reason_code,
                "stored_receipt_identity": None,
                "mismatch_reason": "CURRENT_RECEIPT_ABSENT",
                "receipt": None,
            }
        return {
            "result_class": "CONNECTION_RECORD_INVALID",
            "reason_code": "CONNECTION_RECORD_INVALID",
            "stored_receipt_identity": None,
            "mismatch_reason": error.reason_code,
            "receipt": None,
        }
    try:
        current = observer(
            paths,
            reference="default",
            proof_request_id=stored["proof"]["proof_request_id"],
        )
    except InspectorError as error:
        result_class = (
            "CONNECTION_SERVICE_UNAVAILABLE"
            if error.reason_code == "CONNECTION_SERVICE_UNAVAILABLE"
            else "CONNECTION_STALE"
        )
        return {
            "result_class": result_class,
            "reason_code": error.reason_code,
            "stored_receipt_identity": stored["receipt_identity"],
            "mismatch_reason": error.message,
            "receipt": None,
        }
    comparisons = {
        "public_origin": (
            stored["service"]["public_origin"],
            current["public_origin"],
        ),
        "default_alias": (
            stored["model"]["default_alias"],
            current["default_alias"],
        ),
        "resolved_immutable_model_id": (
            stored["model"]["resolved_immutable_model_id"],
            current["resolved_immutable_model_id"],
        ),
        "artifact_version_id": (
            stored["model"]["artifact_version_id"],
            current["artifact_version_id"],
        ),
        "capability_manifest_identity": (
            stored["model"]["capability_manifest_identity"],
            current["capability_manifest_identity"],
        ),
        "non_secret_key_id": (
            stored["authentication"]["non_secret_key_id"],
            current["non_secret_key_id"],
        ),
    }
    mismatches = _receipt_mismatch_names(stored, current)
    if mismatches:
        return {
            "result_class": "CONNECTION_STALE",
            "reason_code": "CONNECTION_STALE",
            "stored_receipt_identity": stored["receipt_identity"],
            "mismatch_reason": ",".join(sorted(mismatches)),
            "receipt": None,
        }
    return {
        "result_class": "CONNECTION_READY",
        "reason_code": "CONNECTION_READY",
        "stored_receipt_identity": stored["receipt_identity"],
        "mismatch_reason": None,
        "receipt": stored,
    }


def render_connection(receipt: dict[str, Any]) -> str:
    validated = validate_receipt(receipt)
    heading = (
        "SYSTEM X API CONNECTION READY"
        if validated["receipt_source"]
        == "EXISTING_ACCEPTED_READY_BASELINE"
        else "SYSTEM X GGUF DEPLOYMENT COMPLETE"
    )
    return "\n".join(
        (
            heading,
            "",
            "STATUS:",
            f"  {validated['service']['service_readiness']}",
            "",
            "SYSTEM X BASE URL:",
            "  "
            + validated["connections"]["system_x_native"]["base_url"],
            "",
            "OPENAI-COMPATIBLE BASE URL:",
            "  "
            + validated["connections"]["openai_compatible"]["base_url"],
            "",
            "MESSAGES-COMPATIBLE BASE URL:",
            "  "
            + validated["connections"]["messages_compatible"]["base_url"],
            "",
            "RECOMMENDED MODEL:",
            "  " + validated["model"]["recommended_reference"],
            "",
            "RESOLVED IMMUTABLE MODEL:",
            "  " + validated["model"]["resolved_immutable_model_id"],
            "",
            "AUTHENTICATION:",
            "  x-api-key",
            "  or Authorization: Bearer",
            "",
            "NON-SECRET KEY ID:",
            "  " + validated["authentication"]["non_secret_key_id"],
            "",
            "RAW API KEY:",
            "  not printed",
            "",
            "SERVICE:",
            "  "
            + (
                "RUNNING"
                if validated["service"]["desired_state"] == "RUNNING"
                else "STOPPED"
            ),
            "  " + validated["service"]["service_readiness"],
            "",
            "PROOF REQUEST:",
            "  HTTP "
            + str(validated["proof"]["proof_request_http_status"]),
        )
    )


def _receipt_mismatch_names(
    stored: dict[str, Any], current: dict[str, Any]
) -> tuple[str, ...]:
    comparisons = {
        "public_origin": (
            stored["service"]["public_origin"],
            current["public_origin"],
        ),
        "default_alias": (
            stored["model"]["default_alias"],
            current["default_alias"],
        ),
        "resolved_immutable_model_id": (
            stored["model"]["resolved_immutable_model_id"],
            current["resolved_immutable_model_id"],
        ),
        "artifact_version_id": (
            stored["model"]["artifact_version_id"],
            current["artifact_version_id"],
        ),
        "capability_manifest_identity": (
            stored["model"]["capability_manifest_identity"],
            current["capability_manifest_identity"],
        ),
        "non_secret_key_id": (
            stored["authentication"]["non_secret_key_id"],
            current["non_secret_key_id"],
        ),
    }
    return tuple(
        name
        for name, (stored_value, current_value) in comparisons.items()
        if stored_value != current_value
    )


def bootstrap_current_receipt(
    paths: InspectorPaths,
    *,
    observer: Callable[..., dict[str, Any]] = observe_current_connection,
    publisher: Callable[..., str] = publish_current_receipt,
    transaction_id_factory: Callable[[], str] = _transaction_id,
    transition_observer: Callable[
        [str, dict[str, Any]], None
    ]
    | None = None,
) -> tuple[str, dict[str, Any], str]:
    existing: dict[str, Any] | None = None
    expected_previous_identity: str | None = None
    if paths.current_connection_status.exists() or (
        paths.current_connection_status.is_symlink()
    ):
        existing = load_current_receipt(paths)
        expected_previous_identity = str(existing["receipt_identity"])
        current = observer(
            paths,
            reference="default",
            proof_request_id=existing["proof"]["proof_request_id"],
        )
        if not _receipt_mismatch_names(existing, current):
            return (
                str(existing["deployment_id"]),
                existing,
                str(existing["receipt_identity"]),
            )
    transaction_id = transaction_id_factory()
    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="initialize-connection",
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        raise _fail(
            "CONNECTION_INITIALIZATION_ACTIVE",
            "connection initialization could not acquire ownership",
        ) from error
    start_utc = utc_now()
    transaction = {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "initialize-connection",
        "start_utc": start_utc,
        "finish_utc": None,
        "state": "GENERATING_CONNECTION_RECEIPT",
        "reason_code": "OK",
        "input_target_name": None,
        "intake_snapshot_identity": None,
        "owner_identity": {
            key: owner.get(key)
            for key in (
                "pid",
                "process_start_identity",
                "boot_identity",
                "inspector_root_identity",
            )
        },
        "status_record_identity": None,
        "connection_receipt_id": None,
        "connection_receipt_identity": None,
        "connection_status_previous_identity": expected_previous_identity,
    }
    try:
        status = _status_value(
            paths,
            state="GENERATING_CONNECTION_RECEIPT",
            reason_code="OK",
            active_transaction_id=transaction_id,
            last_transaction_id=None,
        )
        transaction["status_record_identity"] = _write_status(
            paths, status, transition_observer
        )
        _write_transaction(paths, transaction, transition_observer)
        observation = observer(
            paths,
            reference="default",
            proof_request_id=(
                existing["proof"]["proof_request_id"]
                if existing is not None
                else None
            ),
        )
        if (
            observation["desired_state"] != "RUNNING"
            or observation["service_readiness"] != "READY"
            or observation["model_service_state"] != "READY"
            or observation["warm"] is not True
            or observation["default_target"]
            != observation["resolved_immutable_model_id"]
            or observation["recovery_state"] != "IDLE"
        ):
            raise _fail(
                "CONNECTION_SERVICE_UNAVAILABLE",
                "accepted baseline predicates are incomplete",
            )
        if existing is None:
            receipt_source = "EXISTING_ACCEPTED_READY_BASELINE"
            deployment_id = transaction_id
            deployment_result_identity = None
            recommended_reference = observation["default_alias"]
            promotion_result = None
            rollback_result = None
            retirement_result = None
        else:
            receipt_source = existing["receipt_source"]
            deployment_id = existing["deployment_id"]
            deployment_result_identity = existing[
                "deployment_result_identity"
            ]
            recommended_reference = existing["model"][
                "recommended_reference"
            ]
            promotion_result = existing["lifecycle"]["promotion_result"]
            rollback_result = existing["lifecycle"]["rollback_result"]
            retirement_result = existing["lifecycle"]["retirement_result"]
        receipt = build_receipt(
            observation,
            receipt_source=receipt_source,
            deployment_id=deployment_id,
            deployment_result_identity=deployment_result_identity,
            recommended_reference=recommended_reference,
            promotion_result=promotion_result,
            rollback_result=rollback_result,
            retirement_result=retirement_result,
            current_receipt_updated=True,
        )
        identity = publisher(
            paths,
            receipt,
            expected_previous_identity=expected_previous_identity,
        )
        transaction.update(
            {
                "finish_utc": utc_now(),
                "state": "COMPLETE",
                "reason_code": "CONNECTION_READY",
                "connection_receipt_id": receipt["receipt_id"],
                "connection_receipt_identity": identity,
            }
        )
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        transaction["status_record_identity"] = _write_status(
            paths, idle, transition_observer
        )
        _write_transaction(paths, transaction, transition_observer)
        return str(deployment_id), receipt, identity
    except BaseException:
        failed = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "FAILED_CLEAN",
            "reason_code": "CONNECTION_INITIALIZATION_FAILED",
        }
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        failed["status_record_identity"] = _write_status(
            paths, idle, transition_observer
        )
        _write_transaction(paths, failed, transition_observer)
        raise
    finally:
        lock.release()
