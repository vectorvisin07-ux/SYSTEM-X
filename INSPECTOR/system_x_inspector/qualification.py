"""Candidate-specific GGUF qualification contracts and runtime entry point."""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import ipaddress
import json
import os
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import quote

from .capabilities import (
    load_binding,
    load_capability_record,
    verify_installed_tuple,
)
from .constants import (
    QUALIFICATION_CHECK_STATUSES,
    QUALIFICATION_PROFILES,
    QUALIFICATION_REASON_CODES,
    QUALIFICATION_RESULT_CLASSES,
    SCHEMA_IDENTITIES,
)
from .decision import (
    build_decision_record,
    load_inspection_result,
    publish_decision_record,
    resolve_decision,
    validate_decision_record,
)
from .errors import InspectorError
from .handoff import (
    DecisionAuthorization,
    DestinationPlan,
    ManagedPolicy,
    PublishedArtifact,
    SourceEvidence,
    StagedArtifact,
    authenticate_handoff_decision,
    create_staged_artifact,
    prepare_handoff_destination,
    publish_staged_artifact,
    revalidate_handoff_source,
)
from .locking import TransactionLock, inspect_active_lock
from .paths import BranchHandoffPaths, InspectorPaths
from .records import (
    atomic_create_json,
    canonical_json_bytes,
    fsync_directory,
    read_json_record,
)
from .results import utc_now
from .runtime import (
    _inspection_stage,
    _status_value,
    _transaction_id,
    _write_status,
    _write_transaction,
)
from .service_publication import (
    HttpResult,
    LoopbackJsonClient,
    SecretCredential,
    ServiceSnapshot,
    read_local_credential,
)


SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
QUALIFICATION_ID_PATTERN = re.compile(
    r"qualification-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "qualification_id",
        "transaction_id",
        "created_utc",
        "completed_utc",
        "inspection",
        "input_decision",
        "requested_profile",
        "installed_tuple",
        "incumbent",
        "candidate_runtime",
        "checks",
        "supported_profiles",
        "observed_capabilities",
        "result_class",
        "reason_codes",
        "restoration",
        "cleanup",
        "validity_predicate",
        "result_identity",
    }
)
CHECK_FIELDS = frozenset(
    {
        "check_name",
        "required",
        "protocol_family",
        "status",
        "request_id",
        "http_status",
        "finish_or_terminal_state",
        "usage",
        "capability_observation",
        "bounded_evidence_identity",
        "reason_code",
    }
)
FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "raw_api_key",
        "credential_verifier",
        "pepper",
        "private_router_url",
        "model_child_port",
        "absolute_managed_path",
        "prompt",
        "answer",
        "reasoning",
        "tool_arguments",
        "tool_result",
    }
)
DECISION_FILE_PATTERN = re.compile(
    r"decision-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\.json\Z"
)
INSPECTION_ID_PATTERN = re.compile(
    r"inspection-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
QUALIFICATION_MANAGED_NAME_PATTERN = re.compile(
    r"qualification-candidate-[0-9a-f]{16}-[0-9a-f]{16}\.gguf\Z"
)
MAX_CONTROL_JSON_BYTES = 2 * 1024 * 1024
MAX_PROFILE_EVIDENCE_BYTES = 64 * 1024
MAX_STREAM_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_OPERATION_LOG_SCAN_BYTES = 4 * 1024 * 1024
MAX_CONTROL_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
PROFILE_HTTP_TIMEOUT_SECONDS = 60.0
PROFILE_MAX_OUTPUT_TOKENS = 1024
OPERATION_RECORD_WAIT_SECONDS = 15.0
REGISTRATION_WAIT_SECONDS = 45.0
RESTORATION_WAIT_SECONDS = 45.0
MAX_MANAGER_WAIT_SECONDS = 3600.0
MANAGER_RESTORE_TIMEOUT_SECONDS = 180.0
REQUEST_ID_PATTERN = re.compile(r"sx_req_[0-9a-f]{32}\Z")

CORE_CHAT_CHECKS = (
    ("candidate_registry_ready", True, "registry"),
    ("candidate_public_model_identity", True, "system_x"),
    ("candidate_model_detail", True, "system_x"),
    ("candidate_capability_manifest", True, "registry"),
    ("native_token_count", True, "system_x"),
    ("native_nonstream_chat", True, "system_x"),
    ("native_final_content", True, "system_x"),
    ("native_response_model", True, "system_x"),
    ("native_stream_sequence", True, "system_x"),
    ("request_cancellation", True, "system_x"),
    ("later_request", True, "system_x"),
    ("request_record_correlation", True, "manager"),
)
EXTENDED_CHAT_CHECKS = (
    ("reasoning_output", True, "system_x"),
    ("reasoning_final_separation", True, "system_x"),
    ("structured_json_output", True, "system_x"),
    ("structured_schema_validation", True, "system_x"),
)
AGENT_CHECKS = (
    ("strict_tool_schema", True, "system_x"),
    ("forced_registered_tool_call", True, "system_x"),
    ("tool_call_id_preserved", True, "system_x"),
    ("tool_arguments_json", True, "system_x"),
    ("api_did_not_execute_tool", True, "system_x"),
    ("typed_external_result_continuation", True, "system_x"),
    ("agent_final_content", True, "system_x"),
)
FULL_PRODUCT_CHECKS = (
    ("openai_model_listing", True, "openai"),
    ("openai_nonstream_request", True, "openai"),
    ("openai_stream_sequence", True, "openai"),
    ("messages_model_listing", True, "messages"),
    ("messages_nonstream_request", True, "messages"),
    ("messages_stream_sequence", True, "messages"),
    ("capability_gating_reasoning", True, "system_x"),
    ("capability_gating_tool_calling", True, "system_x"),
    ("capability_gating_structured_output", True, "system_x"),
    ("heychat_adapter_compatibility", True, "heychat"),
)
ALL_QUALIFICATION_CHECKS = (
    CORE_CHAT_CHECKS
    + EXTENDED_CHAT_CHECKS
    + AGENT_CHECKS
    + FULL_PRODUCT_CHECKS
)
PROFILE_CHECKS = {
    "CORE_CHAT": CORE_CHAT_CHECKS,
    "EXTENDED_CHAT": CORE_CHAT_CHECKS + EXTENDED_CHAT_CHECKS,
    "AGENT": CORE_CHAT_CHECKS + AGENT_CHECKS,
    "FULL_PRODUCT": CORE_CHAT_CHECKS + FULL_PRODUCT_CHECKS,
}
DIRECT_PROFILE_REQUIREMENTS = {
    "CORE_CHAT": frozenset(
        {
            "generate/chat",
            "Responses",
            "token count",
            "streaming",
        }
    ),
    "EXTENDED_CHAT": frozenset(
        {
            "generate/chat",
            "Responses",
            "token count",
            "streaming",
            "reasoning output",
            "accepted reasoning control mode",
            "structured output",
        }
    ),
    "AGENT": frozenset(
        {
            "generate/chat",
            "Responses",
            "token count",
            "streaming",
            "tool calling",
            "structured output",
        }
    ),
    "FULL_PRODUCT": frozenset(
        {
            "generate/chat",
            "Responses",
            "token count",
            "streaming",
            "OpenAI-compatible",
            "Messages-compatible",
            "HEYCHAT-compatible",
        }
    ),
}
SUCCESS_CHECK_STATUSES = frozenset({"PASSED", "PASSED_AVAILABLE"})
ACCURATE_GATING_STATUSES = frozenset(
    {"PASSED", "PASSED_AVAILABLE", "PASSED_GATED_UNAVAILABLE"}
)
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


@dataclass(frozen=True)
class QualificationAuthorization:
    requested_profile: str
    decision_authorization: DecisionAuthorization
    branch_paths: BranchHandoffPaths
    source: SourceEvidence
    installed_tuple_evidence: dict[str, Any]


@dataclass(frozen=True)
class IncumbentSnapshot:
    present: bool
    default_alias: str | None
    public_model_id: str | None
    artifact_version_id: str | None
    capability_manifest_identity: str | None
    managed_location_identity: str | None
    warm_before: dict[str, Any] | None
    registry_generation: int
    credential_key_id: str
    profile_identity: str
    service_readiness: str
    recovery_state: str
    api_service_transaction_id: str | None
    router_transaction_id: str | None
    model_child_identity: dict[str, Any] | None
    historical_registry_locations: tuple[str, ...]

    def result_projection(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "default_alias": self.default_alias,
            "public_model_id": self.public_model_id,
            "artifact_version_id": self.artifact_version_id,
            "capability_manifest_identity": self.capability_manifest_identity,
            "managed_location_identity": self.managed_location_identity,
            "warm_before": self.warm_before,
            "warm_after": None,
        }


@dataclass(frozen=True)
class QualificationAdmission:
    plan: DestinationPlan
    staged: StagedArtifact
    published: PublishedArtifact


@dataclass(frozen=True)
class ProfileRun:
    requested_profile: str
    checks: tuple[dict[str, Any], ...]
    supported_profiles: tuple[str, ...]
    observed_capabilities: dict[str, list[str]]
    result_class: str
    reason_codes: tuple[str, ...]


class QualificationProbeAdapter(Protocol):
    def probe(
        self,
        check_name: str,
        *,
        model_id: str,
        artifact_version_id: str,
        capability_manifest_identity: str,
    ) -> dict[str, Any]:
        """Return one content-free check observation."""


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _system_x_source_evidence(paths: InspectorPaths) -> dict[str, str]:
    system_x_root = paths.source_root.parent.parent
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(system_x_root),
            "rev-parse",
            "HEAD",
            "HEAD^{tree}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    lines = completed.stdout.decode("ascii", errors="replace").splitlines()
    if (
        completed.returncode != 0
        or len(lines) != 2
        or any(GIT_OBJECT_PATTERN.fullmatch(item) is None for item in lines)
    ):
        raise _qualification_error(
            "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
            "System X source commit and tree cannot be authenticated",
        )
    source_manifest = []
    source_roots = (
        paths.source_root,
        paths.source_root.parent / "schemas",
    )
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*")):
            if path.suffix not in {".py", ".json"}:
                continue
            try:
                details = path.lstat()
            except FileNotFoundError as error:
                raise _qualification_error(
                    "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
                    "Inspector source graph changed during authentication",
                ) from error
            if (
                stat.S_ISLNK(details.st_mode)
                or not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
            ):
                raise _qualification_error(
                    "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
                    "Inspector source graph contains an unsafe entry",
                )
            content = path.read_bytes()
            source_manifest.append(
                {
                    "path": str(path.relative_to(system_x_root)),
                    "byte_count": len(content),
                    "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
                }
            )
    if not source_manifest:
        raise _qualification_error(
            "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
            "Inspector source graph is empty",
        )
    return {
        "system_x_source_commit": lines[0],
        "system_x_source_tree": lines[1],
        "inspector_source_identity": _identity(source_manifest),
    }


def _operating_profile_identity(
    value: object,
    *,
    reason_code: str,
) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _qualification_error(
            reason_code,
            "operating profile cannot be canonically identified",
        ) from error
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def qualification_result_identity(value: dict[str, Any]) -> str:
    if "result_identity" not in value:
        raise InspectorError(
            "QUALIFICATION_RESULT_INVALID",
            "qualification result identity field is absent",
        )
    return _identity(
        {key: value[key] for key in sorted(value) if key != "result_identity"}
    )


def _reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_RESULT_KEYS:
                raise InspectorError(
                    "QUALIFICATION_RESULT_INVALID",
                    f"qualification result contains prohibited field: {key}",
                )
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def validate_qualification_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_FIELDS:
        raise InspectorError(
            "QUALIFICATION_RESULT_INVALID",
            "qualification result fields are not closed",
        )
    record = value
    _reject_forbidden(record)
    if (
        record["schema_version"]
        != SCHEMA_IDENTITIES["gguf_qualification_result"]
        or not isinstance(record["qualification_id"], str)
        or QUALIFICATION_ID_PATTERN.fullmatch(record["qualification_id"])
        is None
        or record["requested_profile"] not in QUALIFICATION_PROFILES
        or record["result_class"] not in QUALIFICATION_RESULT_CLASSES
    ):
        raise InspectorError(
            "QUALIFICATION_RESULT_INVALID",
            "qualification result identity or enum is invalid",
        )
    checks = record["checks"]
    if not isinstance(checks, list):
        raise InspectorError(
            "QUALIFICATION_RESULT_INVALID",
            "qualification checks are not an array",
        )
    for check in checks:
        if (
            not isinstance(check, dict)
            or set(check) != CHECK_FIELDS
            or check["status"] not in QUALIFICATION_CHECK_STATUSES
            or check["reason_code"] not in QUALIFICATION_REASON_CODES
        ):
            raise InspectorError(
                "QUALIFICATION_RESULT_INVALID",
                "qualification check contract is invalid",
            )
    reasons = record["reason_codes"]
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(item not in QUALIFICATION_REASON_CODES for item in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise InspectorError(
            "QUALIFICATION_RESULT_INVALID",
            "qualification reason codes are invalid",
        )
    expected = qualification_result_identity(record)
    if (
        not isinstance(record["result_identity"], str)
        or SHA256_PATTERN.fullmatch(record["result_identity"]) is None
        or record["result_identity"] != expected
    ):
        raise InspectorError(
            "QUALIFICATION_RESULT_INVALID",
            "qualification result identity is invalid",
        )
    return record


def profile_check_names(profile: str) -> tuple[str, ...]:
    try:
        definitions = PROFILE_CHECKS[profile]
    except KeyError as error:
        raise _qualification_error(
            "QUALIFICATION_INPUT_INVALID",
            "qualification profile is invalid",
        ) from error
    return tuple(item[0] for item in definitions)


def _validate_usage_projection(value: object) -> dict[str, int | None] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }:
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            "profile check usage is not closed",
        )
    for item in value.values():
        if item is not None and (type(item) is not int or item < 0):
            raise _qualification_error(
                "QUALIFICATION_PROFILE_FAILED",
                "profile check usage is invalid",
            )
    if (
        value["input_tokens"] is not None
        and value["output_tokens"] is not None
        and value["total_tokens"] is not None
        and value["total_tokens"]
        != value["input_tokens"] + value["output_tokens"]
    ):
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            "profile check usage total is inconsistent",
        )
    return {
        "input_tokens": value["input_tokens"],
        "output_tokens": value["output_tokens"],
        "total_tokens": value["total_tokens"],
    }


def _validate_capability_projection(
    value: object,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"capability", "available", "accurately_gated"}
        or not isinstance(value["capability"], str)
        or not value["capability"]
        or value["available"] not in {True, False, None}
        or value["accurately_gated"] not in {True, False, None}
    ):
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            "profile capability observation is invalid",
        )
    return {
        "capability": value["capability"],
        "available": value["available"],
        "accurately_gated": value["accurately_gated"],
    }


def _normalize_check(
    definition: tuple[str, bool, str], value: object
) -> dict[str, Any]:
    name, required, protocol_family = definition
    allowed = {
        "status",
        "request_id",
        "http_status",
        "finish_or_terminal_state",
        "usage",
        "capability_observation",
        "evidence",
        "reason_code",
    }
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"profile check result is invalid: {name}",
        )
    _reject_forbidden(value)
    status_value = value.get("status")
    if status_value not in QUALIFICATION_CHECK_STATUSES:
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"profile check status is invalid: {name}",
        )
    request_id = value.get("request_id")
    http_status = value.get("http_status")
    terminal = value.get("finish_or_terminal_state")
    if (
        request_id is not None
        and (
            not isinstance(request_id, str)
            or REQUEST_ID_PATTERN.fullmatch(request_id) is None
        )
    ):
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"profile check request identity is invalid: {name}",
        )
    if http_status is not None and (
        type(http_status) is not int or not 100 <= http_status <= 599
    ):
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"profile check HTTP status is invalid: {name}",
        )
    if terminal is not None and (
        not isinstance(terminal, str) or not terminal
    ):
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"profile check terminal state is invalid: {name}",
        )
    reason = value.get(
        "reason_code",
        {
            "PASSED": "CHECK_PASSED",
            "PASSED_AVAILABLE": "CHECK_AVAILABLE",
            "PASSED_GATED_UNAVAILABLE": "CHECK_GATED_UNAVAILABLE",
            "UNAVAILABLE": "CHECK_UNAVAILABLE",
            "FAILED": "CHECK_FAILED",
            "NOT_APPLICABLE": "CHECK_NOT_APPLICABLE",
            "NOT_REQUESTED": "CHECK_NOT_REQUESTED",
        }[status_value],
    )
    if reason not in QUALIFICATION_REASON_CODES:
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"profile check reason is invalid: {name}",
        )
    capability = _validate_capability_projection(
        value.get("capability_observation")
    )
    if (
        status_value == "PASSED_AVAILABLE"
        and (
            capability is None
            or capability["available"] is not True
        )
    ):
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"available check omitted availability evidence: {name}",
        )
    if (
        status_value == "PASSED_GATED_UNAVAILABLE"
        and (
            capability is None
            or capability["available"] is not False
            or capability["accurately_gated"] is not True
        )
    ):
        raise _qualification_error(
            "QUALIFICATION_PROFILE_FAILED",
            f"gated check omitted gating evidence: {name}",
        )
    evidence = value.get("evidence")
    bounded_identity = None
    if evidence is not None:
        try:
            encoded = canonical_json_bytes(evidence)
        except (TypeError, ValueError) as error:
            raise _qualification_error(
                "QUALIFICATION_PROFILE_FAILED",
                f"profile evidence is not canonical: {name}",
            ) from error
        if len(encoded) > MAX_PROFILE_EVIDENCE_BYTES:
            raise _qualification_error(
                "QUALIFICATION_PROFILE_FAILED",
                f"profile evidence exceeds its bound: {name}",
            )
        bounded_identity = (
            "sha256:" + hashlib.sha256(encoded).hexdigest()
        )
    return {
        "check_name": name,
        "required": required,
        "protocol_family": protocol_family,
        "status": status_value,
        "request_id": request_id,
        "http_status": http_status,
        "finish_or_terminal_state": terminal,
        "usage": _validate_usage_projection(value.get("usage")),
        "capability_observation": capability,
        "bounded_evidence_identity": bounded_identity,
        "reason_code": reason,
    }


def _not_requested_check(
    definition: tuple[str, bool, str],
) -> dict[str, Any]:
    return _normalize_check(
        definition,
        {
            "status": "NOT_REQUESTED",
            "reason_code": "CHECK_NOT_REQUESTED",
            "evidence": {
                "check_name": definition[0],
                "selected": False,
            },
        },
    )


def _core_passed(checks: Iterable[dict[str, Any]]) -> bool:
    by_name = {item["check_name"]: item for item in checks}
    return all(
        by_name[name]["status"] in SUCCESS_CHECK_STATUSES
        for name, _required, _protocol in CORE_CHAT_CHECKS
    )


def classify_qualification_result(
    requested_profile: str,
    checks: Iterable[dict[str, Any]],
    *,
    runtime_outcome: str = "READY",
    cleanup_proved: bool = True,
    restoration_proved: bool = True,
    ownership_certain: bool = True,
) -> tuple[str, tuple[str, ...]]:
    if requested_profile not in QUALIFICATION_PROFILES:
        raise _qualification_error(
            "QUALIFICATION_INPUT_INVALID",
            "qualification profile is invalid",
        )
    if (
        not ownership_certain
        or not cleanup_proved
        or not restoration_proved
    ):
        return (
            "QUALIFICATION_FAIL_CLOSED",
            ("QUALIFICATION_FAIL_CLOSED",),
        )
    if runtime_outcome == "UNSUPPORTED":
        return "UNSUPPORTED", ("QUALIFICATION_PROFILE_UNSUPPORTED",)
    if runtime_outcome == "REJECTED":
        return "REJECTED", ("QUALIFICATION_REJECTED",)
    if runtime_outcome != "READY":
        return (
            "QUALIFICATION_FAILED_CLEAN",
            ("QUALIFICATION_FAILED_CLEAN",),
        )
    records = tuple(checks)
    if not _core_passed(records):
        return (
            "QUALIFICATION_FAILED_CLEAN",
            ("QUALIFICATION_PROFILE_FAILED",),
        )
    selected = set(profile_check_names(requested_profile))
    selected_records = [
        item for item in records if item["check_name"] in selected
    ]
    if requested_profile == "CORE_CHAT":
        return (
            "SUPPORTED_FOR_CURRENT_TUPLE",
            ("QUALIFICATION_PROFILE_SUPPORTED",),
        )
    extras = [
        item
        for item in selected_records
        if item["check_name"]
        not in {definition[0] for definition in CORE_CHAT_CHECKS}
    ]
    if requested_profile == "FULL_PRODUCT":
        if all(
            item["status"] in ACCURATE_GATING_STATUSES
            for item in extras
        ):
            return (
                "SUPPORTED_FOR_CURRENT_TUPLE",
                ("QUALIFICATION_PROFILE_SUPPORTED",),
            )
        return (
            "QUALIFICATION_FAILED_CLEAN",
            ("QUALIFICATION_PROFILE_FAILED",),
        )
    if any(item["status"] == "FAILED" for item in extras):
        return (
            "QUALIFICATION_FAILED_CLEAN",
            ("QUALIFICATION_PROFILE_FAILED",),
        )
    if all(item["status"] in SUCCESS_CHECK_STATUSES for item in extras):
        return (
            "SUPPORTED_FOR_CURRENT_TUPLE",
            ("QUALIFICATION_PROFILE_SUPPORTED",),
        )
    return "PARTIALLY_SUPPORTED", ("QUALIFICATION_PROFILE_PARTIAL",)


def _observed_capabilities(
    checks: Iterable[dict[str, Any]],
) -> dict[str, list[str]]:
    available: set[str] = set()
    gated: set[str] = set()
    unavailable: set[str] = set()
    for check in checks:
        observation = check["capability_observation"]
        if not isinstance(observation, dict):
            continue
        name = observation["capability"]
        if observation["available"] is True:
            available.add(name)
        elif (
            observation["available"] is False
            and observation["accurately_gated"] is True
        ):
            gated.add(name)
        elif observation["available"] is False:
            unavailable.add(name)
    return {
        "available": sorted(available),
        "gated_unavailable": sorted(gated),
        "unavailable": sorted(unavailable),
    }


def run_capability_profile(
    adapter: QualificationProbeAdapter,
    *,
    requested_profile: str,
    model_id: str,
    artifact_version_id: str,
    capability_manifest_identity: str,
) -> ProfileRun:
    if (
        requested_profile not in QUALIFICATION_PROFILES
        or not isinstance(model_id, str)
        or not model_id
        or not isinstance(artifact_version_id, str)
        or not artifact_version_id
        or SHA256_PATTERN.fullmatch(capability_manifest_identity) is None
    ):
        raise _qualification_error(
            "QUALIFICATION_INPUT_INVALID",
            "profile runner input is invalid",
        )
    selected = set(profile_check_names(requested_profile))
    checks: list[dict[str, Any]] = []
    for definition in ALL_QUALIFICATION_CHECKS:
        name = definition[0]
        if name not in selected:
            checks.append(_not_requested_check(definition))
            continue
        try:
            value = adapter.probe(
                name,
                model_id=model_id,
                artifact_version_id=artifact_version_id,
                capability_manifest_identity=(
                    capability_manifest_identity
                ),
            )
            checks.append(_normalize_check(definition, value))
        except InspectorError as error:
            reason = (
                error.reason_code
                if error.reason_code in QUALIFICATION_REASON_CODES
                else "QUALIFICATION_INTERNAL_ERROR"
            )
            checks.append(
                _normalize_check(
                    definition,
                    {
                        "status": "FAILED",
                        "reason_code": reason,
                        "evidence": {
                            "check_name": name,
                            "error_reason_code": reason,
                        },
                    },
                )
            )
        except Exception:
            checks.append(
                _normalize_check(
                    definition,
                    {
                        "status": "FAILED",
                        "reason_code": "QUALIFICATION_INTERNAL_ERROR",
                        "evidence": {
                            "check_name": name,
                            "error_reason_code": (
                                "QUALIFICATION_INTERNAL_ERROR"
                            ),
                        },
                    },
                )
            )
    result_class, reasons = classify_qualification_result(
        requested_profile, checks
    )
    supported: list[str] = []
    if _core_passed(checks):
        supported.append("CORE_CHAT")
    if (
        requested_profile != "CORE_CHAT"
        and result_class == "SUPPORTED_FOR_CURRENT_TUPLE"
    ):
        supported.append(requested_profile)
    return ProfileRun(
        requested_profile=requested_profile,
        checks=tuple(checks),
        supported_profiles=tuple(supported),
        observed_capabilities=_observed_capabilities(checks),
        result_class=result_class,
        reason_codes=reasons,
    )


def _check_result(
    *,
    status: str,
    evidence: dict[str, Any],
    request_id: str | None = None,
    http_status: int | None = None,
    terminal: str | None = None,
    usage: dict[str, int | None] | None = None,
    capability: str | None = None,
    available: bool | None = None,
    accurately_gated: bool | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "status": status,
        "request_id": request_id,
        "http_status": http_status,
        "finish_or_terminal_state": terminal,
        "usage": usage,
        "capability_observation": (
            {
                "capability": capability,
                "available": available,
                "accurately_gated": accurately_gated,
            }
            if capability is not None
            else None
        ),
        "evidence": evidence,
    }
    if reason_code is not None:
        value["reason_code"] = reason_code
    return value


def _response_request_id(result: HttpResult) -> str:
    request_id = result.body.get("request_id")
    if (
        not isinstance(request_id, str)
        or REQUEST_ID_PATTERN.fullmatch(request_id) is None
        or result.request_id_header != request_id
    ):
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "public response request identity is invalid",
        )
    return request_id


def _usage_projection(value: object) -> dict[str, int | None]:
    if not isinstance(value, dict):
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "public response usage is absent",
        )
    projected = {
        "input_tokens": value.get("input_tokens"),
        "output_tokens": value.get("output_tokens"),
        "total_tokens": value.get("total_tokens"),
    }
    validated = _validate_usage_projection(projected)
    if validated is None:
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "public response usage is invalid",
        )
    return validated


def parse_system_x_stream(
    raw: bytes, *, expected_model_id: str
) -> dict[str, Any]:
    if (
        not isinstance(raw, bytes)
        or not raw
        or len(raw) > MAX_STREAM_RESPONSE_BYTES
        or not isinstance(expected_model_id, str)
        or not expected_model_id
    ):
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "native stream is absent or exceeds its bound",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "native stream is not UTF-8",
        ) from error
    text = text.replace("\r\n", "\n")
    blocks = [item for item in text.split("\n\n") if item.strip()]
    events: list[dict[str, Any]] = []
    request_id: str | None = None
    output_digest = hashlib.sha256()
    output_bytes = 0
    terminal_count = 0
    usage: dict[str, int | None] | None = None
    terminal_state: str | None = None
    allowed_types = {
        "response.started",
        "response.reasoning.delta",
        "response.output_text.delta",
        "response.tool_call.added",
        "response.tool_call.arguments.delta",
        "response.tool_call.done",
        "response.usage",
        "response.requires_action",
        "response.completed",
        "response.incomplete",
        "response.failed",
    }
    terminal_types = {
        "response.requires_action",
        "response.completed",
        "response.incomplete",
        "response.failed",
    }
    for expected_sequence, block in enumerate(blocks):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ": " not in line:
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "native stream framing is invalid",
                )
            key, value = line.split(": ", 1)
            if key in fields or key not in {"event", "id", "data"}:
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "native stream fields are not closed",
                )
            fields[key] = value
        if set(fields) != {"event", "id", "data"}:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream event is incomplete",
            )
        try:
            data = json.loads(fields["data"])
        except json.JSONDecodeError as error:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream data is invalid",
            ) from error
        if not isinstance(data, dict):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream data is not an object",
            )
        observed_request_id = data.get("request_id")
        event_type = data.get("type")
        if (
            not isinstance(observed_request_id, str)
            or REQUEST_ID_PATTERN.fullmatch(observed_request_id) is None
            or event_type not in allowed_types
            or event_type != fields["event"]
            or data.get("sequence") != expected_sequence
            or data.get("model") != expected_model_id
            or fields["id"]
            != f"{observed_request_id}:{expected_sequence}"
        ):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream identity or sequence is invalid",
            )
        if request_id is None:
            request_id = observed_request_id
        elif request_id != observed_request_id:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream changed request identity",
            )
        if expected_sequence == 0 and event_type != "response.started":
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream did not begin with response.started",
            )
        if terminal_count:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream emitted after its terminal event",
            )
        if event_type == "response.output_text.delta":
            delta = data.get("delta")
            if not isinstance(delta, str) or not delta:
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "native stream output delta is invalid",
                )
            encoded = delta.encode("utf-8")
            output_digest.update(encoded)
            output_bytes += len(encoded)
        if event_type == "response.usage":
            usage = _usage_projection(data.get("usage"))
        if event_type in terminal_types:
            terminal_count += 1
            state = data.get("status")
            finish = data.get("finish_reason")
            if (
                not isinstance(state, str)
                or not state
                or not isinstance(finish, str)
                or not finish
            ):
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "native stream terminal state is invalid",
                )
            terminal_state = f"{state}:{finish}"
        events.append(
            {
                "sequence": expected_sequence,
                "type": event_type,
            }
        )
    if (
        request_id is None
        or terminal_count != 1
        or terminal_state is None
        or events[-1]["type"] not in terminal_types
        or usage is None
    ):
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "native stream did not complete one legal sequence",
        )
    return {
        "request_id": request_id,
        "event_count": len(events),
        "event_types": [item["type"] for item in events],
        "terminal_state": terminal_state,
        "usage": usage,
        "content_present": output_bytes > 0,
        "content_bytes": output_bytes,
        "content_sha256": (
            "sha256:" + output_digest.hexdigest()
            if output_bytes
            else None
        ),
    }


@dataclass(frozen=True)
class OperationExpectation:
    request_id: str
    protocol_family: str
    endpoint: str
    operation: str
    streamed: bool
    http_status: int
    operation_state: str | None
    public_model_id: str | None
    artifact_version_id: str | None


class PublicProfileProbeAdapter:
    """Content-free profile probes over the accepted public System X origin."""

    def __init__(
        self,
        service: ServiceSnapshot,
        credential: SecretCredential,
        *,
        registry_states: Iterable[str],
        public_model_id: str,
        artifact_version_id: str,
        capability_manifest_identity: str,
        compatibility_probe: (
            Callable[[str, str, str, str], dict[str, Any]] | None
        ) = None,
    ) -> None:
        address = ipaddress.ip_address(service.host)
        if (
            not address.is_loopback
            or str(address) != service.host
            or not isinstance(credential.key_id, str)
            or not credential.key_id
            or not isinstance(public_model_id, str)
            or not public_model_id
            or not isinstance(artifact_version_id, str)
            or not artifact_version_id
            or SHA256_PATTERN.fullmatch(capability_manifest_identity) is None
        ):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "public profile adapter input is invalid",
            )
        details = service.operation_log.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
        ):
            raise _qualification_error(
                "QUALIFICATION_REQUEST_RECORD_NOT_FOUND",
                "manager-owned operation log is unsafe",
            )
        self.service = service
        self.credential = credential
        self.client = LoopbackJsonClient(service)
        self.registry_states = tuple(registry_states)
        self.public_model_id = public_model_id
        self.artifact_version_id = artifact_version_id
        self.capability_manifest_identity = capability_manifest_identity
        self.compatibility_probe = compatibility_probe
        self.log_start_offset = details.st_size
        self.expectations: dict[str, OperationExpectation] = {}
        self.cache: dict[str, Any] = {}

    def _require_target(
        self,
        model_id: str,
        artifact_version_id: str,
        capability_manifest_identity: str,
    ) -> None:
        if (
            model_id != self.public_model_id
            or artifact_version_id != self.artifact_version_id
            or capability_manifest_identity
            != self.capability_manifest_identity
        ):
            raise _qualification_error(
                "QUALIFICATION_REQUEST_RECORD_MISMATCH",
                "profile target differs from observed candidate",
            )

    def _expect(
        self,
        request_id: str,
        *,
        endpoint: str,
        operation: str,
        streamed: bool,
        http_status: int,
        operation_state: str | None,
        protocol_family: str = "system_x",
        resolved_model: bool = True,
    ) -> None:
        if request_id in self.expectations:
            return
        self.expectations[request_id] = OperationExpectation(
            request_id=request_id,
            protocol_family=protocol_family,
            endpoint=endpoint,
            operation=operation,
            streamed=streamed,
            http_status=http_status,
            operation_state=operation_state,
            public_model_id=(
                self.public_model_id if resolved_model else None
            ),
            artifact_version_id=(
                self.artifact_version_id if resolved_model else None
            ),
        )

    def _catalogue(self) -> dict[str, Any]:
        cached = self.cache.get("catalogue")
        if isinstance(cached, dict):
            return cached
        result = self.client.request(
            "GET",
            "/system/v1/models",
            credential=self.credential,
        )
        if result.status in {401, 403}:
            raise _qualification_error(
                "QUALIFICATION_AUTHENTICATION_REJECTED",
                "public credential was rejected",
            )
        request_id = _response_request_id(result)
        models = result.body.get("models")
        matches = (
            [
                item
                for item in models
                if isinstance(item, dict)
                and item.get("id") == self.public_model_id
            ]
            if isinstance(models, list)
            else []
        )
        if result.status != 200 or len(matches) != 1:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "candidate is absent from the public catalogue",
            )
        self._expect(
            request_id,
            endpoint="/system/v1/models",
            operation="model.list",
            streamed=False,
            http_status=result.status,
            operation_state="completed",
            resolved_model=False,
        )
        cached = {
            "request_id": request_id,
            "http_status": result.status,
            "registry_generation": result.body.get(
                "registry_generation"
            ),
            "model": matches[0],
        }
        self.cache["catalogue"] = cached
        return cached

    def _detail(self) -> dict[str, Any]:
        cached = self.cache.get("detail")
        if isinstance(cached, dict):
            return cached
        target = quote(self.public_model_id, safe="")
        result = self.client.request(
            "GET",
            f"/system/v1/models/{target}",
            credential=self.credential,
        )
        if result.status in {401, 403}:
            raise _qualification_error(
                "QUALIFICATION_AUTHENTICATION_REJECTED",
                "public credential was rejected",
            )
        request_id = _response_request_id(result)
        model = result.body.get("model")
        if (
            result.status != 200
            or not isinstance(model, dict)
            or model.get("public_model_id") != self.public_model_id
            or model.get("resolved_model_id") != self.public_model_id
            or model.get("artifact_version_id")
            != self.artifact_version_id
            or model.get("state") != "ready"
            or not isinstance(model.get("capabilities"), dict)
        ):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "candidate public model detail is inconsistent",
            )
        self._expect(
            request_id,
            endpoint="/system/v1/models/{model_id}",
            operation="model.detail",
            streamed=False,
            http_status=result.status,
            operation_state="completed",
        )
        cached = {
            "request_id": request_id,
            "http_status": result.status,
            "registry_generation": result.body.get(
                "registry_generation"
            ),
            "model": model,
        }
        self.cache["detail"] = cached
        return cached

    def _native_body(
        self, instruction: str, *, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self.public_model_id,
            "messages": [{"role": "user", "content": instruction}],
            "max_output_tokens": PROFILE_MAX_OUTPUT_TOKENS,
            "stream": stream,
            "temperature": 0.0,
        }

    def _json_chat(
        self, cache_key: str, instruction: str
    ) -> dict[str, Any]:
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            return cached
        result = self.client.request(
            "POST",
            "/system/v1/chat",
            credential=self.credential,
            body=self._native_body(instruction, stream=False),
        )
        if result.status in {401, 403}:
            raise _qualification_error(
                "QUALIFICATION_AUTHENTICATION_REJECTED",
                "public credential was rejected",
            )
        request_id = _response_request_id(result)
        output = result.body.get("output")
        usage = _usage_projection(result.body.get("usage"))
        content = (
            output.get("content") if isinstance(output, dict) else None
        )
        if (
            result.status != 200
            or result.body.get("model") != self.public_model_id
            or result.body.get("status") != "completed"
            or not isinstance(result.body.get("finish_reason"), str)
            or not isinstance(content, str)
            or not content
        ):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native chat response is invalid",
            )
        encoded = content.encode("utf-8")
        self._expect(
            request_id,
            endpoint="/system/v1/chat",
            operation="chat",
            streamed=False,
            http_status=result.status,
            operation_state="completed",
        )
        cached = {
            "request_id": request_id,
            "http_status": result.status,
            "status": result.body["status"],
            "finish_reason": result.body["finish_reason"],
            "model": result.body["model"],
            "usage": usage,
            "content_present": True,
            "content_bytes": len(encoded),
            "content_sha256": (
                "sha256:" + hashlib.sha256(encoded).hexdigest()
            ),
        }
        self.cache[cache_key] = cached
        return cached

    def _token_count(self) -> dict[str, Any]:
        cached = self.cache.get("token_count")
        if isinstance(cached, dict):
            return cached
        result = self.client.request(
            "POST",
            "/system/v1/tokens/count",
            credential=self.credential,
            body={
                "model": self.public_model_id,
                "operation": "chat",
                "messages": [
                    {
                        "role": "user",
                        "content": "Count this bounded qualification input.",
                    }
                ],
            },
        )
        if result.status in {401, 403}:
            raise _qualification_error(
                "QUALIFICATION_AUTHENTICATION_REJECTED",
                "public credential was rejected",
            )
        request_id = _response_request_id(result)
        count = result.body.get("input_tokens")
        if (
            result.status != 200
            or result.body.get("model") != self.public_model_id
            or result.body.get("operation") != "chat"
            or type(count) is not int
            or count < 1
        ):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native token count response is invalid",
            )
        self._expect(
            request_id,
            endpoint="/system/v1/tokens/count",
            operation="tokens.count",
            streamed=False,
            http_status=result.status,
            operation_state="completed",
        )
        cached = {
            "request_id": request_id,
            "http_status": result.status,
            "input_tokens": count,
        }
        self.cache["token_count"] = cached
        return cached

    def _native_stream(self) -> dict[str, Any]:
        cached = self.cache.get("native_stream")
        if isinstance(cached, dict):
            return cached
        encoded = canonical_json_bytes(
            self._native_body(
                (
                    "Return exactly the text OK as the final answer. "
                    "Do not explain or reason."
                ),
                stream=True,
            )
        )
        connection = http.client.HTTPConnection(
            self.service.host,
            self.service.port,
            timeout=PROFILE_HTTP_TIMEOUT_SECONDS,
        )
        try:
            connection.request(
                "POST",
                "/system/v1/chat",
                body=encoded,
                headers={
                    "Accept": "text/event-stream",
                    "Authorization": "Bearer " + self.credential.raw,
                    "Connection": "close",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            status_code = response.status
            media_type = response.getheader("Content-Type", "")
            request_id_header = response.getheader(
                "X-System-X-Request-ID"
            )
            raw = response.read(MAX_STREAM_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as error:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native streaming request failed",
            ) from error
        finally:
            connection.close()
        if (
            status_code != 200
            or not media_type.startswith("text/event-stream")
            or len(raw) > MAX_STREAM_RESPONSE_BYTES
        ):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native streaming response is invalid",
            )
        parsed = parse_system_x_stream(
            raw, expected_model_id=self.public_model_id
        )
        if request_id_header != parsed["request_id"]:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "native stream header identity is invalid",
            )
        state, _separator, finish = parsed["terminal_state"].partition(":")
        self._expect(
            parsed["request_id"],
            endpoint="/system/v1/chat",
            operation="chat",
            streamed=True,
            http_status=status_code,
            operation_state=state,
        )
        parsed["http_status"] = status_code
        parsed["finish_reason"] = finish
        self.cache["native_stream"] = parsed
        return parsed

    def _compatibility_exchange(
        self,
        protocol: str,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        if protocol not in {"openai", "messages"}:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "compatibility protocol is invalid",
            )
        encoded = canonical_json_bytes(body) if body is not None else None
        headers = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Connection": "close",
        }
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        if protocol == "openai":
            headers["Authorization"] = "Bearer " + self.credential.raw
        else:
            headers["x-api-key"] = self.credential.raw
            headers["anthropic-version"] = "2023-06-01"
        connection = http.client.HTTPConnection(
            self.service.host,
            self.service.port,
            timeout=PROFILE_HTTP_TIMEOUT_SECONDS,
        )
        try:
            connection.request(method, path, body=encoded, headers=headers)
            response = connection.getresponse()
            status_code = response.status
            media_type = response.getheader("Content-Type", "")
            request_id = response.getheader("X-System-X-Request-ID")
            family_request_id = response.getheader(
                "x-request-id" if protocol == "openai" else "request-id"
            )
            compatibility = response.getheader(
                "X-System-X-Compatibility-Version"
                if protocol == "openai"
                else "X-System-X-Anthropic-Compatibility"
            )
            streaming_identity = (
                response.getheader(
                    "X-System-X-OpenAI-Streaming"
                    if protocol == "openai"
                    else "X-System-X-Anthropic-Streaming"
                )
                if stream
                else None
            )
            raw = response.read(MAX_STREAM_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as error:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "compatibility request failed",
            ) from error
        finally:
            connection.close()
        if status_code in {401, 403}:
            raise _qualification_error(
                "QUALIFICATION_AUTHENTICATION_REJECTED",
                "compatibility credential was rejected",
            )
        suffix = (
            request_id.removeprefix("sx_req_")
            if isinstance(request_id, str)
            else ""
        )
        expected_family_request_id = (
            request_id if protocol == "openai" else "req_sx_" + suffix
        )
        expected_compatibility = (
            "system-x.openai-compatible.v1"
            if protocol == "openai"
            else "system-x.anthropic-compatible.v1"
        )
        expected_streaming = (
            "system-x.openai-streaming.v1"
            if protocol == "openai"
            else "system-x.anthropic-streaming.v1"
        )
        if (
            not isinstance(request_id, str)
            or REQUEST_ID_PATTERN.fullmatch(request_id) is None
            or family_request_id != expected_family_request_id
            or compatibility != expected_compatibility
            or (stream and streaming_identity != expected_streaming)
            or len(raw) > MAX_STREAM_RESPONSE_BYTES
            or (
                stream
                and not media_type.startswith("text/event-stream")
            )
            or (
                not stream
                and not media_type.startswith("application/json")
            )
        ):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "compatibility response identity is invalid",
            )
        parsed = None
        if not stream:
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "compatibility JSON response is invalid",
                ) from error
            if not isinstance(parsed, dict):
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "compatibility JSON response is not an object",
                )
        return {
            "status": status_code,
            "request_id": request_id,
            "compatibility_identity": compatibility,
            "streaming_identity": streaming_identity,
            "body": parsed,
            "raw": raw if stream else None,
        }

    @staticmethod
    def _first_stream_event(
        raw: bytes, *, request_id: str, model_id: str
    ) -> dict[str, Any]:
        try:
            block = raw.decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError as error:
            raise _qualification_error(
                "QUALIFICATION_CANCELLATION_FAILED",
                "cancellation stream is not UTF-8",
            ) from error
        fields: dict[str, str] = {}
        for line in block.strip().splitlines():
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            fields[key] = value
        try:
            data = json.loads(fields["data"])
        except (KeyError, json.JSONDecodeError) as error:
            raise _qualification_error(
                "QUALIFICATION_CANCELLATION_FAILED",
                "cancellation stream did not emit a valid first event",
            ) from error
        if (
            fields.get("event") != "response.started"
            or fields.get("id") != f"{request_id}:0"
            or not isinstance(data, dict)
            or data.get("type") != "response.started"
            or data.get("sequence") != 0
            or data.get("request_id") != request_id
            or data.get("model") != model_id
        ):
            raise _qualification_error(
                "QUALIFICATION_CANCELLATION_FAILED",
                "cancellation stream first event is invalid",
            )
        return {
            "type": "response.started",
            "sequence": 0,
            "request_id": request_id,
            "model": model_id,
        }

    def _cancel_request(self) -> dict[str, Any]:
        cached = self.cache.get("cancel")
        if isinstance(cached, dict):
            return cached
        body = self._native_body(
            "Produce a long bounded numbered sequence for cancellation.",
            stream=True,
        )
        body["max_output_tokens"] = 2048
        encoded = canonical_json_bytes(body)
        connection = http.client.HTTPConnection(
            self.service.host,
            self.service.port,
            timeout=PROFILE_HTTP_TIMEOUT_SECONDS,
        )
        response: http.client.HTTPResponse | None = None
        status_code = 0
        request_id: str | None = None
        first = bytearray()
        try:
            connection.request(
                "POST",
                "/system/v1/chat",
                body=encoded,
                headers={
                    "Accept": "text/event-stream",
                    "Authorization": "Bearer " + self.credential.raw,
                    "Connection": "close",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            status_code = response.status
            request_id = response.getheader("X-System-X-Request-ID")
            while len(first) <= MAX_PROFILE_EVIDENCE_BYTES:
                line = response.readline()
                if not line:
                    break
                first.extend(line)
                if first.endswith(b"\n\n") or first.endswith(b"\r\n\r\n"):
                    break
        except (OSError, http.client.HTTPException) as error:
            raise _qualification_error(
                "QUALIFICATION_CANCELLATION_FAILED",
                "candidate cancellation request failed",
            ) from error
        finally:
            if response is not None:
                response.close()
            connection.close()
        if (
            status_code != 200
            or not isinstance(request_id, str)
            or REQUEST_ID_PATTERN.fullmatch(request_id) is None
            or not first
            or len(first) > MAX_PROFILE_EVIDENCE_BYTES
        ):
            raise _qualification_error(
                "QUALIFICATION_CANCELLATION_FAILED",
                "candidate cancellation response is invalid",
            )
        first_event = self._first_stream_event(
            bytes(first),
            request_id=request_id,
            model_id=self.public_model_id,
        )
        self._expect(
            request_id,
            endpoint="/system/v1/chat",
            operation="chat",
            streamed=True,
            http_status=status_code,
            operation_state="cancelled",
        )
        cached = {
            "request_id": request_id,
            "http_status": status_code,
            "first_event": first_event,
            "connection_closed_by_client": True,
        }
        self.cache["cancel"] = cached
        return cached

    def _capability_state(self, name: str) -> str:
        detail = self._detail()["model"]
        capabilities = detail.get("capabilities")
        state = (
            capabilities.get(name)
            if isinstance(capabilities, dict)
            else None
        )
        if state not in {
            "available",
            "unavailable",
            "not_tested",
            "not_exposed",
        }:
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "candidate capability state is invalid",
            )
        return state

    def _gated(self, capability: str, state: str) -> dict[str, Any]:
        return _check_result(
            status="PASSED_GATED_UNAVAILABLE",
            capability=capability,
            available=False,
            accurately_gated=True,
            reason_code="CHECK_GATED_UNAVAILABLE",
            evidence={
                "capability": capability,
                "reported_state": state,
                "request_issued": False,
            },
        )

    def _reasoning(self) -> dict[str, Any]:
        cached = self.cache.get("reasoning")
        if isinstance(cached, dict):
            return cached
        state = self._capability_state("reasoning_output")
        if state in {"unavailable", "not_exposed"}:
            cached = {"gated": self._gated("reasoning_output", state)}
            self.cache["reasoning"] = cached
            return cached
        body = self._native_body(
            "Reason briefly, then return one short final confirmation.",
            stream=False,
        )
        body["reasoning"] = {"mode": "standard"}
        result = self.client.request(
            "POST",
            "/system/v1/chat",
            credential=self.credential,
            body=body,
        )
        request_id = _response_request_id(result)
        output = result.body.get("output")
        reasoning = (
            output.get("reasoning") if isinstance(output, dict) else None
        )
        content = (
            output.get("content") if isinstance(output, dict) else None
        )
        usage = _usage_projection(result.body.get("usage"))
        if (
            result.status != 200
            or result.body.get("status") != "completed"
            or result.body.get("model") != self.public_model_id
            or not isinstance(reasoning, list)
            or not reasoning
            or any(not isinstance(item, str) or not item for item in reasoning)
            or not isinstance(content, str)
            or not content
        ):
            raise _qualification_error(
                "QUALIFICATION_PROFILE_FAILED",
                "reasoning output or final separation is invalid",
            )
        reasoning_bytes = sum(
            len(item.encode("utf-8")) for item in reasoning
        )
        final_encoded = content.encode("utf-8")
        self._expect(
            request_id,
            endpoint="/system/v1/chat",
            operation="chat",
            streamed=False,
            http_status=result.status,
            operation_state="completed",
        )
        cached = {
            "request_id": request_id,
            "http_status": result.status,
            "terminal": (
                f"{result.body['status']}:"
                f"{result.body.get('finish_reason')}"
            ),
            "usage": usage,
            "reasoning_item_count": len(reasoning),
            "reasoning_bytes": reasoning_bytes,
            "final_content_bytes": len(final_encoded),
            "final_content_sha256": (
                "sha256:" + hashlib.sha256(final_encoded).hexdigest()
            ),
        }
        self.cache["reasoning"] = cached
        return cached

    def _structured(self) -> dict[str, Any]:
        cached = self.cache.get("structured")
        if isinstance(cached, dict):
            return cached
        state = self._capability_state("structured_output")
        if state in {"unavailable", "not_exposed"}:
            cached = {
                "gated": self._gated("structured_output", state)
            }
            self.cache["structured"] = cached
            return cached
        body = self._native_body(
            "Return a JSON object whose marker is ok.", stream=False
        )
        body["output_format"] = {
            "type": "json_schema",
            "name": "qualification_marker",
            "schema": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string", "const": "ok"}
                },
                "required": ["marker"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        result = self.client.request(
            "POST",
            "/system/v1/chat",
            credential=self.credential,
            body=body,
        )
        request_id = _response_request_id(result)
        output = result.body.get("output")
        structured = (
            output.get("structured") if isinstance(output, dict) else None
        )
        usage = _usage_projection(result.body.get("usage"))
        if (
            result.status != 200
            or result.body.get("status") != "completed"
            or result.body.get("model") != self.public_model_id
            or structured != {"marker": "ok"}
        ):
            raise _qualification_error(
                "QUALIFICATION_PROFILE_FAILED",
                "strict structured output failed schema validation",
            )
        self._expect(
            request_id,
            endpoint="/system/v1/chat",
            operation="chat",
            streamed=False,
            http_status=result.status,
            operation_state="completed",
        )
        cached = {
            "request_id": request_id,
            "http_status": result.status,
            "terminal": (
                f"{result.body['status']}:"
                f"{result.body.get('finish_reason')}"
            ),
            "usage": usage,
            "structured_identity": _identity(structured),
            "schema_valid": True,
        }
        self.cache["structured"] = cached
        return cached

    def _agent(self) -> dict[str, Any]:
        cached = self.cache.get("agent")
        if isinstance(cached, dict):
            return cached
        state = self._capability_state("tool_calling")
        if state in {"unavailable", "not_exposed"}:
            cached = {"gated": self._gated("tool_calling", state)}
            self.cache["agent"] = cached
            return cached
        tool = {
            "type": "function",
            "name": "qualification_marker",
            "description": "Return one harmless logical marker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string"}
                },
                "required": ["marker"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        body = self._native_body(
            "Call the registered qualification marker tool.",
            stream=False,
        )
        body["tools"] = [tool]
        body["tool_choice"] = {
            "type": "function",
            "name": "qualification_marker",
        }
        first = self.client.request(
            "POST",
            "/system/v1/chat",
            credential=self.credential,
            body=body,
        )
        first_request_id = _response_request_id(first)
        output = first.body.get("output")
        calls = (
            output.get("tool_calls") if isinstance(output, dict) else None
        )
        if (
            first.status != 200
            or first.body.get("status") != "requires_action"
            or first.body.get("finish_reason") != "tool_call"
            or first.body.get("model") != self.public_model_id
            or not isinstance(calls, list)
            or len(calls) != 1
            or not isinstance(calls[0], dict)
            or calls[0].get("type") != "function"
            or calls[0].get("name") != "qualification_marker"
            or not isinstance(calls[0].get("arguments"), dict)
            or not isinstance(calls[0].get("id"), str)
            or re.fullmatch(
                r"sx_call_[0-9a-f]{32}", calls[0]["id"]
            )
            is None
        ):
            raise _qualification_error(
                "QUALIFICATION_PROFILE_FAILED",
                "forced logical tool call is invalid",
            )
        call = calls[0]
        self._expect(
            first_request_id,
            endpoint="/system/v1/chat",
            operation="chat",
            streamed=False,
            http_status=first.status,
            operation_state="requires_action",
        )
        continuation = self._native_body(
            "Continue after the external logical result.", stream=False
        )
        continuation["messages"] = [
            {
                "role": "user",
                "content": "Call the registered qualification marker tool.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call],
            },
            {
                "role": "tool",
                "content": {"marker": "ok"},
                "tool_call_id": call["id"],
                "name": "qualification_marker",
                "is_error": False,
            },
        ]
        second = self.client.request(
            "POST",
            "/system/v1/chat",
            credential=self.credential,
            body=continuation,
        )
        second_request_id = _response_request_id(second)
        second_output = second.body.get("output")
        final_content = (
            second_output.get("content")
            if isinstance(second_output, dict)
            else None
        )
        if (
            second.status != 200
            or second.body.get("status") != "completed"
            or second.body.get("model") != self.public_model_id
            or not isinstance(final_content, str)
            or not final_content
        ):
            raise _qualification_error(
                "QUALIFICATION_PROFILE_FAILED",
                "typed external result continuation failed",
            )
        self._expect(
            second_request_id,
            endpoint="/system/v1/chat",
            operation="chat",
            streamed=False,
            http_status=second.status,
            operation_state="completed",
        )
        final_encoded = final_content.encode("utf-8")
        cached = {
            "first_request_id": first_request_id,
            "second_request_id": second_request_id,
            "http_status": second.status,
            "tool_call_id": call["id"],
            "tool_name": call["name"],
            "arguments_identity": _identity(call["arguments"]),
            "arguments_json_valid": True,
            "api_executed_tool": False,
            "continuation_accepted": True,
            "final_content_bytes": len(final_encoded),
            "final_content_sha256": (
                "sha256:" + hashlib.sha256(final_encoded).hexdigest()
            ),
            "terminal": (
                f"{second.body['status']}:"
                f"{second.body.get('finish_reason')}"
            ),
            "usage": _usage_projection(second.body.get("usage")),
        }
        self.cache["agent"] = cached
        return cached

    @staticmethod
    def _operation_record_from_line(
        line: bytes,
    ) -> dict[str, Any] | None:
        marker = b"system_x_operation "
        position = line.find(marker)
        if position < 0:
            return None
        try:
            value = json.loads(line[position + len(marker) :].strip())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _correlate_operations(self) -> dict[str, Any]:
        if not self.expectations:
            raise _qualification_error(
                "QUALIFICATION_REQUEST_RECORD_NOT_FOUND",
                "profile produced no request identities",
            )
        deadline = time.monotonic() + OPERATION_RECORD_WAIT_SECONDS
        matches: dict[str, dict[str, Any]] = {}
        while True:
            try:
                details = self.service.operation_log.lstat()
                if (
                    stat.S_ISLNK(details.st_mode)
                    or not stat.S_ISREG(details.st_mode)
                    or details.st_nlink != 1
                    or details.st_size < self.log_start_offset
                ):
                    raise OSError("unsafe operation log")
                lower = max(
                    self.log_start_offset,
                    details.st_size - MAX_OPERATION_LOG_SCAN_BYTES,
                )
                with self.service.operation_log.open("rb") as handle:
                    handle.seek(lower)
                    raw = handle.read(MAX_OPERATION_LOG_SCAN_BYTES + 1)
            except OSError as error:
                raise _qualification_error(
                    "QUALIFICATION_REQUEST_RECORD_NOT_FOUND",
                    "manager-owned operation log became unavailable",
                ) from error
            if len(raw) > MAX_OPERATION_LOG_SCAN_BYTES:
                raw = raw[-MAX_OPERATION_LOG_SCAN_BYTES:]
            matches = {}
            duplicates: set[str] = set()
            for line in raw.splitlines():
                record = self._operation_record_from_line(line)
                if not isinstance(record, dict):
                    continue
                request_id = record.get("request_id")
                if request_id not in self.expectations:
                    continue
                if request_id in matches:
                    duplicates.add(request_id)
                matches[request_id] = record
            if duplicates:
                raise _qualification_error(
                    "QUALIFICATION_REQUEST_RECORD_MISMATCH",
                    "multiple operation records exist for one request",
                )
            if set(matches) == set(self.expectations):
                break
            if time.monotonic() >= deadline:
                raise _qualification_error(
                    "QUALIFICATION_REQUEST_RECORD_NOT_FOUND",
                    "one or more operation records were not observed",
                )
            time.sleep(0.1)
        identities: list[str] = []
        for request_id, expected in sorted(self.expectations.items()):
            record = matches[request_id]
            if (
                set(record) != OPERATION_RECORD_FIELDS
                or record.get("schema") != "system-x.operation-record.v1"
                or record.get("request_id") != request_id
                or record.get("key_id") != self.credential.key_id
                or record.get("protocol_family")
                != expected.protocol_family
                or record.get("endpoint") != expected.endpoint
                or record.get("operation") != expected.operation
                or record.get("streamed") is not expected.streamed
                or record.get("http_status") != expected.http_status
                or (
                    expected.operation_state is not None
                    and record.get("operation_state")
                    != expected.operation_state
                )
                or (
                    expected.public_model_id is not None
                    and record.get("public_model_id")
                    != expected.public_model_id
                )
                or (
                    expected.artifact_version_id is not None
                    and record.get("artifact_version_id")
                    != expected.artifact_version_id
                )
            ):
                raise _qualification_error(
                    "QUALIFICATION_REQUEST_RECORD_MISMATCH",
                    "operation record does not match profile evidence",
                )
            identities.append(_identity(record))
        return {
            "request_count": len(matches),
            "request_ids": sorted(matches),
            "operation_record_identities": identities,
        }

    def _compatibility(
        self, check_name: str
    ) -> dict[str, Any]:
        if self.compatibility_probe is not None:
            value = self.compatibility_probe(
                check_name,
                self.public_model_id,
                self.artifact_version_id,
                self.capability_manifest_identity,
            )
            if not isinstance(value, dict):
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "compatibility probe returned an invalid result",
                )
            return value
        cached = self.cache.get("compatibility:" + check_name)
        if isinstance(cached, dict):
            return cached

        def passed(
            observed: dict[str, Any],
            *,
            terminal: str,
            usage: dict[str, int | None] | None = None,
        ) -> dict[str, Any]:
            result = _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal=terminal,
                usage=usage,
                evidence=observed,
            )
            self.cache["compatibility:" + check_name] = result
            return result

        if check_name in {"openai_model_listing", "messages_model_listing"}:
            protocol = (
                "openai" if check_name.startswith("openai") else "messages"
            )
            response = self._compatibility_exchange(
                protocol, "GET", "/v1/models"
            )
            body = response["body"]
            data = body.get("data") if isinstance(body, dict) else None
            identifiers = (
                [item.get("id") for item in data if isinstance(item, dict)]
                if isinstance(data, list)
                else []
            )
            if (
                response["status"] != 200
                or identifiers.count("default") != 1
                or identifiers.count(self.public_model_id) != 1
                or (
                    protocol == "openai"
                    and body.get("object") != "list"
                )
                or (
                    protocol == "messages"
                    and body.get("has_more") is not False
                )
            ):
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "compatibility model listing is invalid",
                )
            self._expect(
                response["request_id"],
                endpoint="/v1/models",
                operation="model.list",
                streamed=False,
                http_status=200,
                operation_state="completed",
                protocol_family=(
                    "openai_compatible"
                    if protocol == "openai"
                    else "messages_compatible"
                ),
                resolved_model=False,
            )
            return passed(
                {
                    "request_id": response["request_id"],
                    "http_status": 200,
                    "default_present": True,
                    "resolved_model_present": True,
                    "model_count": len(identifiers),
                    "compatibility_identity": response[
                        "compatibility_identity"
                    ],
                },
                terminal="completed",
            )

        if check_name in {
            "openai_nonstream_request",
            "messages_nonstream_request",
        }:
            protocol = (
                "openai" if check_name.startswith("openai") else "messages"
            )
            path = (
                "/v1/chat/completions"
                if protocol == "openai"
                else "/v1/messages"
            )
            request_body = {
                "model": self.public_model_id,
                "messages": [
                    {"role": "user", "content": "Return exactly OK."}
                ],
                "stream": False,
                "temperature": 0.0,
            }
            request_body[
                "max_completion_tokens"
                if protocol == "openai"
                else "max_tokens"
            ] = PROFILE_MAX_OUTPUT_TOKENS
            response = self._compatibility_exchange(
                protocol, "POST", path, body=request_body
            )
            body = response["body"]
            if protocol == "openai":
                choices = body.get("choices") if isinstance(body, dict) else None
                choice = (
                    choices[0]
                    if isinstance(choices, list) and len(choices) == 1
                    else None
                )
                message = choice.get("message") if isinstance(choice, dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                raw_usage = body.get("usage") if isinstance(body, dict) else None
                usage = {
                    "input_tokens": raw_usage.get("prompt_tokens"),
                    "output_tokens": raw_usage.get("completion_tokens"),
                    "total_tokens": raw_usage.get("total_tokens"),
                } if isinstance(raw_usage, dict) else None
                finish = choice.get("finish_reason") if isinstance(choice, dict) else None
                valid = (
                    body.get("object") == "chat.completion"
                    and body.get("model") == self.public_model_id
                    and finish in {"stop", "length"}
                )
            else:
                blocks = body.get("content") if isinstance(body, dict) else None
                content = (
                    "".join(
                        item.get("text", "")
                        for item in blocks
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                    if isinstance(blocks, list)
                    else None
                )
                raw_usage = body.get("usage") if isinstance(body, dict) else None
                usage = {
                    "input_tokens": raw_usage.get("input_tokens"),
                    "output_tokens": raw_usage.get("output_tokens"),
                    "total_tokens": (
                        raw_usage.get("input_tokens")
                        + raw_usage.get("output_tokens")
                    ),
                } if (
                    isinstance(raw_usage, dict)
                    and type(raw_usage.get("input_tokens")) is int
                    and type(raw_usage.get("output_tokens")) is int
                ) else None
                finish = body.get("stop_reason") if isinstance(body, dict) else None
                valid = (
                    body.get("type") == "message"
                    and body.get("model") == self.public_model_id
                    and finish
                    in {
                        "end_turn",
                        "max_tokens",
                        "model_context_window_exceeded",
                    }
                )
            validated_usage = _validate_usage_projection(usage)
            if (
                response["status"] != 200
                or not valid
                or not isinstance(content, str)
                or not content
                or validated_usage is None
            ):
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "compatibility non-stream response is invalid",
                )
            encoded = content.encode("utf-8")
            family = (
                "openai_compatible"
                if protocol == "openai"
                else "messages_compatible"
            )
            self._expect(
                response["request_id"], endpoint=path, operation="chat",
                streamed=False, http_status=200,
                operation_state="completed", protocol_family=family,
            )
            return passed(
                {
                    "request_id": response["request_id"],
                    "http_status": 200,
                    "model": self.public_model_id,
                    "finish_reason": finish,
                    "usage": validated_usage,
                    "content_present": True,
                    "content_bytes": len(encoded),
                    "content_sha256": (
                        "sha256:" + hashlib.sha256(encoded).hexdigest()
                    ),
                    "compatibility_identity": response[
                        "compatibility_identity"
                    ],
                },
                terminal="completed:" + str(finish),
                usage=validated_usage,
            )

        if check_name in {
            "openai_stream_sequence",
            "messages_stream_sequence",
        }:
            protocol = (
                "openai" if check_name.startswith("openai") else "messages"
            )
            path = (
                "/v1/chat/completions"
                if protocol == "openai"
                else "/v1/messages"
            )
            request_body = {
                "model": self.public_model_id,
                "messages": [
                    {"role": "user", "content": "Return exactly OK."}
                ],
                "stream": True,
                "temperature": 0.0,
            }
            request_body[
                "max_completion_tokens"
                if protocol == "openai"
                else "max_tokens"
            ] = PROFILE_MAX_OUTPUT_TOKENS
            if protocol == "openai":
                request_body["stream_options"] = {"include_usage": True}
            response = self._compatibility_exchange(
                protocol, "POST", path, body=request_body, stream=True
            )
            try:
                blocks = [
                    item
                    for item in response["raw"].decode("utf-8")
                    .replace("\r\n", "\n").split("\n\n")
                    if item
                ]
            except UnicodeDecodeError as error:
                raise _qualification_error(
                    "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                    "compatibility stream is not UTF-8",
                ) from error
            digest = hashlib.sha256()
            content_bytes = 0
            finish: str | None = None
            usage: dict[str, int | None] | None = None
            event_types: list[str] = []
            if protocol == "openai":
                done = 0
                for index, block in enumerate(blocks):
                    if not block.startswith("data: ") or "\n" in block:
                        raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "OpenAI stream framing is invalid")
                    payload = block[6:]
                    if payload == "[DONE]":
                        done += 1
                        if index != len(blocks) - 1:
                            raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "OpenAI DONE is not terminal")
                        continue
                    try:
                        frame = json.loads(payload)
                    except json.JSONDecodeError as error:
                        raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "OpenAI stream data is invalid") from error
                    if not isinstance(frame, dict) or frame.get("object") != "chat.completion.chunk" or frame.get("model") != self.public_model_id:
                        raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "OpenAI stream identity is invalid")
                    choices = frame.get("choices")
                    if isinstance(choices, list) and len(choices) == 1:
                        choice = choices[0]
                        delta = choice.get("delta") if isinstance(choice, dict) else None
                        text = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(text, str) and text:
                            encoded = text.encode("utf-8"); digest.update(encoded); content_bytes += len(encoded)
                        terminal = choice.get("finish_reason") if isinstance(choice, dict) else None
                        if terminal is not None:
                            finish = terminal
                    elif choices == [] and isinstance(frame.get("usage"), dict):
                        raw_usage = frame["usage"]
                        usage = {"input_tokens": raw_usage.get("prompt_tokens"), "output_tokens": raw_usage.get("completion_tokens"), "total_tokens": raw_usage.get("total_tokens")}
                    else:
                        raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "OpenAI stream chunk is invalid")
                    event_types.append("chunk")
                complete = done == 1 and finish in {"stop", "length"}
            else:
                input_tokens = None; output_tokens = None
                for block in blocks:
                    lines = block.splitlines()
                    if len(lines) != 2 or not lines[0].startswith("event: ") or not lines[1].startswith("data: "):
                        raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "Messages stream framing is invalid")
                    event_type = lines[0][7:]
                    try:
                        frame = json.loads(lines[1][6:])
                    except json.JSONDecodeError as error:
                        raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "Messages stream data is invalid") from error
                    if not isinstance(frame, dict) or frame.get("type") != event_type or event_type == "error":
                        raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "Messages stream event is invalid")
                    if event_type == "message_start":
                        message = frame.get("message"); raw_usage = message.get("usage") if isinstance(message, dict) else None
                        if not isinstance(message, dict) or message.get("model") != self.public_model_id or not isinstance(raw_usage, dict):
                            raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "Messages start event is invalid")
                        input_tokens = raw_usage.get("input_tokens")
                    if event_type == "content_block_delta":
                        delta = frame.get("delta"); text = delta.get("text") if isinstance(delta, dict) and delta.get("type") == "text_delta" else None
                        if isinstance(text, str) and text:
                            encoded = text.encode("utf-8"); digest.update(encoded); content_bytes += len(encoded)
                    if event_type == "message_delta":
                        delta = frame.get("delta"); raw_usage = frame.get("usage")
                        finish = delta.get("stop_reason") if isinstance(delta, dict) else None
                        output_tokens = raw_usage.get("output_tokens") if isinstance(raw_usage, dict) else None
                    event_types.append(event_type)
                usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens} if type(input_tokens) is int and type(output_tokens) is int else None
                complete = bool(event_types and event_types[0] == "message_start" and event_types[-1] == "message_stop" and event_types.count("message_start") == 1 and event_types.count("message_stop") == 1 and finish in {"end_turn", "max_tokens", "model_context_window_exceeded"})
            validated_usage = _validate_usage_projection(usage)
            if response["status"] != 200 or not complete or content_bytes < 1 or validated_usage is None:
                raise _qualification_error("QUALIFICATION_PUBLIC_REQUEST_FAILED", "compatibility stream did not complete")
            family = "openai_compatible" if protocol == "openai" else "messages_compatible"
            self._expect(response["request_id"], endpoint=path, operation="chat", streamed=True, http_status=200, operation_state="completed", protocol_family=family)
            return passed({"request_id": response["request_id"], "http_status": 200, "event_count": len(event_types), "event_types": event_types, "terminal_state": "completed:" + str(finish), "usage": validated_usage, "content_present": True, "content_bytes": content_bytes, "content_sha256": "sha256:" + digest.hexdigest(), "streaming_identity": response["streaming_identity"]}, terminal="completed:" + str(finish), usage=validated_usage)

        if check_name == "heychat_adapter_compatibility":
            models = self._compatibility("openai_model_listing")
            chat = self._compatibility("openai_nonstream_request")
            observed = {"request_id": chat["request_id"], "http_status": chat["http_status"], "adapter_protocol": "openai_compatible", "base_path": "/v1", "authentication": "bearer", "model_reference": self.public_model_id, "model_listing_request_id": models["request_id"], "inference_request_id": chat["request_id"], "default_present": True, "resolved_model_present": True}
            return passed(observed, terminal=str(chat["finish_or_terminal_state"]), usage=chat["usage"])
        raise _qualification_error(
            "QUALIFICATION_INTERNAL_ERROR",
            "unknown compatibility profile check",
        )
        value = None
        if not isinstance(value, dict):
            raise _qualification_error(
                "QUALIFICATION_PUBLIC_REQUEST_FAILED",
                "compatibility probe returned an invalid result",
            )
        return value

    def probe(
        self,
        check_name: str,
        *,
        model_id: str,
        artifact_version_id: str,
        capability_manifest_identity: str,
    ) -> dict[str, Any]:
        self._require_target(
            model_id,
            artifact_version_id,
            capability_manifest_identity,
        )
        if check_name == "candidate_registry_ready":
            ready = "READY" in self.registry_states
            return _check_result(
                status="PASSED" if ready else "FAILED",
                reason_code=(
                    "CHECK_PASSED" if ready else "QUALIFICATION_REGISTRY_UNAVAILABLE"
                ),
                terminal="READY" if ready else None,
                evidence={
                    "registry_states_observed": list(
                        self.registry_states
                    ),
                    "ready": ready,
                },
            )
        if check_name == "candidate_public_model_identity":
            observed = self._catalogue()
            return _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal="ready",
                evidence={
                    "public_model_id": self.public_model_id,
                    "registry_generation": observed[
                        "registry_generation"
                    ],
                },
            )
        if check_name == "candidate_model_detail":
            observed = self._detail()
            return _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal="ready",
                evidence={
                    "public_model_id": self.public_model_id,
                    "artifact_version_id": self.artifact_version_id,
                    "registry_generation": observed[
                        "registry_generation"
                    ],
                },
            )
        if check_name == "candidate_capability_manifest":
            return _check_result(
                status="PASSED",
                evidence={
                    "capability_manifest_identity": (
                        self.capability_manifest_identity
                    ),
                },
            )
        if check_name == "native_token_count":
            observed = self._token_count()
            return _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal="completed",
                usage={
                    "input_tokens": observed["input_tokens"],
                    "output_tokens": None,
                    "total_tokens": None,
                },
                evidence=observed,
            )
        if check_name in {
            "native_nonstream_chat",
            "native_final_content",
            "native_response_model",
        }:
            observed = self._json_chat(
                "core_chat",
                (
                    "Return exactly the text OK as the final answer. "
                    "Do not explain or reason."
                ),
            )
            return _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal=(
                    f"{observed['status']}:"
                    f"{observed['finish_reason']}"
                ),
                usage=observed["usage"],
                evidence=observed,
            )
        if check_name == "native_stream_sequence":
            observed = self._native_stream()
            return _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal=observed["terminal_state"],
                usage=observed["usage"],
                evidence=observed,
            )
        if check_name == "request_cancellation":
            observed = self._cancel_request()
            return _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal="cancelled",
                evidence=observed,
            )
        if check_name == "later_request":
            observed = self._json_chat(
                "later_chat",
                (
                    "Return exactly the text OK as the final answer. "
                    "Do not explain or reason."
                ),
            )
            return _check_result(
                status="PASSED",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal=(
                    f"{observed['status']}:"
                    f"{observed['finish_reason']}"
                ),
                usage=observed["usage"],
                evidence=observed,
            )
        if check_name == "request_record_correlation":
            observed = self._correlate_operations()
            return _check_result(
                status="PASSED",
                terminal="correlated",
                evidence=observed,
            )
        if check_name in {
            "reasoning_output",
            "reasoning_final_separation",
        }:
            observed = self._reasoning()
            if "gated" in observed:
                return observed["gated"]
            return _check_result(
                status="PASSED_AVAILABLE",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal=observed["terminal"],
                usage=observed["usage"],
                capability="reasoning_output",
                available=True,
                accurately_gated=None,
                evidence=observed,
            )
        if check_name in {
            "structured_json_output",
            "structured_schema_validation",
        }:
            observed = self._structured()
            if "gated" in observed:
                return observed["gated"]
            return _check_result(
                status="PASSED_AVAILABLE",
                request_id=observed["request_id"],
                http_status=observed["http_status"],
                terminal=observed["terminal"],
                usage=observed["usage"],
                capability="structured_output",
                available=True,
                accurately_gated=None,
                evidence=observed,
            )
        if check_name in {
            "strict_tool_schema",
            "forced_registered_tool_call",
            "tool_call_id_preserved",
            "tool_arguments_json",
            "api_did_not_execute_tool",
            "typed_external_result_continuation",
            "agent_final_content",
        }:
            observed = self._agent()
            if "gated" in observed:
                return observed["gated"]
            return _check_result(
                status="PASSED_AVAILABLE",
                request_id=(
                    observed["second_request_id"]
                    if check_name
                    in {
                        "typed_external_result_continuation",
                        "agent_final_content",
                    }
                    else observed["first_request_id"]
                ),
                http_status=observed["http_status"],
                terminal=observed["terminal"],
                usage=observed["usage"],
                capability="tool_calling",
                available=True,
                accurately_gated=None,
                evidence=observed,
            )
        if check_name in {
            "capability_gating_reasoning",
            "capability_gating_tool_calling",
            "capability_gating_structured_output",
        }:
            capability = {
                "capability_gating_reasoning": "reasoning_output",
                "capability_gating_tool_calling": "tool_calling",
                "capability_gating_structured_output": (
                    "structured_output"
                ),
            }[check_name]
            state = self._capability_state(capability)
            if state == "available":
                return _check_result(
                    status="PASSED_AVAILABLE",
                    capability=capability,
                    available=True,
                    accurately_gated=None,
                    evidence={
                        "capability": capability,
                        "reported_state": state,
                    },
                )
            return self._gated(capability, state)
        if check_name in {
            "openai_model_listing",
            "openai_nonstream_request",
            "openai_stream_sequence",
            "messages_model_listing",
            "messages_nonstream_request",
            "messages_stream_sequence",
            "heychat_adapter_compatibility",
        }:
            return self._compatibility(check_name)
        raise _qualification_error(
            "QUALIFICATION_INTERNAL_ERROR",
            "unknown profile check",
        )


def qualification_result_path(
    paths: InspectorPaths, qualification_id: str
) -> Path:
    if QUALIFICATION_ID_PATTERN.fullmatch(qualification_id) is None:
        raise InspectorError(
            "QUALIFICATION_RESULT_INVALID",
            "qualification ID is not canonical",
        )
    return paths.qualification_results / f"{qualification_id}.json"


def _qualification_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return f"qualification-{stamp}-{secrets.token_hex(8)}"


def _nullable_sha256(value: object) -> str | None:
    return (
        value
        if isinstance(value, str)
        and SHA256_PATTERN.fullmatch(value) is not None
        else None
    )


def _inspection_projection(
    authorization: QualificationAuthorization,
) -> dict[str, Any]:
    authenticated = authorization.decision_authorization
    inspection = authenticated.inspection
    normalized = inspection["normalized"]
    quantization = normalized.get("quantization")
    gguf = inspection.get("format", {}).get("gguf")
    if not isinstance(quantization, dict) or not isinstance(gguf, dict):
        raise _qualification_error(
            "QUALIFICATION_INSPECTION_INVALID",
            "GGUF inspection omits normalized qualification evidence",
        )
    histogram = quantization.get("tensor_type_histogram")
    if not isinstance(histogram, dict):
        raise _qualification_error(
            "QUALIFICATION_INSPECTION_INVALID",
            "GGUF inspection omits tensor type evidence",
        )
    architectures = normalized.get("architectures")
    architecture = (
        architectures[0]
        if isinstance(architectures, list)
        and architectures
        and isinstance(architectures[0], str)
        else None
    )
    return {
        "inspection_id": inspection["inspection_id"],
        "inspection_result_identity": (
            authenticated.inspection_result_identity
        ),
        "artifact_identity": inspection["artifact"]["identity"],
        "artifact_size": inspection["artifact"]["byte_count"],
        "physical_format": "GGUF",
        "model_type": "model",
        "architecture": architecture,
        "quantization_summary": {
            "kind": quantization.get("kind"),
            "general_file_type": quantization.get(
                "general_file_type"
            ),
            "quantization_version": quantization.get(
                "quantization_version"
            ),
            "mixed_tensor_types": quantization.get(
                "mixed_tensor_types"
            ),
            "tensor_types": sorted(
                key for key in histogram if isinstance(key, str)
            ),
        },
        "tokenizer_identity": _nullable_sha256(
            gguf.get("tokenizer_token_identity")
        ),
        "chat_template_identity": _nullable_sha256(
            gguf.get("chat_template_identity")
        ),
    }


def qualification_validity_predicate(
    authorization: QualificationAuthorization,
) -> dict[str, Any]:
    tuple_evidence = authorization.installed_tuple_evidence
    capability = authorization.decision_authorization.capability_record
    binding = authorization.decision_authorization.binding
    basis = {
        "artifact_identity": authorization.source.artifact_identity,
        "capability_record_identity": capability[
            "capability_record_identity"
        ],
        "binding_identity": binding["binding_identity"],
        "installed_tuple_verification_identity": tuple_evidence[
            "installed_tuple_verification_identity"
        ],
        "llama_cpp_commit": tuple_evidence["llama_cpp_commit"],
        "llama_server_sha256": tuple_evidence[
            "llama_server_sha256"
        ],
        "connected_source_identity": tuple_evidence[
            "connected_source_identity"
        ],
        "system_x_source_commit": tuple_evidence["system_x_source_commit"],
        "system_x_source_tree": tuple_evidence["system_x_source_tree"],
        "inspector_source_identity": tuple_evidence["inspector_source_identity"],
    }
    return {**basis, "predicate_identity": _identity(basis)}


def build_qualification_record(
    *,
    qualification_id: str,
    transaction_id: str,
    created_utc: str,
    completed_utc: str,
    authorization: QualificationAuthorization,
    incumbent: IncumbentSnapshot,
    candidate_runtime: dict[str, Any],
    profile_run: ProfileRun,
    result_class: str,
    reason_codes: Iterable[str],
    warm_after: dict[str, Any] | None,
    restoration: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    reasons = list(dict.fromkeys(reason_codes))
    if (
        QUALIFICATION_ID_PATTERN.fullmatch(qualification_id) is None
        or result_class not in QUALIFICATION_RESULT_CLASSES
        or not reasons
        or any(
            reason not in QUALIFICATION_REASON_CODES
            for reason in reasons
        )
    ):
        raise _qualification_error(
            "QUALIFICATION_RESULT_INVALID",
            "qualification terminal identity is invalid",
        )
    decision = authorization.decision_authorization.decision
    installed = {
        key: authorization.installed_tuple_evidence[key]
        for key in (
            "branch_capability_record_identity",
            "capability_binding_identity",
            "installed_tuple_verification_identity",
            "llama_cpp_commit",
            "llama_server_sha256",
            "branch_controller_identity",
            "api_service_controller_identity",
            "api_source_manifest_identity",
            "supervisor_identity",
            "system_x_source_commit",
            "system_x_source_tree",
            "inspector_source_identity",
        )
    }
    incumbent_projection = incumbent.result_projection()
    incumbent_projection["warm_after"] = warm_after
    basis = {
        "schema_version": SCHEMA_IDENTITIES[
            "gguf_qualification_result"
        ],
        "qualification_id": qualification_id,
        "transaction_id": transaction_id,
        "created_utc": created_utc,
        "completed_utc": completed_utc,
        "inspection": _inspection_projection(authorization),
        "input_decision": {
            "decision_id": decision["decision_id"],
            "decision_result_identity": decision["result_identity"],
            "capability_result": decision["capability"][
                "capability_result"
            ],
            "reason_code": decision["reason_code"],
        },
        "requested_profile": authorization.requested_profile,
        "installed_tuple": installed,
        "incumbent": incumbent_projection,
        "candidate_runtime": candidate_runtime,
        "checks": [dict(item) for item in profile_run.checks],
        "supported_profiles": list(profile_run.supported_profiles),
        "observed_capabilities": dict(
            profile_run.observed_capabilities
        ),
        "result_class": result_class,
        "reason_codes": reasons,
        "restoration": restoration,
        "cleanup": cleanup,
        "validity_predicate": qualification_validity_predicate(
            authorization
        ),
    }
    record = {
        **basis,
        "result_identity": _identity(basis),
    }
    return validate_qualification_record(record)


def publish_qualification_record(
    paths: InspectorPaths, record: dict[str, Any]
) -> Path:
    validated = validate_qualification_record(record)
    path = qualification_result_path(
        paths, validated["qualification_id"]
    )
    try:
        atomic_create_json(path, validated, mode=0o600)
    except InspectorError as error:
        if error.reason_code == "INSPECTION_RECORD_COLLISION":
            raise _qualification_error(
                "QUALIFICATION_RESULT_COLLISION",
                "qualification result target already exists",
            ) from error
        raise
    details = path.lstat()
    observed = validate_qualification_record(read_json_record(path))
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or observed != validated
    ):
        raise _qualification_error(
            "QUALIFICATION_RESULT_INVALID",
            "immutable qualification result did not round-trip",
        )
    return path


def find_idempotent_qualification(
    paths: InspectorPaths,
    authorization: QualificationAuthorization,
) -> tuple[dict[str, Any], Path] | None:
    expected_predicate = qualification_validity_predicate(
        authorization
    )
    matches: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(paths.qualification_results.iterdir()):
        if (
            path.name.startswith(".")
            or re.fullmatch(
                r"qualification-[0-9]{8}T[0-9]{12}Z-"
                r"[0-9a-f]{16}\.json",
                path.name,
            )
            is None
        ):
            raise _qualification_error(
                "QUALIFICATION_RESULT_INVALID",
                "qualification result store contains an unknown entry",
            )
        details = path.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
        ):
            raise _qualification_error(
                "QUALIFICATION_RESULT_INVALID",
                "qualification result has an unsafe physical type",
            )
        record = validate_qualification_record(read_json_record(path))
        if (
            record["inspection"]["inspection_id"]
            == authorization.decision_authorization.inspection[
                "inspection_id"
            ]
            and record["inspection"]["artifact_identity"]
            == authorization.source.artifact_identity
            and record["requested_profile"]
            == authorization.requested_profile
            and record["input_decision"]["decision_id"]
            == authorization.decision_authorization.decision[
                "decision_id"
            ]
            and record["input_decision"][
                "decision_result_identity"
            ]
            == authorization.decision_authorization.decision[
                "result_identity"
            ]
            and record["input_decision"]["capability_result"]
            == authorization.decision_authorization.decision[
                "capability"
            ]["capability_result"]
            and record["validity_predicate"] == expected_predicate
        ):
            matches.append((record, path))
    if len(matches) > 1:
        raise _qualification_error(
            "QUALIFICATION_RESULT_COLLISION",
            "multiple immutable results match one qualification input",
        )
    return matches[0] if matches else None


def _qualification_error(
    reason: str, message: str, *, data: dict[str, Any] | None = None
) -> InspectorError:
    return InspectorError(reason, message, data=data)


def _safe_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _qualification_error(reason, f"required record is absent: {path.name}") from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_size > MAX_CONTROL_JSON_BYTES
    ):
        raise _qualification_error(reason, f"required record is unsafe: {path.name}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _qualification_error(reason, f"required record is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise _qualification_error(reason, f"required record is not an object: {path.name}")
    return value


def _decision_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"decision-{stamp}-{secrets.token_hex(8)}"


def _active_transaction_gate(
    paths: InspectorPaths, expected_transaction_id: str
) -> None:
    observed = inspect_active_lock(paths.locks / "active.json")
    if observed.get("state") == "absent":
        return
    record = observed.get("record")
    if (
        observed.get("state") == "active"
        and isinstance(record, dict)
        and record.get("transaction_id") == expected_transaction_id
        and record.get("operation") == "qualify-gguf"
    ):
        return
    raise _qualification_error(
        "QUALIFICATION_ACTIVE_TRANSACTION",
        "another or uncertain Inspector transaction blocks qualification",
    )


def _decision_records(
    paths: InspectorPaths, inspection_id: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(paths.decision_results.iterdir()):
        if DECISION_FILE_PATTERN.fullmatch(path.name) is None:
            raise _qualification_error(
                "QUALIFICATION_DECISION_INVALID",
                "decision result store contains an unknown entry",
            )
        details = path.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
        ):
            raise _qualification_error(
                "QUALIFICATION_DECISION_INVALID",
                "decision result has an unsafe physical type",
            )
        try:
            record = validate_decision_record(read_json_record(path))
        except InspectorError as error:
            raise _qualification_error(
                "QUALIFICATION_DECISION_INVALID",
                "decision result authentication failed",
            ) from error
        if record["inspection"]["inspection_id"] == inspection_id:
            records.append(record)
    return records


def _current_decision(
    paths: InspectorPaths,
    inspection_id: str,
    transaction_id: str,
    *,
    decision_id_factory: Callable[[], str] = _decision_id,
    decision_publisher: Callable[
        [InspectorPaths, dict[str, Any]], Path
    ] = publish_decision_record,
) -> dict[str, Any]:
    try:
        outcome = resolve_decision(paths, inspection_id)
    except InspectorError as error:
        raise _qualification_error(
            "QUALIFICATION_DECISION_INVALID",
            "current decision cannot be resolved",
        ) from error
    matching = [
        record
        for record in _decision_records(paths, inspection_id)
        if record["decision_basis_identity"] == outcome["decision_basis_identity"]
        and record["capability"] == outcome["capability"]
        and record["selected_branch"] == outcome["selected_branch"]
        and record["handoff_allowed"] == outcome["handoff_allowed"]
        and record["spawn_allowed"] == outcome["spawn_allowed"]
        and record["reason_code"] == outcome["reason_code"]
    ]
    if matching:
        return sorted(
            matching, key=lambda item: (item["decision_timestamp_utc"], item["decision_id"])
        )[-1]
    record = build_decision_record(
        outcome,
        decision_id=decision_id_factory(),
        transaction_id=transaction_id,
        decision_timestamp_utc=utc_now(),
    )
    path = decision_publisher(paths, record)
    try:
        observed = validate_decision_record(read_json_record(path))
    except InspectorError as error:
        raise _qualification_error(
            "QUALIFICATION_DECISION_INVALID",
            "created decision did not authenticate",
        ) from error
    if observed != record:
        raise _qualification_error(
            "QUALIFICATION_DECISION_INVALID",
            "created decision did not round-trip",
        )
    return record


def _map_handoff_error(error: InspectorError) -> InspectorError:
    mapping = {
        "HANDOFF_SOURCE_NOT_FOUND": "QUALIFICATION_SOURCE_NOT_FOUND",
        "HANDOFF_SOURCE_INVALID": "QUALIFICATION_SOURCE_INVALID",
        "HANDOFF_SOURCE_SYMLINK": "QUALIFICATION_SOURCE_SYMLINK",
        "HANDOFF_SOURCE_HARDLINK_REJECTED": (
            "QUALIFICATION_SOURCE_HARDLINK_REJECTED"
        ),
        "HANDOFF_SOURCE_IDENTITY_MISMATCH": (
            "QUALIFICATION_ARTIFACT_IDENTITY_MISMATCH"
        ),
        "HANDOFF_SOURCE_CHANGED": "QUALIFICATION_SOURCE_CHANGED",
        "HANDOFF_TARGET_COLLISION": "QUALIFICATION_TARGET_COLLISION",
        "HANDOFF_REGISTRY_LOCATION_COLLISION": (
            "QUALIFICATION_TARGET_COLLISION"
        ),
        "HANDOFF_STAGING_COLLISION": "QUALIFICATION_STAGING_COLLISION",
        "HANDOFF_STAGING_INVALID": "QUALIFICATION_STAGING_COLLISION",
        "HANDOFF_INSUFFICIENT_STORAGE": (
            "QUALIFICATION_INSUFFICIENT_STORAGE"
        ),
        "HANDOFF_COPY_FAILED": "QUALIFICATION_COPY_FAILED",
        "HANDOFF_STAGED_IDENTITY_MISMATCH": "QUALIFICATION_COPY_FAILED",
        "HANDOFF_PUBLICATION_CONFLICT": "QUALIFICATION_TARGET_COLLISION",
        "HANDOFF_PUBLICATION_FAILED": "QUALIFICATION_PUBLICATION_FAILED",
    }
    reason = mapping.get(error.reason_code, "QUALIFICATION_SOURCE_INVALID")
    return _qualification_error(reason, "qualification admission validation failed")


def installed_tuple_evidence(
    paths: InspectorPaths,
    capability: dict[str, Any],
    binding: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    installed = capability["installed_tuple"]
    if not isinstance(installed, dict):
        raise _qualification_error(
            "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
            "GGUF capability has no installed tuple",
        )
    components = {
        item["name"]: item for item in installed["components"]
    }
    manifests = {item["name"]: item for item in installed["manifests"]}
    required_components = {
        "llama_server_binary",
        "branch_controller",
        "api_service_controller",
        "service_control_supervisor",
    }
    if not required_components <= set(components) or "api_source_graph" not in manifests:
        raise _qualification_error(
            "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
            "installed tuple omits a qualification-bound component",
        )
    verification_identity = _identity(verification)
    connected_basis = {
        "components": [
            {"name": name, "sha256": item["sha256"]}
            for name, item in sorted(components.items())
        ],
        "manifests": [
            {"name": name, "identity": item["identity"]}
            for name, item in sorted(manifests.items())
        ],
    }
    source_evidence = _system_x_source_evidence(paths)
    return {
        "branch_capability_record_identity": capability[
            "capability_record_identity"
        ],
        "capability_binding_identity": binding["binding_identity"],
        "installed_tuple_verification_identity": verification_identity,
        "llama_cpp_commit": installed["source_commit"],
        "llama_server_sha256": components["llama_server_binary"]["sha256"],
        "branch_controller_identity": components["branch_controller"]["sha256"],
        "api_service_controller_identity": components[
            "api_service_controller"
        ]["sha256"],
        "api_source_manifest_identity": manifests["api_source_graph"][
            "identity"
        ],
        "supervisor_identity": components["service_control_supervisor"][
            "sha256"
        ],
        "connected_source_identity": _identity(connected_basis),
        **source_evidence,
    }


def authenticate_qualification(
    paths: InspectorPaths,
    inspection_id: str,
    candidate_artifact_identity: str,
    required_capability_profile: str,
    *,
    transaction_id: str,
    decision_id_factory: Callable[[], str] = _decision_id,
    decision_publisher: Callable[
        [InspectorPaths, dict[str, Any]], Path
    ] = publish_decision_record,
) -> QualificationAuthorization:
    if (
        not isinstance(inspection_id, str)
        or INSPECTION_ID_PATTERN.fullmatch(inspection_id) is None
        or not isinstance(candidate_artifact_identity, str)
        or SHA256_PATTERN.fullmatch(candidate_artifact_identity) is None
        or required_capability_profile not in QUALIFICATION_PROFILES
        or not isinstance(transaction_id, str)
        or not transaction_id
    ):
        raise _qualification_error(
            "QUALIFICATION_INPUT_INVALID", "qualification input is invalid"
        )
    _active_transaction_gate(paths, transaction_id)
    try:
        inspection, inspection_identity = load_inspection_result(
            paths, inspection_id
        )
    except InspectorError as error:
        reason = (
            "QUALIFICATION_INSPECTION_NOT_FOUND"
            if error.reason_code == "INSPECTION_RECORD_NOT_FOUND"
            else "QUALIFICATION_INSPECTION_IDENTITY_MISMATCH"
            if error.reason_code == "INSPECTION_RESULT_IDENTITY_MISMATCH"
            else "QUALIFICATION_INSPECTION_INVALID"
        )
        raise _qualification_error(
            reason, "qualification inspection authentication failed"
        ) from error
    terminal = inspection["classification"]["terminal_class"]
    if terminal != "GGUF":
        raise _qualification_error(
            "QUALIFICATION_PHYSICAL_FORMAT_REJECTED",
            "qualification accepts only an authenticated GGUF",
        )
    if inspection["normalized"]["model_type"] != "model":
        raise _qualification_error(
            "QUALIFICATION_MODEL_TYPE_REJECTED",
            "qualification accepts only a primary model",
        )
    if (
        inspection["artifact"]["identity"] != candidate_artifact_identity
        or inspection["normalized"]["artifact_identity"]
        != candidate_artifact_identity
    ):
        raise _qualification_error(
            "QUALIFICATION_ARTIFACT_IDENTITY_MISMATCH",
            "candidate identity does not match inspection evidence",
        )
    decision = _current_decision(
        paths,
        inspection_id,
        transaction_id,
        decision_id_factory=decision_id_factory,
        decision_publisher=decision_publisher,
    )
    linked = decision["inspection"]
    if (
        linked["inspection_result_identity"] != inspection_identity
        or linked["artifact_identity"] != candidate_artifact_identity
        or linked["physical_format"] != "GGUF"
        or linked["source_target_name"]
        != inspection["source"]["candidate_name"]
    ):
        raise _qualification_error(
            "QUALIFICATION_DECISION_STALE",
            "current decision does not authenticate the inspection",
        )
    capability_surface = decision["capability"]
    direct_supported = bool(
        capability_surface["capability_result"] == "SUPPORTED"
        and decision["reason_code"]
        == "GGUF_ACCEPTED_CAPABILITY_MATCH"
        and decision["selected_branch"] == "model-api-gguf"
        and decision["handoff_allowed"] is True
        and decision["spawn_allowed"] is True
    )
    if direct_supported:
        try:
            decision_authorization = authenticate_handoff_decision(
                paths, decision["decision_id"]
            )
        except InspectorError as error:
            raise _qualification_error(
                "QUALIFICATION_DECISION_STALE",
                "direct-supported decision authentication failed",
            ) from error
        if (
            decision_authorization.decision != decision
            or decision_authorization.inspection_result_identity
            != inspection_identity
        ):
            raise _qualification_error(
                "QUALIFICATION_DECISION_STALE",
                "direct-supported evidence changed during authentication",
            )
        branch_paths = BranchHandoffPaths.discover(paths)
        try:
            source = revalidate_handoff_source(
                paths,
                branch_paths,
                decision_authorization,
                inspection["source"]["candidate_name"],
            )
        except InspectorError as error:
            raise _map_handoff_error(error) from error
        if source.artifact_identity != candidate_artifact_identity:
            raise _qualification_error(
                "QUALIFICATION_ARTIFACT_IDENTITY_MISMATCH",
                "fresh source identity differs from direct evidence",
            )
        return QualificationAuthorization(
            requested_profile=required_capability_profile,
            decision_authorization=decision_authorization,
            branch_paths=branch_paths,
            source=source,
            installed_tuple_evidence=installed_tuple_evidence(
                paths,
                decision_authorization.capability_record,
                decision_authorization.binding,
                decision_authorization.installed_tuple_verification,
            ),
        )
    if (
        capability_surface["capability_result"] != "RUNTIME_SMOKE_REQUIRED"
        or decision["reason_code"] != "GGUF_RUNTIME_SMOKE_REQUIRED"
        or decision["selected_branch"] is not None
        or decision["handoff_allowed"] is not False
        or decision["spawn_allowed"] is not False
    ):
        raise _qualification_error(
            "QUALIFICATION_DECISION_NOT_RUNTIME_SMOKE_REQUIRED",
            "decision is outside the qualification-only gate",
        )
    try:
        binding = load_binding(paths, "model-api-gguf")
        capability = load_capability_record(
            paths, binding["capability_record_id"]
        )
    except InspectorError as error:
        raise _qualification_error(
            "QUALIFICATION_CAPABILITY_BINDING_INVALID",
            "current GGUF capability graph is invalid",
        ) from error
    if (
        capability_surface["branch_identity"] != "model-api-gguf"
        or capability_surface["binding_identity"] != binding["binding_identity"]
        or capability_surface["capability_record_id"]
        != capability["capability_record_id"]
        or capability_surface["capability_record_identity"]
        != capability["capability_record_identity"]
        or binding["capability_record_identity"]
        != capability["capability_record_identity"]
        or capability["unproven_valid_policy"] != "RUNTIME_SMOKE_REQUIRED"
    ):
        raise _qualification_error(
            "QUALIFICATION_CAPABILITY_BINDING_INVALID",
            "decision does not bind the current GGUF capability",
        )
    try:
        verification = verify_installed_tuple(paths, capability)
    except InspectorError as error:
        raise _qualification_error(
            "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
            "installed tuple cannot be verified",
        ) from error
    if verification.get("verified") is not True:
        raise _qualification_error(
            "QUALIFICATION_INSTALLED_TUPLE_MISMATCH",
            "installed tuple differs from accepted capability",
        )
    branch_paths = BranchHandoffPaths.discover(paths)
    decision_authorization = DecisionAuthorization(
        decision=decision,
        inspection=inspection,
        inspection_result_identity=inspection_identity,
        capability_record=capability,
        binding=binding,
        installed_tuple_verification=verification,
    )
    try:
        source = revalidate_handoff_source(
            paths,
            branch_paths,
            decision_authorization,
            inspection["source"]["candidate_name"],
        )
    except InspectorError as error:
        raise _map_handoff_error(error) from error
    if source.artifact_identity != candidate_artifact_identity:
        raise _qualification_error(
            "QUALIFICATION_ARTIFACT_IDENTITY_MISMATCH",
            "fresh source identity differs from qualification input",
        )
    return QualificationAuthorization(
        requested_profile=required_capability_profile,
        decision_authorization=decision_authorization,
        branch_paths=branch_paths,
        source=source,
        installed_tuple_evidence=installed_tuple_evidence(
            paths,
            capability, binding, verification
        ),
    )


def direct_supported_profile_run(
    authorization: QualificationAuthorization,
) -> ProfileRun:
    """Project exact accepted branch evidence without claiming a runtime run."""

    decision = authorization.decision_authorization.decision
    if decision["capability"]["capability_result"] != "SUPPORTED":
        raise _qualification_error(
            "QUALIFICATION_DECISION_INVALID",
            "direct attestation requires one SUPPORTED decision",
        )
    supported_evidence = (
        authorization.decision_authorization.capability_record[
            "supported_evidence"
        ]
    )
    accepted_value = supported_evidence.get(
        "accepted_runtime_capabilities"
    )
    if not isinstance(accepted_value, list) or any(
        not isinstance(item, str) or not item
        for item in accepted_value
    ):
        raise _qualification_error(
            "QUALIFICATION_CAPABILITY_BINDING_INVALID",
            "direct capability evidence omits accepted runtime names",
        )
    accepted = frozenset(accepted_value)
    supported_profiles = tuple(
        profile
        for profile in QUALIFICATION_PROFILES
        if DIRECT_PROFILE_REQUIREMENTS[profile] <= accepted
    )
    requested = authorization.requested_profile
    required = DIRECT_PROFILE_REQUIREMENTS[requested]
    checks: list[dict[str, Any]] = []
    for index, capability in enumerate(sorted(required), start=1):
        available = capability in accepted
        checks.append(
            _normalize_check(
                (
                    f"accepted_branch_capability_{index:02d}",
                    True,
                    "none",
                ),
                {
                    "status": (
                        "PASSED_AVAILABLE"
                        if available
                        else "UNAVAILABLE"
                    ),
                    "capability_observation": {
                        "capability": capability,
                        "available": available,
                        "accurately_gated": None,
                    },
                    "finish_or_terminal_state": (
                        "ACCEPTED_BRANCH_EVIDENCE"
                    ),
                    "evidence": {
                        "evidence_source": (
                            "ACCEPTED_BRANCH_CAPABILITY_RECORD"
                        ),
                        "capability_record_identity": (
                            authorization.decision_authorization.
                            capability_record[
                                "capability_record_identity"
                            ]
                        ),
                        "binding_identity": (
                            authorization.decision_authorization.binding[
                                "binding_identity"
                            ]
                        ),
                        "artifact_identity": (
                            authorization.source.artifact_identity
                        ),
                        "capability": capability,
                    },
                    "reason_code": (
                        "CHECK_AVAILABLE"
                        if available
                        else "CHECK_UNAVAILABLE"
                    ),
                },
            )
        )
    complete = requested in supported_profiles
    return ProfileRun(
        requested_profile=requested,
        checks=tuple(checks),
        supported_profiles=supported_profiles,
        observed_capabilities={
            "available": sorted(accepted),
            "gated_unavailable": [],
            "unavailable": sorted(required - accepted),
        },
        result_class=(
            "SUPPORTED_FOR_CURRENT_TUPLE"
            if complete
            else "UNSUPPORTED"
        ),
        reason_codes=(
            ("QUALIFICATION_PROFILE_SUPPORTED",)
            if complete
            else ("QUALIFICATION_PROFILE_UNSUPPORTED",)
        ),
    )


def observe_qualification_candidate(
    branch_root: Path,
    managed_name: str,
    artifact_identity: str,
) -> dict[str, Any]:
    if (
        not isinstance(managed_name, str)
        or QUALIFICATION_MANAGED_NAME_PATTERN.fullmatch(managed_name)
        is None
        or SHA256_PATTERN.fullmatch(artifact_identity) is None
    ):
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "candidate registry query input is invalid",
        )
    database = (
        branch_root
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
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro", uri=True, timeout=5.0
        )
    except (OSError, sqlite3.Error) as error:
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "registry is unavailable for candidate observation",
        ) from error
    connection.row_factory = sqlite3.Row
    expected_bundle = (
        "bundle-" + artifact_identity.removeprefix("sha256:")
    )
    state_order = (
        "REGISTERED",
        "PROBING",
        "READY",
        "REJECTED",
        "UNAVAILABLE",
        "REMOVED",
    )
    event_state = {
        "model_registered": "REGISTERED",
        "replacement_candidate_registered": "REGISTERED",
        "capability_probe_started": "PROBING",
        "capability_ready": "READY",
        "artifact_rejected": "REJECTED",
        "capability_probe_unavailable": "UNAVAILABLE",
        "artifact_location_invalid": "UNAVAILABLE",
        "artifact_location_removed": "REMOVED",
    }
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise sqlite3.DatabaseError("query_only unavailable")
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute(
                "SELECT key,value FROM registry_metadata"
            )
        }
        generation = int(metadata["registry_generation"])
        rejection = connection.execute(
            """
            SELECT reason_code,detail_json,first_seen_utc,last_seen_utc
            FROM artifact_rejections WHERE relative_path=?
            """,
            (managed_name,),
        ).fetchone()
        location = connection.execute(
            """
            SELECT current_bundle_id,present,first_seen_utc,last_seen_utc
            FROM artifact_locations WHERE relative_root=?
            """,
            (managed_name,),
        ).fetchone()
        rows = list(
            connection.execute(
                """
                SELECT mv.model_version_id,mv.bundle_id,mv.state,
                       cm.manifest_sha256
                FROM model_version_locations AS mvl
                JOIN model_versions AS mv
                  ON mv.model_version_id=mvl.model_version_id
                LEFT JOIN capability_manifests AS cm
                  ON cm.model_version_id=mv.model_version_id
                WHERE mvl.relative_root=?
                ORDER BY mv.created_utc,mv.model_version_id
                """,
                (managed_name,),
            )
        )
        current_rows = [
            row for row in rows if row["bundle_id"] == expected_bundle
        ]
        if len(current_rows) > 1:
            raise _qualification_error(
                "QUALIFICATION_REGISTRY_UNAVAILABLE",
                "candidate location resolves to multiple model versions",
            )
        current = current_rows[0] if current_rows else None
        subject_ids = {managed_name}
        subject_ids.update(str(row["model_version_id"]) for row in rows)
        events = [
            {
                "generation": int(row["generation"]),
                "event_type": str(row["event_type"]),
                "subject_id": (
                    str(row["subject_id"])
                    if row["subject_id"] is not None
                    else None
                ),
            }
            for row in connection.execute(
                """
                SELECT generation,event_type,subject_id
                FROM registry_events
                ORDER BY generation,event_id
                """
            )
            if row["subject_id"] in subject_ids
        ]
        observed = {
            event_state[item["event_type"]]
            for item in events
            if item["event_type"] in event_state
        }
        if rejection is not None:
            observed.add("REJECTED")
        if current is not None and current["state"] in state_order:
            observed.add(str(current["state"]))
        present = bool(location is not None and location["present"] == 1)
        if (
            location is not None
            and present
            and location["current_bundle_id"] != expected_bundle
        ):
            raise _qualification_error(
                "QUALIFICATION_REGISTRY_UNAVAILABLE",
                "candidate location bundle identity is contradictory",
            )
        aliases: list[str] = []
        if current is not None:
            aliases = [
                str(row["alias"])
                for row in connection.execute(
                    """
                    SELECT alias FROM aliases
                    WHERE model_version_id=? ORDER BY alias
                    """,
                    (current["model_version_id"],),
                )
            ]
        terminal: str | None = None
        public_model_id: str | None = None
        artifact_version_id: str | None = None
        manifest_identity: str | None = None
        if rejection is not None:
            terminal = "REJECTED"
        if current is not None:
            state = str(current["state"])
            if state in {"REJECTED", "UNAVAILABLE", "REMOVED"}:
                terminal = state
                if state in {"UNAVAILABLE", "REMOVED"}:
                    manifest = current["manifest_sha256"]
                    if (
                        not isinstance(manifest, str)
                        or re.fullmatch(r"[0-9a-f]{64}", manifest) is None
                    ):
                        raise _qualification_error(
                            "QUALIFICATION_REGISTRY_UNAVAILABLE",
                            "removed candidate lacks authenticated identities",
                        )
                    public_model_id = str(current["model_version_id"])
                    artifact_version_id = str(current["bundle_id"])
                    manifest_identity = "sha256:" + manifest
            elif state == "READY":
                manifest = current["manifest_sha256"]
                if (
                    not isinstance(manifest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", manifest) is None
                ):
                    raise _qualification_error(
                        "QUALIFICATION_REGISTRY_UNAVAILABLE",
                        "candidate lacks an authenticated capability manifest",
                    )
                terminal = "READY" if present else "REMOVED"
                public_model_id = str(current["model_version_id"])
                artifact_version_id = str(current["bundle_id"])
                manifest_identity = "sha256:" + manifest
        if connection.total_changes != 0:
            raise _qualification_error(
                "QUALIFICATION_REGISTRY_UNAVAILABLE",
                "read-only candidate observation changed registry state",
            )
        return {
            "registry_generation": generation,
            "present": present,
            "terminal": terminal,
            "states_observed": [
                state for state in state_order if state in observed
            ],
            "public_model_id": public_model_id,
            "artifact_version_id": artifact_version_id,
            "capability_manifest_identity": manifest_identity,
            "aliases": aliases,
            "default_bound": "default" in aliases,
            "rejection_reason_code": (
                str(rejection["reason_code"])
                if rejection is not None
                else None
            ),
            "rejection_detail_identity": (
                "sha256:"
                + hashlib.sha256(
                    str(rejection["detail_json"]).encode("utf-8")
                ).hexdigest()
                if rejection is not None
                else None
            ),
            "events": events,
        }
    except (KeyError, ValueError, sqlite3.Error) as error:
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "candidate registry observation failed",
        ) from error
    finally:
        connection.close()


def _accepted_registration_wait_seconds(branch_root: Path) -> float:
    service = _safe_json(
        branch_root / "RUNTIME" / "api" / "status" / "service.json",
        "QUALIFICATION_REGISTRY_UNAVAILABLE",
    )
    service_timeout = service.get("service_start_timeout_seconds")
    model_timeout = service.get(
        "private_backend_model_timeout_seconds"
    )
    if (
        service.get("lifecycle_state") != "STARTED"
        or type(service_timeout) not in {int, float}
        or type(model_timeout) not in {int, float}
        or not 1.0
        <= float(model_timeout)
        <= float(service_timeout)
        <= MAX_MANAGER_WAIT_SECONDS
    ):
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "manager candidate-registration timeout is invalid",
        )
    return max(REGISTRATION_WAIT_SECONDS, float(service_timeout))


def wait_for_qualification_candidate(
    branch_root: Path,
    managed_name: str,
    artifact_identity: str,
    *,
    timeout_seconds: float | None = None,
    observer: Callable[
        [Path, str, str], dict[str, Any]
    ] = observe_qualification_candidate,
) -> dict[str, Any]:
    if timeout_seconds is None:
        timeout_seconds = _accepted_registration_wait_seconds(
            branch_root
        )
    if (
        type(timeout_seconds) not in {int, float}
        or not 0.0
        <= float(timeout_seconds)
        <= MAX_MANAGER_WAIT_SECONDS
    ):
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "candidate-registration wait bound is invalid",
        )
    deadline = time.monotonic() + timeout_seconds
    observed_states: list[str] = []
    latest: dict[str, Any] | None = None
    while True:
        latest = observer(
            branch_root, managed_name, artifact_identity
        )
        for state in latest["states_observed"]:
            if state not in observed_states:
                observed_states.append(state)
        terminal = latest.get("terminal")
        if terminal in {"READY", "REJECTED", "UNAVAILABLE"}:
            return {
                **latest,
                "states_observed": observed_states,
            }
        if time.monotonic() >= deadline:
            return {
                **latest,
                "terminal": "TIMEOUT",
                "states_observed": observed_states,
                "rejection_reason_code": (
                    "QUALIFICATION_REGISTRATION_TIMEOUT"
                ),
            }
        time.sleep(0.1)


def wait_for_candidate_removal(
    branch_root: Path,
    managed_name: str,
    artifact_identity: str,
    *,
    timeout_seconds: float | None = None,
    observer: Callable[
        [Path, str, str], dict[str, Any]
    ] = observe_qualification_candidate,
) -> dict[str, Any]:
    if timeout_seconds is None:
        timeout_seconds = _accepted_registration_wait_seconds(
            branch_root
        )
    if (
        type(timeout_seconds) not in {int, float}
        or not 0.0
        <= float(timeout_seconds)
        <= MAX_MANAGER_WAIT_SECONDS
    ):
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "candidate-removal wait bound is invalid",
        )
    deadline = time.monotonic() + timeout_seconds
    while True:
        observed = observer(
            branch_root, managed_name, artifact_identity
        )
        if observed["present"] is False:
            states = list(observed["states_observed"])
            if "REMOVED" not in states:
                states.append("REMOVED")
            return {
                **observed,
                "states_observed": states,
                "registry_location_removed": True,
            }
        if time.monotonic() >= deadline:
            return {
                **observed,
                "registry_location_removed": False,
            }
        time.sleep(0.1)


def _observe_default_registry(branch_root: Path) -> dict[str, Any]:
    database = (
        branch_root
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
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro", uri=True, timeout=5.0
        )
    except (OSError, sqlite3.Error) as error:
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "registry is unavailable for read-only observation",
        ) from error
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute(
                "SELECT key,value FROM registry_metadata"
            )
        }
        generation = int(metadata["registry_generation"])
        historical = tuple(
            sorted(
                str(row["relative_root"])
                for row in connection.execute(
                    "SELECT relative_root FROM artifact_locations"
                )
            )
        )
        defaults = list(
            connection.execute(
                """
                SELECT a.alias,a.model_version_id,mv.bundle_id,mv.state
                FROM aliases AS a
                JOIN model_versions AS mv
                  ON mv.model_version_id=a.model_version_id
                WHERE a.alias_kind='default'
                """
            )
        )
        if not defaults:
            return {
                "present": False,
                "generation": generation,
                "default_alias": None,
                "public_model_id": None,
                "artifact_version_id": None,
                "capability_manifest_identity": None,
                "managed_location_identity": None,
                "historical_locations": historical,
            }
        if len(defaults) != 1 or defaults[0]["state"] != "READY":
            raise _qualification_error(
                "QUALIFICATION_REGISTRY_UNAVAILABLE",
                "registry default state is contradictory",
            )
        row = defaults[0]
        manifest = connection.execute(
            "SELECT manifest_sha256 FROM capability_manifests "
            "WHERE model_version_id=?",
            (row["model_version_id"],),
        ).fetchone()
        locations = list(
            connection.execute(
                """
                SELECT mvl.relative_root
                FROM model_version_locations AS mvl
                JOIN artifact_locations AS al
                  ON al.relative_root=mvl.relative_root
                WHERE mvl.model_version_id=?
                  AND al.present=1
                  AND al.current_bundle_id=?
                """,
                (row["model_version_id"], row["bundle_id"]),
            )
        )
        if (
            manifest is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(manifest["manifest_sha256"])
            )
            is None
            or len(locations) != 1
        ):
            raise _qualification_error(
                "QUALIFICATION_REGISTRY_UNAVAILABLE",
                "registry default evidence is incomplete",
            )
        location_basis = {
            "relative_root": str(locations[0]["relative_root"]),
            "bundle_id": str(row["bundle_id"]),
        }
        if connection.total_changes != 0:
            raise _qualification_error(
                "QUALIFICATION_REGISTRY_UNAVAILABLE",
                "read-only registry observation changed state",
            )
        return {
            "present": True,
            "generation": generation,
            "default_alias": str(row["alias"]),
            "public_model_id": str(row["model_version_id"]),
            "artifact_version_id": str(row["bundle_id"]),
            "capability_manifest_identity": (
                "sha256:" + str(manifest["manifest_sha256"])
            ),
            "managed_location_identity": _identity(location_basis),
            "historical_locations": historical,
        }
    except (KeyError, ValueError, sqlite3.Error) as error:
        raise _qualification_error(
            "QUALIFICATION_REGISTRY_UNAVAILABLE",
            "registry read-only observation failed",
        ) from error
    finally:
        connection.close()


def _observe_service_state(
    branch_root: Path, registry: dict[str, Any]
) -> dict[str, Any]:
    control = branch_root / "RUNTIME" / "service_control"
    profile = _safe_json(
        control / "operating-profile.json",
        "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
    )
    desired = _safe_json(
        control / "desired-state.json",
        "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
    )
    supervisor = _safe_json(
        control / "status" / "supervisor.json",
        "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
    )
    profile_identity = _operating_profile_identity(
        profile,
        reason_code="QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
    )
    recovery = supervisor.get("recovery_status")
    api = supervisor.get("observed_api_service")
    router = supervisor.get("observed_private_router")
    warm = supervisor.get("warm_model_identity")
    readiness = supervisor.get("service_readiness_state")
    if (
        desired.get("desired_state") != "RUNNING"
        or desired.get("profile_identity") != profile_identity
        or not isinstance(recovery, dict)
        or recovery.get("recovery_state") != "IDLE"
        or recovery.get("fail_closed_latched") is not False
        or not isinstance(api, dict)
        or api.get("active") is not True
        or api.get("consistent") is not True
        or api.get("listener_owned") is not True
        or not isinstance(router, dict)
        or router.get("active") is not True
        or router.get("consistent") is not True
        or router.get("listener_owned") is not True
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "service ownership is not coherent before qualification",
        )
    if registry["present"]:
        if (
            readiness != "READY"
            or not isinstance(warm, dict)
            or warm.get("health_state") != "ready"
            or warm.get("resolved_public_model_id")
            != registry["public_model_id"]
            or warm.get("artifact_version_id")
            != registry["artifact_version_id"]
        ):
            raise _qualification_error(
                "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
                "incumbent is not coherently READY and warm",
            )
    elif readiness != "WAITING_FOR_MODEL" or warm is not None:
        raise _qualification_error(
            "QUALIFICATION_WAITING_FOR_MODEL_RESTORATION_FAILED",
            "healthy no-model service is not WAITING_FOR_MODEL",
        )
    warm_projection = (
        {
            "public_model_id": warm.get("resolved_public_model_id"),
            "artifact_version_id": warm.get("artifact_version_id"),
            "capability_manifest_identity": (
                "sha256:" + warm["capability_manifest_identity"]
                if isinstance(warm.get("capability_manifest_identity"), str)
                and not warm["capability_manifest_identity"].startswith("sha256:")
                else warm.get("capability_manifest_identity")
            ),
            "health_state": warm.get("health_state"),
        }
        if isinstance(warm, dict)
        else None
    )
    return {
        "profile_identity": profile_identity,
        "service_readiness": str(readiness),
        "recovery_state": str(recovery["recovery_state"]),
        "warm": warm_projection,
        "api_service_transaction_id": api.get("transaction_id"),
        "router_transaction_id": router.get("transaction_id"),
        "model_child_identity": supervisor.get("observed_model_child"),
    }


def capture_incumbent_snapshot(
    paths: InspectorPaths,
    branch_paths: BranchHandoffPaths,
    *,
    registry_reader: Callable[[Path], dict[str, Any]] = (
        _observe_default_registry
    ),
    service_reader: Callable[
        [Path, dict[str, Any]], dict[str, Any]
    ] = _observe_service_state,
    credential_reader: Callable[[Path], SecretCredential] = (
        read_local_credential
    ),
) -> IncumbentSnapshot:
    registry = registry_reader(branch_paths.branch_root)
    service = service_reader(branch_paths.branch_root, registry)
    credential = credential_reader(branch_paths.branch_root)
    if (
        not isinstance(credential.key_id, str)
        or not credential.key_id
        or not isinstance(registry.get("historical_locations"), tuple)
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "incumbent snapshot evidence is invalid",
        )
    return IncumbentSnapshot(
        present=bool(registry["present"]),
        default_alias=registry["default_alias"],
        public_model_id=registry["public_model_id"],
        artifact_version_id=registry["artifact_version_id"],
        capability_manifest_identity=registry[
            "capability_manifest_identity"
        ],
        managed_location_identity=registry["managed_location_identity"],
        warm_before=service["warm"],
        registry_generation=int(registry["generation"]),
        credential_key_id=credential.key_id,
        profile_identity=service["profile_identity"],
        service_readiness=service["service_readiness"],
        recovery_state=service["recovery_state"],
        api_service_transaction_id=service[
            "api_service_transaction_id"
        ],
        router_transaction_id=service["router_transaction_id"],
        model_child_identity=service["model_child_identity"],
        historical_registry_locations=registry["historical_locations"],
    )


def qualification_managed_name(
    artifact_identity: str, transaction_id: str
) -> str:
    if SHA256_PATTERN.fullmatch(artifact_identity) is None:
        raise _qualification_error(
            "QUALIFICATION_INPUT_INVALID",
            "candidate artifact identity is invalid",
        )
    if not isinstance(transaction_id, str) or not transaction_id:
        raise _qualification_error(
            "QUALIFICATION_INPUT_INVALID",
            "qualification transaction identity is invalid",
        )
    digest = artifact_identity.removeprefix("sha256:")
    owner = hashlib.sha256(transaction_id.encode("utf-8")).hexdigest()
    return (
        f"qualification-candidate-{owner[:16]}-{digest[:16]}.gguf"
    )


def stage_qualification_candidate(
    authorization: QualificationAuthorization,
    incumbent: IncumbentSnapshot,
    *,
    transaction_id: str,
    safety_margin_bytes: int | None = None,
    stager: Callable[..., StagedArtifact] = create_staged_artifact,
    publisher: Callable[
        [DestinationPlan, StagedArtifact], PublishedArtifact
    ] = publish_staged_artifact,
) -> QualificationAdmission:
    managed_name = qualification_managed_name(
        authorization.source.artifact_identity, transaction_id
    )
    try:
        plan = prepare_handoff_destination(
            authorization.branch_paths,
            transaction_id=transaction_id,
            managed_name=managed_name,
            artifact_identity=authorization.source.artifact_identity,
            historical_registry_locations=(
                incumbent.historical_registry_locations
            ),
        )
        staged = stager(
            plan,
            authorization.source,
            safety_margin_bytes=safety_margin_bytes,
        )
        published = publisher(plan, staged)
    except InspectorError as error:
        raise _map_handoff_error(error) from error
    if (
        published.sha256 != authorization.source.artifact_identity
        or published.size_bytes
        != authorization.source.snapshot["size_bytes"]
        or published.link_count != 1
        or published.path != plan.managed_target
        or published.relative_path != plan.managed_relative_path
    ):
        raise _qualification_error(
            "QUALIFICATION_PUBLICATION_FAILED",
            "qualification publication identity is inconsistent",
        )
    return QualificationAdmission(
        plan=plan, staged=staged, published=published
    )


def _accepted_control_script(
    branch_root: Path, relative_path: tuple[str, ...]
) -> Path:
    candidate = branch_root.joinpath(*relative_path)
    try:
        details = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager control script is unavailable",
        ) from error
    if (
        resolved != candidate
        or not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or details.st_nlink != 1
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager control script is unsafe",
        )
    return candidate


def _run_accepted_control_command(
    branch_root: Path,
    script: Path,
    *arguments: str,
) -> dict[str, Any]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-B", str(script), *arguments],
            cwd=branch_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=MANAGER_RESTORE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager control command did not complete",
        ) from error
    if (
        completed.returncode != 0
        or len(completed.stdout) > MAX_CONTROL_COMMAND_OUTPUT_BYTES
        or len(completed.stderr) > MAX_CONTROL_COMMAND_OUTPUT_BYTES
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager control command failed",
        )
    try:
        lines = [
            line
            for line in completed.stdout.decode("utf-8").splitlines()
            if line.strip()
        ]
        result = json.loads(lines[-1]) if len(lines) == 1 else None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager control result is invalid",
        ) from error
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager control result was not successful",
        )
    return result


def _wait_for_restarted_api_service(
    branch_root: Path, previous_transaction_id: str
) -> str:
    deadline = time.monotonic() + MANAGER_RESTORE_TIMEOUT_SECONDS
    status_path = (
        branch_root / "RUNTIME" / "api" / "status" / "service.json"
    )
    while True:
        try:
            service = _safe_json(
                status_path,
                "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            )
        except InspectorError:
            service = {}
        transaction_id = service.get("transaction_id")
        if (
            service.get("lifecycle_state") == "STARTED"
            and isinstance(transaction_id, str)
            and transaction_id
            and transaction_id != previous_transaction_id
        ):
            return transaction_id
        if time.monotonic() >= deadline:
            raise _qualification_error(
                "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
                "restarted API service did not become active",
            )
        time.sleep(0.25)


def restore_with_accepted_platform_manager(
    branch_root: Path,
) -> dict[str, Any]:
    service_before = _safe_json(
        branch_root / "RUNTIME" / "api" / "status" / "service.json",
        "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
    )
    previous_api_transaction_id = service_before.get("transaction_id")
    desired = _safe_json(
        branch_root
        / "RUNTIME"
        / "service_control"
        / "desired-state.json",
        "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
    )
    generation = desired.get("generation")
    if (
        not isinstance(previous_api_transaction_id, str)
        or not previous_api_transaction_id
        or service_before.get("lifecycle_state") != "STARTED"
        or desired.get("desired_state") != "RUNNING"
        or type(generation) is not int
        or generation < 0
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager fallback requires RUNNING desired state",
        )
    managed_root = branch_root / "MODEL" / "SUPERMODEL"
    try:
        qualification_targets = [
            child
            for child in managed_root.iterdir()
            if child.name.startswith("qualification-candidate-")
        ]
    except OSError as error:
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "managed model root is unavailable for restoration",
        ) from error
    if qualification_targets:
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "qualification target remains before manager fallback",
        )
    adapter = _accepted_control_script(
        branch_root,
        (
            "service_control",
            "platform_adapters",
            "linux_systemd_user.py",
        ),
    )
    api_controller = _accepted_control_script(
        branch_root,
        ("api_service_controller", "controller.py"),
    )
    branch_controller = _accepted_control_script(
        branch_root,
        ("branch_controller", "controller.py"),
    )
    stopped = _run_accepted_control_command(
        branch_root,
        adapter,
        "stop",
        "--expected-generation",
        str(generation),
        "--wait-timeout-seconds",
        str(MANAGER_RESTORE_TIMEOUT_SECONDS),
    )
    stopped_generation = stopped.get("desired_state_generation")
    if (
        stopped.get("desired_state") != "STOPPED"
        or stopped_generation != generation + 1
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager stop transition is inconsistent",
        )
    api_reconcile = _run_accepted_control_command(
        branch_root, api_controller, "reconcile"
    )
    router_reconcile = _run_accepted_control_command(
        branch_root, branch_controller, "reconcile"
    )
    started = _run_accepted_control_command(
        branch_root,
        adapter,
        "start",
        "--expected-generation",
        str(stopped_generation),
        "--wait-timeout-seconds",
        str(MANAGER_RESTORE_TIMEOUT_SECONDS),
    )
    if (
        started.get("desired_state") != "RUNNING"
        or started.get("desired_state_generation")
        != stopped_generation + 1
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager start transition is inconsistent",
        )
    restarted_api_transaction_id = _wait_for_restarted_api_service(
        branch_root, previous_api_transaction_id
    )
    return {
        "used": True,
        "stop_generation": stopped_generation,
        "start_generation": started["desired_state_generation"],
        "restarted_api_transaction_id": restarted_api_transaction_id,
        "api_reconcile_reason_code": api_reconcile.get("reason_code"),
        "router_reconcile_reason_code": router_reconcile.get(
            "reason_code"
        ),
    }


def recover_with_accepted_platform_manager(
    branch_root: Path,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    managed_root = branch_root / "MODEL" / "SUPERMODEL"
    if any(
        child.name.startswith("qualification-candidate-")
        for child in managed_root.iterdir()
    ):
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "qualification target remains before interrupted recovery",
        )
    adapter = _accepted_control_script(
        branch_root,
        ("service_control", "platform_adapters", "linux_systemd_user.py"),
    )
    api_controller = _accepted_control_script(
        branch_root, ("api_service_controller", "controller.py")
    )
    branch_controller = _accepted_control_script(
        branch_root, ("branch_controller", "controller.py")
    )

    def invoke(
        script: Path,
        *arguments: str,
        accepted: tuple[int, ...] = (0,),
    ) -> dict[str, Any]:
        completed = subprocess.run(
            [sys.executable, "-B", str(script), *arguments],
            cwd=branch_root,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=MANAGER_RESTORE_TIMEOUT_SECONDS,
        )
        try:
            lines = [
                line
                for line in completed.stdout.decode("utf-8").splitlines()
                if line.strip()
            ]
            value = json.loads(lines[-1]) if len(lines) == 1 else None
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _qualification_error(
                "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
                "interrupted recovery control output is invalid",
            ) from error
        if completed.returncode not in accepted or not isinstance(value, dict):
            raise _qualification_error(
                "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
                "interrupted recovery control command failed",
            )
        return value

    stopped = invoke(
        adapter,
        "stop",
        "--wait-timeout-seconds",
        str(MANAGER_RESTORE_TIMEOUT_SECONDS),
        accepted=(0, 3),
    )
    api_reconcile = invoke(api_controller, "reconcile")
    router_reconcile = invoke(branch_controller, "reconcile")
    sleeper(20.0)
    attempts = []
    started = None
    for _attempt in range(6):
        candidate = invoke(
            adapter,
            "start",
            "--wait-timeout-seconds",
            str(MANAGER_RESTORE_TIMEOUT_SECONDS),
            accepted=(0, 3, 4),
        )
        attempts.append(candidate.get("reason_code"))
        if candidate.get("ok") is True:
            started = candidate
            break
        if candidate.get("reason_code") not in {
            "ENDPOINT_CONFLICT",
            "ADAPTER_ACTIVATION_FAILED",
        }:
            break
        sleeper(10.0)
    if started is None:
        raise _qualification_error(
            "QUALIFICATION_INCUMBENT_RESTORATION_FAILED",
            "accepted manager could not resume interrupted recovery",
        )
    return {
        "used": True,
        "stop_reason_code": stopped.get("reason_code"),
        "start_reason_code": started.get("reason_code"),
        "start_attempt_reason_codes": attempts,
        "api_reconcile_reason_code": api_reconcile.get("reason_code"),
        "router_reconcile_reason_code": router_reconcile.get("reason_code"),
    }


def qualification_owned_cold_default(
    admission: QualificationAdmission,
    incumbent: IncumbentSnapshot,
    observation: dict[str, Any],
) -> bool:
    return bool(
        not incumbent.present
        and observation.get("terminal") == "READY"
        and observation.get("present") is True
        and observation.get("default_bound") is True
        and observation.get("aliases") == ["default"]
        and observation.get("artifact_version_id")
        == admission.published.sha256.replace("sha256:", "bundle-", 1)
        and observation.get("public_model_id") is not None
        and observation.get("capability_manifest_identity") is not None
    )


def clear_qualification_default(
    branch_root: Path,
    managed_name: str,
    artifact_identity: str,
    observation: dict[str, Any],
    transaction_id: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    observer: Callable[
        [Path, str, str], dict[str, Any]
    ] = observe_qualification_candidate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    refreshed = observer(branch_root, managed_name, artifact_identity)
    if (
        refreshed.get("public_model_id")
        != observation.get("public_model_id")
        or refreshed.get("artifact_version_id")
        != observation.get("artifact_version_id")
        or SHA256_PATTERN.fullmatch(
            str(refreshed.get("capability_manifest_identity"))
        ) is None
    ):
        raise _qualification_error(
            "QUALIFICATION_DEFAULT_CHANGED",
            "qualification candidate identity changed before alias clear",
        )
    observation = refreshed
    public_model_id = observation.get("public_model_id")
    artifact_version_id = observation.get("artifact_version_id")
    generation = observation.get("registry_generation")
    aliases = observation.get("aliases")
    if (
        observation.get("terminal") != "READY"
        or observation.get("present") is not True
        or observation.get("default_bound") is not True
        or aliases != ["default"]
        or not isinstance(public_model_id, str)
        or not isinstance(generation, int)
        or QUALIFICATION_MANAGED_NAME_PATTERN.fullmatch(managed_name) is None
        or SHA256_PATTERN.fullmatch(artifact_identity) is None
        or artifact_version_id
        != artifact_identity.replace("sha256:", "bundle-", 1)
    ):
        raise _qualification_error(
            "QUALIFICATION_DEFAULT_CHANGED",
            "qualification default is not an exact owned cold-install alias",
        )
    request = {
        "schema_version": "system-x.gguf-alias-transaction.v1",
        "action": "clear",
        "promotion_transaction_id": transaction_id,
        "alias": "default",
        "expected_current_target": public_model_id,
        "new_target": None,
        "expected_registry_generation": generation,
        "target_artifact_version_id": None,
        "target_capability_manifest_identity": None,
        "target_relative_root": None,
        "promotion_alias_event_identity": None,
    }
    completed = runner(
        [
            str(branch_root / "api_service" / ".venv" / "bin" / "python"),
            "-B",
            str(branch_root / "api_service_controller" / "controller.py"),
            "alias-transaction",
        ],
        input=canonical_json_bytes(request),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=branch_root,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        timeout=30.0,
        check=False,
    )
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _qualification_error(
            "QUALIFICATION_DEFAULT_CHANGED",
            "branch alias transaction emitted invalid JSON",
        ) from error
    alias = value.get("alias_transaction") if isinstance(value, dict) else None
    if (
        completed.returncode != 0
        or not isinstance(value, dict)
        or value.get("ok") is not True
        or not isinstance(alias, dict)
        or alias.get("action") != "clear"
        or alias.get("alias") != "default"
        or alias.get("previous_target") != public_model_id
        or alias.get("new_target") is not None
        or alias.get("new_registry_generation") != generation + 1
    ):
        raise _qualification_error(
            "QUALIFICATION_DEFAULT_CHANGED",
            "branch alias transaction did not prove the exact clear",
        )
    after = observer(branch_root, managed_name, artifact_identity)
    if (
        after.get("terminal") != "READY"
        or after.get("present") is not True
        or after.get("default_bound") is not False
        or after.get("aliases") != []
        or after.get("public_model_id") != public_model_id
        or after.get("artifact_version_id") != artifact_version_id
        or after.get("registry_generation") != generation + 1
        or after.get("capability_manifest_identity")
        != observation.get("capability_manifest_identity")
    ):
        raise _qualification_error(
            "QUALIFICATION_DEFAULT_CHANGED",
            "qualification default clear did not round-trip exactly",
        )
    return dict(alias), after


def cleanup_qualification_candidate(
    admission: QualificationAdmission,
    registry_observation: dict[str, Any],
    *,
    removal_waiter: Callable[
        [Path, str, str], dict[str, Any]
    ] = wait_for_candidate_removal,
    manager_restorer: Callable[
        [Path], dict[str, Any]
    ] = restore_with_accepted_platform_manager,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if registry_observation.get("default_bound") is True:
        raise _qualification_error(
            "QUALIFICATION_DEFAULT_CHANGED",
            "qualification candidate became default-bound",
        )
    target = admission.published.path
    try:
        descriptor = os.open(
            target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as error:
        raise _qualification_error(
            "QUALIFICATION_OWNERSHIP_UNCERTAIN",
            "qualification managed target cannot be reopened",
        ) from error
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    expected = admission.published
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != expected.device
        or before.st_ino != expected.inode
        or before.st_size != expected.size_bytes
        or before.st_nlink != 1
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or byte_count != expected.size_bytes
        or "sha256:" + digest.hexdigest() != expected.sha256
        or target.parent != admission.plan.branch_paths.managed_root
        or target.name != admission.plan.managed_name
    ):
        raise _qualification_error(
            "QUALIFICATION_OWNERSHIP_UNCERTAIN",
            "qualification target ownership cannot be proved",
        )
    target.unlink()
    fsync_directory(target.parent)
    if target.exists() or target.is_symlink():
        raise _qualification_error(
            "QUALIFICATION_CLEANUP_FAILED",
            "qualification target remained after cleanup",
        )
    if (
        admission.plan.staging_path.exists()
        or admission.plan.staging_path.is_symlink()
    ):
        raise _qualification_error(
            "QUALIFICATION_CLEANUP_FAILED",
            "qualification staging remained after publication",
        )
    removed = removal_waiter(
        admission.plan.branch_paths.branch_root,
        admission.plan.managed_name,
        expected.sha256,
    )
    if removed.get("registry_location_removed") is False:
        manager_restorer(admission.plan.branch_paths.branch_root)
        removed = removal_waiter(
            admission.plan.branch_paths.branch_root,
            admission.plan.managed_name,
            expected.sha256,
        )
    cleanup = {
        "staging_absent": True,
        "managed_target_absent": True,
        "registry_location_removed": removed.get(
            "registry_location_removed"
        ),
        "source_removed_if_packet_owned": None,
        "ownership_certain": (
            removed.get("registry_location_removed") is True
        ),
    }
    return cleanup, removed


def _recovery_manifest_identities(
    record: dict[str, Any],
) -> frozenset[str]:
    runtime = record.get("candidate_runtime")
    primary = (
        runtime.get("capability_manifest_identity")
        if isinstance(runtime, dict)
        else None
    )
    identities = {primary}
    warm_after = record.get("incumbent", {}).get("warm_after")
    if isinstance(warm_after, dict):
        identities.add(warm_after.get("capability_manifest_identity"))
    if any(
        not isinstance(value, str)
        or SHA256_PATTERN.fullmatch(value) is None
        for value in identities
    ):
        raise _qualification_error(
            "QUALIFICATION_OWNERSHIP_UNCERTAIN",
            "failed qualification manifest lineage is invalid",
        )
    return frozenset(identities)


def _recovery_admission(
    paths: InspectorPaths,
    failed_transaction_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    QualificationAdmission | None,
    dict[str, Any],
]:
    if re.fullmatch(
        r"tx-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}",
        failed_transaction_id,
    ) is None:
        raise _qualification_error(
            "QUALIFICATION_INPUT_INVALID",
            "failed qualification transaction ID is invalid",
        )
    failed_path = paths.transactions / f"{failed_transaction_id}.json"
    failed = _safe_json(
        failed_path, "QUALIFICATION_OWNERSHIP_UNCERTAIN"
    )
    record_value = failed.get("qualification_record_candidate")
    try:
        record = validate_qualification_record(record_value)
    except InspectorError as error:
        raise _qualification_error(
            "QUALIFICATION_OWNERSHIP_UNCERTAIN",
            "failed qualification result cannot be authenticated",
        ) from error
    result_path = paths.qualification_results / f"{record['qualification_id']}.json"
    persisted = validate_qualification_record(read_json_record(result_path))
    managed_name = failed.get("managed_name")
    managed_relative = failed.get("managed_relative_path")
    staging_relative = failed.get("staging_relative_path")
    artifact_identity = failed.get("candidate_artifact_identity")
    incumbent = failed.get("incumbent_snapshot")
    if (
        failed.get("transaction_id") != failed_transaction_id
        or failed.get("operation") != "qualify-gguf"
        or failed.get("state") != "FAIL_CLOSED"
        or failed.get("reason_code") != "QUALIFICATION_FAIL_CLOSED"
        or failed.get("result_class") != "QUALIFICATION_FAIL_CLOSED"
        or not isinstance(incumbent, dict)
        or incumbent.get("present") is not False
        or incumbent.get("default_alias") is not None
        or not isinstance(managed_name, str)
        or QUALIFICATION_MANAGED_NAME_PATTERN.fullmatch(managed_name) is None
        or managed_relative != f"MODEL/SUPERMODEL/{managed_name}"
        or not isinstance(staging_relative, str)
        or not staging_relative.startswith(
            "RUNTIME/api/replacement-staging/." + failed_transaction_id + "."
        )
        or not staging_relative.endswith(".partial-staging.gguf")
        or SHA256_PATTERN.fullmatch(artifact_identity or "") is None
        or failed.get("artifact_identity") != artifact_identity
        or failed.get("publication_sha256") != artifact_identity
        or failed.get("publication_mode") != "0640"
        or failed.get("publication_link_count") != 1
        or not isinstance(failed.get("publication_device"), int)
        or not isinstance(failed.get("publication_inode"), int)
        or not isinstance(failed.get("publication_size"), int)
        or failed.get("publication_size") <= 0
        or persisted != record
        or record.get("transaction_id") != failed_transaction_id
        or record.get("result_identity")
        != failed.get("qualification_result_identity")
        or "QUALIFICATION_DEFAULT_CHANGED" not in record.get("reason_codes", [])
        or record.get("cleanup", {}).get("managed_target_absent") is not False
        or record.get("cleanup", {}).get("ownership_certain") is not False
        or record.get("candidate_runtime", {}).get("managed_relative_path")
        != managed_relative
    ):
        raise _qualification_error(
            "QUALIFICATION_OWNERSHIP_UNCERTAIN",
            "failed qualification does not describe one recoverable cold install",
        )
    branch = BranchHandoffPaths.discover(paths)
    accepted_manifest_identities = _recovery_manifest_identities(record)
    target = branch.managed_root / managed_name
    staging = branch.branch_root / staging_relative
    root_details = branch.managed_root.lstat()
    try:
        target_details = target.lstat()
    except FileNotFoundError as error:
        if target.exists() or target.is_symlink() or staging.exists() or staging.is_symlink():
            raise _qualification_error(
                "QUALIFICATION_OWNERSHIP_UNCERTAIN",
                "interrupted qualification cleanup has unsafe residue",
            ) from error
        observation = observe_qualification_candidate(
            branch.branch_root, managed_name, artifact_identity
        )
        runtime = record["candidate_runtime"]
        if (
            observation.get("default_bound") is not False
            or observation.get("aliases") != []
            or observation.get("public_model_id") != runtime["public_model_id"]
            or observation.get("artifact_version_id")
            != runtime["artifact_version_id"]
            or observation.get("capability_manifest_identity")
            not in accepted_manifest_identities
            or observation.get("present") not in {True, False}
        ):
            raise _qualification_error(
                "QUALIFICATION_OWNERSHIP_UNCERTAIN",
                "interrupted qualification cleanup is not exactly resumable",
            )
        failed_reconciliations = [
            value
            for path in paths.transactions.glob("*.json")
            if (value := _safe_json(path, "QUALIFICATION_OWNERSHIP_UNCERTAIN")).get("operation")
            == "reconcile-qualification"
            and value.get("failed_qualification_transaction_id")
            == failed_transaction_id
            and value.get("state") == "FAIL_CLOSED"
        ]
        if not failed_reconciliations:
            raise _qualification_error(
                "QUALIFICATION_OWNERSHIP_UNCERTAIN",
                "interrupted cleanup lacks an Inspector reconciliation record",
            )
        return failed, record, None, observation
    if (
        stat.S_ISLNK(target_details.st_mode)
        or not stat.S_ISREG(target_details.st_mode)
        or target_details.st_dev != failed["publication_device"]
        or target_details.st_ino != failed["publication_inode"]
        or target_details.st_size != failed["publication_size"]
        or stat.S_IMODE(target_details.st_mode) != 0o640
        or target_details.st_nlink != 1
        or staging.exists()
        or staging.is_symlink()
    ):
        raise _qualification_error(
            "QUALIFICATION_OWNERSHIP_UNCERTAIN",
            "failed qualification physical ownership changed",
        )
    plan = DestinationPlan(
        branch_paths=branch,
        transaction_id=failed_transaction_id,
        managed_name=managed_name,
        managed_relative_path=managed_relative,
        managed_target=target,
        staging_name=staging.name,
        staging_relative_path=staging_relative,
        staging_path=staging,
        policy=ManagedPolicy(
            mode=0o640,
            owner_uid=target_details.st_uid,
            owner_gid=target_details.st_gid,
            reference_names=(),
        ),
        managed_root_identity=(
            root_details.st_dev,
            root_details.st_ino,
            stat.S_IMODE(root_details.st_mode),
        ),
    )
    staged = StagedArtifact(
        path=staging,
        relative_path=staging_relative,
        transfer_method=str(failed.get("transfer_method")),
        device=target_details.st_dev,
        inode=target_details.st_ino,
        mode=0o640,
        link_count=1,
        size_bytes=target_details.st_size,
        sha256=artifact_identity,
        source_snapshot_identity=str(failed.get("intake_snapshot_identity")),
    )
    published = PublishedArtifact(
        path=target,
        relative_path=managed_relative,
        device=target_details.st_dev,
        inode=target_details.st_ino,
        mode=0o640,
        link_count=1,
        size_bytes=target_details.st_size,
        sha256=artifact_identity,
    )
    admission = QualificationAdmission(
        plan=plan, staged=staged, published=published
    )
    observation = observe_qualification_candidate(
        branch.branch_root, managed_name, artifact_identity
    )
    runtime = record["candidate_runtime"]
    if (
        observation.get("terminal") != "READY"
        or observation.get("present") is not True
        or observation.get("default_bound") is not True
        or observation.get("aliases") != ["default"]
        or observation.get("public_model_id") != runtime["public_model_id"]
        or observation.get("artifact_version_id")
        != runtime["artifact_version_id"]
        or observation.get("capability_manifest_identity")
        not in accepted_manifest_identities
    ):
        raise _qualification_error(
            "QUALIFICATION_DEFAULT_CHANGED",
            "failed qualification runtime no longer matches its owned candidate",
        )
    return failed, record, admission, observation


def _prove_cold_waiting(
    branch_root: Path,
    *,
    timeout_seconds: float = RESTORATION_WAIT_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest_registry = None
    latest_service = None
    while True:
        try:
            latest_registry = _observe_default_registry(branch_root)
            latest_service = _observe_service_state(
                branch_root, latest_registry
            )
            if (
                latest_registry.get("present") is False
                and latest_registry.get("default_alias") is None
                and latest_service.get("service_readiness")
                == "WAITING_FOR_MODEL"
                and latest_service.get("recovery_state") == "IDLE"
                and latest_service.get("warm") is None
            ):
                return {
                    "default_absent": True,
                    "service_readiness": "WAITING_FOR_MODEL",
                    "recovery_state": "IDLE",
                    "warm": None,
                    "proved": True,
                }
        except InspectorError:
            pass
        if time.monotonic() >= deadline:
            raise _qualification_error(
                "QUALIFICATION_WAITING_FOR_MODEL_RESTORATION_FAILED",
                "cold-install recovery did not restore waiting state",
            )
        time.sleep(0.1)


def reconcile_qualification_transaction(
    paths: InspectorPaths,
    failed_transaction_id: str,
    *,
    transaction_id_factory: Callable[[], str] = _transaction_id,
    transition_observer: (
        Callable[[str, dict[str, Any]], None] | None
    ) = None,
) -> tuple[str, dict[str, Any]]:
    transaction_id = transaction_id_factory()
    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="reconcile-qualification",
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
    transaction = {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "reconcile-qualification",
        "start_utc": utc_now(),
        "finish_utc": None,
        "state": "RECOVERY_RECONCILING",
        "reason_code": "OK",
        "owner_identity": {
            key: owner.get(key)
            for key in (
                "pid",
                "process_start_identity",
                "boot_identity",
                "inspector_root_identity",
            )
        },
        "failed_qualification_transaction_id": failed_transaction_id,
        "reconciliation": None,
    }
    try:
        transaction = _inspection_stage(
            paths,
            transaction,
            state="RECOVERY_RECONCILING",
            reason_code="OK",
            observer=transition_observer,
        )
        failed, record, admission, observation = _recovery_admission(
            paths, failed_transaction_id
        )
        if admission is not None:
            branch_root = admission.plan.branch_paths.branch_root
            managed_name = admission.plan.managed_name
            artifact_identity = admission.published.sha256
            managed_path = str(admission.published.path)
            transaction = _inspection_stage(
                paths,
                transaction,
                state="DEFAULT_ALIAS_CLEARING",
                reason_code="OK",
                observer=transition_observer,
            )
            alias, after_clear = clear_qualification_default(
                branch_root,
                managed_name,
                artifact_identity,
                observation,
                transaction_id,
            )
        else:
            branch_paths = BranchHandoffPaths.discover(paths)
            branch_root = branch_paths.branch_root
            managed_name = str(failed["managed_name"])
            artifact_identity = str(failed["candidate_artifact_identity"])
            managed_path = str(branch_paths.managed_root / managed_name)
            alias = {
                "action": "clear",
                "alias": "default",
                "changed": False,
                "new_target": None,
                "previous_target": observation["public_model_id"],
                "resumed_from_failed_reconciliation": True,
            }
            after_clear = observation
        transaction = _inspection_stage(
            paths,
            transaction,
            state="CLEANING_CANDIDATE",
            reason_code="OK",
            observer=transition_observer,
        )
        if admission is not None:
            cleanup, removal = cleanup_qualification_candidate(
                admission, after_clear
            )
            manager_recovery = {"used": False}
        else:
            if observation.get("present") is False:
                manager_recovery = {
                    "used": False,
                    "already_converged": True,
                }
                removal = {
                    **observation,
                    "registry_location_removed": True,
                }
            else:
                manager_recovery = recover_with_accepted_platform_manager(
                    branch_root
                )
                removal = wait_for_candidate_removal(
                    branch_root, managed_name, artifact_identity
                )
            cleanup = {
                "staging_absent": True,
                "managed_target_absent": True,
                "registry_location_removed": removal.get(
                    "registry_location_removed"
                ),
                "source_removed_if_packet_owned": None,
                "ownership_certain": (
                    removal.get("registry_location_removed") is True
                ),
            }
        restoration = _prove_cold_waiting(branch_root)
        proof = {
            "failed_qualification_transaction_id": failed_transaction_id,
            "qualification_id": record["qualification_id"],
            "artifact_identity": artifact_identity,
            "managed_path": managed_path,
            "public_model_id": observation["public_model_id"],
            "source_state": failed["state"],
            "alias_transaction": alias,
            "cleanup": cleanup,
            "removal": removal,
            "manager_recovery": manager_recovery,
            "restoration": restoration,
        }
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        idle_identity = _write_status(
            paths, idle, transition_observer
        )
        terminal = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "COMPLETE",
            "reason_code": "QUALIFICATION_CANDIDATE_CLEANED",
            "status_record_identity": idle_identity,
            "reconciliation": proof,
        }
        _write_transaction(paths, terminal, transition_observer)
        return transaction_id, proof
    except Exception as error:
        failed_status = _status_value(
            paths,
            state="FAIL_CLOSED",
            reason_code="QUALIFICATION_FAIL_CLOSED",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        status_identity = _write_status(
            paths, failed_status, transition_observer
        )
        _write_transaction(
            paths,
            {
                **transaction,
                "finish_utc": utc_now(),
                "state": "FAIL_CLOSED",
                "reason_code": "QUALIFICATION_FAIL_CLOSED",
                "status_record_identity": status_identity,
            },
            transition_observer,
        )
        if isinstance(error, InspectorError):
            error.data = {**error.data, "transaction_id": transaction_id}
            raise
        raise _qualification_error(
            "QUALIFICATION_FAIL_CLOSED",
            "unexpected qualification reconciliation failure",
            data={"transaction_id": transaction_id},
        ) from error
    finally:
        lock.release()


def prove_incumbent_restoration(
    authorization: QualificationAuthorization,
    incumbent: IncumbentSnapshot,
    *,
    timeout_seconds: float = RESTORATION_WAIT_SECONDS,
    registry_reader: Callable[[Path], dict[str, Any]] = (
        _observe_default_registry
    ),
    service_reader: Callable[
        [Path, dict[str, Any]], dict[str, Any]
    ] = _observe_service_state,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    deadline = time.monotonic() + timeout_seconds
    latest_registry: dict[str, Any] | None = None
    latest_service: dict[str, Any] | None = None
    proved = False
    while True:
        try:
            latest_registry = registry_reader(
                authorization.branch_paths.branch_root
            )
            latest_service = service_reader(
                authorization.branch_paths.branch_root,
                latest_registry,
            )
            if incumbent.present:
                proved = bool(
                    latest_registry["present"] is True
                    and latest_registry["default_alias"]
                    == incumbent.default_alias
                    and latest_registry["public_model_id"]
                    == incumbent.public_model_id
                    and latest_registry["artifact_version_id"]
                    == incumbent.artifact_version_id
                    and latest_registry[
                        "capability_manifest_identity"
                    ]
                    == incumbent.capability_manifest_identity
                    and latest_registry[
                        "managed_location_identity"
                    ]
                    == incumbent.managed_location_identity
                    and latest_service["service_readiness"] == "READY"
                    and latest_service["recovery_state"] == "IDLE"
                    and latest_service["warm"] == incumbent.warm_before
                )
            else:
                proved = bool(
                    latest_registry["present"] is False
                    and latest_registry["default_alias"] is None
                    and latest_service["service_readiness"]
                    == "WAITING_FOR_MODEL"
                    and latest_service["recovery_state"] == "IDLE"
                    and latest_service["warm"] is None
                )
        except InspectorError:
            proved = False
        if proved or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    warm_after = (
        latest_service.get("warm")
        if isinstance(latest_service, dict)
        else None
    )
    restoration = {
        "required_state": (
            "READY" if incumbent.present else "WAITING_FOR_MODEL"
        ),
        "default_unchanged": (
            bool(
                latest_registry is not None
                and latest_registry.get("default_alias")
                == incumbent.default_alias
                and latest_registry.get("public_model_id")
                == incumbent.public_model_id
                and latest_registry.get("artifact_version_id")
                == incumbent.artifact_version_id
            )
            if incumbent.present
            else bool(
                latest_registry is not None
                and latest_registry.get("present") is False
                and latest_registry.get("default_alias") is None
            )
        ),
        "incumbent_ready": (
            bool(
                incumbent.present
                and latest_service is not None
                and latest_service.get("service_readiness") == "READY"
            )
            if incumbent.present
            else None
        ),
        "incumbent_warm": (
            bool(
                incumbent.present
                and latest_service is not None
                and latest_service.get("warm")
                == incumbent.warm_before
            )
            if incumbent.present
            else None
        ),
        "waiting_for_model": (
            bool(
                not incumbent.present
                and latest_service is not None
                and latest_service.get("service_readiness")
                == "WAITING_FOR_MODEL"
                and latest_service.get("warm") is None
            )
            if not incumbent.present
            else None
        ),
        "recovery_idle": bool(
            latest_service is not None
            and latest_service.get("recovery_state") == "IDLE"
        ),
        "proved": proved,
    }
    return restoration, warm_after


def qualification_service_snapshot(
    authorization: QualificationAuthorization,
    incumbent: IncumbentSnapshot,
) -> ServiceSnapshot:
    branch_root = authorization.branch_paths.branch_root
    control = branch_root / "RUNTIME" / "service_control"
    profile = _safe_json(
        control / "operating-profile.json",
        "QUALIFICATION_PUBLIC_REQUEST_FAILED",
    )
    profile_identity = _operating_profile_identity(
        profile,
        reason_code="QUALIFICATION_PUBLIC_REQUEST_FAILED",
    )
    public = profile.get("public_endpoint")
    if (
        profile_identity != incumbent.profile_identity
        or not isinstance(public, dict)
    ):
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "operating profile changed during qualification",
        )
    host = public.get("host")
    port = public.get("port")
    try:
        address = ipaddress.ip_address(host)
    except (TypeError, ValueError) as error:
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "public origin is invalid",
        ) from error
    if (
        not address.is_loopback
        or str(address) != host
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "public origin is not exact numeric loopback",
        )
    service_status = _safe_json(
        branch_root / "RUNTIME" / "api" / "status" / "service.json",
        "QUALIFICATION_PUBLIC_REQUEST_FAILED",
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
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "public service activation is inconsistent",
        )
    log_path = Path(log_value)
    expected_log_root = (
        branch_root / "RUNTIME" / "api" / "logs"
    ).resolve(strict=True)
    try:
        log_details = log_path.lstat()
        resolved_log = log_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise _qualification_error(
            "QUALIFICATION_REQUEST_RECORD_NOT_FOUND",
            "manager-owned operation log is unavailable",
        ) from error
    if (
        stat.S_ISLNK(log_details.st_mode)
        or not stat.S_ISREG(log_details.st_mode)
        or log_details.st_nlink != 1
        or resolved_log.parent != expected_log_root
        or resolved_log.name != f"{transaction_id}.log"
    ):
        raise _qualification_error(
            "QUALIFICATION_REQUEST_RECORD_NOT_FOUND",
            "manager-owned operation log is unsafe",
        )
    default_alias = profile.get("default_model_alias")
    if not isinstance(default_alias, str) or not default_alias:
        raise _qualification_error(
            "QUALIFICATION_PUBLIC_REQUEST_FAILED",
            "operating profile default alias is invalid",
        )
    return ServiceSnapshot(
        profile_identity=profile_identity,
        host=host,
        port=port,
        base_url=f"http://{host}:{port}",
        default_alias=default_alias,
        service_transaction_id=transaction_id,
        operation_log=resolved_log,
        readiness_state="READY",
    )


def terminal_profile_run(
    requested_profile: str, runtime_outcome: str
) -> ProfileRun:
    selected = set(profile_check_names(requested_profile))
    checks: list[dict[str, Any]] = []
    for definition in ALL_QUALIFICATION_CHECKS:
        if definition[0] in selected:
            checks.append(
                _normalize_check(
                    definition,
                    {
                        "status": "NOT_APPLICABLE",
                        "reason_code": "CHECK_NOT_APPLICABLE",
                        "evidence": {
                            "check_name": definition[0],
                            "runtime_outcome": runtime_outcome,
                        },
                    },
                )
            )
        else:
            checks.append(_not_requested_check(definition))
    result_class, reasons = classify_qualification_result(
        requested_profile,
        checks,
        runtime_outcome=runtime_outcome,
    )
    return ProfileRun(
        requested_profile=requested_profile,
        checks=tuple(checks),
        supported_profiles=(),
        observed_capabilities={
            "available": [],
            "gated_unavailable": [],
            "unavailable": [],
        },
        result_class=result_class,
        reason_codes=reasons,
    )


def _prepublication_cleanup(
    authorization: QualificationAuthorization,
    transaction_id: str,
) -> dict[str, Any]:
    """Remove only an exact transaction staging file; never guess at a target."""

    managed_name = qualification_managed_name(
        authorization.source.artifact_identity, transaction_id
    )
    name_identity = hashlib.sha256(
        managed_name.encode("utf-8")
    ).hexdigest()[:16]
    staging_name = (
        f".{transaction_id}.{name_identity}.partial-staging.gguf"
    )
    staging_path = (
        authorization.branch_paths.branch_staging_root / staging_name
    )
    managed_target = (
        authorization.branch_paths.managed_root / managed_name
    )
    if (
        staging_path.parent
        != authorization.branch_paths.branch_staging_root
        or managed_target.parent
        != authorization.branch_paths.managed_root
    ):
        raise _qualification_error(
            "QUALIFICATION_OWNERSHIP_UNCERTAIN",
            "derived qualification cleanup path escaped its root",
        )
    try:
        staging_details = staging_path.lstat()
    except FileNotFoundError:
        staging_details = None
    if staging_details is not None:
        if (
            stat.S_ISLNK(staging_details.st_mode)
            or not stat.S_ISREG(staging_details.st_mode)
            or staging_details.st_nlink != 1
        ):
            raise _qualification_error(
                "QUALIFICATION_OWNERSHIP_UNCERTAIN",
                "transaction staging path has an unsafe physical type",
            )
        staging_path.unlink()
        fsync_directory(staging_path.parent)
    staging_absent = (
        not staging_path.exists() and not staging_path.is_symlink()
    )
    managed_absent = (
        not managed_target.exists() and not managed_target.is_symlink()
    )
    return {
        "staging_absent": staging_absent,
        "managed_target_absent": managed_absent,
        "registry_location_removed": None,
        "source_removed_if_packet_owned": None,
        "ownership_certain": staging_absent and managed_absent,
    }


def _failed_restoration(
    incumbent: IncumbentSnapshot,
) -> dict[str, Any]:
    return {
        "required_state": (
            "READY" if incumbent.present else "WAITING_FOR_MODEL"
        ),
        "default_unchanged": False,
        "incumbent_ready": False if incumbent.present else None,
        "incumbent_warm": False if incumbent.present else None,
        "waiting_for_model": False if not incumbent.present else None,
        "recovery_idle": False,
        "proved": False,
    }


def _candidate_runtime_projection(
    admission: QualificationAdmission | None,
    observation: dict[str, Any] | None,
    removal: dict[str, Any] | None,
) -> dict[str, Any]:
    states: list[str] = []
    for source in (observation, removal):
        if not isinstance(source, dict):
            continue
        for state in source.get("states_observed", []):
            if (
                state
                in {
                    "REGISTERED",
                    "PROBING",
                    "READY",
                    "REJECTED",
                    "UNAVAILABLE",
                    "REMOVED",
                }
                and state not in states
            ):
                states.append(state)
    source = observation if isinstance(observation, dict) else {}
    return {
        "managed_relative_path": (
            admission.plan.managed_relative_path
            if admission is not None
            else None
        ),
        "registry_states_observed": states,
        "public_model_id": source.get("public_model_id"),
        "artifact_version_id": source.get("artifact_version_id"),
        "capability_manifest_identity": source.get(
            "capability_manifest_identity"
        ),
    }


def _incumbent_transaction_projection(
    incumbent: IncumbentSnapshot,
) -> dict[str, Any]:
    return {
        **incumbent.result_projection(),
        "registry_generation": incumbent.registry_generation,
        "credential_key_id": incumbent.credential_key_id,
        "profile_identity": incumbent.profile_identity,
        "service_readiness": incumbent.service_readiness,
        "recovery_state": incumbent.recovery_state,
        "api_service_transaction_id": (
            incumbent.api_service_transaction_id
        ),
        "router_transaction_id": incumbent.router_transaction_id,
        "model_child_identity": incumbent.model_child_identity,
    }


def _public_profile_adapter(
    *,
    service: ServiceSnapshot,
    credential: SecretCredential,
    observation: dict[str, Any],
) -> QualificationProbeAdapter:
    return PublicProfileProbeAdapter(
        service,
        credential,
        registry_states=observation["states_observed"],
        public_model_id=observation["public_model_id"],
        artifact_version_id=observation["artifact_version_id"],
        capability_manifest_identity=observation[
            "capability_manifest_identity"
        ],
    )


def _qualification_terminal_state(
    result_class: str,
) -> tuple[str, str]:
    if result_class == "REJECTED":
        return "REJECTED", "QUALIFICATION_REJECTED"
    if result_class == "QUALIFICATION_FAILED_CLEAN":
        return "FAILED_CLEAN", "QUALIFICATION_FAILED_CLEAN"
    if result_class == "QUALIFICATION_FAIL_CLOSED":
        return "FAIL_CLOSED", "QUALIFICATION_FAIL_CLOSED"
    return "COMPLETE", "QUALIFICATION_COMPLETE"


def qualify_transaction(
    paths: InspectorPaths,
    inspection_id: str,
    candidate_artifact_identity: str,
    required_capability_profile: str,
    *,
    transaction_id_factory: Callable[[], str] = _transaction_id,
    qualification_id_factory: Callable[[], str] = _qualification_id,
    authorization_factory: Callable[
        ..., QualificationAuthorization
    ] = authenticate_qualification,
    incumbent_factory: Callable[..., IncumbentSnapshot] = (
        capture_incumbent_snapshot
    ),
    admission_factory: Callable[..., QualificationAdmission] = (
        stage_qualification_candidate
    ),
    idempotence_finder: Callable[
        [InspectorPaths, QualificationAuthorization],
        tuple[dict[str, Any], Path] | None,
    ] = find_idempotent_qualification,
    registry_waiter: Callable[..., dict[str, Any]] = (
        wait_for_qualification_candidate
    ),
    service_factory: Callable[
        [QualificationAuthorization, IncumbentSnapshot],
        ServiceSnapshot,
    ] = qualification_service_snapshot,
    credential_reader: Callable[[Path], SecretCredential] = (
        read_local_credential
    ),
    profile_adapter_factory: Callable[
        ..., QualificationProbeAdapter
    ] = _public_profile_adapter,
    profile_runner: Callable[..., ProfileRun] = run_capability_profile,
    cleanup_factory: Callable[
        ..., tuple[dict[str, Any], dict[str, Any]]
    ] = cleanup_qualification_candidate,
    default_clearer: Callable[
        [Path, str, str, dict[str, Any], str],
        tuple[dict[str, Any], dict[str, Any]],
    ] = clear_qualification_default,
    restoration_factory: Callable[
        ..., tuple[dict[str, Any], dict[str, Any] | None]
    ] = prove_incumbent_restoration,
    record_builder: Callable[..., dict[str, Any]] = (
        build_qualification_record
    ),
    result_publisher: Callable[
        [InspectorPaths, dict[str, Any]], Path
    ] = publish_qualification_record,
    transition_observer: (
        Callable[[str, dict[str, Any]], None] | None
    ) = None,
) -> tuple[str, dict[str, Any], Path, str]:
    if (
        not isinstance(inspection_id, str)
        or INSPECTION_ID_PATTERN.fullmatch(inspection_id) is None
        or SHA256_PATTERN.fullmatch(candidate_artifact_identity) is None
        or required_capability_profile not in QUALIFICATION_PROFILES
    ):
        raise InspectorError(
            "QUALIFICATION_INPUT_INVALID",
            "qualification input is invalid",
        )
    transaction_id = transaction_id_factory()
    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="qualify-gguf",
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        reason = (
            "QUALIFICATION_CONCURRENCY_REJECTED"
            if error.reason_code == "TRANSACTION_LOCK_ACTIVE"
            else "QUALIFICATION_ACTIVE_TRANSACTION"
        )
        raise _qualification_error(
            reason,
            "qualification could not acquire exclusive transaction ownership",
            data={"transaction_id": transaction_id},
        ) from error

    start_utc = utc_now()
    owner_identity = {
        key: owner.get(key)
        for key in (
            "pid",
            "process_start_identity",
            "boot_identity",
            "inspector_root_identity",
        )
    }
    transaction = {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "qualify-gguf",
        "start_utc": start_utc,
        "finish_utc": None,
        "state": "VALIDATING_QUALIFICATION",
        "reason_code": "OK",
        "input_target_name": None,
        "intake_snapshot_identity": None,
        "owner_identity": owner_identity,
        "status_record_identity": None,
        "artifact_identity": candidate_artifact_identity,
        "inspection_id": inspection_id,
        "qualification_id": None,
        "candidate_artifact_identity": candidate_artifact_identity,
        "requested_profile": required_capability_profile,
        "incumbent_snapshot": None,
        "registry_states_observed": None,
        "qualification_result_identity": None,
        "qualification_result_path": None,
        "qualification_record_candidate": None,
        "result_class": None,
    }
    authorization: QualificationAuthorization | None = None
    incumbent: IncumbentSnapshot | None = None
    branch_mutation_started = False
    result_published = False
    cleanup_proved = False
    restoration_proved = False
    try:
        transaction = _inspection_stage(
            paths,
            transaction,
            state="VALIDATING_QUALIFICATION",
            reason_code="OK",
            observer=transition_observer,
        )
        authorization = authorization_factory(
            paths,
            inspection_id,
            candidate_artifact_identity,
            required_capability_profile,
            transaction_id=transaction_id,
        )
        transaction = _inspection_stage(
            paths,
            transaction,
            state="SNAPSHOTTING_INCUMBENT",
            reason_code="OK",
            observer=transition_observer,
            input_target_name=authorization.source.relative_name,
            intake_snapshot_identity=authorization.source.snapshot_identity,
        )
        incumbent = incumbent_factory(
            paths, authorization.branch_paths
        )
        incumbent_projection = _incumbent_transaction_projection(
            incumbent
        )
        transaction = {
            **transaction,
            "incumbent_snapshot": incumbent_projection,
        }
        _write_transaction(paths, transaction, transition_observer)

        duplicate = idempotence_finder(paths, authorization)
        if duplicate is not None:
            record, result_path = duplicate
            idle = _status_value(
                paths,
                state="IDLE",
                reason_code="OK",
                active_transaction_id=None,
                last_transaction_id=transaction_id,
            )
            idle_identity = _write_status(
                paths, idle, transition_observer
            )
            terminal = {
                **transaction,
                "finish_utc": utc_now(),
                "state": "COMPLETE",
                "reason_code": "QUALIFICATION_IDEMPOTENT_RESULT",
                "status_record_identity": idle_identity,
                "qualification_id": record["qualification_id"],
                "registry_states_observed": record[
                    "candidate_runtime"
                ]["registry_states_observed"],
                "qualification_result_identity": record[
                    "result_identity"
                ],
                "qualification_result_path": str(result_path),
                "result_class": record["result_class"],
            }
            _write_transaction(paths, terminal, transition_observer)
            return (
                transaction_id,
                record,
                result_path,
                record["result_identity"],
            )

        qualification_id = qualification_id_factory()
        if (
            authorization.decision_authorization.decision[
                "capability"
            ]["capability_result"]
            == "SUPPORTED"
        ):
            transaction = _inspection_stage(
                paths,
                transaction,
                state="QUALIFIED",
                reason_code="OK",
                observer=transition_observer,
                qualification_id=qualification_id,
            )
            profile_run = direct_supported_profile_run(authorization)
            incumbent_warm = (
                bool(
                    isinstance(incumbent.warm_before, dict)
                    and incumbent.warm_before.get("health_state")
                    == "ready"
                )
                if incumbent.present
                else None
            )
            restoration = {
                "required_state": (
                    "READY"
                    if incumbent.present
                    else "WAITING_FOR_MODEL"
                ),
                "default_unchanged": True,
                "incumbent_ready": (
                    incumbent.service_readiness == "READY"
                    if incumbent.present
                    else None
                ),
                "incumbent_warm": incumbent_warm,
                "waiting_for_model": (
                    incumbent.service_readiness
                    == "WAITING_FOR_MODEL"
                    if not incumbent.present
                    else None
                ),
                "recovery_idle": incumbent.recovery_state == "IDLE",
                "proved": bool(
                    incumbent.recovery_state == "IDLE"
                    and (
                        (
                            incumbent.present
                            and incumbent.service_readiness == "READY"
                            and incumbent_warm is True
                        )
                        or (
                            not incumbent.present
                            and incumbent.service_readiness
                            == "WAITING_FOR_MODEL"
                        )
                    )
                ),
            }
            cleanup = {
                "staging_absent": True,
                "managed_target_absent": True,
                "registry_location_removed": None,
                "source_removed_if_packet_owned": None,
                "ownership_certain": True,
            }
            candidate_runtime = {
                "managed_relative_path": None,
                "registry_states_observed": [],
                "public_model_id": None,
                "artifact_version_id": None,
                "capability_manifest_identity": None,
            }
            record = record_builder(
                qualification_id=qualification_id,
                transaction_id=transaction_id,
                created_utc=start_utc,
                completed_utc=utc_now(),
                authorization=authorization,
                incumbent=incumbent,
                candidate_runtime=candidate_runtime,
                profile_run=profile_run,
                result_class=profile_run.result_class,
                reason_codes=profile_run.reason_codes,
                warm_after=incumbent.warm_before,
                restoration=restoration,
                cleanup=cleanup,
            )
            result_path = result_publisher(paths, record)
            result_published = True
            terminal_status = _status_value(
                paths,
                state="IDLE",
                reason_code="OK",
                active_transaction_id=None,
                last_transaction_id=transaction_id,
            )
            terminal_status_identity = _write_status(
                paths, terminal_status, transition_observer
            )
            terminal_transaction = {
                **transaction,
                "finish_utc": record["completed_utc"],
                "state": "COMPLETE",
                "reason_code": "QUALIFICATION_COMPLETE",
                "status_record_identity": terminal_status_identity,
                "registry_states_observed": [],
                "qualification_result_identity": record[
                    "result_identity"
                ],
                "qualification_result_path": str(result_path),
                "qualification_record_candidate": record,
                "result_class": profile_run.result_class,
            }
            _write_transaction(
                paths, terminal_transaction, transition_observer
            )
            return (
                transaction_id,
                record,
                result_path,
                record["result_identity"],
            )
        transaction = _inspection_stage(
            paths,
            transaction,
            state="STAGING_CANDIDATE",
            reason_code="OK",
            observer=transition_observer,
            qualification_id=qualification_id,
        )
        admission: QualificationAdmission | None = None
        observation: dict[str, Any] | None = None
        removal: dict[str, Any] | None = None
        operational_reason: str | None = None
        runtime_outcome = "UNAVAILABLE"
        profile_run = terminal_profile_run(
            required_capability_profile, runtime_outcome
        )
        try:
            branch_mutation_started = True
            admission = admission_factory(
                authorization,
                incumbent,
                transaction_id=transaction_id,
            )
        except InspectorError as error:
            operational_reason = (
                error.reason_code
                if error.reason_code in QUALIFICATION_REASON_CODES
                else "QUALIFICATION_INTERNAL_ERROR"
            )

        if admission is not None:
            transaction = _inspection_stage(
                paths,
                transaction,
                state="WAITING_FOR_REGISTRATION",
                reason_code="OK",
                observer=transition_observer,
                managed_name=admission.plan.managed_name,
                staging_relative_path=(
                    admission.plan.staging_relative_path
                ),
                managed_relative_path=(
                    admission.plan.managed_relative_path
                ),
                transfer_method=admission.staged.transfer_method,
                publication_device=admission.published.device,
                publication_inode=admission.published.inode,
                publication_size=admission.published.size_bytes,
                publication_sha256=admission.published.sha256,
                publication_mode=f"{admission.published.mode:04o}",
                publication_link_count=(
                    admission.published.link_count
                ),
            )
            try:
                observation = registry_waiter(
                    authorization.branch_paths.branch_root,
                    admission.plan.managed_name,
                    candidate_artifact_identity,
                )
            except InspectorError as error:
                operational_reason = (
                    error.reason_code
                    if error.reason_code in QUALIFICATION_REASON_CODES
                    else "QUALIFICATION_REGISTRY_UNAVAILABLE"
                )
            if observation is not None:
                transaction = {
                    **transaction,
                    "registry_states_observed": list(
                        observation.get("states_observed", [])
                    ),
                }
                _write_transaction(
                    paths, transaction, transition_observer
                )
                terminal_observation = observation.get("terminal")
                cold_default = qualification_owned_cold_default(
                    admission, incumbent, observation
                )
                if (
                    observation.get("default_bound") is True
                    and not cold_default
                ):
                    operational_reason = "QUALIFICATION_DEFAULT_CHANGED"
                    runtime_outcome = "UNAVAILABLE"
                elif terminal_observation == "READY":
                    runtime_outcome = "READY"
                    transaction = _inspection_stage(
                        paths,
                        transaction,
                        state="CANDIDATE_READY",
                        reason_code="QUALIFICATION_REGISTRY_READY",
                        observer=transition_observer,
                    )
                    transaction = _inspection_stage(
                        paths,
                        transaction,
                        state="RUNNING_PROFILE",
                        reason_code="OK",
                        observer=transition_observer,
                    )
                    try:
                        service = service_factory(
                            authorization, incumbent
                        )
                        credential = credential_reader(
                            authorization.branch_paths.branch_root
                        )
                        if credential.key_id != incumbent.credential_key_id:
                            raise _qualification_error(
                                "QUALIFICATION_AUTHENTICATION_REJECTED",
                                "credential identity changed during qualification",
                            )
                        adapter = profile_adapter_factory(
                            service=service,
                            credential=credential,
                            observation=observation,
                        )
                        profile_run = profile_runner(
                            adapter,
                            requested_profile=(
                                required_capability_profile
                            ),
                            model_id=observation["public_model_id"],
                            artifact_version_id=observation[
                                "artifact_version_id"
                            ],
                            capability_manifest_identity=observation[
                                "capability_manifest_identity"
                            ],
                        )
                    except InspectorError as error:
                        operational_reason = (
                            error.reason_code
                            if error.reason_code
                            in QUALIFICATION_REASON_CODES
                            else "QUALIFICATION_PROFILE_FAILED"
                        )
                        runtime_outcome = "UNAVAILABLE"
                        profile_run = terminal_profile_run(
                            required_capability_profile,
                            runtime_outcome,
                        )
                elif terminal_observation == "REJECTED":
                    runtime_outcome = "REJECTED"
                    operational_reason = (
                        "QUALIFICATION_REGISTRY_REJECTED"
                    )
                    profile_run = terminal_profile_run(
                        required_capability_profile, runtime_outcome
                    )
                elif terminal_observation == "UNSUPPORTED":
                    runtime_outcome = "UNSUPPORTED"
                    operational_reason = "QUALIFICATION_UNSUPPORTED"
                    profile_run = terminal_profile_run(
                        required_capability_profile, runtime_outcome
                    )
                else:
                    runtime_outcome = "UNAVAILABLE"
                    operational_reason = (
                        "QUALIFICATION_REGISTRATION_TIMEOUT"
                        if terminal_observation == "TIMEOUT"
                        else "QUALIFICATION_REGISTRY_UNAVAILABLE"
                    )
                    profile_run = terminal_profile_run(
                        required_capability_profile, runtime_outcome
                    )

        transaction = _inspection_stage(
            paths,
            transaction,
            state="CLEANING_CANDIDATE",
            reason_code="OK",
            observer=transition_observer,
        )
        if (
            admission is not None
            and observation is not None
            and observation.get("default_bound") is True
        ):
            try:
                _alias, observation = default_clearer(
                    authorization.branch_paths.branch_root,
                    admission.plan.managed_name,
                    admission.published.sha256,
                    observation,
                    transaction_id,
                )
            except InspectorError as error:
                operational_reason = (
                    error.reason_code
                    if error.reason_code in QUALIFICATION_REASON_CODES
                    else "QUALIFICATION_DEFAULT_CHANGED"
                )
        if admission is None:
            try:
                cleanup = _prepublication_cleanup(
                    authorization, transaction_id
                )
            except InspectorError as error:
                cleanup = {
                    "staging_absent": False,
                    "managed_target_absent": False,
                    "registry_location_removed": None,
                    "source_removed_if_packet_owned": None,
                    "ownership_certain": False,
                }
                operational_reason = error.reason_code
        elif (
            observation is None
            or observation.get("default_bound") is True
        ):
            cleanup = {
                "staging_absent": (
                    not admission.plan.staging_path.exists()
                    and not admission.plan.staging_path.is_symlink()
                ),
                "managed_target_absent": False,
                "registry_location_removed": None,
                "source_removed_if_packet_owned": None,
                "ownership_certain": False,
            }
        else:
            try:
                cleanup, removal = cleanup_factory(
                    admission, observation
                )
            except InspectorError as error:
                cleanup = {
                    "staging_absent": (
                        not admission.plan.staging_path.exists()
                        and not admission.plan.staging_path.is_symlink()
                    ),
                    "managed_target_absent": (
                        not admission.plan.managed_target.exists()
                        and not admission.plan.managed_target.is_symlink()
                    ),
                    "registry_location_removed": None,
                    "source_removed_if_packet_owned": None,
                    "ownership_certain": False,
                }
                operational_reason = (
                    error.reason_code
                    if error.reason_code in QUALIFICATION_REASON_CODES
                    else "QUALIFICATION_CLEANUP_FAILED"
                )
        cleanup_proved = bool(
            cleanup["staging_absent"]
            and cleanup["managed_target_absent"]
            and cleanup["registry_location_removed"] is not False
        )

        transaction = _inspection_stage(
            paths,
            transaction,
            state="RESTORING_INCUMBENT",
            reason_code="OK",
            observer=transition_observer,
        )
        try:
            restoration, warm_after = restoration_factory(
                authorization, incumbent
            )
        except InspectorError as error:
            restoration = _failed_restoration(incumbent)
            warm_after = None
            operational_reason = (
                error.reason_code
                if error.reason_code in QUALIFICATION_REASON_CODES
                else "QUALIFICATION_INCUMBENT_RESTORATION_FAILED"
            )
        restoration_proved = restoration["proved"] is True
        final_class, final_reasons = classify_qualification_result(
            required_capability_profile,
            profile_run.checks,
            runtime_outcome=runtime_outcome,
            cleanup_proved=cleanup_proved,
            restoration_proved=restoration_proved,
            ownership_certain=cleanup["ownership_certain"],
        )
        reasons = list(final_reasons)
        if (
            operational_reason in QUALIFICATION_REASON_CODES
            and operational_reason not in reasons
        ):
            reasons.append(operational_reason)
        candidate_runtime = _candidate_runtime_projection(
            admission, observation, removal
        )
        completed_utc = utc_now()
        record = record_builder(
            qualification_id=qualification_id,
            transaction_id=transaction_id,
            created_utc=start_utc,
            completed_utc=completed_utc,
            authorization=authorization,
            incumbent=incumbent,
            candidate_runtime=candidate_runtime,
            profile_run=profile_run,
            result_class=final_class,
            reason_codes=reasons,
            warm_after=warm_after,
            restoration=restoration,
            cleanup=cleanup,
        )
        result_path = result_publisher(paths, record)
        result_published = True
        terminal_state, terminal_reason = (
            _qualification_terminal_state(final_class)
        )
        safe_terminal = (
            cleanup_proved
            and restoration_proved
            and cleanup["ownership_certain"] is True
        )
        terminal_status = _status_value(
            paths,
            state="IDLE" if safe_terminal else "FAIL_CLOSED",
            reason_code=(
                "OK"
                if safe_terminal
                else "QUALIFICATION_FAIL_CLOSED"
            ),
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        terminal_status_identity = _write_status(
            paths, terminal_status, transition_observer
        )
        terminal_transaction = {
            **transaction,
            "finish_utc": completed_utc,
            "state": terminal_state,
            "reason_code": terminal_reason,
            "status_record_identity": terminal_status_identity,
            "registry_states_observed": candidate_runtime[
                "registry_states_observed"
            ],
            "qualification_result_identity": record[
                "result_identity"
            ],
            "qualification_result_path": str(result_path),
            "qualification_record_candidate": record,
            "result_class": final_class,
        }
        _write_transaction(
            paths, terminal_transaction, transition_observer
        )
        return (
            transaction_id,
            record,
            result_path,
            record["result_identity"],
        )
    except InspectorError as error:
        reason = (
            error.reason_code
            if error.reason_code in QUALIFICATION_REASON_CODES
            else "QUALIFICATION_INTERNAL_ERROR"
        )
        unsafe = branch_mutation_started and not (
            result_published
            and cleanup_proved
            and restoration_proved
        )
        status = _status_value(
            paths,
            state="FAIL_CLOSED" if unsafe else "FAILED",
            reason_code=(
                "QUALIFICATION_FAIL_CLOSED" if unsafe else reason
            ),
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        status_identity = _write_status(
            paths, status, transition_observer
        )
        failed_transaction = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "FAIL_CLOSED" if unsafe else "FAILED",
            "reason_code": (
                "QUALIFICATION_FAIL_CLOSED" if unsafe else reason
            ),
            "status_record_identity": status_identity,
        }
        _write_transaction(
            paths, failed_transaction, transition_observer
        )
        if not unsafe:
            idle = _status_value(
                paths,
                state="IDLE",
                reason_code="OK",
                active_transaction_id=None,
                last_transaction_id=transaction_id,
            )
            _write_status(paths, idle, transition_observer)
        error.data = {
            **error.data,
            "transaction_id": transaction_id,
        }
        raise
    except Exception as error:
        unsafe = branch_mutation_started and not (
            result_published
            and cleanup_proved
            and restoration_proved
        )
        reason = (
            "QUALIFICATION_FAIL_CLOSED"
            if unsafe
            else "QUALIFICATION_INTERNAL_ERROR"
        )
        status = _status_value(
            paths,
            state="FAIL_CLOSED" if unsafe else "FAILED",
            reason_code=reason,
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        status_identity = _write_status(
            paths, status, transition_observer
        )
        failed_transaction = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "FAIL_CLOSED" if unsafe else "FAILED",
            "reason_code": reason,
            "status_record_identity": status_identity,
        }
        _write_transaction(
            paths, failed_transaction, transition_observer
        )
        if not unsafe:
            idle = _status_value(
                paths,
                state="IDLE",
                reason_code="OK",
                active_transaction_id=None,
                last_transaction_id=transaction_id,
            )
            _write_status(paths, idle, transition_observer)
        raise _qualification_error(
            reason,
            "unexpected qualification runtime failure",
            data={"transaction_id": transaction_id},
        ) from error
    finally:
        lock.release()
