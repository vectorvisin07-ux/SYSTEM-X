"""Pure deterministic branch-decision resolution."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any, Callable

from .capabilities import (
    binding_path,
    load_capability_record,
    validate_binding,
    verify_installed_tuple,
)
from .classifier import NORMALIZED_FIELDS
from .constants import SCHEMA_IDENTITIES
from .errors import InspectorError
from .paths import InspectorPaths
from .records import atomic_create_json, canonical_json_bytes, read_json_record


INSPECTION_ID_PATTERN = re.compile(
    r"inspection-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
INSPECTION_FIELDS = frozenset(
    {
        "schema_version",
        "inspection_id",
        "transaction_id",
        "created_utc",
        "source",
        "artifact",
        "classification",
        "format",
        "normalized",
        "evidence",
        "warnings",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "candidate_name",
        "candidate_kind",
        "relative_path",
        "realpath",
        "device",
        "inode",
        "intake_snapshot_identity",
        "pre_inspection_snapshot_identity",
        "post_inspection_snapshot_identity",
    }
)
ARTIFACT_FIELDS = frozenset(
    {
        "identity",
        "algorithm",
        "byte_count",
        "file_count",
        "content_manifest_identity",
        "files",
        "file_manifest_truncated",
    }
)
CLASSIFICATION_FIELDS = frozenset(
    {
        "terminal_class",
        "detected_family",
        "inspection_confidence",
        "reason_codes",
    }
)
FORMAT_FIELDS = frozenset(
    {"definition_identity", "endianness", "gguf", "native", "version"}
)
INVALID_CLASS_REASON = {
    "UNKNOWN": "PHYSICAL_FORMAT_UNKNOWN",
    "CONTRADICTORY": "PHYSICAL_FORMAT_CONTRADICTORY",
    "CORRUPT": "PHYSICAL_FORMAT_CORRUPT",
    "INCOMPLETE": "PHYSICAL_FORMAT_INCOMPLETE",
}
DECISION_REASON_ORDER = (
    "DECISION_COMPLETE",
    "INSPECTION_RECORD_NOT_FOUND",
    "INSPECTION_RECORD_INVALID",
    "INSPECTION_RESULT_IDENTITY_MISMATCH",
    "INSPECTION_RECORD_CHANGED_DURING_DECISION",
    "CAPABILITY_BINDING_NOT_FOUND",
    "CAPABILITY_BINDING_INVALID",
    "CAPABILITY_BINDING_AMBIGUOUS",
    "CAPABILITY_RECORD_NOT_FOUND",
    "CAPABILITY_RECORD_INVALID",
    "CAPABILITY_RECORD_IDENTITY_MISMATCH",
    "CAPABILITY_INSTALLED_TUPLE_MISMATCH",
    "CAPABILITY_EVIDENCE_UNVERIFIED",
    "DECISION_RECORD_COLLISION",
    "GGUF_ACCEPTED_CAPABILITY_MATCH",
    "GGUF_RUNTIME_SMOKE_REQUIRED",
    "GGUF_REQUIREMENT_UNSUPPORTED",
    "NATIVE_BRANCH_UNAVAILABLE",
    "PHYSICAL_FORMAT_UNKNOWN",
    "PHYSICAL_FORMAT_CONTRADICTORY",
    "PHYSICAL_FORMAT_CORRUPT",
    "PHYSICAL_FORMAT_INCOMPLETE",
    "EXTENSION_ONLY_EVIDENCE_IGNORED",
)
_REASON_RANK = {
    reason: index for index, reason in enumerate(DECISION_REASON_ORDER)
}
DECISION_ID_PATTERN = re.compile(
    r"decision-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "decision_id",
        "transaction_id",
        "decision_timestamp_utc",
        "decision_basis_identity",
        "result_identity",
        "inspection",
        "capability",
        "selected_branch",
        "handoff_allowed",
        "spawn_allowed",
        "reason_code",
        "reason_codes",
        "warnings",
    }
)
DECISION_INSPECTION_FIELDS = frozenset(
    {
        "inspection_id",
        "inspection_result_identity",
        "artifact_identity",
        "physical_format",
        "source_target_name",
    }
)
DECISION_CAPABILITY_FIELDS = frozenset(
    {
        "branch_identity",
        "binding_identity",
        "capability_record_id",
        "capability_record_identity",
        "capability_result",
        "evaluated",
    }
)
FORBIDDEN_DECISION_KEYS = frozenset(
    {
        "api_key",
        "credential_path",
        "private_router_url",
        "model_child_port",
        "pid",
        "process_start_identity",
        "physical_model_path",
        "launch_argv",
    }
)


def _error(reason: str, message: str) -> InspectorError:
    return InspectorError(reason, message)


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ordered_decision_reasons(*reasons: str) -> list[str]:
    return sorted(
        set(reasons),
        key=lambda reason: (_REASON_RANK.get(reason, len(_REASON_RANK)), reason),
    )


def _exact(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _error(
            "INSPECTION_RECORD_INVALID", f"{label} fields are not closed"
        )
    return value


def _private_file(path: Path, reason: str) -> bytes:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _error(reason, f"record is absent: {path.name}") from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
    ):
        raise _error(reason, f"record is physically unsafe: {path.name}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise _error(reason, f"record cannot be read: {path.name}") from error


def _inspection_path(paths: InspectorPaths, inspection_id: str) -> Path:
    if (
        not isinstance(inspection_id, str)
        or INSPECTION_ID_PATTERN.fullmatch(inspection_id) is None
        or "/" in inspection_id
        or "\\" in inspection_id
    ):
        raise _error(
            "INSPECTION_RECORD_INVALID", "inspection ID is not canonical"
        )
    return paths.inspection_results / f"{inspection_id}.json"


def _validate_inspection(value: object, inspection_id: str) -> dict[str, Any]:
    record = _exact(value, INSPECTION_FIELDS, "inspection record")
    if record["schema_version"] != SCHEMA_IDENTITIES["inspection_result"]:
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "inspection schema identity is invalid",
        )
    if record["inspection_id"] != inspection_id:
        raise _error(
            "INSPECTION_RECORD_INVALID", "inspection ID does not match"
        )
    if not isinstance(record["transaction_id"], str):
        raise _error(
            "INSPECTION_RECORD_INVALID", "inspection transaction is invalid"
        )
    source = _exact(record["source"], SOURCE_FIELDS, "inspection source")
    artifact = _exact(
        record["artifact"], ARTIFACT_FIELDS, "inspection artifact"
    )
    classification = _exact(
        record["classification"],
        CLASSIFICATION_FIELDS,
        "inspection classification",
    )
    format_value = _exact(
        record["format"], FORMAT_FIELDS, "inspection format"
    )
    normalized = _exact(
        record["normalized"],
        frozenset(NORMALIZED_FIELDS),
        "inspection normalized surface",
    )
    if (
        not isinstance(record["evidence"], list)
        or not isinstance(record["warnings"], list)
        or not isinstance(classification["reason_codes"], list)
    ):
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "inspection lists have invalid types",
        )
    if (
        source["candidate_name"] != source["relative_path"]
        or source["pre_inspection_snapshot_identity"]
        != source["post_inspection_snapshot_identity"]
    ):
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "inspection immutable source references do not agree",
        )
    artifact_identity = artifact["identity"]
    if (
        not isinstance(artifact_identity, str)
        or SHA256_PATTERN.fullmatch(artifact_identity) is None
        or normalized["artifact_identity"] != artifact_identity
        or artifact["algorithm"] != "sha256"
    ):
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "inspection artifact identities do not agree",
        )
    terminal = classification["terminal_class"]
    if terminal not in {
        "GGUF",
        "NATIVE",
        "UNKNOWN",
        "CONTRADICTORY",
        "CORRUPT",
        "INCOMPLETE",
    }:
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "inspection terminal class is invalid",
        )
    physical = normalized["physical_format"]
    if terminal == "GGUF" and physical != "GGUF":
        raise _error(
            "INSPECTION_RECORD_INVALID", "GGUF physical evidence conflicts"
        )
    if terminal == "NATIVE" and physical != "NATIVE":
        raise _error(
            "INSPECTION_RECORD_INVALID", "native physical evidence conflicts"
        )
    if terminal == "UNKNOWN" and physical != "unknown":
        raise _error(
            "INSPECTION_RECORD_INVALID", "unknown physical evidence conflicts"
        )
    if terminal == "CONTRADICTORY" and physical != "contradictory":
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "contradictory physical evidence conflicts",
        )
    if terminal == "GGUF":
        if (
            not isinstance(format_value["gguf"], dict)
            or format_value["native"] is not None
        ):
            raise _error(
                "INSPECTION_RECORD_INVALID", "GGUF format summary is invalid"
            )
    if terminal == "NATIVE":
        if (
            not isinstance(format_value["native"], dict)
            or format_value["gguf"] is not None
        ):
            raise _error(
                "INSPECTION_RECORD_INVALID",
                "native format summary is invalid",
            )
    return record


def load_inspection_result(
    paths: InspectorPaths, inspection_id: str
) -> tuple[dict[str, Any], str]:
    path = _inspection_path(paths, inspection_id)
    before = _private_file(path, "INSPECTION_RECORD_NOT_FOUND")
    observed_identity = "sha256:" + hashlib.sha256(before).hexdigest()
    try:
        value = json.loads(before)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _error(
            "INSPECTION_RECORD_INVALID", "inspection JSON is invalid"
        ) from error
    record = _validate_inspection(value, inspection_id)
    transaction_id = record["transaction_id"]
    if (
        not isinstance(transaction_id, str)
        or Path(transaction_id).name != transaction_id
    ):
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "inspection transaction reference is unsafe",
        )
    transaction_path = paths.transactions / f"{transaction_id}.json"
    transaction_raw = _private_file(
        transaction_path, "INSPECTION_RECORD_INVALID"
    )
    try:
        transaction = json.loads(transaction_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _error(
            "INSPECTION_RECORD_INVALID",
            "inspection transaction JSON is invalid",
        ) from error
    if (
        not isinstance(transaction, dict)
        or transaction.get("schema_version") != SCHEMA_IDENTITIES["transaction"]
        or transaction.get("transaction_id") != transaction_id
        or transaction.get("operation") != "inspect"
        or transaction.get("state") != "COMPLETED"
        or transaction.get("reason_code") != "INSPECTION_COMPLETE"
        or transaction.get("inspection_result_identity") != observed_identity
        or transaction.get("inspection_result_path") != str(path)
        or transaction.get("artifact_identity")
        != record["artifact"]["identity"]
        or transaction.get("terminal_class")
        != record["classification"]["terminal_class"]
    ):
        raise _error(
            "INSPECTION_RESULT_IDENTITY_MISMATCH",
            "inspection transaction does not authenticate the result",
        )
    after = _private_file(path, "INSPECTION_RECORD_NOT_FOUND")
    if after != before:
        raise _error(
            "INSPECTION_RECORD_CHANGED_DURING_DECISION",
            "inspection record changed during decision",
        )
    return record, observed_identity


def _binding_candidates(
    paths: InspectorPaths, branch_identity: str
) -> list[dict[str, Any]]:
    root = paths.capability_bindings
    try:
        details = root.lstat()
    except FileNotFoundError:
        return []
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise _error(
            "CAPABILITY_BINDING_INVALID",
            "capability binding root is unsafe",
        )
    candidates = []
    for path in sorted(root.iterdir()):
        if path.suffix != ".json":
            raise _error(
                "CAPABILITY_BINDING_INVALID",
                "unknown binding-root content is present",
            )
        raw = _private_file(path, "CAPABILITY_BINDING_INVALID")
        try:
            value = validate_binding(json.loads(raw))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            InspectorError,
        ) as error:
            raise _error(
                "CAPABILITY_BINDING_INVALID",
                f"capability binding is invalid: {path.name}",
            ) from error
        if value["branch_identity"] == branch_identity:
            candidates.append({"path": path, "value": value})
    return candidates


def _capability_section(
    *,
    branch_identity: str | None,
    binding_identity: str | None,
    capability_record_id: str | None,
    capability_record_identity: str | None,
    capability_result: str | None,
    evaluated: bool,
) -> dict[str, Any]:
    return {
        "branch_identity": branch_identity,
        "binding_identity": binding_identity,
        "capability_record_id": capability_record_id,
        "capability_record_identity": capability_record_identity,
        "capability_result": capability_result,
        "evaluated": evaluated,
    }


def _basis(outcome: dict[str, Any]) -> str:
    return _identity(
        {
            "inspection_result_identity": outcome["inspection"][
                "inspection_result_identity"
            ],
            "artifact_identity": outcome["inspection"]["artifact_identity"],
            "physical_format": outcome["inspection"]["physical_format"],
            "binding_identity": outcome["capability"]["binding_identity"],
            "capability_record_identity": outcome["capability"][
                "capability_record_identity"
            ],
            "capability_result": outcome["capability"]["capability_result"],
            "selected_branch": outcome["selected_branch"],
            "handoff_allowed": outcome["handoff_allowed"],
            "spawn_allowed": outcome["spawn_allowed"],
            "reason_codes": outcome["reason_codes"],
        }
    )


def _outcome(
    *,
    inspection: dict[str, Any],
    inspection_identity: str,
    capability: dict[str, Any],
    selected_branch: str | None,
    handoff_allowed: bool,
    spawn_allowed: bool,
    reason_code: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    result = {
        "inspection": {
            "inspection_id": inspection["inspection_id"],
            "inspection_result_identity": inspection_identity,
            "artifact_identity": inspection["artifact"]["identity"],
            "physical_format": inspection["classification"]["terminal_class"],
            "source_target_name": inspection["source"]["candidate_name"],
        },
        "capability": capability,
        "selected_branch": selected_branch,
        "handoff_allowed": handoff_allowed,
        "spawn_allowed": spawn_allowed,
        "reason_code": reason_code,
        "reason_codes": reason_codes,
        "warnings": [],
    }
    result["decision_basis_identity"] = _basis(result)
    return result


def _profile_agrees(
    inspection: dict[str, Any], capability: dict[str, Any]
) -> bool:
    supported = capability["supported_evidence"]
    normalized = inspection["normalized"]
    gguf = inspection["format"]["gguf"]
    if not isinstance(gguf, dict):
        return False
    return all(
        (
            inspection["format"]["version"]
            in supported["accepted_format_versions"],
            all(
                item in supported["accepted_architectures"]
                for item in normalized["architectures"]
            ),
            normalized["model_type"]
            in supported["accepted_primary_model_types"],
            all(
                item in supported["accepted_modalities"]
                for item in normalized["modalities"]
            ),
            sorted(gguf.get("tensor_type_histogram", {}))
            == sorted(supported["accepted_tensor_type_evidence"]),
            gguf.get("tokenizer_token_identity")
            == supported["accepted_tokenizer_evidence"],
            gguf.get("chat_template_identity")
            == supported["accepted_chat_template_evidence"],
        )
    )


def resolve_decision(
    paths: InspectorPaths,
    inspection_id: str,
    *,
    installed_tuple_verifier: Callable[
        [InspectorPaths, dict[str, Any]], dict[str, Any]
    ] = verify_installed_tuple,
) -> dict[str, Any]:
    inspection, inspection_identity = load_inspection_result(
        paths, inspection_id
    )
    terminal = inspection["classification"]["terminal_class"]
    if terminal in INVALID_CLASS_REASON:
        reasons = [
            "DECISION_COMPLETE",
            INVALID_CLASS_REASON[terminal],
        ]
        if "MISLEADING_EXTENSION_IGNORED" in inspection["classification"][
            "reason_codes"
        ]:
            reasons.append("EXTENSION_ONLY_EVIDENCE_IGNORED")
        reasons = ordered_decision_reasons(*reasons)
        return _outcome(
            inspection=inspection,
            inspection_identity=inspection_identity,
            capability=_capability_section(
                branch_identity=None,
                binding_identity=None,
                capability_record_id=None,
                capability_record_identity=None,
                capability_result=None,
                evaluated=False,
            ),
            selected_branch=None,
            handoff_allowed=False,
            spawn_allowed=False,
            reason_code=INVALID_CLASS_REASON[terminal],
            reason_codes=reasons,
        )
    branch_identity = (
        "model-api-gguf" if terminal == "GGUF" else "model-api-native"
    )
    candidates = _binding_candidates(paths, branch_identity)
    if len(candidates) > 1:
        reason = "CAPABILITY_BINDING_AMBIGUOUS"
        return _outcome(
            inspection=inspection,
            inspection_identity=inspection_identity,
            capability=_capability_section(
                branch_identity=branch_identity,
                binding_identity=None,
                capability_record_id=None,
                capability_record_identity=None,
                capability_result="AMBIGUOUS",
                evaluated=True,
            ),
            selected_branch=None,
            handoff_allowed=False,
            spawn_allowed=False,
            reason_code=reason,
            reason_codes=ordered_decision_reasons(
                "DECISION_COMPLETE", reason
            ),
        )
    if not candidates:
        reason = "CAPABILITY_BINDING_NOT_FOUND"
        return _outcome(
            inspection=inspection,
            inspection_identity=inspection_identity,
            capability=_capability_section(
                branch_identity=branch_identity,
                binding_identity=None,
                capability_record_id=None,
                capability_record_identity=None,
                capability_result="UNAVAILABLE",
                evaluated=True,
            ),
            selected_branch=None,
            handoff_allowed=False,
            spawn_allowed=False,
            reason_code=reason,
            reason_codes=ordered_decision_reasons(
                "DECISION_COMPLETE", reason
            ),
        )
    candidate = candidates[0]
    binding = candidate["value"]
    if candidate["path"] != binding_path(paths, branch_identity):
        raise _error(
            "CAPABILITY_BINDING_INVALID",
            "sole current binding has a non-canonical filename",
        )
    record = load_capability_record(
        paths, binding["capability_record_id"]
    )
    if (
        record["capability_record_identity"]
        != binding["capability_record_identity"]
        or record["branch_identity"] != binding["branch_identity"]
    ):
        raise _error(
            "CAPABILITY_RECORD_IDENTITY_MISMATCH",
            "binding does not authenticate its capability record",
        )
    common = {
        "branch_identity": branch_identity,
        "binding_identity": binding["binding_identity"],
        "capability_record_id": record["capability_record_id"],
        "capability_record_identity": record["capability_record_identity"],
        "evaluated": True,
    }
    if record["availability"] == "UNAVAILABLE":
        reason = "NATIVE_BRANCH_UNAVAILABLE"
        return _outcome(
            inspection=inspection,
            inspection_identity=inspection_identity,
            capability=_capability_section(
                **common, capability_result="UNAVAILABLE"
            ),
            selected_branch=None,
            handoff_allowed=False,
            spawn_allowed=False,
            reason_code=reason,
            reason_codes=ordered_decision_reasons(
                "DECISION_COMPLETE", reason
            ),
        )
    verification = installed_tuple_verifier(paths, record)
    if not verification.get("verified"):
        reason = "CAPABILITY_INSTALLED_TUPLE_MISMATCH"
        return _outcome(
            inspection=inspection,
            inspection_identity=inspection_identity,
            capability=_capability_section(
                **common, capability_result="AMBIGUOUS"
            ),
            selected_branch=None,
            handoff_allowed=False,
            spawn_allowed=False,
            reason_code=reason,
            reason_codes=ordered_decision_reasons(
                "DECISION_COMPLETE", reason
            ),
        )
    artifact_identity = inspection["artifact"]["identity"]
    supported = record["supported_evidence"]
    if artifact_identity in supported["supported_exact_artifact_identities"]:
        if not _profile_agrees(inspection, record):
            reason = "CAPABILITY_EVIDENCE_UNVERIFIED"
            return _outcome(
                inspection=inspection,
                inspection_identity=inspection_identity,
                capability=_capability_section(
                    **common, capability_result="AMBIGUOUS"
                ),
                selected_branch=None,
                handoff_allowed=False,
                spawn_allowed=False,
                reason_code=reason,
                reason_codes=ordered_decision_reasons(
                    "DECISION_COMPLETE", reason
                ),
            )
        reason = "GGUF_ACCEPTED_CAPABILITY_MATCH"
        return _outcome(
            inspection=inspection,
            inspection_identity=inspection_identity,
            capability=_capability_section(
                **common, capability_result="SUPPORTED"
            ),
            selected_branch="model-api-gguf",
            handoff_allowed=True,
            spawn_allowed=True,
            reason_code=reason,
            reason_codes=ordered_decision_reasons(
                "DECISION_COMPLETE", reason
            ),
        )
    if (
        inspection["normalized"]["model_type"]
        in record["unsupported_primary_artifact_roles"]
    ):
        reason = "GGUF_REQUIREMENT_UNSUPPORTED"
        return _outcome(
            inspection=inspection,
            inspection_identity=inspection_identity,
            capability=_capability_section(
                **common, capability_result="UNSUPPORTED"
            ),
            selected_branch=None,
            handoff_allowed=False,
            spawn_allowed=False,
            reason_code=reason,
            reason_codes=ordered_decision_reasons(
                "DECISION_COMPLETE", reason
            ),
        )
    reason = "GGUF_RUNTIME_SMOKE_REQUIRED"
    return _outcome(
        inspection=inspection,
        inspection_identity=inspection_identity,
        capability=_capability_section(
            **common, capability_result="RUNTIME_SMOKE_REQUIRED"
        ),
        selected_branch=None,
        handoff_allowed=False,
        spawn_allowed=False,
        reason_code=reason,
        reason_codes=ordered_decision_reasons("DECISION_COMPLETE", reason),
    )


def decision_result_identity(value: dict[str, Any]) -> str:
    try:
        basis = {key: value[key] for key in DECISION_FIELDS if key != "result_identity"}
    except KeyError as error:
        raise _error(
            "DECISION_RECORD_COLLISION",
            f"decision result field is missing: {error.args[0]}",
        ) from error
    return _identity(basis)


def _reject_decision_secrets(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_DECISION_KEYS:
                raise _error(
                    "DECISION_RECORD_COLLISION",
                    f"decision contains prohibited field: {key}",
                )
            _reject_decision_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _reject_decision_secrets(child)


def build_decision_record(
    outcome: dict[str, Any],
    *,
    decision_id: str,
    transaction_id: str,
    decision_timestamp_utc: str,
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_IDENTITIES["branch_decision"],
        "decision_id": decision_id,
        "transaction_id": transaction_id,
        "decision_timestamp_utc": decision_timestamp_utc,
        "decision_basis_identity": outcome["decision_basis_identity"],
        "result_identity": None,
        "inspection": outcome["inspection"],
        "capability": outcome["capability"],
        "selected_branch": outcome["selected_branch"],
        "handoff_allowed": outcome["handoff_allowed"],
        "spawn_allowed": outcome["spawn_allowed"],
        "reason_code": outcome["reason_code"],
        "reason_codes": outcome["reason_codes"],
        "warnings": outcome["warnings"],
    }
    value["result_identity"] = decision_result_identity(value)
    return validate_decision_record(value)


def validate_decision_record(value: object) -> dict[str, Any]:
    record = (
        value
        if isinstance(value, dict) and set(value) == DECISION_FIELDS
        else None
    )
    if record is None:
        raise _error(
            "DECISION_RECORD_COLLISION", "decision fields are not closed"
        )
    _reject_decision_secrets(record)
    if record["schema_version"] != SCHEMA_IDENTITIES["branch_decision"]:
        raise _error(
            "DECISION_RECORD_COLLISION", "decision schema is invalid"
        )
    if (
        not isinstance(record["decision_id"], str)
        or DECISION_ID_PATTERN.fullmatch(record["decision_id"]) is None
    ):
        raise _error(
            "DECISION_RECORD_COLLISION", "decision ID is invalid"
        )
    if (
        not isinstance(record["transaction_id"], str)
        or not record["transaction_id"]
        or not isinstance(record["decision_timestamp_utc"], str)
        or not record["decision_timestamp_utc"]
    ):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision transaction or time is invalid",
        )
    for key in ("decision_basis_identity", "result_identity"):
        if (
            not isinstance(record[key], str)
            or SHA256_PATTERN.fullmatch(record[key]) is None
        ):
            raise _error(
                "DECISION_RECORD_COLLISION",
                f"decision {key} is invalid",
            )
    inspection = record["inspection"]
    capability = record["capability"]
    if (
        not isinstance(inspection, dict)
        or set(inspection) != DECISION_INSPECTION_FIELDS
        or not isinstance(capability, dict)
        or set(capability) != DECISION_CAPABILITY_FIELDS
    ):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision nested fields are not closed",
        )
    for key in ("inspection_result_identity", "artifact_identity"):
        if (
            not isinstance(inspection[key], str)
            or SHA256_PATTERN.fullmatch(inspection[key]) is None
        ):
            raise _error(
                "DECISION_RECORD_COLLISION",
                f"decision inspection {key} is invalid",
            )
    if (
        not isinstance(inspection["inspection_id"], str)
        or INSPECTION_ID_PATTERN.fullmatch(inspection["inspection_id"]) is None
        or inspection["physical_format"]
        not in {
            "GGUF",
            "NATIVE",
            "UNKNOWN",
            "CONTRADICTORY",
            "CORRUPT",
            "INCOMPLETE",
        }
        or not isinstance(inspection["source_target_name"], str)
    ):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision inspection surface is invalid",
        )
    if not isinstance(capability["evaluated"], bool):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision capability evaluation flag is invalid",
        )
    if capability["capability_result"] not in {
        "SUPPORTED",
        "RUNTIME_SMOKE_REQUIRED",
        "UNSUPPORTED",
        "UNAVAILABLE",
        "AMBIGUOUS",
        None,
    }:
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision capability result is invalid",
        )
    for key in ("binding_identity", "capability_record_identity"):
        if capability[key] is not None and (
            not isinstance(capability[key], str)
            or SHA256_PATTERN.fullmatch(capability[key]) is None
        ):
            raise _error(
                "DECISION_RECORD_COLLISION",
                f"decision capability {key} is invalid",
            )
    if capability["branch_identity"] not in {
        "model-api-gguf",
        "model-api-native",
        None,
    }:
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision capability branch is invalid",
        )
    if capability["capability_record_id"] is not None and (
        not isinstance(capability["capability_record_id"], str)
        or not capability["capability_record_id"].startswith("capability-")
    ):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision capability record ID is invalid",
        )
    if (
        not isinstance(record["handoff_allowed"], bool)
        or not isinstance(record["spawn_allowed"], bool)
        or record["selected_branch"]
        not in {"model-api-gguf", "model-api-native", None}
        or not isinstance(record["reason_code"], str)
        or not isinstance(record["reason_codes"], list)
        or not isinstance(record["warnings"], list)
        or any(not isinstance(item, str) for item in record["reason_codes"])
        or any(not isinstance(item, str) for item in record["warnings"])
        or record["reason_codes"]
        != ordered_decision_reasons(*record["reason_codes"])
    ):
        raise _error(
            "DECISION_RECORD_COLLISION", "decision outcome is invalid"
        )
    result = capability["capability_result"]
    if result == "SUPPORTED":
        if (
            record["selected_branch"] != "model-api-gguf"
            or record["handoff_allowed"] is not True
            or record["spawn_allowed"] is not True
        ):
            raise _error(
                "DECISION_RECORD_COLLISION",
                "supported decision authorization is invalid",
            )
    elif (
        record["selected_branch"] is not None
        or record["handoff_allowed"] is not False
        or record["spawn_allowed"] is not False
    ):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "no-spawn decision authorization is invalid",
        )
    if not capability["evaluated"] and any(
        capability[key] is not None
        for key in (
            "branch_identity",
            "binding_identity",
            "capability_record_id",
            "capability_record_identity",
            "capability_result",
        )
    ):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "unevaluated capability fields must be null",
        )
    expected_basis = _basis(
        {
            "inspection": inspection,
            "capability": capability,
            "selected_branch": record["selected_branch"],
            "handoff_allowed": record["handoff_allowed"],
            "spawn_allowed": record["spawn_allowed"],
            "reason_codes": record["reason_codes"],
        }
    )
    if record["decision_basis_identity"] != expected_basis:
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision basis identity is invalid",
        )
    if record["result_identity"] != decision_result_identity(record):
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision result identity is invalid",
        )
    return record


def publish_decision_record(
    paths: InspectorPaths, value: dict[str, Any]
) -> Path:
    record = validate_decision_record(value)
    path = paths.decision_results / f"{record['decision_id']}.json"
    if path.parent != paths.decision_results:
        raise _error(
            "DECISION_RECORD_COLLISION", "decision path escaped its root"
        )
    if path.exists() or path.is_symlink():
        existing = _private_file(path, "DECISION_RECORD_COLLISION")
        if existing == canonical_json_bytes(record):
            return path
        raise _error(
            "DECISION_RECORD_COLLISION",
            "different immutable decision record already exists",
        )
    try:
        atomic_create_json(path, record, mode=0o600)
    except InspectorError as error:
        if path.exists() and not path.is_symlink():
            existing = _private_file(path, "DECISION_RECORD_COLLISION")
            if existing == canonical_json_bytes(record):
                return path
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision record atomic publication collided",
        ) from error
    if validate_decision_record(read_json_record(path)) != record:
        raise _error(
            "DECISION_RECORD_COLLISION",
            "decision record did not round-trip",
        )
    return path
