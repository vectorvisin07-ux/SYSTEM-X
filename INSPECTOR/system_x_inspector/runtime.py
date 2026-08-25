"""Foundation layout, status, and stateful intake transactions."""

from __future__ import annotations

import datetime as dt
import secrets
from pathlib import Path
from typing import Any, Callable

from .classifier import ClassificationResult, classify_artifact
from .config import ValidatedConfiguration
from .content_identity import (
    ArtifactIdentity,
    artifact_source_snapshot,
    identify_artifact,
)
from .constants import SCHEMA_IDENTITIES
from .errors import InspectorError
from .intake import validate_intake
from .locking import TransactionLock, inspect_active_lock
from .paths import InspectorPaths, physical_state
from .records import (
    atomic_create_json,
    atomic_write_json,
    read_json_record,
    record_identity,
)
from .results import utc_now
from .decision import (
    build_decision_record,
    publish_decision_record,
    resolve_decision,
)
from .gguf import FORMAT_DEFINITION_IDENTITY as GGUF_DEFINITION_IDENTITY
from .native import FORMAT_DEFINITION_IDENTITY as NATIVE_DEFINITION_IDENTITY


def layout_report(paths: InspectorPaths) -> dict[str, Any]:
    context_roots = {"source_root", "user_config_root"}
    states = {
        key: physical_state(
            path,
            path.parent if key in context_roots else paths.inspector_root,
        )
        for key, path in paths.as_mapping().items()
    }
    expected_files = {"environment_lock"}
    valid = True
    for key, state in states.items():
        if key == "source_root":
            expected = "regular_directory"
            matches = state["state"] == expected
        elif key == "user_config_root":
            expected = "regular_directory_or_absent"
            matches = state["state"] in {"regular_directory", "absent"}
        elif key in expected_files:
            expected = "regular_file"
            matches = state["state"] == expected
        else:
            expected = "regular_directory"
            matches = state["state"] == expected
        state["expected_state"] = expected
        state["matches_expected"] = matches
        valid = valid and matches
    if not valid:
        raise InspectorError(
            "LAYOUT_INVALID",
            "Inspector foundation layout is incomplete or unsafe",
            data={"layout": states},
        )
    return {"layout_valid": True, "entries": states}


def status_report(paths: InspectorPaths) -> dict[str, Any]:
    current_path = paths.status / "current.json"
    if not current_path.exists():
        raise InspectorError(
            "INSPECTOR_NOT_CONFIGURED",
            "current Inspector status record does not exist",
        )
    current = read_json_record(current_path)
    result_readiness = {
        name: {
            "path": str(path),
            "ready": (
                path.is_dir()
                and not path.is_symlink()
                and path.resolve(strict=True).is_relative_to(
                    paths.inspector_root
                )
            ),
            "entry_count": (
                sum(1 for _ in path.iterdir())
                if path.is_dir() and not path.is_symlink()
                else None
            ),
        }
        for name, path in (
            ("inspection", paths.inspection_results),
            ("decision", paths.decision_results),
            ("handoff", paths.handoff_results),
            ("publication", paths.publication_results),
            ("qualification", paths.qualification_results),
            ("promotion", paths.promotion_results),
            ("retirement", paths.retirement_results),
            ("deployment", paths.deployment_results),
        )
    }
    return {
        "current": current,
        "current_status_identity": record_identity(current_path),
        "active_lock": inspect_active_lock(paths.locks / "active.json"),
        "last_transaction_reference": current.get("last_transaction_id"),
        "result_root_readiness": result_readiness,
    }


def _transaction_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"tx-{stamp}-{secrets.token_hex(6)}"


def _decision_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"decision-{stamp}-{secrets.token_hex(8)}"


def _status_value(
    paths: InspectorPaths,
    *,
    state: str,
    reason_code: str,
    active_transaction_id: str | None,
    last_transaction_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_IDENTITIES["status"],
        "state": state,
        "reason_code": reason_code,
        "updated_utc": utc_now(),
        "inspector_root": str(paths.inspector_root),
        "active_transaction_id": active_transaction_id,
        "last_transaction_id": last_transaction_id,
    }


def _write_status(
    paths: InspectorPaths,
    value: dict[str, Any],
    observer: Callable[[str, dict[str, Any]], None] | None,
) -> str:
    identity = atomic_write_json(
        paths.status / "current.json", value, mode=0o600
    )
    observed = read_json_record(paths.status / "current.json")
    if observed != value:
        raise RuntimeError("status record did not round-trip atomically")
    if observer is not None:
        observer("status", observed)
    return identity


def _write_transaction(
    paths: InspectorPaths,
    value: dict[str, Any],
    observer: Callable[[str, dict[str, Any]], None] | None,
) -> str:
    path = paths.transactions / f"{value['transaction_id']}.json"
    identity = atomic_write_json(path, value, mode=0o600)
    observed = read_json_record(path)
    if observed != value:
        raise RuntimeError("transaction record did not round-trip atomically")
    if observer is not None:
        observer("transaction", observed)
    return identity


def validate_intake_transaction(
    paths: InspectorPaths,
    configuration: ValidatedConfiguration,
    target_name: str | None,
    *,
    validator: Callable[..., dict[str, Any]] = validate_intake,
    transition_observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    transaction_id = _transaction_id()
    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="validate-intake",
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
    try:
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
        start_status = _status_value(
            paths,
            state="VALIDATING_INTAKE",
            reason_code="OK",
            active_transaction_id=transaction_id,
            last_transaction_id=None,
        )
        start_status_identity = _write_status(
            paths, start_status, transition_observer
        )
        start_transaction = {
            "schema_version": SCHEMA_IDENTITIES["transaction"],
            "transaction_id": transaction_id,
            "operation": "validate-intake",
            "start_utc": start_utc,
            "finish_utc": None,
            "state": "VALIDATING_INTAKE",
            "reason_code": "OK",
            "input_target_name": target_name,
            "intake_snapshot_identity": None,
            "owner_identity": owner_identity,
            "status_record_identity": start_status_identity,
        }
        _write_transaction(paths, start_transaction, transition_observer)
    except Exception:
        lock.release()
        raise
    try:
        candidate = validator(
            paths,
            configuration.values["intake_bounds"],
            target_name,
        )
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        idle_identity = _write_status(paths, idle, transition_observer)
        terminal = {
            **start_transaction,
            "finish_utc": utc_now(),
            "state": "COMPLETED",
            "reason_code": "OK",
            "input_target_name": candidate["target_name"],
            "intake_snapshot_identity": candidate[
                "intake_snapshot_identity"
            ],
            "status_record_identity": idle_identity,
        }
        _write_transaction(paths, terminal, transition_observer)
        return transaction_id, candidate
    except InspectorError as error:
        failed = _status_value(
            paths,
            state="FAILED",
            reason_code=error.reason_code,
            active_transaction_id=transaction_id,
            last_transaction_id=transaction_id,
        )
        failed_identity = _write_status(paths, failed, transition_observer)
        terminal = {
            **start_transaction,
            "finish_utc": utc_now(),
            "state": "FAILED",
            "reason_code": error.reason_code,
            "status_record_identity": failed_identity,
        }
        _write_transaction(paths, terminal, transition_observer)
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        _write_status(paths, idle, transition_observer)
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
    except Exception as error:
        failed = _status_value(
            paths,
            state="FAILED",
            reason_code="INTERNAL_ERROR",
            active_transaction_id=transaction_id,
            last_transaction_id=transaction_id,
        )
        failed_identity = _write_status(paths, failed, transition_observer)
        terminal = {
            **start_transaction,
            "finish_utc": utc_now(),
            "state": "FAILED",
            "reason_code": "INTERNAL_ERROR",
            "status_record_identity": failed_identity,
        }
        _write_transaction(paths, terminal, transition_observer)
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        _write_status(paths, idle, transition_observer)
        raise InspectorError(
            "INTERNAL_ERROR",
            "Unexpected Inspector internal failure",
            data={"transaction_id": transaction_id},
            exit_status=70,
        ) from error
    finally:
        lock.release()


def _inspection_stage(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    state: str,
    reason_code: str,
    observer: Callable[[str, dict[str, Any]], None] | None,
    **updates: Any,
) -> dict[str, Any]:
    status = _status_value(
        paths,
        state=state,
        reason_code=reason_code,
        active_transaction_id=transaction["transaction_id"],
        last_transaction_id=None,
    )
    status_identity = _write_status(paths, status, observer)
    changed = {
        **transaction,
        **updates,
        "state": state,
        "reason_code": reason_code,
        "status_record_identity": status_identity,
    }
    _write_transaction(paths, changed, observer)
    return changed


def _manifest_entry(
    identity: ArtifactIdentity, relative_path: str
) -> dict[str, Any] | None:
    for item in identity.files:
        if item["relative_path"] == relative_path:
            return dict(item)
    return None


def _source_descriptor(
    *,
    kind: str,
    relative_path: str,
    entry: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "relative_path": relative_path,
        "byte_count": entry.get("byte_count") if entry else None,
        "sha256": (
            "sha256:" + entry["sha256"]
            if entry and isinstance(entry.get("sha256"), str)
            else None
        ),
        "token_count": extra.get("token_count"),
        "token_identity": extra.get("token_identity"),
        "chat_template_present": extra.get("chat_template_present"),
        "chat_template_identity": extra.get("chat_template_identity"),
    }


def _detected_family(classification: ClassificationResult) -> str:
    physical = classification.normalized["physical_format"]
    if physical in {"GGUF", "NATIVE"}:
        return physical
    if classification.terminal_class == "CONTRADICTORY":
        return "mixed"
    evidence = classification.parser_evidence
    if "gguf_issue" in evidence:
        return "GGUF"
    if "native_issue" in evidence:
        return "NATIVE"
    return "unknown"


def _enriched_normalized(
    classification: ClassificationResult,
    identity: ArtifactIdentity,
    candidate_name: str,
) -> dict[str, Any]:
    normalized = dict(classification.normalized)
    family = _detected_family(classification)
    evidence = classification.parser_evidence
    if family == "GGUF":
        member = evidence.get("container_member")
        relative = (
            member
            if isinstance(member, str)
            else identity.files[0]["relative_path"]
            if identity.files
            else candidate_name
        )
        entry = _manifest_entry(identity, relative)
        normalized["configuration_source"] = _source_descriptor(
            kind="embedded_gguf_metadata",
            relative_path=relative,
            entry=entry,
        )
        if evidence.get("tokenizer_metadata_present"):
            normalized["tokenizer_source"] = [
                _source_descriptor(
                    kind="embedded_gguf_tokenizer_metadata",
                    relative_path=relative,
                    entry=entry,
                    token_count=evidence.get("tokenizer_token_count"),
                    token_identity=evidence.get("tokenizer_token_identity"),
                    chat_template_present=evidence.get(
                        "chat_template_present"
                    ),
                    chat_template_identity=evidence.get(
                        "chat_template_identity"
                    ),
                )
            ]
        else:
            normalized["tokenizer_source"] = []
    elif family == "NATIVE":
        entry = _manifest_entry(identity, "config.json")
        normalized["configuration_source"] = (
            _source_descriptor(
                kind="native_configuration",
                relative_path="config.json",
                entry=entry,
            )
            if entry is not None
            else None
        )
        native_evidence = evidence
        if "native" in evidence and isinstance(evidence["native"], dict):
            native_evidence = evidence["native"]
        sources = native_evidence.get("tokenizer_source", [])
        normalized["tokenizer_source"] = [
            _source_descriptor(
                kind="native_tokenizer_source",
                relative_path=name,
                entry=_manifest_entry(identity, name),
            )
            for name in sources
            if isinstance(name, str)
        ]
    return normalized


def _format_summary(
    classification: ClassificationResult,
) -> dict[str, Any]:
    family = _detected_family(classification)
    evidence = classification.parser_evidence
    gguf = None
    native = None
    definition_identity = None
    endianness = None
    if family == "GGUF" and "version" in evidence:
        definition_identity = evidence.get(
            "format_definition_identity", GGUF_DEFINITION_IDENTITY
        )
        endianness = "little"
        gguf = {
            key: evidence.get(key)
            for key in (
                "tensor_count",
                "metadata_count",
                "alignment",
                "tensor_data_offset",
                "tensor_data_byte_count",
                "architecture",
                "model_type",
                "general_file_type",
                "quantization_version",
                "tensor_type_histogram",
                "tokenizer_metadata_present",
                "tokenizer_token_count",
                "tokenizer_token_identity",
                "chat_template_present",
                "chat_template_identity",
                "tensor_name_count",
                "tensor_name_identity",
                "tensor_payload_bytes_read",
            )
        }
    elif family == "NATIVE":
        native_evidence = evidence
        if "native" in evidence and isinstance(evidence["native"], dict):
            native_evidence = evidence["native"]
        if "weight_format" in native_evidence:
            definition_identity = NATIVE_DEFINITION_IDENTITY
            native = {
                key: native_evidence.get(key)
                for key in (
                    "model_type",
                    "architectures",
                    "quantization",
                    "declared_dtype",
                    "configuration_source",
                    "tokenizer_source",
                    "weight_format",
                    "shard_count",
                    "tensor_count",
                    "tensor_bytes",
                    "dtype_histogram",
                    "payload_deserialized",
                    "code_executed",
                )
            }
    return {
        "version": classification.normalized.get("format_version"),
        "endianness": endianness,
        "definition_identity": definition_identity,
        "gguf": gguf,
        "native": native,
    }


def _bounded_evidence(
    classification: ClassificationResult,
    identity: ArtifactIdentity,
    candidate_name: str,
) -> list[dict[str, Any]]:
    items = [
        {
            "code": "INSPECTION_COMPLETE",
            "source_kind": "complete_artifact_identity",
            "relative_path": candidate_name,
            "field": "artifact.identity",
            "byte_offset": None,
            "encoded_length": identity.byte_count,
            "value_type": "sha256",
            "bounded_summary": identity.identity,
            "value_sha256": identity.identity,
            "validation_state": "validated",
        }
    ]
    for reason in classification.normalized["reason_codes"]:
        if reason == "INSPECTION_COMPLETE":
            continue
        items.append(
            {
                "code": reason,
                "source_kind": "physical_format_parser",
                "relative_path": candidate_name,
                "field": None,
                "byte_offset": None,
                "encoded_length": None,
                "value_type": "reason_code",
                "bounded_summary": reason,
                "value_sha256": None,
                "validation_state": "observed",
            }
        )
    return items[:512]


def _warnings(classification: ClassificationResult) -> list[str]:
    evidence = classification.parser_evidence
    if "native" in evidence and isinstance(evidence["native"], dict):
        evidence = evidence["native"]
    value = evidence.get("warnings", [])
    return sorted(
        {
            item[:4096]
            for item in value
            if isinstance(item, str) and item
        }
    )[:512]


def _inspection_record(
    *,
    inspection_id: str,
    transaction_id: str,
    candidate: dict[str, Any],
    target: Path,
    identity: ArtifactIdentity,
    classification: ClassificationResult,
    source_before: str,
    source_after: str,
) -> dict[str, Any]:
    details = target.lstat()
    normalized = _enriched_normalized(
        classification, identity, candidate["target_name"]
    )
    files = [dict(item) for item in identity.files[:512]]
    family = _detected_family(classification)
    return {
        "schema_version": SCHEMA_IDENTITIES["inspection_result"],
        "inspection_id": inspection_id,
        "transaction_id": transaction_id,
        "created_utc": utc_now(),
        "source": {
            "candidate_name": candidate["target_name"],
            "candidate_kind": candidate["root_type"],
            "relative_path": candidate["target_name"],
            "realpath": str(target.resolve(strict=True)),
            "device": details.st_dev,
            "inode": details.st_ino,
            "intake_snapshot_identity": candidate[
                "intake_snapshot_identity"
            ],
            "pre_inspection_snapshot_identity": source_before,
            "post_inspection_snapshot_identity": source_after,
        },
        "artifact": {
            "identity": identity.identity,
            "algorithm": "sha256",
            "byte_count": identity.byte_count,
            "file_count": identity.file_count,
            "content_manifest_identity": identity.content_manifest_identity,
            "files": files,
            "file_manifest_truncated": len(identity.files) > len(files),
        },
        "classification": {
            "terminal_class": classification.terminal_class,
            "detected_family": family,
            "inspection_confidence": normalized["inspection_confidence"],
            "reason_codes": list(normalized["reason_codes"]),
        },
        "format": _format_summary(classification),
        "normalized": normalized,
        "evidence": _bounded_evidence(
            classification, identity, candidate["target_name"]
        ),
        "warnings": _warnings(classification),
    }


def inspect_transaction(
    paths: InspectorPaths,
    configuration: ValidatedConfiguration,
    target_name: str | None,
    *,
    validator: Callable[..., dict[str, Any]] = validate_intake,
    identifier: Callable[..., ArtifactIdentity] | None = None,
    classifier: Callable[..., ClassificationResult] | None = None,
    source_snapshotter: Callable[..., str] | None = None,
    publisher: Callable[..., str] = atomic_create_json,
    transition_observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any], Path, str]:
    transaction_id = _transaction_id()
    inspection_id = (
        "inspection-"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + secrets.token_hex(8)
    )
    lock = TransactionLock(
        paths, transaction_id=transaction_id, operation="inspect"
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
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
        "operation": "inspect",
        "start_utc": start_utc,
        "finish_utc": None,
        "state": "VALIDATING_INTAKE",
        "reason_code": "OK",
        "input_target_name": target_name,
        "intake_snapshot_identity": None,
        "owner_identity": owner_identity,
        "status_record_identity": None,
        "artifact_identity": None,
        "terminal_class": None,
        "inspection_result_identity": None,
        "inspection_result_path": None,
    }
    try:
        transaction = _inspection_stage(
            paths,
            transaction,
            state="VALIDATING_INTAKE",
            reason_code="OK",
            observer=transition_observer,
        )
        bounds = configuration.values["intake_bounds"]
        candidate = validator(paths, bounds, target_name)
        target = paths.intake_root / candidate["target_name"]
        snapshot = source_snapshotter or artifact_source_snapshot
        identify = identifier or identify_artifact
        classify = classifier or classify_artifact
        source_before = snapshot(
            target,
            maximum_entries=bounds["maximum_entry_count"],
            maximum_depth=bounds["maximum_directory_depth"],
        )
        transaction = _inspection_stage(
            paths,
            transaction,
            state="HASHING_ARTIFACT",
            reason_code="OK",
            observer=transition_observer,
            input_target_name=candidate["target_name"],
            intake_snapshot_identity=candidate[
                "intake_snapshot_identity"
            ],
        )
        identity = identify(
            target,
            maximum_entries=bounds["maximum_entry_count"],
            maximum_depth=bounds["maximum_directory_depth"],
        )
        transaction = _inspection_stage(
            paths,
            transaction,
            state="INSPECTING_FORMAT",
            reason_code="OK",
            observer=transition_observer,
            artifact_identity=identity.identity,
        )
        classification = classify(
            target,
            artifact_identity=identity.identity,
            artifact_size=identity.byte_count,
        )
        source_after = snapshot(
            target,
            maximum_entries=bounds["maximum_entry_count"],
            maximum_depth=bounds["maximum_directory_depth"],
        )
        if source_before != source_after:
            raise InspectorError(
                "ARTIFACT_CHANGED_DURING_INSPECTION",
                "artifact metadata identity changed during inspection",
            )
        transaction = _inspection_stage(
            paths,
            transaction,
            state="PERSISTING_RESULT",
            reason_code="OK",
            observer=transition_observer,
            terminal_class=classification.terminal_class,
        )
        record = _inspection_record(
            inspection_id=inspection_id,
            transaction_id=transaction_id,
            candidate=candidate,
            target=target,
            identity=identity,
            classification=classification,
            source_before=source_before,
            source_after=source_after,
        )
        result_path = paths.inspection_results / f"{inspection_id}.json"
        result_identity = publisher(
            result_path, record, mode=0o600
        )
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
            "state": "COMPLETED",
            "reason_code": "INSPECTION_COMPLETE",
            "status_record_identity": idle_identity,
            "inspection_result_identity": result_identity,
            "inspection_result_path": str(result_path),
        }
        _write_transaction(paths, terminal, transition_observer)
        return transaction_id, record, result_path, result_identity
    except InspectorError as error:
        failed = _status_value(
            paths,
            state="FAILED",
            reason_code=error.reason_code,
            active_transaction_id=transaction_id,
            last_transaction_id=transaction_id,
        )
        failed_identity = _write_status(
            paths, failed, transition_observer
        )
        terminal = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "FAILED",
            "reason_code": error.reason_code,
            "status_record_identity": failed_identity,
        }
        _write_transaction(paths, terminal, transition_observer)
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        _write_status(paths, idle, transition_observer)
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
    except Exception as error:
        reason_code = "INSPECTION_INTERNAL_ERROR"
        failed = _status_value(
            paths,
            state="FAILED",
            reason_code=reason_code,
            active_transaction_id=transaction_id,
            last_transaction_id=transaction_id,
        )
        failed_identity = _write_status(
            paths, failed, transition_observer
        )
        terminal = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "FAILED",
            "reason_code": reason_code,
            "status_record_identity": failed_identity,
        }
        _write_transaction(paths, terminal, transition_observer)
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        _write_status(paths, idle, transition_observer)
        raise InspectorError(
            reason_code,
            "Unexpected Inspector inspection failure",
            data={"transaction_id": transaction_id},
            exit_status=70,
        ) from error
    finally:
        lock.release()


def decide_transaction(
    paths: InspectorPaths,
    inspection_id: str,
    *,
    resolver: Callable[..., dict[str, Any]] = resolve_decision,
    publisher: Callable[..., Path] = publish_decision_record,
    decision_id_factory: Callable[[], str] = _decision_id,
    transition_observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any], Path, str]:
    transaction_id = _transaction_id()
    lock = TransactionLock(
        paths, transaction_id=transaction_id, operation="decide"
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
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
    transaction: dict[str, Any] = {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "decide",
        "start_utc": start_utc,
        "finish_utc": None,
        "state": "RESOLVING_CAPABILITY",
        "reason_code": "OK",
        "input_target_name": None,
        "intake_snapshot_identity": None,
        "owner_identity": owner_identity,
        "status_record_identity": None,
        "artifact_identity": None,
        "terminal_class": None,
        "inspection_result_identity": None,
        "inspection_result_path": None,
        "inspection_id": inspection_id,
        "capability_record_identity": None,
        "decision_id": None,
        "decision_result_identity": None,
        "decision_result_path": None,
    }
    try:
        transaction = _inspection_stage(
            paths,
            transaction,
            state="RESOLVING_CAPABILITY",
            reason_code="OK",
            observer=transition_observer,
        )
        outcome = resolver(paths, inspection_id)
        decision_id = decision_id_factory()
        transaction = _inspection_stage(
            paths,
            transaction,
            state="DECIDING",
            reason_code="OK",
            observer=transition_observer,
            artifact_identity=outcome["inspection"]["artifact_identity"],
            terminal_class=outcome["inspection"]["physical_format"],
            inspection_result_identity=outcome["inspection"][
                "inspection_result_identity"
            ],
            inspection_result_path=str(
                paths.inspection_results / f"{inspection_id}.json"
            ),
            capability_record_identity=outcome["capability"][
                "capability_record_identity"
            ],
            decision_id=decision_id,
        )
        record = build_decision_record(
            outcome,
            decision_id=decision_id,
            transaction_id=transaction_id,
            decision_timestamp_utc=utc_now(),
        )
        result_path = publisher(paths, record)
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
            "state": "COMPLETED",
            "reason_code": record["reason_code"],
            "status_record_identity": idle_identity,
            "decision_result_identity": record["result_identity"],
            "decision_result_path": str(result_path),
        }
        _write_transaction(paths, terminal, transition_observer)
        return (
            transaction_id,
            record,
            result_path,
            record["result_identity"],
        )
    except InspectorError as error:
        failed = _status_value(
            paths,
            state="FAILED",
            reason_code=error.reason_code,
            active_transaction_id=transaction_id,
            last_transaction_id=transaction_id,
        )
        failed_identity = _write_status(
            paths, failed, transition_observer
        )
        terminal = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "FAILED",
            "reason_code": error.reason_code,
            "status_record_identity": failed_identity,
        }
        _write_transaction(paths, terminal, transition_observer)
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        _write_status(paths, idle, transition_observer)
        error.data = {**error.data, "transaction_id": transaction_id}
        raise
    except Exception as error:
        reason_code = "INTERNAL_ERROR"
        failed = _status_value(
            paths,
            state="FAILED",
            reason_code=reason_code,
            active_transaction_id=transaction_id,
            last_transaction_id=transaction_id,
        )
        failed_identity = _write_status(
            paths, failed, transition_observer
        )
        terminal = {
            **transaction,
            "finish_utc": utc_now(),
            "state": "FAILED",
            "reason_code": reason_code,
            "status_record_identity": failed_identity,
        }
        _write_transaction(paths, terminal, transition_observer)
        idle = _status_value(
            paths,
            state="IDLE",
            reason_code="OK",
            active_transaction_id=None,
            last_transaction_id=transaction_id,
        )
        _write_status(paths, idle, transition_observer)
        raise InspectorError(
            reason_code,
            "Unexpected Inspector decision failure",
            data={"transaction_id": transaction_id},
            exit_status=70,
        ) from error
    finally:
        lock.release()
