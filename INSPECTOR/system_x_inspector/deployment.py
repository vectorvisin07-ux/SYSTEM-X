"""Transactional composition for one bounded GGUF deployment."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Callable, Protocol

from .connection_receipt import (
    build_receipt,
    load_current_receipt,
    publish_current_receipt,
    validate_receipt,
)
from .constants import QUALIFICATION_PROFILES, SCHEMA_IDENTITIES
from .content_identity import identify_artifact
from .errors import InspectorError
from .locking import TransactionLock, inspect_active_lock
from .paths import InspectorPaths
from .records import (
    atomic_create_json,
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


DEPLOYMENT_ID_PATTERN = re.compile(
    r"deployment-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
CANDIDATE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
DEPLOYMENT_MODES = ("install-first", "add", "replace-default")
RETIREMENT_POLICIES = (
    "retain-incumbent",
    "retire-incumbent-after-acceptance",
)
RESULT_CLASSES = (
    "DEPLOYMENT_COMPLETE",
    "DEPLOYMENT_ROLLED_BACK",
    "DEPLOYMENT_FAILED_CLEAN",
    "DEPLOYMENT_FAIL_CLOSED",
)
INPUT_FIELDS = frozenset(
    {
        "candidate_name",
        "deployment_mode",
        "required_capability_profile",
        "retirement_policy",
    }
)
RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "deployment_id",
        "transaction_id",
        "created_utc",
        "completed_utc",
        "result_class",
        "reason_code",
        "deployment_mode",
        "required_capability_profile",
        "retirement_policy",
        "input_identity",
        "source_candidate",
        "incumbent_snapshot",
        "candidate_identity",
        "child_results",
        "final_model_state",
        "promotion_result",
        "rollback_result",
        "retirement_result",
        "connection_receipt",
        "cleanup",
        "warnings",
        "result_identity",
    }
)
CHILD_NAMES = (
    "inspection",
    "decision",
    "qualification",
    "handoff",
    "publication",
    "promotion",
    "retirement",
)
CHILD_ID_KEYS = {
    "inspection": "inspection_id",
    "decision": "decision_id",
    "qualification": "qualification_id",
    "handoff": "handoff_id",
    "publication": "publication_id",
    "promotion": "promotion_id",
    "retirement": "retirement_id",
}
FORBIDDEN_KEYS = frozenset(
    {
        "raw_api_key",
        "authorization_value",
        "x_api_key_value",
        "credential_verifier",
        "pepper",
        "private_router_url",
        "private_router_port",
        "physical_gguf_path",
        "model_child_pid",
        "model_child_port",
        "process_environment",
        "prompt_content",
        "answer_content",
        "reasoning_content",
        "tool_content",
    }
)
NONTERMINAL_STATES = frozenset(
    {
        "PREPARING",
        "VALIDATING_INTAKE",
        "INSPECTING",
        "CLASSIFIED",
        "RESOLVING_CAPABILITY",
        "QUALIFYING",
        "QUALIFIED",
        "HANDING_OFF",
        "REGISTERED",
        "PROBING",
        "CANDIDATE_READY",
        "PUBLISHING_CANDIDATE",
        "CANDIDATE_REQUEST_PROVEN",
        "PROMOTING_DEFAULT",
        "DEFAULT_PROMOTED",
        "STABILITY_OBSERVING",
        "RESTART_VERIFYING",
        "RETIRING_INCUMBENT",
        "GENERATING_CONNECTION_RECEIPT",
        "PUBLISHING_DEPLOYMENT_RESULT",
        "ROLLING_BACK",
    }
)


class DeploymentInterruption(BaseException):
    """Testable crash boundary; leaves an authenticated resumable record."""


class DeploymentAdapter(Protocol):
    def source_snapshot(
        self, paths: InspectorPaths, candidate_name: str
    ) -> dict[str, Any]: ...

    def capture_prestate(self, paths: InspectorPaths) -> dict[str, Any]: ...

    def inspect(
        self, paths: InspectorPaths, candidate_name: str
    ) -> dict[str, Any]: ...

    def decide(
        self, paths: InspectorPaths, inspection: dict[str, Any]
    ) -> dict[str, Any]: ...

    def qualify(
        self,
        paths: InspectorPaths,
        inspection: dict[str, Any],
        decision: dict[str, Any],
        requested_profile: str,
    ) -> dict[str, Any]: ...

    def handoff(
        self,
        paths: InspectorPaths,
        source_candidate: str,
        decision: dict[str, Any],
        qualification: dict[str, Any],
    ) -> dict[str, Any]: ...

    def publish_candidate(
        self, paths: InspectorPaths, handoff: dict[str, Any]
    ) -> dict[str, Any]: ...

    def promote(
        self,
        paths: InspectorPaths,
        candidate_name: str,
        qualification: dict[str, Any],
    ) -> dict[str, Any]: ...

    def retire(
        self,
        paths: InspectorPaths,
        incumbent: dict[str, Any],
    ) -> dict[str, Any]: ...

    def observe_connection(
        self,
        paths: InspectorPaths,
        reference: str,
        proof_request_id: str | None,
    ) -> dict[str, Any]: ...

    def rollback(
        self,
        paths: InspectorPaths,
        runtime: dict[str, Any],
    ) -> dict[str, Any]: ...

    def cleanup_failed(
        self,
        paths: InspectorPaths,
        runtime: dict[str, Any],
    ) -> dict[str, Any]: ...

    def remove_source(
        self,
        paths: InspectorPaths,
        source: dict[str, Any],
    ) -> dict[str, Any]: ...

    def authenticate_child(
        self,
        paths: InspectorPaths,
        name: str,
        projection: dict[str, str],
        data: dict[str, Any],
    ) -> dict[str, Any]: ...


def _error(
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


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise _error(
                    "DEPLOYMENT_RESULT_INVALID",
                    f"deployment record contains prohibited field: {key}",
                )
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def validate_deploy_input(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != INPUT_FIELDS:
        raise _error(
            "DEPLOYMENT_INPUT_INVALID",
            "deploy-gguf input fields are not closed",
        )
    candidate = value["candidate_name"]
    mode = value["deployment_mode"]
    profile = value["required_capability_profile"]
    retirement = value["retirement_policy"]
    if (
        not isinstance(candidate, str)
        or CANDIDATE_PATTERN.fullmatch(candidate) is None
        or Path(candidate).name != candidate
        or candidate.startswith(".")
        or any(ord(character) < 32 for character in candidate)
        or mode not in DEPLOYMENT_MODES
        or profile not in QUALIFICATION_PROFILES
        or retirement not in RETIREMENT_POLICIES
    ):
        raise _error(
            "DEPLOYMENT_INPUT_INVALID",
            "deploy-gguf input value is invalid",
        )
    if (
        mode in {"install-first", "add"}
        and retirement != "retain-incumbent"
    ):
        raise _error(
            "DEPLOYMENT_RETIREMENT_POLICY_INVALID",
            "deployment mode cannot retire an incumbent",
        )
    return {
        "candidate_name": candidate,
        "deployment_mode": mode,
        "required_capability_profile": profile,
        "retirement_policy": retirement,
    }


def deployment_result_identity(value: dict[str, Any]) -> str:
    if set(value) != RESULT_FIELDS:
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "deployment result fields are not closed",
        )
    return _identity(
        {
            key: value[key]
            for key in sorted(value)
            if key != "result_identity"
        }
    )


def _child_projection(value: object, name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"id", "identity"}:
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            f"{name} child result projection is invalid",
        )
    if (
        not isinstance(value["id"], str)
        or not value["id"]
        or not isinstance(value["identity"], str)
        or SHA256_PATTERN.fullmatch(value["identity"]) is None
    ):
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            f"{name} child result identity is invalid",
        )
    return value


def validate_deployment_result(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS:
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "deployment result fields are not closed",
        )
    record = value
    _reject_forbidden(record)
    if (
        record["schema_version"]
        != SCHEMA_IDENTITIES["gguf_deployment_result"]
        or not isinstance(record["deployment_id"], str)
        or DEPLOYMENT_ID_PATTERN.fullmatch(record["deployment_id"]) is None
        or not isinstance(record["transaction_id"], str)
        or not record["transaction_id"]
        or record["result_class"] not in RESULT_CLASSES
        or record["deployment_mode"] not in DEPLOYMENT_MODES
        or record["required_capability_profile"]
        not in QUALIFICATION_PROFILES
        or record["retirement_policy"] not in RETIREMENT_POLICIES
        or not isinstance(record["input_identity"], str)
        or SHA256_PATTERN.fullmatch(record["input_identity"]) is None
    ):
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "deployment result identity or enum is invalid",
        )
    for name in ("created_utc", "completed_utc", "reason_code"):
        if not isinstance(record[name], str) or not record[name]:
            raise _error(
                "DEPLOYMENT_RESULT_INVALID",
                f"deployment {name} is invalid",
            )
    source = record["source_candidate"]
    if (
        not isinstance(source, dict)
        or set(source) != {"candidate_name", "artifact_identity", "size"}
        or CANDIDATE_PATTERN.fullmatch(
            str(source.get("candidate_name", ""))
        )
        is None
        or SHA256_PATTERN.fullmatch(
            str(source.get("artifact_identity", ""))
        )
        is None
        or type(source.get("size")) is not int
        or source["size"] < 1
    ):
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "deployment source projection is invalid",
        )
    children = record["child_results"]
    if not isinstance(children, dict) or set(children) != set(CHILD_NAMES):
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "deployment child results are not closed",
        )
    for name in CHILD_NAMES:
        _child_projection(children[name], name)
    for name in (
        "incumbent_snapshot",
        "candidate_identity",
        "final_model_state",
        "cleanup",
    ):
        if not isinstance(record[name], dict):
            raise _error(
                "DEPLOYMENT_RESULT_INVALID",
                f"deployment {name} is not an object",
            )
    if record["connection_receipt"] is not None:
        validate_receipt(record["connection_receipt"])
    if (
        not isinstance(record["warnings"], list)
        or any(
            not isinstance(item, str) or not item
            for item in record["warnings"]
        )
        or not isinstance(record["result_identity"], str)
        or record["result_identity"] != deployment_result_identity(record)
    ):
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "deployment terminal identity is invalid",
        )
    return record


def deployment_result_path(
    paths: InspectorPaths, deployment_id: str
) -> Path:
    if DEPLOYMENT_ID_PATTERN.fullmatch(deployment_id) is None:
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "deployment ID is not canonical",
        )
    return paths.deployment_results / f"{deployment_id}.json"


def publish_deployment_result(
    paths: InspectorPaths, record: dict[str, Any]
) -> tuple[Path, str]:
    validated = validate_deployment_result(record)
    path = deployment_result_path(paths, validated["deployment_id"])
    try:
        atomic_create_json(path, validated, mode=0o600)
    except InspectorError as error:
        if error.reason_code == "INSPECTION_RECORD_COLLISION":
            raise _error(
                "DEPLOYMENT_RESULT_COLLISION",
                "immutable deployment result already exists",
            ) from error
        raise
    details = path.lstat()
    observed = validate_deployment_result(read_json_record(path))
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or observed != validated
    ):
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "immutable deployment result did not round-trip",
            internal=True,
        )
    return path, observed["result_identity"]


def _new_deployment_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"deployment-{stamp}-{secrets.token_hex(8)}"


def _projection(value: dict[str, Any], id_key: str) -> dict[str, str]:
    identifier = value.get(id_key)
    identity = value.get("identity")
    if (
        not isinstance(identifier, str)
        or not identifier
        or not isinstance(identity, str)
        or SHA256_PATTERN.fullmatch(identity) is None
    ):
        raise _error(
            "DEPLOYMENT_CHILD_RESULT_INVALID",
            f"{id_key} child result is invalid",
        )
    return {"id": identifier, "identity": identity}


def _preconditions(mode: str, state: dict[str, Any]) -> None:
    required = {
        "desired_state",
        "model_service_state",
        "ready_model_count",
        "default_alias",
        "default_target",
        "warm_model_id",
        "operating_profile_identity",
        "capability_binding_identity",
        "non_secret_key_id",
    }
    if not required <= set(state):
        raise _error(
            "DEPLOYMENT_PRECONDITION_FAILED",
            "deployment prestate is incomplete",
        )
    if state["desired_state"] != "RUNNING":
        raise _error(
            "DEPLOYMENT_PRECONDITION_FAILED",
            "System X desired state is not RUNNING",
        )
    if mode == "install-first":
        valid = (
            state["model_service_state"] == "WAITING_FOR_MODEL"
            and state["ready_model_count"] == 0
            and state["default_alias"] is None
            and state["default_target"] is None
            and state["warm_model_id"] is None
        )
    else:
        valid = (
            state["model_service_state"] == "READY"
            and type(state["ready_model_count"]) is int
            and state["ready_model_count"] >= 1
            and isinstance(state["default_alias"], str)
            and isinstance(state["default_target"], str)
            and (
                mode == "add"
                or state["warm_model_id"] == state["default_target"]
            )
        )
    if not valid:
        raise _error(
            "DEPLOYMENT_PRECONDITION_FAILED",
            f"{mode} deployment preconditions are not satisfied",
        )


def _input_identity(
    request: dict[str, str],
    source: dict[str, Any],
    prestate: dict[str, Any],
) -> str:
    return _identity(
        {
            "candidate_artifact_identity": source["artifact_identity"],
            "deployment_mode": request["deployment_mode"],
            "required_capability_profile": request[
                "required_capability_profile"
            ],
            "retirement_policy": request["retirement_policy"],
            "current_capability_binding_identity": prestate[
                "capability_binding_identity"
            ],
            "current_operating_profile_identity": prestate[
                "operating_profile_identity"
            ],
        }
    )


def _request_matches(
    transaction: dict[str, Any],
    request: dict[str, str],
    source: dict[str, Any],
) -> bool:
    runtime = transaction.get("deployment_runtime")
    if not isinstance(runtime, dict):
        return False
    return (
        transaction.get("source_candidate") == request["candidate_name"]
        and transaction.get("deployment_mode")
        == request["deployment_mode"]
        and transaction.get("requested_profile")
        == request["required_capability_profile"]
        and transaction.get("retirement_policy")
        == request["retirement_policy"]
        and runtime.get("source", {}).get("artifact_identity")
        == source["artifact_identity"]
    )


def _request_parameters_match(
    transaction: dict[str, Any],
    request: dict[str, str],
) -> bool:
    return (
        transaction.get("source_candidate")
        == request["candidate_name"]
        and transaction.get("deployment_mode")
        == request["deployment_mode"]
        and transaction.get("requested_profile")
        == request["required_capability_profile"]
        and transaction.get("retirement_policy")
        == request["retirement_policy"]
    )


def _find_recoverable(
    paths: InspectorPaths,
    request: dict[str, str],
    source: dict[str, Any],
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    other_active: list[dict[str, Any]] = []
    for path in sorted(paths.transactions.glob("tx-*.json")):
        try:
            value = read_json_record(path)
        except (OSError, InspectorError):
            continue
        if (
            value.get("operation") == "deploy-gguf"
            and value.get("state") in NONTERMINAL_STATES
        ):
            if _request_parameters_match(value, request):
                runtime = value.get("deployment_runtime")
                prior_identity = (
                    runtime.get("source", {}).get(
                        "artifact_identity"
                    )
                    if isinstance(runtime, dict)
                    else None
                )
                if prior_identity != source["artifact_identity"]:
                    raise _error(
                        "DEPLOYMENT_SOURCE_CHANGED",
                        "candidate identity changed during active deployment",
                    )
                if _request_matches(value, request, source):
                    matches.append(value)
            else:
                other_active.append(value)
    if len(matches) > 1:
        raise _error(
            "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
            "multiple resumable deployments match one input",
        )
    if not matches and other_active:
        raise _error(
            "DEPLOYMENT_ACTIVE",
            "a different deployment transaction is active",
        )
    return matches[0] if matches else None


def _find_completed(
    paths: InspectorPaths,
    request: dict[str, str],
    source: dict[str, Any] | None,
) -> tuple[dict[str, Any], Path] | None:
    matches: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(paths.deployment_results.glob("deployment-*.json")):
        try:
            value = validate_deployment_result(read_json_record(path))
        except (OSError, InspectorError):
            raise _error(
                "DEPLOYMENT_RESULT_INVALID",
                "deployment result store contains invalid evidence",
            )
        transaction_path = (
            paths.transactions / f"{value['transaction_id']}.json"
        )
        try:
            transaction = read_json_record(transaction_path)
        except (OSError, InspectorError) as error:
            raise _error(
                "DEPLOYMENT_RESULT_INVALID",
                "deployment result lacks its terminal transaction",
            ) from error
        expected_terminal = {
            "DEPLOYMENT_COMPLETE": "COMPLETE",
            "DEPLOYMENT_ROLLED_BACK": "ROLLED_BACK",
            "DEPLOYMENT_FAILED_CLEAN": "FAILED_CLEAN",
            "DEPLOYMENT_FAIL_CLOSED": "FAIL_CLOSED",
        }[value["result_class"]]
        if (
            transaction.get("operation") != "deploy-gguf"
            or transaction.get("deployment_id")
            != value["deployment_id"]
            or transaction.get("state") != expected_terminal
            or transaction.get("deployment_result_identity")
            != value["result_identity"]
            or transaction.get("deployment_result_path") != str(path)
        ):
            raise _error(
                "DEPLOYMENT_RESULT_INVALID",
                "deployment result and terminal transaction disagree",
            )
        candidate = value["source_candidate"]
        if (
            candidate["candidate_name"] == request["candidate_name"]
            and value["deployment_mode"] == request["deployment_mode"]
            and value["required_capability_profile"]
            == request["required_capability_profile"]
            and value["retirement_policy"] == request["retirement_policy"]
            and (
                source is None
                or candidate["artifact_identity"]
                == source["artifact_identity"]
            )
        ):
            matches.append((value, path))
    if len(matches) > 1:
        raise _error(
            "DEPLOYMENT_RESULT_INVALID",
            "multiple immutable deployments match one input",
        )
    return matches[0] if matches else None


def _clear_exact_stale_deployment_lock(
    paths: InspectorPaths, transaction_id: str
) -> None:
    state = inspect_active_lock(paths.deployment_lock)
    if state["state"] == "absent":
        return
    record = state.get("record")
    if (
        state["state"] != "stale"
        or not isinstance(record, dict)
        or record.get("transaction_id") != transaction_id
        or record.get("operation") != "deploy-gguf"
    ):
        raise _error(
            "DEPLOYMENT_ACTIVE",
            "another or uncertain deployment owns the deployment lock",
        )
    details = paths.deployment_lock.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise _error(
            "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
            "stale deployment lock has an unsafe physical type",
        )
    paths.deployment_lock.unlink()
    fsync_directory(paths.deployment_lock.parent)


def _authenticate_resume_children(
    paths: InspectorPaths,
    runtime: dict[str, Any],
    adapter: DeploymentAdapter,
) -> None:
    children = runtime.get("child_results")
    child_data = runtime.get("child_data")
    if not isinstance(children, dict) or not isinstance(child_data, dict):
        raise _error(
            "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
            "resumable child evidence is incomplete",
        )
    for name, projection in children.items():
        if name not in CHILD_NAMES or projection is None:
            continue
        if (
            not isinstance(projection, dict)
            or not isinstance(child_data.get(name), dict)
        ):
            raise _error(
                "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
                f"resumable {name} evidence is incomplete",
            )
        data = child_data[name]
        if _projection(data, CHILD_ID_KEYS[name]) != projection:
            raise _error(
                "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
                f"resumable {name} projection changed",
            )
        observed = adapter.authenticate_child(
            paths, name, projection, data
        )
        if observed != data:
            raise _error(
                "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
                f"resumable {name} child result changed",
            )


def _publish_or_authenticate_deployment_result(
    paths: InspectorPaths,
    result: dict[str, Any],
    publisher: Callable[
        [InspectorPaths, dict[str, Any]], tuple[Path, str]
    ],
) -> tuple[Path, str]:
    expected = deployment_result_path(paths, result["deployment_id"])
    if expected.exists() or expected.is_symlink():
        try:
            observed = validate_deployment_result(
                read_json_record(expected)
            )
        except (OSError, InspectorError) as error:
            raise _error(
                "DEPLOYMENT_RESULT_INVALID",
                "existing deployment result cannot authenticate",
            ) from error
        if observed != result:
            raise _error(
                "DEPLOYMENT_RESULT_COLLISION",
                "existing deployment result differs from resume state",
            )
        return expected, observed["result_identity"]
    try:
        return publisher(paths, result)
    except BaseException:
        if expected.exists() and not expected.is_symlink():
            try:
                observed = validate_deployment_result(
                    read_json_record(expected)
                )
            except (OSError, InspectorError):
                observed = None
            if observed == result:
                return expected, result["result_identity"]
        raise


def _publish_or_authenticate_current_receipt(
    paths: InspectorPaths,
    receipt: dict[str, Any],
    *,
    expected_previous_identity: str | None,
    publisher: Callable[..., str],
) -> str:
    try:
        current = load_current_receipt(paths)
    except InspectorError as error:
        if error.reason_code != "CONNECTION_NOT_INITIALIZED":
            raise
        current = None
    if current == receipt:
        return receipt["receipt_identity"]
    observed_identity = (
        current["receipt_identity"] if current is not None else None
    )
    if observed_identity != expected_previous_identity:
        raise _error(
            "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
            "current receipt changed before deployment publication",
        )
    return publisher(
        paths,
        receipt,
        expected_previous_identity=expected_previous_identity,
    )


def _checkpoint(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    state: str,
    *,
    observer: Callable[[str, dict[str, Any]], None] | None,
    reason_code: str = "OK",
) -> dict[str, Any]:
    value = {**transaction, "state": state, "reason_code": reason_code}
    status = _status_value(
        paths,
        state=state,
        reason_code=reason_code,
        active_transaction_id=value["transaction_id"],
        last_transaction_id=None,
    )
    value["status_record_identity"] = _write_status(paths, status, None)
    _write_transaction(paths, value, None)
    if observer is not None:
        observer(state, value)
    return value


def _terminal_idle(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    state: str,
    reason_code: str,
) -> dict[str, Any]:
    idle = _status_value(
        paths,
        state="IDLE",
        reason_code="OK",
        active_transaction_id=None,
        last_transaction_id=transaction["transaction_id"],
    )
    value = {
        **transaction,
        "finish_utc": utc_now(),
        "state": state,
        "reason_code": reason_code,
    }
    value["status_record_identity"] = _write_status(paths, idle, None)
    _write_transaction(paths, value, None)
    return value


def _build_result(
    transaction: dict[str, Any],
    runtime: dict[str, Any],
    *,
    result_class: str,
    reason_code: str,
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    request = runtime["request"]
    children = runtime["child_results"]
    basis = {
        "schema_version": SCHEMA_IDENTITIES["gguf_deployment_result"],
        "deployment_id": transaction["deployment_id"],
        "transaction_id": transaction["transaction_id"],
        "created_utc": transaction["start_utc"],
        "completed_utc": utc_now(),
        "result_class": result_class,
        "reason_code": reason_code,
        "deployment_mode": request["deployment_mode"],
        "required_capability_profile": request[
            "required_capability_profile"
        ],
        "retirement_policy": request["retirement_policy"],
        "input_identity": transaction["deployment_input_identity"],
        "source_candidate": {
            key: runtime["source"][key]
            for key in ("candidate_name", "artifact_identity", "size")
        },
        "incumbent_snapshot": dict(runtime["prestate"]),
        "candidate_identity": dict(runtime.get("candidate_identity") or {}),
        "child_results": {
            name: children.get(name) for name in CHILD_NAMES
        },
        "final_model_state": dict(
            runtime.get("final_model_state") or {}
        ),
        "promotion_result": runtime.get("promotion_result"),
        "rollback_result": runtime.get("rollback_result"),
        "retirement_result": runtime.get("retirement_result"),
        "connection_receipt": runtime.get("connection_receipt"),
        "cleanup": cleanup,
        "warnings": list(runtime.get("warnings") or []),
    }
    return validate_deployment_result(
        {**basis, "result_identity": _identity(basis)}
    )


def _restore_previous_receipt(
    paths: InspectorPaths,
    runtime: dict[str, Any],
    *,
    publisher: Callable[..., str],
) -> None:
    current_identity = runtime.get("connection_receipt_identity")
    if current_identity is None:
        return
    previous = runtime.get("previous_connection_receipt")
    if previous is not None:
        publisher(
            paths,
            previous,
            expected_previous_identity=current_identity,
        )
        return
    current = load_current_receipt(paths)
    if current["receipt_identity"] != current_identity:
        raise _error(
            "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
            "current receipt changed before rollback",
        )
    path = paths.current_connection_status
    details = path.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
    ):
        raise _error(
            "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
            "current receipt is unsafe during rollback",
        )
    path.unlink()
    fsync_directory(path.parent)


def deploy_transaction(
    paths: InspectorPaths,
    request_value: object,
    *,
    adapter: DeploymentAdapter,
    result_publisher: Callable[
        [InspectorPaths, dict[str, Any]], tuple[Path, str]
    ] = publish_deployment_result,
    receipt_publisher: Callable[..., str] = publish_current_receipt,
    transaction_id_factory: Callable[[], str] = _transaction_id,
    deployment_id_factory: Callable[[], str] = _new_deployment_id,
    transition_observer: Callable[
        [str, dict[str, Any]], None
    ]
    | None = None,
) -> tuple[str, dict[str, Any], Path, str]:
    """Compose child operations without holding their global operation lock."""

    request = validate_deploy_input(request_value)
    source_path = paths.intake_root / request["candidate_name"]
    source: dict[str, Any] | None
    try:
        source = adapter.source_snapshot(paths, request["candidate_name"])
    except InspectorError:
        source = None
    if source is None:
        duplicate = _find_completed(paths, request, None)
        if duplicate is not None:
            record, path = duplicate
            return (
                record["transaction_id"],
                record,
                path,
                record["result_identity"],
            )
        raise _error(
            "DEPLOYMENT_SOURCE_INVALID",
            "deployment candidate is absent or unsafe",
        )
    if (
        source.get("candidate_name") != request["candidate_name"]
        or SHA256_PATTERN.fullmatch(
            str(source.get("artifact_identity", ""))
        )
        is None
        or type(source.get("size")) is not int
        or source["size"] < 1
        or source_path.parent != paths.intake_root
    ):
        raise _error(
            "DEPLOYMENT_SOURCE_INVALID",
            "deployment candidate snapshot is invalid",
        )
    recoverable = _find_recoverable(paths, request, source)
    resuming = recoverable is not None
    if recoverable is None:
        duplicate = _find_completed(paths, request, source)
        if duplicate is not None:
            record, path = duplicate
            return (
                record["transaction_id"],
                record,
                path,
                record["result_identity"],
            )
        prestate = adapter.capture_prestate(paths)
        _preconditions(request["deployment_mode"], prestate)
        try:
            previous_receipt = load_current_receipt(paths)
        except InspectorError as error:
            if error.reason_code != "CONNECTION_NOT_INITIALIZED":
                raise
            previous_receipt = None
        transaction_id = transaction_id_factory()
        deployment_id = deployment_id_factory()
        input_identity = _input_identity(request, source, prestate)
        runtime = {
            "request": request,
            "source": source,
            "prestate": prestate,
            "child_results": {},
            "child_data": {},
            "candidate_identity": {},
            "final_model_state": {},
            "promotion_result": None,
            "rollback_result": None,
            "retirement_result": None,
            "connection_receipt": None,
            "connection_receipt_identity": None,
            "current_connection_receipt": None,
            "current_connection_receipt_identity": None,
            "previous_connection_receipt": previous_receipt,
            "previous_connection_receipt_identity": (
                previous_receipt["receipt_identity"]
                if previous_receipt is not None
                else None
            ),
            "connection_observation": None,
            "deployment_result": None,
            "deployment_result_published": False,
            "source_cleanup": None,
            "warnings": [],
            "irreversible_steps": [],
        }
        transaction = {
            "schema_version": SCHEMA_IDENTITIES["transaction"],
            "transaction_id": transaction_id,
            "operation": "deploy-gguf",
            "start_utc": utc_now(),
            "finish_utc": None,
            "state": "PREPARING",
            "reason_code": "OK",
            "input_target_name": request["candidate_name"],
            "intake_snapshot_identity": source.get("snapshot_identity"),
            "owner_identity": None,
            "status_record_identity": None,
            "deployment_id": deployment_id,
            "deployment_input_identity": input_identity,
            "deployment_mode": request["deployment_mode"],
            "retirement_policy": request["retirement_policy"],
            "requested_profile": request[
                "required_capability_profile"
            ],
            "source_candidate": request["candidate_name"],
            "deployment_result_identity": None,
            "deployment_result_path": None,
            "deployment_runtime": runtime,
        }
    else:
        transaction = recoverable
        transaction_id = transaction["transaction_id"]
        deployment_id = transaction["deployment_id"]
        input_identity = transaction["deployment_input_identity"]
        runtime = dict(transaction["deployment_runtime"])
        if runtime["source"]["artifact_identity"] != source[
            "artifact_identity"
        ]:
            raise _error(
                "DEPLOYMENT_SOURCE_CHANGED",
                "candidate identity changed before deployment resume",
            )
        _clear_exact_stale_deployment_lock(paths, transaction_id)

    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="deploy-gguf",
    )
    lock.path = paths.deployment_lock
    try:
        owner = lock.acquire()
    except InspectorError as error:
        raise _error(
            "DEPLOYMENT_ACTIVE",
            "deployment lock is already owned",
            data={"transaction_id": transaction_id},
        ) from error
    transaction["owner_identity"] = {
        key: owner.get(key)
        for key in (
            "pid",
            "process_start_identity",
            "boot_identity",
            "inspector_root_identity",
        )
    }
    result_published = False
    try:
        if resuming:
            try:
                _authenticate_resume_children(
                    paths, runtime, adapter
                )
            except InspectorError:
                runtime["connection_receipt"] = None
                runtime["rollback_result"] = {
                    "result_class": "ROLLBACK_FAIL_CLOSED",
                    "reason_code": (
                        "DEPLOYMENT_OWNERSHIP_UNCERTAIN"
                    ),
                }
                transaction["deployment_runtime"] = runtime
                result = _build_result(
                    transaction,
                    runtime,
                    result_class="DEPLOYMENT_FAIL_CLOSED",
                    reason_code="DEPLOYMENT_FAIL_CLOSED",
                    cleanup={
                        "failed_residue_absent": False,
                        "ownership_certain": False,
                        "source_removal_committed": False,
                    },
                )
                result_path, result_identity = (
                    _publish_or_authenticate_deployment_result(
                        paths, result, result_publisher
                    )
                )
                transaction.update(
                    {
                        "deployment_result_identity": result_identity,
                        "deployment_result_path": str(result_path),
                    }
                )
                _terminal_idle(
                    paths,
                    transaction,
                    state="FAIL_CLOSED",
                    reason_code="DEPLOYMENT_FAIL_CLOSED",
                )
                return (
                    transaction_id,
                    result,
                    result_path,
                    result_identity,
                )
        transaction["deployment_runtime"] = runtime
        transaction = _checkpoint(
            paths,
            transaction,
            "VALIDATING_INTAKE",
            observer=transition_observer,
        )
        children = runtime["child_results"]
        child_data = runtime["child_data"]

        if "inspection" not in children:
            transaction = _checkpoint(
                paths, transaction, "INSPECTING",
                observer=transition_observer,
            )
            inspection = adapter.inspect(
                paths, request["candidate_name"]
            )
            children["inspection"] = _projection(
                inspection, "inspection_id"
            )
            child_data["inspection"] = inspection
            if (
                inspection.get("artifact_identity")
                != source["artifact_identity"]
                or inspection.get("size") != source["size"]
            ):
                raise _error(
                    "DEPLOYMENT_SOURCE_CHANGED",
                    "inspection does not match the deployment source",
                )
            if inspection.get("terminal_class") != "GGUF":
                raise _error(
                    "DEPLOYMENT_PHYSICAL_FORMAT_REJECTED",
                    "candidate is not one accepted GGUF primary model",
                )
            transaction["deployment_runtime"] = runtime
            transaction = _checkpoint(
                paths, transaction, "CLASSIFIED",
                observer=transition_observer,
            )
        inspection = child_data["inspection"]

        if "decision" not in children:
            transaction = _checkpoint(
                paths, transaction, "RESOLVING_CAPABILITY",
                observer=transition_observer,
            )
            decision = adapter.decide(paths, inspection)
            children["decision"] = _projection(
                decision, "decision_id"
            )
            child_data["decision"] = decision
            transaction["deployment_runtime"] = runtime
            _write_transaction(paths, transaction, None)
        decision = child_data["decision"]
        if decision.get("capability_result") not in {
            "SUPPORTED",
            "RUNTIME_SMOKE_REQUIRED",
        }:
            raise _error(
                "DEPLOYMENT_CAPABILITY_REJECTED",
                "candidate capability decision blocks handoff",
            )

        if "qualification" not in children:
            transaction = _checkpoint(
                paths, transaction, "QUALIFYING",
                observer=transition_observer,
            )
            qualification = adapter.qualify(
                paths,
                inspection,
                decision,
                request["required_capability_profile"],
            )
            children["qualification"] = _projection(
                qualification, "qualification_id"
            )
            child_data["qualification"] = qualification
            transaction["deployment_runtime"] = runtime
            transaction = _checkpoint(
                paths, transaction, "QUALIFIED",
                observer=transition_observer,
            )
        qualification = child_data["qualification"]
        if (
            qualification.get("result_class")
            != "SUPPORTED_FOR_CURRENT_TUPLE"
            or request["required_capability_profile"]
            not in qualification.get("supported_profiles", [])
            or qualification.get("artifact_identity")
            != source["artifact_identity"]
            or qualification.get("capability_result")
            != decision.get("capability_result")
        ):
            raise _error(
                "DEPLOYMENT_QUALIFICATION_REJECTED",
                "candidate qualification is not promotable",
            )

        if "handoff" not in children:
            transaction = _checkpoint(
                paths, transaction, "HANDING_OFF",
                observer=transition_observer,
            )
            handoff = adapter.handoff(
                paths,
                request["candidate_name"],
                decision,
                qualification,
            )
            children["handoff"] = _projection(handoff, "handoff_id")
            child_data["handoff"] = handoff
            runtime["irreversible_steps"].append("handoff")
            transaction["deployment_runtime"] = runtime
            transaction = _checkpoint(
                paths, transaction, "REGISTERED",
                observer=transition_observer,
            )
        handoff = child_data["handoff"]

        if "publication" not in children:
            transaction = _checkpoint(
                paths, transaction, "PROBING",
                observer=transition_observer,
            )
            publication = adapter.publish_candidate(paths, handoff)
            children["publication"] = _projection(
                publication, "publication_id"
            )
            child_data["publication"] = publication
            runtime["candidate_identity"] = dict(
                publication["candidate_identity"]
            )
            transaction["deployment_runtime"] = runtime
            transaction = _checkpoint(
                paths, transaction, "CANDIDATE_READY",
                observer=transition_observer,
            )
            transaction = _checkpoint(
                paths, transaction, "PUBLISHING_CANDIDATE",
                observer=transition_observer,
            )
            transaction = _checkpoint(
                paths, transaction, "CANDIDATE_REQUEST_PROVEN",
                observer=transition_observer,
            )
        publication = child_data["publication"]
        if (
            publication.get("artifact_identity")
            != source["artifact_identity"]
        ):
            raise _error(
                "DEPLOYMENT_CHILD_RESULT_INVALID",
                "publication artifact differs from deployment source",
            )

        mode = request["deployment_mode"]
        if mode in {"install-first", "replace-default"}:
            if "promotion" not in children:
                transaction = _checkpoint(
                    paths, transaction, "PROMOTING_DEFAULT",
                    observer=transition_observer,
                )
                promotion = adapter.promote(
                    paths,
                    request["candidate_name"],
                    qualification,
                )
                children["promotion"] = _projection(
                    promotion, "promotion_id"
                )
                child_data["promotion"] = promotion
                runtime["promotion_result"] = promotion[
                    "result_class"
                ]
                runtime["irreversible_steps"].append("promotion")
                transaction["deployment_runtime"] = runtime
                transaction = _checkpoint(
                    paths, transaction, "DEFAULT_PROMOTED",
                    observer=transition_observer,
                )
                transaction = _checkpoint(
                    paths, transaction, "STABILITY_OBSERVING",
                    observer=transition_observer,
                )
                transaction = _checkpoint(
                    paths, transaction, "RESTART_VERIFYING",
                    observer=transition_observer,
                )
            promotion = child_data["promotion"]
            if promotion.get("result_class") != "PROMOTION_COMPLETE":
                raise _error(
                    "DEPLOYMENT_PROMOTION_FAILED",
                    "default promotion did not complete",
                )

        if (
            mode == "replace-default"
            and request["retirement_policy"]
            == "retire-incumbent-after-acceptance"
            and "retirement" not in children
        ):
            transaction = _checkpoint(
                paths, transaction, "RETIRING_INCUMBENT",
                observer=transition_observer,
            )
            retirement = adapter.retire(paths, runtime["prestate"])
            children["retirement"] = _projection(
                retirement, "retirement_id"
            )
            child_data["retirement"] = retirement
            runtime["retirement_result"] = retirement[
                "result_class"
            ]
            runtime["irreversible_steps"].append("retirement")
            transaction["deployment_runtime"] = runtime
            _write_transaction(paths, transaction, None)
            if retirement.get("result_class") != "RETIREMENT_COMPLETE":
                raise _error(
                    "DEPLOYMENT_RETIREMENT_FAILED",
                    "incumbent retirement did not complete",
                )
            if transition_observer is not None:
                transition_observer(
                    "INCUMBENT_RETIRED", transaction
                )

        transaction = _checkpoint(
            paths, transaction, "GENERATING_CONNECTION_RECEIPT",
            observer=transition_observer,
        )
        reference = (
            publication["candidate_identity"][
                "resolved_immutable_model_id"
            ]
            if mode == "add"
            else "default"
        )
        if runtime.get("connection_observation") is None:
            observation = adapter.observe_connection(
                paths,
                reference,
                publication.get("proof_request_id"),
            )
            runtime["connection_observation"] = observation
            receipt = build_receipt(
                observation,
                receipt_source="DEPLOY_GGUF",
                deployment_id=deployment_id,
                deployment_result_identity=None,
                recommended_reference=reference,
                promotion_result=runtime.get("promotion_result"),
                rollback_result=None,
                retirement_result=runtime.get("retirement_result"),
                current_receipt_updated=False,
            )
            runtime["connection_receipt"] = receipt
            runtime["final_model_state"] = {
                "service_readiness": observation["service_readiness"],
                "default_alias": observation["default_alias"],
                "default_target": observation["default_target"],
                "resolved_immutable_model_id": observation[
                    "resolved_immutable_model_id"
                ],
                "artifact_version_id": observation[
                    "artifact_version_id"
                ],
                "warm": observation["warm"],
            }
            transaction["deployment_runtime"] = runtime
            _write_transaction(paths, transaction, None)
            if transition_observer is not None:
                transition_observer(
                    "CONNECTION_RECEIPT_PREPARED", transaction
                )
        observation = runtime["connection_observation"]

        transaction = _checkpoint(
            paths, transaction, "PUBLISHING_DEPLOYMENT_RESULT",
            observer=transition_observer,
        )
        cleanup = {
            "failed_residue_absent": True,
            "ownership_certain": True,
            "source_removal_committed": False,
        }
        if runtime.get("deployment_result") is None:
            runtime["deployment_result"] = _build_result(
                transaction,
                runtime,
                result_class="DEPLOYMENT_COMPLETE",
                reason_code="DEPLOYMENT_COMPLETE",
                cleanup=cleanup,
            )
            transaction["deployment_runtime"] = runtime
            _write_transaction(paths, transaction, None)
        result = runtime["deployment_result"]
        result_path, result_identity = (
            _publish_or_authenticate_deployment_result(
                paths, result, result_publisher
            )
        )
        result_published = True
        runtime["deployment_result_published"] = True
        transaction.update(
            {
                "deployment_result_identity": result_identity,
                "deployment_result_path": str(result_path),
                "deployment_runtime": runtime,
            }
        )
        _write_transaction(paths, transaction, None)
        if transition_observer is not None:
            transition_observer(
                "DEPLOYMENT_RESULT_PUBLISHED", transaction
            )

        if mode != "add":
            if runtime.get("current_connection_receipt") is None:
                runtime["current_connection_receipt"] = build_receipt(
                    observation,
                    receipt_source="DEPLOY_GGUF",
                    deployment_id=deployment_id,
                    deployment_result_identity=result_identity,
                    recommended_reference=reference,
                    promotion_result=runtime.get("promotion_result"),
                    rollback_result=None,
                    retirement_result=runtime.get(
                        "retirement_result"
                    ),
                    current_receipt_updated=True,
                )
                transaction["deployment_runtime"] = runtime
                _write_transaction(paths, transaction, None)
            current_receipt = runtime["current_connection_receipt"]
            current_identity = (
                _publish_or_authenticate_current_receipt(
                    paths,
                    current_receipt,
                    expected_previous_identity=runtime[
                        "previous_connection_receipt_identity"
                    ],
                    publisher=receipt_publisher,
                )
            )
            runtime["current_connection_receipt_identity"] = (
                current_identity
            )
            runtime["connection_receipt_identity"] = current_identity
            transaction["deployment_runtime"] = runtime
            _write_transaction(paths, transaction, None)
            if transition_observer is not None:
                transition_observer(
                    "CONNECTION_RECEIPT_PUBLISHED", transaction
                )
        else:
            try:
                unchanged = load_current_receipt(paths)
            except InspectorError as error:
                if error.reason_code != "CONNECTION_NOT_INITIALIZED":
                    raise
                unchanged = None
            if unchanged != runtime["previous_connection_receipt"]:
                raise _error(
                    "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
                    "add deployment changed the current receipt",
                )

        if runtime.get("source_cleanup") is None:
            runtime["source_cleanup"] = adapter.remove_source(
                paths, runtime["source"]
            )
            transaction["deployment_runtime"] = runtime
            _write_transaction(paths, transaction, None)
        transaction["deployment_runtime"] = runtime
        transaction = _terminal_idle(
            paths,
            transaction,
            state="COMPLETE",
            reason_code="DEPLOYMENT_COMPLETE",
        )
        return transaction_id, result, result_path, result_identity
    except DeploymentInterruption:
        transaction["deployment_runtime"] = runtime
        _write_transaction(paths, transaction, None)
        raise
    except BaseException as failure:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise
        reason = (
            failure.reason_code
            if isinstance(failure, InspectorError)
            else "DEPLOYMENT_INTERNAL_ERROR"
        )
        if result_published:
            transaction = _terminal_idle(
                paths,
                transaction,
                state="FAIL_CLOSED",
                reason_code="DEPLOYMENT_FAIL_CLOSED",
            )
            raise _error(
                "DEPLOYMENT_FAIL_CLOSED",
                "failure occurred after immutable result publication",
                internal=True,
            ) from failure
        promoted = "promotion" in runtime.get("irreversible_steps", [])
        result_class = "DEPLOYMENT_FAILED_CLEAN"
        terminal_state = "FAILED_CLEAN"
        cleanup: dict[str, Any]
        try:
            if promoted:
                transaction = _checkpoint(
                    paths, transaction, "ROLLING_BACK",
                    observer=transition_observer,
                    reason_code=reason,
                )
                rollback = adapter.rollback(paths, runtime)
                runtime["rollback_result"] = rollback
                _restore_previous_receipt(
                    paths, runtime, publisher=receipt_publisher
                )
                result_class = "DEPLOYMENT_ROLLED_BACK"
                terminal_state = "ROLLED_BACK"
                cleanup = {
                    "failed_residue_absent": bool(
                        rollback.get("candidate_residue_absent", True)
                    ),
                    "ownership_certain": True,
                    "source_removal_committed": False,
                }
            else:
                cleanup = adapter.cleanup_failed(paths, runtime)
                if cleanup.get("ownership_certain") is not True:
                    raise _error(
                        "DEPLOYMENT_OWNERSHIP_UNCERTAIN",
                        "failed cleanup ownership is uncertain",
                    )
        except BaseException as rollback_failure:
            if isinstance(
                rollback_failure, (KeyboardInterrupt, SystemExit)
            ):
                raise
            result_class = "DEPLOYMENT_FAIL_CLOSED"
            terminal_state = "FAIL_CLOSED"
            cleanup = {
                "failed_residue_absent": False,
                "ownership_certain": False,
                "source_removal_committed": False,
            }
            runtime["rollback_result"] = {
                "result_class": "ROLLBACK_FAIL_CLOSED",
                "reason_code": (
                    rollback_failure.reason_code
                    if isinstance(rollback_failure, InspectorError)
                    else "DEPLOYMENT_OWNERSHIP_UNCERTAIN"
                ),
            }
        runtime["connection_receipt"] = None
        transaction["deployment_runtime"] = runtime
        terminal_reason = {
            "DEPLOYMENT_FAILED_CLEAN": "DEPLOYMENT_FAILED_CLEAN",
            "DEPLOYMENT_ROLLED_BACK": "DEPLOYMENT_ROLLED_BACK",
            "DEPLOYMENT_FAIL_CLOSED": "DEPLOYMENT_FAIL_CLOSED",
        }[result_class]
        result = _build_result(
            transaction,
            runtime,
            result_class=result_class,
            reason_code=terminal_reason,
            cleanup=cleanup,
        )
        result_path, result_identity = result_publisher(paths, result)
        transaction.update(
            {
                "deployment_result_identity": result_identity,
                "deployment_result_path": str(result_path),
            }
        )
        transaction = _terminal_idle(
            paths,
            transaction,
            state=terminal_state,
            reason_code=terminal_reason,
        )
        return transaction_id, result, result_path, result_identity
    finally:
        lock.release()


class CurrentSourceDeploymentAdapter:
    """Narrow adapter that delegates every mutation to accepted child code."""

    def source_snapshot(
        self, paths: InspectorPaths, candidate_name: str
    ) -> dict[str, Any]:
        path = paths.intake_root / candidate_name
        if path.parent != paths.intake_root:
            raise _error(
                "DEPLOYMENT_SOURCE_INVALID", "candidate escaped intake"
            )
        identity = identify_artifact(path)
        return {
            "candidate_name": candidate_name,
            "artifact_identity": identity.identity,
            "size": identity.byte_count,
            "snapshot_identity": identity.post_inspection_snapshot_identity,
        }

    def capture_prestate(self, paths: InspectorPaths) -> dict[str, Any]:
        from .capabilities import load_binding
        from .paths import BranchHandoffPaths
        from .qualification import capture_incumbent_snapshot

        branch = BranchHandoffPaths.discover(paths)
        incumbent = capture_incumbent_snapshot(paths, branch)
        binding = load_binding(paths, "model-api-gguf")
        observation: dict[str, Any] | None = None
        if incumbent.present:
            from .connection_receipt import observe_current_connection

            observation = observe_current_connection(paths)
            if (
                observation["resolved_immutable_model_id"]
                != incumbent.public_model_id
                or observation["artifact_version_id"]
                != incumbent.artifact_version_id
                or observation["capability_manifest_identity"]
                != incumbent.capability_manifest_identity
                or observation["non_secret_key_id"]
                != incumbent.credential_key_id
                or observation["profile_identity"]
                != incumbent.profile_identity
                or observation["warm"] is not True
            ):
                raise _error(
                    "DEPLOYMENT_PRECONDITION_FAILED",
                    "incumbent connection and rollback snapshots disagree",
                )
        return {
            "desired_state": "RUNNING",
            "model_service_state": incumbent.service_readiness,
            "ready_model_count": 1 if incumbent.present else 0,
            "default_alias": incumbent.default_alias,
            "default_target": incumbent.public_model_id,
            "warm_model_id": (
                incumbent.public_model_id
                if incumbent.present
                and isinstance(incumbent.warm_before, dict)
                and incumbent.warm_before.get("health_state")
                == "ready"
                else None
            ),
            "operating_profile_identity": incumbent.profile_identity,
            "capability_binding_identity": binding[
                "binding_identity"
            ],
            "non_secret_key_id": incumbent.credential_key_id,
            "artifact_identity": (
                observation["artifact_sha256"]
                if observation is not None
                else None
            ),
            "artifact_version_id": incumbent.artifact_version_id,
            "capability_manifest_identity": (
                incumbent.capability_manifest_identity
            ),
            "resolved_immutable_model_id": incumbent.public_model_id,
            "managed_location_identity": (
                incumbent.managed_location_identity
            ),
            "registry_generation": incumbent.registry_generation,
            "recovery_state": incumbent.recovery_state,
        }

    def inspect(
        self, paths: InspectorPaths, candidate_name: str
    ) -> dict[str, Any]:
        from .config import validate_configuration_values
        from .constants import SAFETY_MAXIMA
        from .runtime import inspect_transaction

        configuration = validate_configuration_values(
            {
                "schema_version": SCHEMA_IDENTITIES["configuration"],
                "intake_root": str(paths.intake_root),
                "runtime_root": str(paths.runtime_root),
                "intake_bounds": dict(SAFETY_MAXIMA),
                "record_policy": {
                    "status_file_mode": "0600",
                    "transaction_file_mode": "0600",
                    "log_file_mode": "0600",
                },
                "result_roots": {
                    "inspection": str(paths.inspection_results),
                    "decision": str(paths.decision_results),
                    "handoff": str(paths.handoff_results),
                    "publication": str(paths.publication_results),
                },
            },
            paths,
        )
        _tx, record, _path, identity = inspect_transaction(
            paths, configuration, candidate_name
        )
        return {
            "inspection_id": record["inspection_id"],
            "identity": identity,
            "artifact_identity": record["artifact"]["identity"],
            "size": record["artifact"]["byte_count"],
            "terminal_class": record["classification"][
                "terminal_class"
            ],
        }

    def decide(
        self, paths: InspectorPaths, inspection: dict[str, Any]
    ) -> dict[str, Any]:
        from .runtime import decide_transaction

        _tx, record, _path, identity = decide_transaction(
            paths, inspection["inspection_id"]
        )
        return {
            "decision_id": record["decision_id"],
            "identity": identity,
            "capability_result": record["capability"][
                "capability_result"
            ],
        }

    def qualify(
        self,
        paths: InspectorPaths,
        inspection: dict[str, Any],
        decision: dict[str, Any],
        requested_profile: str,
    ) -> dict[str, Any]:
        from .qualification import qualify_transaction

        if decision["capability_result"] not in {
            "SUPPORTED",
            "RUNTIME_SMOKE_REQUIRED",
        }:
            raise _error(
                "DEPLOYMENT_DIRECT_ATTESTATION_REQUIRED",
                "candidate decision cannot authorize qualification",
            )
        _tx, record, _path, identity = qualify_transaction(
            paths,
            inspection["inspection_id"],
            inspection["artifact_identity"],
            requested_profile,
        )
        return {
            "qualification_id": record["qualification_id"],
            "identity": identity,
            "result_class": record["result_class"],
            "supported_profiles": record["supported_profiles"],
            "artifact_identity": record["inspection"][
                "artifact_identity"
            ],
            "capability_result": record["input_decision"][
                "capability_result"
            ],
        }

    def handoff(
        self,
        paths: InspectorPaths,
        source_candidate: str,
        decision: dict[str, Any],
        qualification: dict[str, Any],
    ) -> dict[str, Any]:
        from .handoff import handoff_transaction

        managed_name = "candidate-" + qualification[
            "artifact_identity"
        ].removeprefix("sha256:")[:16] + ".gguf"
        qualification_id = (
            qualification["qualification_id"]
            if decision["capability_result"]
            == "RUNTIME_SMOKE_REQUIRED"
            else None
        )
        _tx, record, _path, identity = handoff_transaction(
            paths,
            decision["decision_id"],
            source_candidate,
            managed_name,
            qualification_id=qualification_id,
        )
        return {
            "handoff_id": record["handoff_id"],
            "identity": identity,
            "managed_relative_path": record["publication"][
                "managed_relative_path"
            ],
        }

    def publish_candidate(
        self, paths: InspectorPaths, handoff: dict[str, Any]
    ) -> dict[str, Any]:
        from .service_publication import publish_service_transaction

        _tx, record, _path, identity = publish_service_transaction(
            paths, handoff["handoff_id"]
        )
        return {
            "publication_id": record["publication_id"],
            "identity": identity,
            "proof_request_id": record["request"]["request_id"],
            "artifact_identity": record["handoff"][
                "artifact_identity"
            ],
            "registry_generation": record["registry"]["generation"],
            "candidate_identity": {
                "resolved_immutable_model_id": record["public_service"][
                    "public_model_id"
                ],
                "artifact_version_id": record["correlation"][
                    "artifact_version_id"
                ],
                "capability_manifest_identity": record["registry"][
                    "capability_manifest_identity"
                ],
            },
        }

    def promote(
        self,
        paths: InspectorPaths,
        candidate_name: str,
        qualification: dict[str, Any],
    ) -> dict[str, Any]:
        from .promotion import promote_transaction

        _tx, record, _path, identity = promote_transaction(
            paths, qualification["qualification_id"], candidate_name
        )
        return {
            "promotion_id": record["promotion_id"],
            "identity": identity,
            "result_class": record["result_class"],
        }

    def retire(
        self,
        paths: InspectorPaths,
        incumbent: dict[str, Any],
    ) -> dict[str, Any]:
        from .retirement import (
            CurrentSourceRetirementAdapter,
            retire_transaction,
        )

        public_model_id = incumbent.get(
            "resolved_immutable_model_id"
        )
        if not isinstance(public_model_id, str):
            raise _error(
                "DEPLOYMENT_RETIREMENT_CONTEXT_INCOMPLETE",
                "incumbent public identity is absent",
            )
        runtime_adapter = CurrentSourceRetirementAdapter(paths)
        target = runtime_adapter._lookup(public_model_id)
        accepted_artifacts = {
            incumbent.get("artifact_identity"),
            incumbent.get("artifact_version_id"),
        }
        if (
            target.public_model_id != public_model_id
            or target.managed_location_identity
            != incumbent.get("managed_location_identity")
            or target.artifact_identity not in accepted_artifacts
        ):
            raise _error(
                "DEPLOYMENT_RETIREMENT_CONTEXT_INCOMPLETE",
                "incumbent retirement identity is stale",
            )
        _tx, record, _path, identity = retire_transaction(
            paths,
            public_model_id=target.public_model_id,
            artifact_identity=target.artifact_identity,
            managed_location_identity=(
                target.managed_location_identity
            ),
            expected_registry_generation=target.registry_generation,
            retirement_reason="accepted deploy-gguf replacement",
            last_model_policy="REJECT",
            adapter=runtime_adapter,
        )
        return {
            "retirement_id": record["retirement_id"],
            "identity": identity,
            "result_class": record["result_class"],
        }

    def observe_connection(
        self,
        paths: InspectorPaths,
        reference: str,
        proof_request_id: str | None,
    ) -> dict[str, Any]:
        from .connection_receipt import observe_current_connection

        return observe_current_connection(
            paths,
            reference=reference,
            proof_request_id=proof_request_id,
        )

    def rollback(
        self,
        paths: InspectorPaths,
        runtime: dict[str, Any],
    ) -> dict[str, Any]:
        from .promotion import (
            CurrentSourcePromotionAdapter,
            _alias_request,
            _incumbent_from_projection,
            authenticate_promotion_qualification,
            load_promotion_record,
        )

        promotion = runtime.get("child_data", {}).get("promotion")
        request = runtime.get("request")
        if not isinstance(promotion, dict) or not isinstance(
            request, dict
        ):
            raise _error(
                "DEPLOYMENT_ROLLBACK_OWNERSHIP_UNCERTAIN",
                "promotion rollback evidence is incomplete",
            )
        record = load_promotion_record(
            paths, promotion["promotion_id"]
        )
        if (
            record["result_identity"] != promotion["identity"]
            or record["result_class"] != "PROMOTION_COMPLETE"
        ):
            raise _error(
                "DEPLOYMENT_ROLLBACK_OWNERSHIP_UNCERTAIN",
                "promotion result cannot authorize rollback",
            )
        authorization = authenticate_promotion_qualification(
            paths,
            record["qualification"]["qualification_id"],
            request["candidate_name"],
        )
        incumbent = _incumbent_from_projection(record["incumbent"])
        adapter = CurrentSourcePromotionAdapter(
            paths, authorization, incumbent
        )
        candidate = dict(record["candidate"])
        candidate_id = candidate.get("public_model_id")
        if not isinstance(candidate_id, str):
            raise _error(
                "DEPLOYMENT_ROLLBACK_OWNERSHIP_UNCERTAIN",
                "promoted candidate identity is absent",
            )
        candidate_state = adapter.observe_exact(
            candidate, expected_default=candidate_id
        )
        alias: dict[str, Any]
        if candidate_state.get("exact") is True:
            promotion_alias = record["alias_promotion"]
            alias = adapter.alias_transaction(
                _alias_request(
                    action="rollback",
                    transaction_id=record["transaction_id"],
                    expected_target=candidate_id,
                    new_target=incumbent.snapshot.public_model_id,
                    expected_generation=int(
                        candidate_state["registry_generation"]
                    ),
                    artifact_version_id=(
                        incumbent.snapshot.artifact_version_id
                    ),
                    capability_manifest_identity=(
                        incumbent.snapshot.
                        capability_manifest_identity
                    ),
                    relative_root=incumbent.relative_root,
                    promotion_alias_event_identity=str(
                        promotion_alias["alias_event_identity"]
                    ),
                )
            )
        else:
            incumbent_state = adapter.observe_exact(
                candidate,
                expected_default=(
                    str(incumbent.snapshot.public_model_id)
                    if incumbent.snapshot.present
                    else candidate_id
                ),
            )
            if (
                not incumbent.snapshot.present
                or incumbent_state.get("exact") is not True
            ):
                raise _error(
                    "DEPLOYMENT_ROLLBACK_OWNERSHIP_UNCERTAIN",
                    "neither candidate nor incumbent is exact",
                )
            alias = {"changed": False, "already_restored": True}
        restoration = adapter.restore_incumbent(incumbent)
        disposition = adapter.retain_candidate(candidate)
        if (
            restoration.get("proved") is not True
            or disposition.get("ownership_certain") is not True
        ):
            raise _error(
                "DEPLOYMENT_ROLLBACK_OWNERSHIP_UNCERTAIN",
                "promotion rollback could not prove ownership",
            )
        return {
            "result_class": "PROMOTION_ROLLED_BACK",
            "reason_code": "DEPLOYMENT_ROLLED_BACK",
            "alias": alias,
            "restoration": restoration,
            "candidate_disposition": disposition,
            "candidate_residue_absent": False,
            "ownership_certain": True,
        }

    def cleanup_failed(
        self,
        paths: InspectorPaths,
        runtime: dict[str, Any],
    ) -> dict[str, Any]:
        children = runtime.get("child_data", {})
        if "handoff" not in children:
            return {
                "failed_residue_absent": True,
                "ownership_certain": True,
                "source_removal_committed": False,
            }
        publication = children.get("publication")
        if not isinstance(publication, dict):
            return {
                "failed_residue_absent": False,
                "ownership_certain": False,
                "source_removal_committed": False,
            }
        from .retirement import (
            CurrentSourceRetirementAdapter,
            retire_transaction,
        )

        model_id = publication.get("candidate_identity", {}).get(
            "resolved_immutable_model_id"
        )
        if not isinstance(model_id, str):
            return {
                "failed_residue_absent": False,
                "ownership_certain": False,
                "source_removal_committed": False,
            }
        runtime_adapter = CurrentSourceRetirementAdapter(paths)
        try:
            target = runtime_adapter._lookup(model_id)
            _tx, record, _path, identity = retire_transaction(
                paths,
                public_model_id=target.public_model_id,
                artifact_identity=target.artifact_identity,
                managed_location_identity=(
                    target.managed_location_identity
                ),
                expected_registry_generation=(
                    target.registry_generation
                ),
                retirement_reason="failed deploy-gguf cleanup",
                last_model_policy=(
                    "ENTER_WAITING_FOR_MODEL"
                    if target.is_last_ready
                    else "REJECT"
                ),
                adapter=runtime_adapter,
            )
        except InspectorError:
            return {
                "failed_residue_absent": False,
                "ownership_certain": False,
                "source_removal_committed": False,
            }
        clean = record["result_class"] in {
            "RETIREMENT_COMPLETE",
            "RETIREMENT_WAITING_FOR_MODEL",
        }
        return {
            "failed_residue_absent": clean,
            "ownership_certain": (
                record["result_class"] != "RETIREMENT_FAIL_CLOSED"
            ),
            "source_removal_committed": False,
            "cleanup_retirement_id": record["retirement_id"],
            "cleanup_retirement_identity": identity,
        }

    def remove_source(
        self,
        paths: InspectorPaths,
        source: dict[str, Any],
    ) -> dict[str, Any]:
        path = paths.intake_root / source["candidate_name"]
        identity = identify_artifact(path)
        if identity.identity != source["artifact_identity"]:
            raise _error(
                "DEPLOYMENT_SOURCE_CHANGED",
                "source changed before successful cleanup",
            )
        if path.is_dir():
            raise _error(
                "DEPLOYMENT_SOURCE_CLEANUP_UNSUPPORTED",
                "directory-bundle cleanup requires explicit ownership",
            )
        path.unlink()
        fsync_directory(paths.intake_root)
        return {
            "source_removed": True,
            "artifact_identity": source["artifact_identity"],
        }

    def authenticate_child(
        self,
        paths: InspectorPaths,
        name: str,
        projection: dict[str, str],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            not isinstance(projection.get("id"), str)
            or Path(projection["id"]).name != projection["id"]
        ):
            raise _error(
                "DEPLOYMENT_CHILD_RESULT_INVALID",
                "child result ID is not one basename",
            )
        if name == "inspection":
            from .decision import load_inspection_result

            record, identity = load_inspection_result(
                paths, projection["id"]
            )
            observed = {
                "inspection_id": record["inspection_id"],
                "identity": identity,
                "artifact_identity": record["artifact"]["identity"],
                "size": record["artifact"]["byte_count"],
                "terminal_class": record["classification"][
                    "terminal_class"
                ],
            }
        elif name == "decision":
            from .decision import validate_decision_record

            record = validate_decision_record(
                read_json_record(
                    paths.decision_results
                    / f"{projection['id']}.json"
                )
            )
            observed = {
                "decision_id": record["decision_id"],
                "identity": record["result_identity"],
                "capability_result": record["capability"][
                    "capability_result"
                ],
            }
        elif name == "qualification":
            from .qualification import (
                qualification_result_path,
                validate_qualification_record,
            )

            record = validate_qualification_record(
                read_json_record(
                    qualification_result_path(paths, projection["id"])
                )
            )
            observed = {
                "qualification_id": record["qualification_id"],
                "identity": record["result_identity"],
                "result_class": record["result_class"],
                "supported_profiles": record["supported_profiles"],
                "artifact_identity": record["inspection"][
                    "artifact_identity"
                ],
                "capability_result": record["input_decision"][
                    "capability_result"
                ],
            }
        elif name == "handoff":
            from .handoff import load_handoff_record

            record, identity = load_handoff_record(
                paths, projection["id"]
            )
            observed = {
                "handoff_id": record["handoff_id"],
                "identity": identity,
                "managed_relative_path": record["publication"][
                    "managed_relative_path"
                ],
            }
        elif name == "publication":
            from .service_publication import (
                publication_result_path,
                validate_publication_record,
            )

            record = validate_publication_record(
                read_json_record(
                    publication_result_path(paths, projection["id"])
                )
            )
            observed = {
                "publication_id": record["publication_id"],
                "identity": record["result_identity"],
                "proof_request_id": record["request"]["request_id"],
                "artifact_identity": record["handoff"][
                    "artifact_identity"
                ],
                "registry_generation": record["registry"][
                    "generation"
                ],
                "candidate_identity": {
                    "resolved_immutable_model_id": record[
                        "public_service"
                    ]["public_model_id"],
                    "artifact_version_id": record["correlation"][
                        "artifact_version_id"
                    ],
                    "capability_manifest_identity": record[
                        "registry"
                    ]["capability_manifest_identity"],
                },
            }
        elif name == "promotion":
            from .promotion import load_promotion_record

            record = load_promotion_record(paths, projection["id"])
            observed = {
                "promotion_id": record["promotion_id"],
                "identity": record["result_identity"],
                "result_class": record["result_class"],
            }
        elif name == "retirement":
            from .retirement import (
                retirement_result_path,
                validate_retirement_record,
            )

            record = validate_retirement_record(
                read_json_record(
                    retirement_result_path(paths, projection["id"])
                )
            )
            observed = {
                "retirement_id": record["retirement_id"],
                "identity": record["result_identity"],
                "result_class": record["result_class"],
            }
        else:
            raise _error(
                "DEPLOYMENT_CHILD_RESULT_INVALID",
                "unknown deployment child result",
            )
        if (
            observed.get("identity") != projection["identity"]
            or _projection(observed, CHILD_ID_KEYS[name])
            != projection
        ):
            raise _error(
                "DEPLOYMENT_CHILD_RESULT_INVALID",
                f"{name} child result identity changed",
            )
        return observed
