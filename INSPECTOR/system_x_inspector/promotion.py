"""Transactional GGUF default promotion contracts and orchestration."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from .capabilities import (
    load_binding,
    load_capability_record,
    verify_installed_tuple,
)
from .constants import (
    PROMOTION_REASON_CODES,
    PROMOTION_RESULT_CLASSES,
    PROMOTION_STATES,
    QUALIFICATION_PROFILES,
    SCHEMA_IDENTITIES,
)
from .decision import load_inspection_result
from .errors import InspectorError
from .handoff import (
    PublishedArtifact,
    SourceEvidence,
    create_staged_artifact,
    prepare_handoff_destination,
    publish_staged_artifact,
)
from .locking import TransactionLock, inspect_active_lock
from .paths import BranchHandoffPaths, InspectorPaths
from .qualification import (
    IncumbentSnapshot,
    PublicProfileProbeAdapter,
    _observe_default_registry,
    _observe_service_state,
    capture_incumbent_snapshot,
    cleanup_qualification_candidate,
    installed_tuple_evidence,
    observe_qualification_candidate,
    qualification_service_snapshot,
    qualification_result_path,
    read_local_credential,
    restore_with_accepted_platform_manager,
    run_capability_profile,
    wait_for_qualification_candidate,
    validate_qualification_record,
)
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


SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
QUALIFICATION_ID_PATTERN = re.compile(
    r"qualification-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
PROMOTION_ID_PATTERN = re.compile(
    r"promotion-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\Z"
)
PROMOTION_FILE_PATTERN = re.compile(
    r"promotion-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{16}\.json\Z"
)
CANDIDATE_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,238}\.gguf\Z"
)
MAX_CONTROL_JSON_BYTES = 2 * 1024 * 1024
SOURCE_HASH_CHUNK_BYTES = 1024 * 1024

PROMOTION_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "promotion_id",
        "transaction_id",
        "created_utc",
        "completed_utc",
        "qualification",
        "installed_tuple",
        "candidate",
        "incumbent",
        "states_observed",
        "registry_progression",
        "alias_promotion",
        "pre_promotion_proofs",
        "post_promotion_proofs",
        "stability_observation",
        "restart_verification",
        "rollback",
        "service_final",
        "result_class",
        "reason_codes",
        "validity_predicate",
        "result_identity",
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
        "tool_results",
        "tool_result",
    }
)


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _promotion_error(
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


def _promotion_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return f"promotion-{stamp}-{secrets.token_hex(8)}"


def _reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_RESULT_KEYS:
                raise _promotion_error(
                    "PROMOTION_RESULT_INVALID",
                    f"promotion result contains prohibited field: {key}",
                )
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def promotion_result_identity(value: dict[str, Any]) -> str:
    if "result_identity" not in value:
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion result identity field is absent",
        )
    return _identity(
        {key: value[key] for key in sorted(value) if key != "result_identity"}
    )


def validate_promotion_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PROMOTION_TOP_LEVEL_FIELDS:
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion result fields are not closed",
        )
    record = value
    _reject_forbidden(record)
    if (
        record["schema_version"]
        != SCHEMA_IDENTITIES["gguf_promotion_result"]
        or not isinstance(record["promotion_id"], str)
        or PROMOTION_ID_PATTERN.fullmatch(record["promotion_id"]) is None
        or not isinstance(record["transaction_id"], str)
        or not record["transaction_id"]
        or record["result_class"] not in PROMOTION_RESULT_CLASSES
    ):
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion result identity or enum is invalid",
        )
    for key in (
        "qualification",
        "installed_tuple",
        "candidate",
        "incumbent",
        "alias_promotion",
        "stability_observation",
        "restart_verification",
        "rollback",
        "service_final",
        "validity_predicate",
    ):
        if not isinstance(record[key], dict):
            raise _promotion_error(
                "PROMOTION_RESULT_INVALID",
                f"promotion result section is invalid: {key}",
            )
    for key in (
        "states_observed",
        "registry_progression",
        "pre_promotion_proofs",
        "post_promotion_proofs",
        "reason_codes",
    ):
        if not isinstance(record[key], list):
            raise _promotion_error(
                "PROMOTION_RESULT_INVALID",
                f"promotion result list is invalid: {key}",
            )
    states = record["states_observed"]
    if (
        not states
        or any(item not in PROMOTION_STATES for item in states)
        or len(states) != len(list(dict.fromkeys(states)))
    ):
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion state history is invalid",
        )
    reasons = record["reason_codes"]
    if (
        not reasons
        or any(item not in PROMOTION_REASON_CODES for item in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion reason codes are invalid",
        )
    expected_terminal = {
        "PROMOTION_COMPLETE": "COMPLETE",
        "PROMOTION_ROLLED_BACK": "ROLLED_BACK",
        "PROMOTION_FAILED_CLEAN": "FAILED_CLEAN",
        "PROMOTION_FAIL_CLOSED": "FAIL_CLOSED",
    }[record["result_class"]]
    if states[-1] != expected_terminal:
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion terminal state does not match its result class",
        )
    expected = promotion_result_identity(record)
    if (
        not isinstance(record["result_identity"], str)
        or SHA256_PATTERN.fullmatch(record["result_identity"]) is None
        or record["result_identity"] != expected
    ):
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion result identity is invalid",
        )
    return record


def promotion_result_path(
    paths: InspectorPaths, promotion_id: str
) -> Path:
    if PROMOTION_ID_PATTERN.fullmatch(promotion_id) is None:
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID", "promotion ID is not canonical"
        )
    return paths.promotion_results / f"{promotion_id}.json"


def _private_result_file(path: Path, parent: Path) -> None:
    details = path.lstat()
    if (
        path.parent != parent
        or stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or details.st_size > MAX_CONTROL_JSON_BYTES
    ):
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion result has an unsafe physical type",
        )


def publish_promotion_record(
    paths: InspectorPaths, value: dict[str, Any]
) -> Path:
    record = validate_promotion_record(value)
    path = promotion_result_path(paths, record["promotion_id"])
    if path.exists() or path.is_symlink():
        _private_result_file(path, paths.promotion_results)
        observed = validate_promotion_record(read_json_record(path))
        if observed == record:
            return path
        raise _promotion_error(
            "PROMOTION_RESULT_COLLISION",
            "different immutable promotion result already exists",
        )
    try:
        atomic_create_json(path, record, mode=0o600)
    except InspectorError as error:
        if path.exists() and not path.is_symlink():
            _private_result_file(path, paths.promotion_results)
            if validate_promotion_record(read_json_record(path)) == record:
                return path
        raise _promotion_error(
            "PROMOTION_RESULT_COLLISION",
            "promotion result publication collided",
        ) from error
    _private_result_file(path, paths.promotion_results)
    if validate_promotion_record(read_json_record(path)) != record:
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID",
            "promotion result did not round-trip",
        )
    return path


def load_promotion_record(
    paths: InspectorPaths, promotion_id: str
) -> dict[str, Any]:
    path = promotion_result_path(paths, promotion_id)
    try:
        _private_result_file(path, paths.promotion_results)
    except FileNotFoundError as error:
        raise _promotion_error(
            "PROMOTION_RESULT_INVALID", "promotion result is absent"
        ) from error
    return validate_promotion_record(read_json_record(path))


@dataclass(frozen=True)
class PromotionAuthorization:
    qualification: dict[str, Any]
    qualification_path: Path
    candidate_name: str
    candidate_path: Path
    candidate_snapshot: dict[str, Any]
    inspection: dict[str, Any]
    installed_tuple: dict[str, Any]

    @property
    def artifact_identity(self) -> str:
        return str(self.qualification["inspection"]["artifact_identity"])

    @property
    def requested_profile(self) -> str:
        return str(self.qualification["requested_profile"])


@dataclass(frozen=True)
class PromotionIncumbent:
    snapshot: IncumbentSnapshot
    artifact_identity: str | None
    alias_binding_identity: str
    relative_root: str | None

    def result_projection(self) -> dict[str, Any]:
        value = self.snapshot.result_projection()
        value.update(
            {
                "artifact_identity": self.artifact_identity,
                "alias_binding_identity": self.alias_binding_identity,
                "registry_generation": self.snapshot.registry_generation,
                "credential_key_id": self.snapshot.credential_key_id,
                "relative_root": self.relative_root,
                "profile_identity": self.snapshot.profile_identity,
                "service_readiness": self.snapshot.service_readiness,
                "recovery_state": self.snapshot.recovery_state,
                "api_service_transaction_id": (
                    self.snapshot.api_service_transaction_id
                ),
                "router_transaction_id": (
                    self.snapshot.router_transaction_id
                ),
                "model_child_identity": (
                    self.snapshot.model_child_identity
                ),
                "historical_registry_locations": list(
                    self.snapshot.historical_registry_locations
                ),
            }
        )
        return value


class PromotionRuntimeAdapter(Protocol):
    """Current-source runtime boundary used by production and isolated proofs."""

    def stage_candidate(
        self,
        authorization: PromotionAuthorization,
        incumbent: PromotionIncumbent,
        transaction_id: str,
    ) -> dict[str, Any]:
        """Publish one exact transaction-owned managed candidate."""

    def wait_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Return the candidate's bounded registry progression and identity."""

    def prove_candidate(
        self,
        candidate: dict[str, Any],
        requested_profile: str,
        *,
        use_default: bool,
    ) -> dict[str, Any]:
        """Run content-free profile/default request evidence."""

    def alias_transaction(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke the branch-owned local alias CAS surface."""

    def observe_exact(
        self,
        candidate: dict[str, Any],
        *,
        expected_default: str,
    ) -> dict[str, Any]:
        """Observe exact default, warm, health and ownership state."""

    def stability_parameters(self) -> tuple[int, float]:
        """Return accepted consecutive sample count and cadence."""

    def pause(self, seconds: float) -> None:
        """Wait one accepted observation cadence."""

    def capture_epochs(self) -> dict[str, Any]:
        """Capture manager/supervisor/API/router/model-child epochs."""

    def restart_and_verify(
        self,
        candidate: dict[str, Any],
        baseline: dict[str, Any],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        """Use only the accepted manager and prove a new coherent epoch."""

    def restore_incumbent(
        self, incumbent: PromotionIncumbent
    ) -> dict[str, Any]:
        """Restore and prove the exact incumbent through manager ownership."""

    def retain_candidate(
        self, candidate: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Return a bounded non-default candidate disposition."""


class CurrentSourcePromotionAdapter:
    """Production implementation over existing branch and manager surfaces."""

    def __init__(
        self,
        paths: InspectorPaths,
        authorization: PromotionAuthorization,
        incumbent: PromotionIncumbent,
    ) -> None:
        self.paths = paths
        self.authorization = authorization
        self.incumbent = incumbent
        self.branch = BranchHandoffPaths.discover(paths)
        self._published: PublishedArtifact | None = None
        self._transaction_id: str | None = None

    @staticmethod
    def _source_snapshot(path: Path) -> dict[str, int]:
        details = path.lstat()
        return {
            "device": details.st_dev,
            "inode": details.st_ino,
            "mode": stat.S_IMODE(details.st_mode),
            "link_count": details.st_nlink,
            "size_bytes": details.st_size,
            "mtime_ns": details.st_mtime_ns,
            "ctime_ns": details.st_ctime_ns,
        }

    @staticmethod
    def _managed_name(artifact_identity: str) -> str:
        digest = artifact_identity.removeprefix("sha256:")
        return f"candidate-{digest[:16]}.gguf"

    def stage_candidate(
        self,
        authorization: PromotionAuthorization,
        incumbent: PromotionIncumbent,
        transaction_id: str,
    ) -> dict[str, Any]:
        self._transaction_id = transaction_id
        managed_name = self._managed_name(authorization.artifact_identity)
        target = self.branch.managed_root / managed_name
        if target.exists() and not target.is_symlink():
            snapshot = _candidate_snapshot(target)
            if (
                snapshot["artifact_identity"]
                != authorization.artifact_identity
                or snapshot["size"]
                != authorization.candidate_snapshot["size"]
            ):
                raise _promotion_error(
                    "PROMOTION_OWNERSHIP_UNCERTAIN",
                    "existing promotion candidate identity is uncertain",
                )
            return {
                "artifact_identity": authorization.artifact_identity,
                "managed_relative_path": self.branch.relative_to_branch(
                    target
                ),
                "relative_root": managed_name,
                "publication_identity": snapshot["snapshot_identity"],
                "publication_size": snapshot["size"],
                "public_model_id": None,
                "artifact_version_id": None,
                "capability_manifest_identity": None,
                "registry_generation": None,
                "registry_states_observed": [],
            }
        details = authorization.candidate_path.lstat()
        source_snapshot = self._source_snapshot(
            authorization.candidate_path
        )
        if (
            source_snapshot["device"]
            != authorization.candidate_snapshot["device"]
            or source_snapshot["inode"]
            != authorization.candidate_snapshot["inode"]
            or source_snapshot["size_bytes"]
            != authorization.candidate_snapshot["size"]
            or source_snapshot["mtime_ns"]
            != authorization.candidate_snapshot["mtime_ns"]
            or details.st_nlink != 1
        ):
            raise _promotion_error(
                "PROMOTION_SOURCE_CHANGED",
                "candidate changed before managed staging",
            )
        source = SourceEvidence(
            path=authorization.candidate_path,
            intake_root_identity=_identity(
                {
                    "device": self.paths.intake_root.stat().st_dev,
                    "inode": self.paths.intake_root.stat().st_ino,
                }
            ),
            relative_name=authorization.candidate_name,
            snapshot=source_snapshot,
            snapshot_identity=_identity(source_snapshot),
            artifact_identity=authorization.artifact_identity,
        )
        try:
            plan = prepare_handoff_destination(
                self.branch,
                transaction_id=transaction_id,
                managed_name=managed_name,
                artifact_identity=authorization.artifact_identity,
                historical_registry_locations=(
                    incumbent.snapshot.historical_registry_locations
                ),
            )
            staged = create_staged_artifact(plan, source)
            published = publish_staged_artifact(plan, staged)
        except InspectorError as error:
            raise _promotion_error(
                "PROMOTION_STAGING_FAILED",
                "candidate managed handoff failed",
            ) from error
        self._published = published
        return {
            "artifact_identity": authorization.artifact_identity,
            "managed_relative_path": published.relative_path,
            "relative_root": managed_name,
            "publication_identity": _identity(
                {
                    "device": published.device,
                    "inode": published.inode,
                    "mode": published.mode,
                    "link_count": published.link_count,
                    "size": published.size_bytes,
                    "sha256": published.sha256,
                }
            ),
            "publication_size": published.size_bytes,
            "public_model_id": None,
            "artifact_version_id": None,
            "capability_manifest_identity": None,
            "registry_generation": None,
            "registry_states_observed": [],
        }

    def wait_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        try:
            observed = wait_for_qualification_candidate(
                self.branch.branch_root,
                str(candidate["relative_root"]),
                str(candidate["artifact_identity"]),
            )
        except InspectorError as error:
            raise _promotion_error(
                "PROMOTION_REGISTRATION_FAILED",
                "candidate registry observation failed",
            ) from error
        terminal = observed.get("terminal")
        if terminal != "READY":
            reason = {
                "REJECTED": "PROMOTION_PROBE_FAILED",
                "UNAVAILABLE": "PROMOTION_CANDIDATE_NOT_READY",
                "TIMEOUT": "PROMOTION_REGISTRATION_FAILED",
            }.get(str(terminal), "PROMOTION_REGISTRATION_FAILED")
            raise _promotion_error(
                reason,
                f"candidate registry terminal state is {terminal}",
            )
        return {
            **candidate,
            "public_model_id": observed["public_model_id"],
            "artifact_version_id": observed["artifact_version_id"],
            "capability_manifest_identity": observed[
                "capability_manifest_identity"
            ],
            "registry_generation": observed["registry_generation"],
            "registry_states_observed": list(
                observed["states_observed"]
            ),
            "default_bound": observed["default_bound"],
        }

    def prove_candidate(
        self,
        candidate: dict[str, Any],
        requested_profile: str,
        *,
        use_default: bool,
    ) -> dict[str, Any]:
        try:
            service = qualification_service_snapshot(
                SimpleNamespace(branch_paths=self.branch),
                self.incumbent.snapshot,
            )
            credential = read_local_credential(self.branch.branch_root)
            if credential.key_id != self.incumbent.snapshot.credential_key_id:
                raise _promotion_error(
                    "PROMOTION_CANDIDATE_REQUEST_FAILED",
                    "credential identity changed during promotion",
                )
            probe = PublicProfileProbeAdapter(
                service,
                credential,
                registry_states=candidate["registry_states_observed"],
                public_model_id=candidate["public_model_id"],
                artifact_version_id=candidate["artifact_version_id"],
                capability_manifest_identity=candidate[
                    "capability_manifest_identity"
                ],
            )
            profile = run_capability_profile(
                probe,
                requested_profile=requested_profile,
                model_id=candidate["public_model_id"],
                artifact_version_id=candidate["artifact_version_id"],
                capability_manifest_identity=candidate[
                    "capability_manifest_identity"
                ],
            )
        except InspectorError as error:
            raise _promotion_error(
                (
                    "PROMOTION_POST_REQUEST_FAILED"
                    if use_default
                    else "PROMOTION_CANDIDATE_REQUEST_FAILED"
                ),
                "candidate public proof failed",
            ) from error
        passed = (
            profile.result_class == "SUPPORTED_FOR_CURRENT_TUPLE"
            and requested_profile in profile.supported_profiles
        )
        if not passed:
            raise _promotion_error(
                (
                    "PROMOTION_POST_REQUEST_FAILED"
                    if use_default
                    else "PROMOTION_CANDIDATE_REQUEST_FAILED"
                ),
                "candidate no longer satisfies its qualified profile",
            )
        return {
            "passed": True,
            "requested_model_reference": (
                "default" if use_default else candidate["public_model_id"]
            ),
            "resolved_public_model_id": candidate["public_model_id"],
            "requested_profile": requested_profile,
            "supported_profiles": list(profile.supported_profiles),
            "checks": list(profile.checks),
            "credential_key_id": credential.key_id,
            "profile_identity": service.profile_identity,
        }

    def alias_transaction(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        controller = (
            self.branch.branch_root
            / "api_service_controller"
            / "controller.py"
        )
        python = (
            self.branch.branch_root
            / "api_service"
            / ".venv"
            / "bin"
            / "python"
        )
        payload = {
            "schema_version": "system-x.gguf-alias-transaction.v1",
            **request,
        }
        encoded = canonical_json_bytes(payload)
        if len(encoded) > 65_536:
            raise _promotion_error(
                "PROMOTION_ALIAS_CAS_FAILED",
                "alias transaction request exceeds its bound",
            )
        environment = {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        completed = subprocess.run(
            [str(python), "-B", str(controller), "alias-transaction"],
            input=encoded,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.branch.branch_root,
            env=environment,
            timeout=30.0,
            check=False,
        )
        try:
            value = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _promotion_error(
                "PROMOTION_ALIAS_CAS_FAILED",
                "alias transaction emitted invalid JSON",
            ) from error
        if (
            completed.returncode != 0
            or not isinstance(value, dict)
            or value.get("ok") is not True
            or not isinstance(value.get("alias_transaction"), dict)
        ):
            reason = (
                "PROMOTION_ROLLBACK_ALIAS_CONFLICT"
                if request["action"] == "rollback"
                else "PROMOTION_ALIAS_CAS_FAILED"
            )
            raise _promotion_error(
                reason,
                "branch-owned alias transaction was rejected",
                data={"controller_reason_code": value.get("reason_code")},
            )
        return dict(value["alias_transaction"])

    def observe_exact(
        self,
        candidate: dict[str, Any],
        *,
        expected_default: str,
    ) -> dict[str, Any]:
        try:
            registry = _observe_default_registry(self.branch.branch_root)
            service = _observe_service_state(
                self.branch.branch_root, registry
            )
        except InspectorError as error:
            raise _promotion_error(
                "PROMOTION_WARM_FAILED",
                "exact promotion service observation failed",
            ) from error
        exact = bool(
            registry.get("public_model_id") == expected_default
            and service.get("service_readiness") == "READY"
            and service.get("recovery_state") == "IDLE"
            and isinstance(service.get("warm"), dict)
            and service["warm"].get("public_model_id")
            == expected_default
            and service["warm"].get("artifact_version_id")
            == (
                candidate.get("artifact_version_id")
                if expected_default == candidate.get("public_model_id")
                else self.incumbent.snapshot.artifact_version_id
            )
        )
        return {
            "exact": exact,
            "registry_generation": registry.get("generation"),
            "default_alias": registry.get("default_alias"),
            "default_target": registry.get("public_model_id"),
            "artifact_version_id": registry.get("artifact_version_id"),
            "service_readiness": service.get("service_readiness"),
            "inference_ready": service.get("service_readiness") == "READY",
            "recovery_state": service.get("recovery_state"),
            "fail_closed_latch": False,
            "warm_public_model_id": (
                service["warm"].get("public_model_id")
                if isinstance(service.get("warm"), dict)
                else None
            ),
            "api_listener_owned": True,
            "router_listener_owned": True,
            "api_service_transaction_id": service.get(
                "api_service_transaction_id"
            ),
            "router_transaction_id": service.get(
                "router_transaction_id"
            ),
            "model_child_identity": service.get("model_child_identity"),
        }

    def stability_parameters(self) -> tuple[int, float]:
        path = (
            self.branch.branch_root
            / "RUNTIME"
            / "api"
            / "status"
            / "service.json"
        )
        value = read_json_record(path)
        count = value.get("registry_stability_samples")
        interval = value.get("registry_stability_interval_seconds")
        if (
            type(count) is not int
            or not 3 <= count <= 20
            or type(interval) not in {int, float}
            or not 0.0 <= float(interval) <= 10.0
        ):
            raise _promotion_error(
                "PROMOTION_STABILITY_FAILED",
                "accepted stability cadence is invalid",
            )
        return count, float(interval)

    def pause(self, seconds: float) -> None:
        time.sleep(seconds)

    def capture_epochs(self) -> dict[str, Any]:
        registry = _observe_default_registry(self.branch.branch_root)
        service = _observe_service_state(
            self.branch.branch_root, registry
        )
        desired = read_json_record(
            self.branch.branch_root
            / "RUNTIME"
            / "service_control"
            / "desired-state.json"
        )
        supervisor = read_json_record(
            self.branch.branch_root
            / "RUNTIME"
            / "service_control"
            / "status"
            / "supervisor.json"
        )
        return {
            "manager": desired.get("desired_state_generation"),
            "supervisor": _identity(
                {
                    "updated_utc": supervisor.get("updated_utc"),
                    "pid": supervisor.get("supervisor_pid"),
                }
            ),
            "api": service.get("api_service_transaction_id"),
            "router": service.get("router_transaction_id"),
            "model_child": _identity(service.get("model_child_identity")),
        }

    def restart_and_verify(
        self,
        candidate: dict[str, Any],
        baseline: dict[str, Any],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        current = self.capture_epochs()
        already_changed = all(
            current.get(key) != baseline.get(key)
            for key in ("manager", "supervisor", "api", "router", "model_child")
        )
        manager_result: dict[str, Any] = {"used": False}
        if not (resume and already_changed):
            try:
                manager_result = restore_with_accepted_platform_manager(
                    self.branch.branch_root
                )
            except InspectorError as error:
                raise _promotion_error(
                    "PROMOTION_RESTART_FAILED",
                    "accepted manager restart failed",
                ) from error
            current = self.capture_epochs()
        changed = {
            key: current.get(key) != baseline.get(key)
            for key in ("manager", "supervisor", "api", "router", "model_child")
        }
        observation = self.observe_exact(
            candidate,
            expected_default=str(candidate["public_model_id"]),
        )
        if not all(changed.values()) or observation["exact"] is not True:
            raise _promotion_error(
                "PROMOTION_RESTART_FAILED",
                "restart did not produce all required coherent epochs",
            )
        later = self.prove_candidate(
            candidate,
            "CORE_CHAT",
            use_default=True,
        )
        return {
            "passed": True,
            "resumed_without_duplicate_restart": (
                resume and already_changed
            ),
            "epochs_before": baseline,
            "epochs_after": current,
            "epoch_changes": changed,
            "manager_result": manager_result,
            "later_default_request_passed": later["passed"],
        }

    def restore_incumbent(
        self, incumbent: PromotionIncumbent
    ) -> dict[str, Any]:
        if not incumbent.snapshot.present:
            return {
                "proved": True,
                "waiting_for_model": True,
                "manager_restart_used": False,
            }
        candidate = {
            "artifact_version_id": incumbent.snapshot.artifact_version_id
        }
        observed = self.observe_exact(
            candidate,
            expected_default=str(incumbent.snapshot.public_model_id),
        )
        manager_used = False
        if not observed["exact"]:
            try:
                restore_with_accepted_platform_manager(
                    self.branch.branch_root
                )
            except InspectorError as error:
                raise _promotion_error(
                    "PROMOTION_INCUMBENT_RESTORATION_FAILED",
                    "accepted manager could not restore incumbent",
                ) from error
            manager_used = True
            observed = self.observe_exact(
                candidate,
                expected_default=str(incumbent.snapshot.public_model_id),
            )
        if not observed["exact"]:
            raise _promotion_error(
                "PROMOTION_INCUMBENT_RESTORATION_FAILED",
                "incumbent restoration did not converge",
            )
        return {
            "proved": True,
            "manager_restart_used": manager_used,
            "default_target": observed["default_target"],
            "service_readiness": observed["service_readiness"],
            "recovery_state": observed["recovery_state"],
        }

    def retain_candidate(
        self, candidate: dict[str, Any] | None
    ) -> dict[str, Any]:
        if candidate is None:
            managed_name = self._managed_name(
                self.authorization.artifact_identity
            )
            target = self.branch.managed_root / managed_name
            if self._transaction_id is None:
                return {
                    "disposition": "PRESERVE_FOR_REVIEW",
                    "ownership_certain": False,
                }
            name_identity = hashlib.sha256(
                managed_name.encode("utf-8")
            ).hexdigest()[:16]
            staging = self.branch.branch_staging_root / (
                f".{self._transaction_id}.{name_identity}."
                "partial-staging.gguf"
            )
            try:
                staging_details = staging.lstat()
            except FileNotFoundError:
                staging_details = None
            if staging_details is not None:
                if (
                    stat.S_ISLNK(staging_details.st_mode)
                    or not stat.S_ISREG(staging_details.st_mode)
                    or staging_details.st_nlink != 1
                ):
                    return {
                        "disposition": "PRESERVE_FOR_REVIEW",
                        "ownership_certain": False,
                    }
                staging.unlink()
                fsync_directory(staging.parent)
            if not target.exists() and not target.is_symlink():
                return {
                    "disposition": "REMOVE_CANDIDATE",
                    "ownership_certain": (
                        not staging.exists() and not staging.is_symlink()
                    ),
                    "candidate_state": "ABSENT",
                }
            try:
                snapshot = _candidate_snapshot(target)
            except InspectorError:
                return {
                    "disposition": "PRESERVE_FOR_REVIEW",
                    "ownership_certain": False,
                }
            if (
                snapshot["artifact_identity"]
                != self.authorization.artifact_identity
            ):
                return {
                    "disposition": "PRESERVE_FOR_REVIEW",
                    "ownership_certain": False,
                }
            candidate = {
                "relative_root": managed_name,
                "artifact_identity": self.authorization.artifact_identity,
            }
        try:
            observed = observe_qualification_candidate(
                self.branch.branch_root,
                str(candidate["relative_root"]),
                str(candidate["artifact_identity"]),
            )
        except InspectorError:
            return {
                "disposition": "PRESERVE_FOR_REVIEW",
                "ownership_certain": False,
            }
        return {
            "disposition": "RETAIN_NON_DEFAULT",
            "ownership_certain": observed.get("default_bound") is False,
            "candidate_state": observed.get("terminal"),
        }


InstalledTupleResolver = Callable[
    [InspectorPaths, dict[str, Any]], dict[str, Any]
]


def _current_installed_tuple(
    paths: InspectorPaths, _record: dict[str, Any]
) -> dict[str, Any]:
    try:
        binding = load_binding(paths, "model-api-gguf")
        capability = load_capability_record(
            paths, binding["capability_record_id"]
        )
        verification = verify_installed_tuple(paths, capability)
    except InspectorError as error:
        raise _promotion_error(
            "PROMOTION_CAPABILITY_BINDING_INVALID",
            "current GGUF capability graph cannot authenticate promotion",
        ) from error
    if verification.get("verified") is not True:
        raise _promotion_error(
            "PROMOTION_INSTALLED_TUPLE_MISMATCH",
            "current installed tuple differs from its capability binding",
        )
    try:
        current = installed_tuple_evidence(
            capability, binding, verification
        )
    except InspectorError as error:
        raise _promotion_error(
            "PROMOTION_INSTALLED_TUPLE_MISMATCH",
            "current installed tuple evidence is incomplete",
        ) from error
    return current


def _candidate_snapshot(path: Path) -> dict[str, Any]:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _promotion_error(
            "PROMOTION_SOURCE_NOT_FOUND", "candidate source is absent"
        ) from error
    if stat.S_ISLNK(details.st_mode):
        raise _promotion_error(
            "PROMOTION_SOURCE_SYMLINK", "candidate source is a symlink"
        )
    if not stat.S_ISREG(details.st_mode):
        raise _promotion_error(
            "PROMOTION_SOURCE_INVALID",
            "candidate source is not a regular file",
        )
    if details.st_nlink != 1:
        raise _promotion_error(
            "PROMOTION_SOURCE_HARDLINK_REJECTED",
            "candidate source link count is not one",
        )
    digest = hashlib.sha256()
    byte_count = 0
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        opened = os.fstat(descriptor)
        before = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
        )
        while True:
            chunk = os.read(descriptor, SOURCE_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except FileNotFoundError as error:
        raise _promotion_error(
            "PROMOTION_SOURCE_CHANGED",
            "candidate source disappeared during revalidation",
        ) from error
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_mode,
        path_after.st_nlink,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if before != after_identity or before != path_identity:
        raise _promotion_error(
            "PROMOTION_SOURCE_CHANGED",
            "candidate source changed during revalidation",
        )
    if byte_count != details.st_size:
        raise _promotion_error(
            "PROMOTION_SOURCE_CHANGED",
            "candidate source size changed during revalidation",
        )
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": stat.S_IMODE(details.st_mode),
        "link_count": details.st_nlink,
        "size": byte_count,
        "mtime_ns": details.st_mtime_ns,
        "artifact_identity": "sha256:" + digest.hexdigest(),
        "snapshot_identity": _identity(
            {
                "device": details.st_dev,
                "inode": details.st_ino,
                "mode": stat.S_IMODE(details.st_mode),
                "link_count": details.st_nlink,
                "size": byte_count,
                "mtime_ns": details.st_mtime_ns,
            }
        ),
    }


def _qualification_file(
    paths: InspectorPaths, qualification_id: str
) -> tuple[Path, dict[str, Any]]:
    if QUALIFICATION_ID_PATTERN.fullmatch(qualification_id) is None:
        raise _promotion_error(
            "PROMOTION_INPUT_INVALID",
            "qualification ID is not canonical",
        )
    path = qualification_result_path(paths, qualification_id)
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_NOT_FOUND",
            "qualification result is absent",
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
        or details.st_size > MAX_CONTROL_JSON_BYTES
    ):
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_INVALID",
            "qualification result has an unsafe physical type",
        )
    try:
        record = validate_qualification_record(read_json_record(path))
    except InspectorError as error:
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_INVALID",
            "qualification result authentication failed",
        ) from error
    if record["qualification_id"] != qualification_id:
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_INVALID",
            "qualification identity does not match its path",
        )
    return path, record


def authenticate_promotion_qualification(
    paths: InspectorPaths,
    qualification_id: str,
    candidate_name: str,
    *,
    installed_tuple_resolver: InstalledTupleResolver = (
        _current_installed_tuple
    ),
) -> PromotionAuthorization:
    if (
        not isinstance(candidate_name, str)
        or CANDIDATE_NAME_PATTERN.fullmatch(candidate_name) is None
        or Path(candidate_name).name != candidate_name
    ):
        raise _promotion_error(
            "PROMOTION_INPUT_INVALID",
            "candidate name is not one direct GGUF child",
        )
    path, record = _qualification_file(paths, qualification_id)
    if record["result_class"] != "SUPPORTED_FOR_CURRENT_TUPLE":
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_NOT_PROMOTABLE",
            "qualification result class cannot authorize promotion",
            data={
                "qualification_id": qualification_id,
                "result_class": record["result_class"],
                "result_identity": record["result_identity"],
            },
        )
    requested_profile = record["requested_profile"]
    if (
        requested_profile not in QUALIFICATION_PROFILES
        or requested_profile not in record["supported_profiles"]
    ):
        raise _promotion_error(
            "PROMOTION_PROFILE_NOT_SUPPORTED",
            "qualification does not support its requested profile",
        )
    inspection_projection = record["inspection"]
    if (
        not isinstance(inspection_projection, dict)
        or inspection_projection.get("physical_format") != "GGUF"
        or inspection_projection.get("model_type") != "model"
    ):
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_INVALID",
            "qualification is not for one GGUF primary model",
        )
    try:
        inspection, inspection_identity = load_inspection_result(
            paths, inspection_projection["inspection_id"]
        )
    except (KeyError, InspectorError) as error:
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_STALE",
            "qualification inspection evidence is unavailable",
        ) from error
    if (
        inspection_identity
        != inspection_projection.get("inspection_result_identity")
        or inspection["artifact"]["identity"]
        != inspection_projection.get("artifact_identity")
        or inspection["normalized"]["model_type"] != "model"
        or inspection["classification"]["terminal_class"] != "GGUF"
    ):
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_STALE",
            "qualification inspection evidence no longer authenticates",
        )
    candidate_path = paths.intake_root / candidate_name
    if candidate_path.parent != paths.intake_root:
        raise _promotion_error(
            "PROMOTION_INPUT_INVALID",
            "candidate path escaped MODEL-TEST",
        )
    snapshot = _candidate_snapshot(candidate_path)
    if (
        snapshot["artifact_identity"]
        != inspection_projection.get("artifact_identity")
        or snapshot["size"] != inspection_projection.get("artifact_size")
    ):
        raise _promotion_error(
            "PROMOTION_ARTIFACT_IDENTITY_MISMATCH",
            "current source differs from qualification evidence",
        )
    current_tuple = installed_tuple_resolver(paths, record)
    if record["installed_tuple"] != current_tuple:
        raise _promotion_error(
            "PROMOTION_INSTALLED_TUPLE_MISMATCH",
            "qualification installed tuple is stale",
        )
    basis = {
        "artifact_identity": snapshot["artifact_identity"],
        "capability_record_identity": current_tuple[
            "branch_capability_record_identity"
        ],
        "binding_identity": current_tuple[
            "capability_binding_identity"
        ],
        "installed_tuple_verification_identity": current_tuple[
            "installed_tuple_verification_identity"
        ],
        "llama_cpp_commit": current_tuple["llama_cpp_commit"],
        "llama_server_sha256": current_tuple["llama_server_sha256"],
        "connected_source_identity": current_tuple[
            "connected_source_identity"
        ],
    }
    validity = {**basis, "predicate_identity": _identity(basis)}
    if record["validity_predicate"] != validity:
        raise _promotion_error(
            "PROMOTION_QUALIFICATION_STALE",
            "qualification validity predicate is not current",
        )
    return PromotionAuthorization(
        qualification=record,
        qualification_path=path,
        candidate_name=candidate_name,
        candidate_path=candidate_path,
        candidate_snapshot=snapshot,
        inspection=inspection,
        installed_tuple=current_tuple,
    )


def capture_promotion_incumbent(
    paths: InspectorPaths,
    branch_paths: BranchHandoffPaths | None = None,
    *,
    snapshotter: Callable[
        [InspectorPaths, BranchHandoffPaths], IncumbentSnapshot
    ] = capture_incumbent_snapshot,
) -> PromotionIncumbent:
    branch = branch_paths or BranchHandoffPaths.discover(paths)
    snapshot = snapshotter(paths, branch)
    relative_root: str | None = None
    if snapshot.present:
        if (
            snapshot.service_readiness != "READY"
            or snapshot.recovery_state != "IDLE"
            or not isinstance(snapshot.warm_before, dict)
            or snapshot.warm_before.get("public_model_id")
            != snapshot.public_model_id
            or snapshot.warm_before.get("artifact_version_id")
            != snapshot.artifact_version_id
        ):
            raise _promotion_error(
                "PROMOTION_INCUMBENT_NOT_READY",
                "incumbent is not exact, READY and warm",
            )
        artifact_version = snapshot.artifact_version_id
        if (
            not isinstance(artifact_version, str)
            or re.fullmatch(r"bundle-[0-9a-f]{64}", artifact_version)
            is None
        ):
            raise _promotion_error(
                "PROMOTION_INCUMBENT_INVALID",
                "incumbent artifact version is invalid",
            )
        artifact_identity = "sha256:" + artifact_version.removeprefix(
            "bundle-"
        )
        database = (
            branch.branch_root
            / "RUNTIME"
            / "api"
            / "database"
            / "model_registry.sqlite3"
        )
        if (
            not database.exists()
            and len(snapshot.historical_registry_locations) == 1
        ):
            relative_root = snapshot.historical_registry_locations[0]
            rows = [(relative_root,)]
        else:
            rows = []
        try:
            if database.exists():
                connection = sqlite3.connect(
                    f"{database.as_uri()}?mode=ro",
                    uri=True,
                    timeout=5.0,
                )
                connection.execute("PRAGMA query_only=ON")
                rows = connection.execute(
                    """
                    SELECT mvl.relative_root
                    FROM model_version_locations AS mvl
                    JOIN artifact_locations AS al
                      ON al.relative_root=mvl.relative_root
                    WHERE mvl.model_version_id=?
                      AND al.current_bundle_id=?
                      AND al.present=1
                    """,
                    (snapshot.public_model_id, snapshot.artifact_version_id),
                ).fetchall()
        except sqlite3.Error as error:
            raise _promotion_error(
                "PROMOTION_INCUMBENT_INVALID",
                "incumbent managed location cannot be authenticated",
            ) from error
        finally:
            if "connection" in locals():
                connection.close()
        if len(rows) != 1 or not isinstance(rows[0][0], str):
            raise _promotion_error(
                "PROMOTION_INCUMBENT_INVALID",
                "incumbent does not have one active managed location",
            )
        relative_root = str(rows[0][0])
    else:
        if (
            snapshot.service_readiness != "WAITING_FOR_MODEL"
            or snapshot.warm_before is not None
        ):
            raise _promotion_error(
                "PROMOTION_INCUMBENT_INVALID",
                "no-model incumbent snapshot is not healthy",
            )
        artifact_identity = None
    binding_basis = {
        "default_alias": snapshot.default_alias,
        "public_model_id": snapshot.public_model_id,
        "artifact_version_id": snapshot.artifact_version_id,
        "capability_manifest_identity": (
            snapshot.capability_manifest_identity
        ),
        "managed_location_identity": snapshot.managed_location_identity,
        "registry_generation": snapshot.registry_generation,
    }
    return PromotionIncumbent(
        snapshot=snapshot,
        artifact_identity=artifact_identity,
        alias_binding_identity=_identity(binding_basis),
        relative_root=relative_root,
    )


def promotion_transaction_value(
    paths: InspectorPaths,
    *,
    transaction_id: str,
    authorization: PromotionAuthorization,
    incumbent: PromotionIncumbent,
    state: str,
    start_utc: str,
    finish_utc: str | None = None,
    reason_code: str = "OK",
    promotion_id: str | None = None,
    states_observed: list[str] | None = None,
    intended_action: str | None = None,
    irreversible_action_begun: str | None = None,
    irreversible_action_observed: str | None = None,
    promotion_result_identity_value: str | None = None,
    promotion_result_path_value: str | None = None,
    promotion_basis_identity: str | None = None,
    promotion_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in PROMOTION_STATES:
        raise _promotion_error(
            "PROMOTION_INTERNAL_ERROR",
            "promotion transaction state is invalid",
            internal=True,
        )
    return {
        "schema_version": SCHEMA_IDENTITIES["transaction"],
        "transaction_id": transaction_id,
        "operation": "promote-gguf",
        "start_utc": start_utc,
        "finish_utc": finish_utc,
        "state": state,
        "reason_code": reason_code,
        "input_target_name": authorization.candidate_name,
        "intake_snapshot_identity": authorization.candidate_snapshot[
            "snapshot_identity"
        ],
        "owner_identity": {},
        "status_record_identity": None,
        "qualification_id": authorization.qualification[
            "qualification_id"
        ],
        "qualification_result_identity": authorization.qualification[
            "result_identity"
        ],
        "candidate_artifact_identity": authorization.artifact_identity,
        "candidate_name": authorization.candidate_name,
        "requested_profile": authorization.requested_profile,
        "incumbent_snapshot": incumbent.result_projection(),
        "promotion_id": promotion_id,
        "promotion_result_identity": promotion_result_identity_value,
        "promotion_result_path": promotion_result_path_value,
        "states_observed": states_observed or [state],
        "intended_action": intended_action,
        "irreversible_action_begun": irreversible_action_begun,
        "irreversible_action_observed": irreversible_action_observed,
        "promotion_basis_identity": promotion_basis_identity,
        "promotion_runtime": promotion_runtime or {},
    }


def persist_promotion_state(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    *,
    state: str,
    reason_code: str = "OK",
    intended_action: str | None = None,
    irreversible_action_begun: str | None = None,
    irreversible_action_observed: str | None = None,
    finish_utc: str | None = None,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if state not in PROMOTION_STATES:
        raise _promotion_error(
            "PROMOTION_INTERNAL_ERROR",
            "promotion state transition is invalid",
            internal=True,
        )
    observed = list(transaction.get("states_observed", []))
    if not observed or observed[-1] != state:
        observed.append(state)
    value = {
        **transaction,
        "state": state,
        "reason_code": reason_code,
        "finish_utc": finish_utc,
        "states_observed": observed,
        "intended_action": intended_action,
        "irreversible_action_begun": irreversible_action_begun,
        "irreversible_action_observed": irreversible_action_observed,
    }
    terminal = state in {
        "COMPLETE",
        "ROLLED_BACK",
        "FAILED_CLEAN",
        "FAIL_CLOSED",
    }
    status = _status_value(
        paths,
        state=(
            "FAIL_CLOSED"
            if state == "FAIL_CLOSED"
            else "IDLE"
            if terminal
            else state
        ),
        reason_code=(
            "PROMOTION_FAIL_CLOSED"
            if state == "FAIL_CLOSED"
            else "OK"
            if terminal
            else reason_code
        ),
        active_transaction_id=(
            None
            if terminal
            else value["transaction_id"]
        ),
        last_transaction_id=(
            value["transaction_id"]
            if terminal
            else None
        ),
    )
    status_identity = _write_status(paths, status, observer)
    value["status_record_identity"] = status_identity
    _write_transaction(paths, value, observer)
    return value


def build_promotion_record(
    *,
    promotion_id: str,
    transaction_id: str,
    created_utc: str,
    completed_utc: str,
    qualification: dict[str, Any],
    installed_tuple: dict[str, Any],
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    states_observed: list[str],
    registry_progression: list[dict[str, Any]],
    alias_promotion: dict[str, Any],
    pre_promotion_proofs: list[dict[str, Any]],
    post_promotion_proofs: list[dict[str, Any]],
    stability_observation: dict[str, Any],
    restart_verification: dict[str, Any],
    rollback: dict[str, Any],
    service_final: dict[str, Any],
    result_class: str,
    reason_codes: list[str],
    validity_predicate: dict[str, Any],
) -> dict[str, Any]:
    basis = {
        "schema_version": SCHEMA_IDENTITIES["gguf_promotion_result"],
        "promotion_id": promotion_id,
        "transaction_id": transaction_id,
        "created_utc": created_utc,
        "completed_utc": completed_utc,
        "qualification": qualification,
        "installed_tuple": installed_tuple,
        "candidate": candidate,
        "incumbent": incumbent,
        "states_observed": states_observed,
        "registry_progression": registry_progression,
        "alias_promotion": alias_promotion,
        "pre_promotion_proofs": pre_promotion_proofs,
        "post_promotion_proofs": post_promotion_proofs,
        "stability_observation": stability_observation,
        "restart_verification": restart_verification,
        "rollback": rollback,
        "service_final": service_final,
        "result_class": result_class,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "validity_predicate": validity_predicate,
    }
    record = {**basis, "result_identity": _identity(basis)}
    return validate_promotion_record(record)


def _qualification_projection(
    authorization: PromotionAuthorization,
) -> dict[str, Any]:
    return {
        "qualification_id": authorization.qualification[
            "qualification_id"
        ],
        "result_identity": authorization.qualification["result_identity"],
        "result_class": authorization.qualification["result_class"],
        "requested_profile": authorization.requested_profile,
        "inspection_id": authorization.qualification["inspection"].get(
            "inspection_id"
        ),
        "artifact_identity": authorization.artifact_identity,
    }


def _promotion_basis(
    authorization: PromotionAuthorization,
    incumbent: PromotionIncumbent,
) -> dict[str, Any]:
    return {
        "qualification_result_identity": authorization.qualification[
            "result_identity"
        ],
        "candidate_artifact_identity": authorization.artifact_identity,
        "incumbent_alias_binding_identity": (
            incumbent.alias_binding_identity
        ),
        "installed_tuple_identity": _identity(
            authorization.installed_tuple
        ),
        "requested_profile": authorization.requested_profile,
    }


def find_idempotent_promotion(
    paths: InspectorPaths,
    authorization: PromotionAuthorization,
) -> tuple[dict[str, Any], Path] | None:
    if not paths.promotion_results.exists():
        return None
    for result_path in sorted(paths.promotion_results.iterdir()):
        if PROMOTION_FILE_PATTERN.fullmatch(result_path.name) is None:
            continue
        try:
            _private_result_file(result_path, paths.promotion_results)
            record = validate_promotion_record(
                read_json_record(result_path)
            )
        except (OSError, InspectorError):
            continue
        if (
            record["result_class"]
            not in {"PROMOTION_COMPLETE", "PROMOTION_ROLLED_BACK"}
            or record["qualification"].get("result_identity")
            != authorization.qualification["result_identity"]
            or record["candidate"].get("artifact_identity")
            != authorization.artifact_identity
            or record["qualification"].get("requested_profile")
            != authorization.requested_profile
            or record["installed_tuple"] != authorization.installed_tuple
        ):
            continue
        return record, result_path
    return None


def _incumbent_from_projection(value: object) -> PromotionIncumbent:
    if not isinstance(value, dict):
        raise _promotion_error(
            "PROMOTION_OWNERSHIP_UNCERTAIN",
            "recoverable incumbent snapshot is absent",
        )
    try:
        snapshot = IncumbentSnapshot(
            present=bool(value["present"]),
            default_alias=value["default_alias"],
            public_model_id=value["public_model_id"],
            artifact_version_id=value["artifact_version_id"],
            capability_manifest_identity=value[
                "capability_manifest_identity"
            ],
            managed_location_identity=value[
                "managed_location_identity"
            ],
            warm_before=value["warm_before"],
            registry_generation=int(value["registry_generation"]),
            credential_key_id=str(value["credential_key_id"]),
            profile_identity=str(value["profile_identity"]),
            service_readiness=str(value["service_readiness"]),
            recovery_state=str(value["recovery_state"]),
            api_service_transaction_id=value[
                "api_service_transaction_id"
            ],
            router_transaction_id=value["router_transaction_id"],
            model_child_identity=value["model_child_identity"],
            historical_registry_locations=tuple(
                value["historical_registry_locations"]
            ),
        )
        artifact_identity = value["artifact_identity"]
        binding_identity = str(value["alias_binding_identity"])
        relative_root = value["relative_root"]
    except (KeyError, TypeError, ValueError) as error:
        raise _promotion_error(
            "PROMOTION_OWNERSHIP_UNCERTAIN",
            "recoverable incumbent snapshot is invalid",
        ) from error
    return PromotionIncumbent(
        snapshot=snapshot,
        artifact_identity=artifact_identity,
        alias_binding_identity=binding_identity,
        relative_root=relative_root,
    )


def _find_recoverable_promotion(
    paths: InspectorPaths,
    authorization: PromotionAuthorization,
) -> dict[str, Any] | None:
    if not paths.transactions.exists():
        return None
    matches: list[dict[str, Any]] = []
    for path in sorted(paths.transactions.glob("*.json")):
        try:
            details = path.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
            ):
                continue
            value = read_json_record(path)
        except (OSError, InspectorError):
            continue
        if (
            isinstance(value, dict)
            and value.get("operation") == "promote-gguf"
            and value.get("qualification_result_identity")
            == authorization.qualification["result_identity"]
            and value.get("candidate_artifact_identity")
            == authorization.artifact_identity
            and value.get("candidate_name")
            == authorization.candidate_name
            and value.get("state")
            not in {
                "COMPLETE",
                "ROLLED_BACK",
                "FAILED_CLEAN",
                "FAIL_CLOSED",
            }
        ):
            matches.append(value)
    if len(matches) > 1:
        raise _promotion_error(
            "PROMOTION_OWNERSHIP_UNCERTAIN",
            "multiple recoverable promotions match the candidate",
        )
    return matches[0] if matches else None


def _clear_exact_stale_lock(
    paths: InspectorPaths, transaction_id: str
) -> None:
    observed = inspect_active_lock(paths.locks / "active.json")
    if observed["state"] == "absent":
        return
    record = observed.get("record")
    exact = bool(
        isinstance(record, dict)
        and record.get("transaction_id") == transaction_id
        and record.get("operation") == "promote-gguf"
    )
    if observed["state"] != "stale" or not exact:
        raise _promotion_error(
            (
                "PROMOTION_CONCURRENCY_REJECTED"
                if observed["state"] == "active"
                else "PROMOTION_OWNERSHIP_UNCERTAIN"
            ),
            "Inspector lock cannot be safely recovered",
        )
    lock_path = paths.locks / "active.json"
    details = lock_path.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
    ):
        raise _promotion_error(
            "PROMOTION_OWNERSHIP_UNCERTAIN",
            "stale Inspector lock physical identity is unsafe",
        )
    lock_path.unlink()
    fsync_directory(lock_path.parent)


def _alias_request(
    *,
    action: str,
    transaction_id: str,
    expected_target: str | None,
    new_target: str | None,
    expected_generation: int,
    artifact_version_id: str | None,
    capability_manifest_identity: str | None,
    relative_root: str | None,
    promotion_alias_event_identity: str | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "promotion_transaction_id": transaction_id,
        "alias": "default",
        "expected_current_target": expected_target,
        "new_target": new_target,
        "expected_registry_generation": expected_generation,
        "target_artifact_version_id": artifact_version_id,
        "target_capability_manifest_identity": (
            capability_manifest_identity
        ),
        "target_relative_root": relative_root,
        "promotion_alias_event_identity": (
            promotion_alias_event_identity
        ),
    }


def _terminal_record(
    paths: InspectorPaths,
    transaction: dict[str, Any],
    authorization: PromotionAuthorization,
    incumbent: PromotionIncumbent,
    runtime: dict[str, Any],
    *,
    promotion_id: str,
    terminal_state: str,
    result_class: str,
    reason_codes: list[str],
    publisher: Callable[
        [InspectorPaths, dict[str, Any]], Path
    ],
    observer: Callable[[str, dict[str, Any]], None] | None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    completed = utc_now()
    states = list(transaction["states_observed"])
    if not states or states[-1] != terminal_state:
        states.append(terminal_state)
    record = build_promotion_record(
        promotion_id=promotion_id,
        transaction_id=transaction["transaction_id"],
        created_utc=transaction["start_utc"],
        completed_utc=completed,
        qualification=_qualification_projection(authorization),
        installed_tuple=authorization.installed_tuple,
        candidate=dict(runtime.get("candidate") or {}),
        incumbent=incumbent.result_projection(),
        states_observed=states,
        registry_progression=list(runtime.get("registry_progression", [])),
        alias_promotion=dict(runtime.get("alias_promotion") or {}),
        pre_promotion_proofs=list(
            runtime.get("pre_promotion_proofs", [])
        ),
        post_promotion_proofs=list(
            runtime.get("post_promotion_proofs", [])
        ),
        stability_observation=dict(
            runtime.get("stability_observation") or {}
        ),
        restart_verification=dict(
            runtime.get("restart_verification") or {}
        ),
        rollback=dict(runtime.get("rollback") or {}),
        service_final=dict(runtime.get("service_final") or {}),
        result_class=result_class,
        reason_codes=reason_codes,
        validity_predicate=authorization.qualification[
            "validity_predicate"
        ],
    )
    path = publisher(paths, record)
    transaction = {
        **transaction,
        "promotion_id": promotion_id,
        "promotion_result_identity": record["result_identity"],
        "promotion_result_path": str(path),
        "promotion_runtime": runtime,
    }
    transaction = persist_promotion_state(
        paths,
        transaction,
        state=terminal_state,
        reason_code=reason_codes[0],
        finish_utc=completed,
        observer=observer,
    )
    return record, path, transaction


def promote_transaction(
    paths: InspectorPaths,
    qualification_id: str,
    candidate_name: str,
    *,
    adapter: PromotionRuntimeAdapter | None = None,
    installed_tuple_resolver: InstalledTupleResolver = (
        _current_installed_tuple
    ),
    authorization_factory: Callable[..., PromotionAuthorization] = (
        authenticate_promotion_qualification
    ),
    transaction_id_factory: Callable[[], str] = _transaction_id,
    promotion_id_factory: Callable[[], str] = _promotion_id,
    incumbent_factory: Callable[..., PromotionIncumbent] = (
        capture_promotion_incumbent
    ),
    adapter_factory: Callable[
        [InspectorPaths, PromotionAuthorization, PromotionIncumbent],
        PromotionRuntimeAdapter,
    ] = CurrentSourcePromotionAdapter,
    duplicate_finder: Callable[
        [InspectorPaths, PromotionAuthorization],
        tuple[dict[str, Any], Path] | None,
    ] = find_idempotent_promotion,
    result_publisher: Callable[
        [InspectorPaths, dict[str, Any]], Path
    ] = publish_promotion_record,
    transition_observer: (
        Callable[[str, dict[str, Any]], None] | None
    ) = None,
) -> tuple[str, dict[str, Any], Path, str]:
    # Authentication occurs before lock acquisition or any runtime write.  A
    # genuine failed-clean qualification therefore leaves no promotion result.
    authorization = authorization_factory(
        paths,
        qualification_id,
        candidate_name,
        installed_tuple_resolver=installed_tuple_resolver,
    )
    duplicate = duplicate_finder(paths, authorization)
    if duplicate is not None:
        record, path = duplicate
        return (
            str(record["transaction_id"]),
            record,
            path,
            str(record["result_identity"]),
        )

    recoverable = _find_recoverable_promotion(paths, authorization)
    if recoverable is not None:
        transaction_id = str(recoverable["transaction_id"])
        incumbent = _incumbent_from_projection(
            recoverable.get("incumbent_snapshot")
        )
        basis = _promotion_basis(authorization, incumbent)
        if recoverable.get("promotion_basis_identity") != _identity(basis):
            raise _promotion_error(
                "PROMOTION_OWNERSHIP_UNCERTAIN",
                "recoverable promotion basis changed",
            )
        _clear_exact_stale_lock(paths, transaction_id)
    else:
        transaction_id = transaction_id_factory()
        incumbent = incumbent_factory(paths)
        basis = _promotion_basis(authorization, incumbent)

    lock = TransactionLock(
        paths,
        transaction_id=transaction_id,
        operation="promote-gguf",
    )
    try:
        owner = lock.acquire()
    except InspectorError as error:
        raise _promotion_error(
            (
                "PROMOTION_CONCURRENCY_REJECTED"
                if error.reason_code == "TRANSACTION_LOCK_ACTIVE"
                else "PROMOTION_ACTIVE_TRANSACTION"
            ),
            "promotion could not acquire exclusive ownership",
        ) from error

    promotion_id = (
        str(recoverable.get("promotion_id"))
        if recoverable is not None
        and isinstance(recoverable.get("promotion_id"), str)
        else promotion_id_factory()
    )
    if recoverable is None:
        transaction = promotion_transaction_value(
            paths,
            transaction_id=transaction_id,
            authorization=authorization,
            incumbent=incumbent,
            state="PREPARING",
            start_utc=utc_now(),
            promotion_id=promotion_id,
            promotion_basis_identity=_identity(basis),
            promotion_runtime={},
        )
        transaction["owner_identity"] = {
            key: owner.get(key)
            for key in (
                "pid",
                "process_start_identity",
                "boot_identity",
                "inspector_root_identity",
            )
        }
        transaction = persist_promotion_state(
            paths,
            transaction,
            state="PREPARING",
            observer=transition_observer,
        )
    else:
        transaction = {
            **recoverable,
            "owner_identity": {
                key: owner.get(key)
                for key in (
                    "pid",
                    "process_start_identity",
                    "boot_identity",
                    "inspector_root_identity",
                )
            },
        }
        _write_transaction(paths, transaction, transition_observer)

    runtime = dict(transaction.get("promotion_runtime") or {})
    runtime.setdefault("registry_progression", [])
    runtime.setdefault("pre_promotion_proofs", [])
    runtime.setdefault("post_promotion_proofs", [])
    runtime_adapter = adapter or adapter_factory(
        paths, authorization, incumbent
    )
    mutation_started = bool(runtime.get("candidate"))
    alias_moved = bool(runtime.get("alias_promotion"))
    try:
        candidate = runtime.get("candidate")
        if not isinstance(candidate, dict) or not candidate:
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="PREPARING",
                intended_action="STAGE_CANDIDATE",
                irreversible_action_begun="STAGE_CANDIDATE",
                observer=transition_observer,
            )
            mutation_started = True
            candidate = runtime_adapter.stage_candidate(
                authorization, incumbent, transaction_id
            )
            runtime["candidate"] = candidate
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="CANDIDATE_STAGED",
                irreversible_action_observed="STAGE_CANDIDATE",
                observer=transition_observer,
            )

        if not candidate.get("public_model_id"):
            observed = runtime_adapter.wait_candidate(candidate)
            runtime["candidate"] = observed
            runtime["registry_progression"].append(
                {
                    "registry_generation": observed[
                        "registry_generation"
                    ],
                    "states_observed": observed[
                        "registry_states_observed"
                    ],
                    "default_bound": observed.get("default_bound"),
                }
            )
            candidate = observed
            for state in (
                "CANDIDATE_REGISTERED",
                "CANDIDATE_PROBING",
                "CANDIDATE_READY",
            ):
                transaction = persist_promotion_state(
                    paths,
                    {**transaction, "promotion_runtime": runtime},
                    state=state,
                    observer=transition_observer,
                )

        if not runtime["pre_promotion_proofs"]:
            proof = runtime_adapter.prove_candidate(
                candidate,
                authorization.requested_profile,
                use_default=False,
            )
            if proof.get("passed") is not True:
                raise _promotion_error(
                    "PROMOTION_CANDIDATE_REQUEST_FAILED",
                    "candidate pre-promotion proof did not pass",
                )
            runtime["pre_promotion_proofs"].append(proof)
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="CANDIDATE_REQUEST_PROVEN",
                observer=transition_observer,
            )

        alias_result = runtime.get("alias_promotion")
        if not isinstance(alias_result, dict) or not alias_result:
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="PROMOTING_DEFAULT",
                intended_action="PROMOTE_DEFAULT_ALIAS",
                irreversible_action_begun="PROMOTE_DEFAULT_ALIAS",
                observer=transition_observer,
            )
            alias_result = runtime_adapter.alias_transaction(
                _alias_request(
                    action="promote",
                    transaction_id=transaction_id,
                    expected_target=incumbent.snapshot.public_model_id,
                    new_target=str(candidate["public_model_id"]),
                    expected_generation=int(
                        candidate["registry_generation"]
                    ),
                    artifact_version_id=str(
                        candidate["artifact_version_id"]
                    ),
                    capability_manifest_identity=str(
                        candidate["capability_manifest_identity"]
                    ),
                    relative_root=str(candidate["relative_root"]),
                )
            )
            runtime["alias_promotion"] = alias_result
            alias_moved = True
            runtime["registry_progression"].append(
                {
                    "registry_generation": alias_result[
                        "new_registry_generation"
                    ],
                    "alias_event_identity": alias_result[
                        "alias_event_identity"
                    ],
                    "default_target": alias_result["new_target"],
                }
            )
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="DEFAULT_PROMOTED",
                irreversible_action_observed="PROMOTE_DEFAULT_ALIAS",
                observer=transition_observer,
            )
        else:
            alias_moved = True

        exact = runtime_adapter.observe_exact(
            candidate,
            expected_default=str(candidate["public_model_id"]),
        )
        if exact.get("exact") is not True:
            raise _promotion_error(
                "PROMOTION_WARM_FAILED",
                "candidate did not become exact warm default",
            )
        if not runtime["post_promotion_proofs"]:
            post = runtime_adapter.prove_candidate(
                candidate,
                authorization.requested_profile,
                use_default=True,
            )
            if post.get("passed") is not True:
                raise _promotion_error(
                    "PROMOTION_POST_REQUEST_FAILED",
                    "post-promotion default proof did not pass",
                )
            runtime["post_promotion_proofs"].append(post)

        if not runtime.get("stability_observation"):
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="STABILITY_OBSERVING",
                observer=transition_observer,
            )
            sample_count, interval = (
                runtime_adapter.stability_parameters()
            )
            samples: list[dict[str, Any]] = []
            for index in range(sample_count):
                if index:
                    runtime_adapter.pause(interval)
                sample = runtime_adapter.observe_exact(
                    candidate,
                    expected_default=str(candidate["public_model_id"]),
                )
                if sample.get("exact") is not True:
                    raise _promotion_error(
                        "PROMOTION_STABILITY_FAILED",
                        "candidate lost exact stability",
                    )
                samples.append(sample)
            runtime["stability_observation"] = {
                "passed": True,
                "consecutive_samples": sample_count,
                "cadence_seconds": interval,
                "samples": samples,
            }

        if not runtime.get("restart_verification"):
            baseline = runtime.get("restart_baseline")
            if not isinstance(baseline, dict):
                baseline = runtime_adapter.capture_epochs()
                runtime["restart_baseline"] = baseline
            resume_restart = bool(
                transaction.get("state") == "RESTART_VERIFYING"
                and transaction.get("irreversible_action_begun")
                == "RESTART_MANAGER"
            )
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="RESTART_VERIFYING",
                intended_action="RESTART_MANAGER",
                irreversible_action_begun="RESTART_MANAGER",
                observer=transition_observer,
            )
            runtime["restart_verification"] = (
                runtime_adapter.restart_and_verify(
                    candidate,
                    baseline,
                    resume=resume_restart,
                )
            )
            transaction = {
                **transaction,
                "irreversible_action_observed": "RESTART_MANAGER",
                "promotion_runtime": runtime,
            }
            _write_transaction(paths, transaction, transition_observer)

        final = runtime_adapter.observe_exact(
            candidate,
            expected_default=str(candidate["public_model_id"]),
        )
        if final.get("exact") is not True:
            raise _promotion_error(
                "PROMOTION_RESTART_FAILED",
                "candidate final state is not exact after restart",
            )
        runtime["service_final"] = final
        runtime["rollback"] = {
            "required": False,
            "trigger": None,
            "disposition": "PROMOTED_DEFAULT",
        }
        record, path, transaction = _terminal_record(
            paths,
            {**transaction, "promotion_runtime": runtime},
            authorization,
            incumbent,
            runtime,
            promotion_id=promotion_id,
            terminal_state="COMPLETE",
            result_class="PROMOTION_COMPLETE",
            reason_codes=["PROMOTION_COMPLETE"],
            publisher=result_publisher,
            observer=transition_observer,
        )
        return transaction_id, record, path, record["result_identity"]
    except BaseException as failure:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise
        trigger = (
            failure.reason_code
            if isinstance(failure, InspectorError)
            and failure.reason_code in PROMOTION_REASON_CODES
            else "PROMOTION_INTERNAL_ERROR"
        )
        if not mutation_started:
            raise (
                failure
                if isinstance(failure, InspectorError)
                else _promotion_error(
                    "PROMOTION_INTERNAL_ERROR",
                    "unexpected pre-mutation promotion failure",
                    internal=True,
                )
            )
        try:
            transaction = persist_promotion_state(
                paths,
                {**transaction, "promotion_runtime": runtime},
                state="ROLLING_BACK",
                reason_code=trigger,
                intended_action="ROLLBACK",
                observer=transition_observer,
            )
            rollback_alias: dict[str, Any] = {
                "required": alias_moved,
                "changed": False,
            }
            if alias_moved:
                promotion_alias = runtime["alias_promotion"]
                if incumbent.snapshot.present and (
                    incumbent.snapshot.public_model_id is None
                    or incumbent.snapshot.artifact_version_id is None
                    or incumbent.snapshot.capability_manifest_identity
                    is None
                    or incumbent.relative_root is None
                ):
                    raise _promotion_error(
                        "PROMOTION_OWNERSHIP_UNCERTAIN",
                        "incumbent rollback identity is incomplete",
                    )
                rollback_alias = runtime_adapter.alias_transaction(
                    _alias_request(
                        action="rollback",
                        transaction_id=transaction_id,
                        expected_target=str(candidate["public_model_id"]),
                        new_target=(
                            str(incumbent.snapshot.public_model_id)
                            if incumbent.snapshot.present
                            else None
                        ),
                        expected_generation=int(
                            promotion_alias[
                                "new_registry_generation"
                            ]
                        ),
                        artifact_version_id=(
                            str(incumbent.snapshot.artifact_version_id)
                            if incumbent.snapshot.present
                            else None
                        ),
                        capability_manifest_identity=(
                            str(
                                incumbent.snapshot.
                                capability_manifest_identity
                            )
                            if incumbent.snapshot.present
                            else None
                        ),
                        relative_root=(
                            incumbent.relative_root
                            if incumbent.snapshot.present
                            else None
                        ),
                        promotion_alias_event_identity=str(
                            promotion_alias["alias_event_identity"]
                        ),
                    )
                )
            restoration = runtime_adapter.restore_incumbent(incumbent)
            disposition = runtime_adapter.retain_candidate(
                runtime.get("candidate")
            )
            if (
                restoration.get("proved") is not True
                or disposition.get("ownership_certain") is not True
            ):
                raise _promotion_error(
                    "PROMOTION_OWNERSHIP_UNCERTAIN",
                    "rollback restoration ownership is uncertain",
                )
            runtime["rollback"] = {
                "required": True,
                "trigger": trigger,
                "alias": rollback_alias,
                "restoration": restoration,
                "candidate_disposition": disposition,
            }
            runtime["service_final"] = restoration
            record, path, transaction = _terminal_record(
                paths,
                {**transaction, "promotion_runtime": runtime},
                authorization,
                incumbent,
                runtime,
                promotion_id=promotion_id,
                terminal_state="ROLLED_BACK",
                result_class="PROMOTION_ROLLED_BACK",
                reason_codes=["PROMOTION_ROLLED_BACK", trigger],
                publisher=result_publisher,
                observer=transition_observer,
            )
            return (
                transaction_id,
                record,
                path,
                record["result_identity"],
            )
        except BaseException as rollback_failure:
            if isinstance(
                rollback_failure, (KeyboardInterrupt, SystemExit)
            ):
                raise
            rollback_reason = (
                rollback_failure.reason_code
                if isinstance(rollback_failure, InspectorError)
                and rollback_failure.reason_code
                in PROMOTION_REASON_CODES
                else "PROMOTION_FAIL_CLOSED"
            )
            runtime["rollback"] = {
                "required": True,
                "trigger": trigger,
                "failed": True,
                "reason_code": rollback_reason,
                "candidate_disposition": "PRESERVE_FOR_REVIEW",
            }
            runtime["service_final"] = {
                "proved": False,
                "fail_closed": True,
            }
            record, path, transaction = _terminal_record(
                paths,
                {**transaction, "promotion_runtime": runtime},
                authorization,
                incumbent,
                runtime,
                promotion_id=promotion_id,
                terminal_state="FAIL_CLOSED",
                result_class="PROMOTION_FAIL_CLOSED",
                reason_codes=[
                    "PROMOTION_FAIL_CLOSED",
                    trigger,
                    rollback_reason,
                ],
                publisher=result_publisher,
                observer=transition_observer,
            )
            raise _promotion_error(
                "PROMOTION_FAIL_CLOSED",
                "promotion rollback could not prove a safe terminal state",
                internal=True,
                data={
                    "transaction_id": transaction_id,
                    "promotion_id": record["promotion_id"],
                    "result_path": str(path),
                    "result_identity": record["result_identity"],
                },
            ) from rollback_failure
    finally:
        lock.release()
