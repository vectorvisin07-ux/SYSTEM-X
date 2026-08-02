#!/usr/bin/env python3
"""Persistent bounded recovery policy for the System X service supervisor."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Any, Mapping


STATUS_SCHEMA = "system-x.service-recovery-status.v1"
TRANSACTION_SCHEMA = "system-x.service-recovery-transaction.v1"
LATCH_SCHEMA = "system-x.service-recovery-fail-closed.v1"
EVENT_SCHEMA = "system-x.service-recovery-history-event.v1"

RECOVERY_STATES = frozenset(
    (
        "IDLE",
        "DETECTED",
        "DELAYING",
        "RECONCILING",
        "RESTARTING_API",
        "RESTARTING_ROUTER",
        "RELOADING_MODEL",
        "VERIFYING",
        "RECOVERED",
        "FAIL_CLOSED",
        "STOPPED",
    )
)
RECOVERY_REASON_CODES = frozenset(
    (
        "API_PROCESS_LOST",
        "PUBLIC_LISTENER_LOST",
        "ROUTER_PROCESS_LOST",
        "PRIVATE_LISTENER_LOST",
        "MODEL_CHILD_LOST",
        "WARM_MODEL_HEALTH_LOST",
        "SUPERVISOR_STATE_STALE",
        "API_STATE_STALE",
        "ROUTER_STATE_STALE",
        "PARTIAL_STARTUP",
        "PARTIAL_SHUTDOWN",
        "ENDPOINT_CONFLICT",
        "OWNERSHIP_UNCERTAIN",
        "RECOVERY_ATTEMPT_FAILED",
        "RECOVERY_STABLE",
        "RECOVERY_LOOP_DETECTED",
        "FAIL_CLOSED_LATCHED",
        "FAIL_CLOSED_RESET",
        "DESIRED_STATE_STOPPED",
        "AUTOMATIC_RECOVERY_DISABLED",
    )
)
OUTCOMES = frozenset(
    ("RECOVERED", "FAILED", "FAIL_CLOSED", "CANCELLED_BY_STOPPED")
)
MAX_RECORD_BYTES = 1_048_576


class RecoveryError(RuntimeError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = str(message)[:4096]


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    automatic_recovery_enabled: bool
    initial_delay_seconds: float
    maximum_delay_seconds: float
    delay_multiplier: float
    maximum_attempts_in_window: int
    attempt_window_seconds: float
    stable_reset_seconds: float

    def __post_init__(self) -> None:
        numeric = (
            self.initial_delay_seconds,
            self.maximum_delay_seconds,
            self.delay_multiplier,
            self.attempt_window_seconds,
            self.stable_reset_seconds,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("recovery policy values must be finite")
        if not 1 <= self.maximum_attempts_in_window <= 16:
            raise ValueError("maximum attempts must be in 1..16")
        if not 1 <= self.attempt_window_seconds <= 3600:
            raise ValueError("attempt window must be in 1..3600")
        if not 1 <= self.stable_reset_seconds <= 3600:
            raise ValueError("stable reset must be in 1..3600")
        if (
            self.initial_delay_seconds < 0
            or self.maximum_delay_seconds <= 0
            or self.initial_delay_seconds > self.maximum_delay_seconds
            or not 1 <= self.delay_multiplier <= 16
        ):
            raise ValueError("recovery delay policy is invalid")

    def delay_for(self, attempt_ordinal: int) -> float:
        if attempt_ordinal < 1:
            raise ValueError("attempt ordinal must be positive")
        return min(
            self.maximum_delay_seconds,
            self.initial_delay_seconds
            * self.delay_multiplier ** (attempt_ordinal - 1),
        )


@dataclass(frozen=True, slots=True)
class RecoveryAttempt:
    recovery_transaction_id: str
    primary_reason_code: str
    attempt_ordinal: int
    delay_seconds: float


@dataclass(frozen=True, slots=True)
class RecoveryPaths:
    root: Path

    @property
    def status_dir(self) -> Path:
        return self.root / "status"

    @property
    def transaction_dir(self) -> Path:
        return self.root / "transactions"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    @property
    def fail_closed_dir(self) -> Path:
        return self.root / "fail-closed"

    @property
    def status(self) -> Path:
        return self.status_dir / "recovery.json"

    @property
    def history(self) -> Path:
        return self.history_dir / "attempts.jsonl"

    @property
    def active_latch(self) -> Path:
        return self.fail_closed_dir / "active.json"

    def transaction(self, transaction_id: str) -> Path:
        return self.transaction_dir / f"{transaction_id}.json"

    def reset_record(self, reset_id: str) -> Path:
        return self.fail_closed_dir / f"{reset_id}.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RecoveryError(
            "OWNERSHIP_UNCERTAIN", "recovery path is not a direct directory"
        )


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_directory(path.parent)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        payload = _canonical(dict(value))
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short recovery-record write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _exclusive_write(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        payload = _canonical(dict(value))
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short recovery-record write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > MAX_RECORD_BYTES
    ):
        raise RecoveryError(
            "OWNERSHIP_UNCERTAIN", "recovery record identity is invalid"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(
            "OWNERSHIP_UNCERTAIN", "recovery record is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RecoveryError(
            "OWNERSHIP_UNCERTAIN", "recovery record must be an object"
        )
    return value


def _append_event(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        payload = _canonical(dict(value))
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short recovery-history write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RecoveryStore:
    """Persist one profile-bound recovery stream and fail-closed latch."""

    def __init__(
        self,
        runtime_root: Path,
        profile_identity: str,
        policy: RecoveryPolicy,
    ) -> None:
        self.paths = RecoveryPaths(Path(runtime_root))
        self.profile_identity = profile_identity
        self.policy = policy
        self._ready_since_monotonic: float | None = None
        for directory in (
            self.paths.status_dir,
            self.paths.transaction_dir,
            self.paths.history_dir,
            self.paths.fail_closed_dir,
        ):
            _ensure_directory(directory)
        if self.paths.active_latch.exists():
            latch = _read(self.paths.active_latch)
            if (
                latch.get("schema_version") != LATCH_SCHEMA
                or latch.get("profile_identity") != profile_identity
            ):
                raise RecoveryError(
                    "OWNERSHIP_UNCERTAIN",
                    "active fail-closed latch identity is invalid",
                )

    def _events(self) -> list[dict[str, Any]]:
        if not self.paths.history.exists():
            return []
        metadata = self.paths.history.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAX_RECORD_BYTES
        ):
            raise RecoveryError(
                "OWNERSHIP_UNCERTAIN", "recovery history identity is invalid"
            )
        events: list[dict[str, Any]] = []
        for line in self.paths.history.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict) or value.get(
                "schema_version"
            ) != EVENT_SCHEMA:
                raise RecoveryError(
                    "OWNERSHIP_UNCERTAIN", "recovery history event is invalid"
                )
            events.append(value)
        return events

    def _attempts_in_window(self, now_epoch: float | None = None) -> int:
        now = time.time() if now_epoch is None else now_epoch
        cutoff = now - self.policy.attempt_window_seconds
        stable_reset_epoch = 0.0
        attempts = []
        for event in self._events():
            if event.get("event") == "STABLE_RESET":
                stable_reset_epoch = max(
                    stable_reset_epoch, float(event.get("epoch_seconds", 0))
                )
            elif event.get("event") == "ATTEMPT":
                attempts.append(float(event.get("epoch_seconds", 0)))
        threshold = max(cutoff, stable_reset_epoch)
        return sum(epoch >= threshold for epoch in attempts)

    def _latch(self, transaction_id: str, attempts: int) -> dict[str, Any]:
        if self.paths.active_latch.exists():
            return _read(self.paths.active_latch)
        value = {
            "schema_version": LATCH_SCHEMA,
            "profile_identity": self.profile_identity,
            "fail_closed_transaction_id": transaction_id,
            "primary_reason_code": "RECOVERY_LOOP_DETECTED",
            "attempts_in_window": attempts,
            "maximum_attempts_in_window": (
                self.policy.maximum_attempts_in_window
            ),
            "attempt_window_seconds": self.policy.attempt_window_seconds,
            "latched_utc": utc_now(),
        }
        _exclusive_write(self.paths.active_latch, value)
        return value

    def _write_status(
        self,
        *,
        desired_state: str,
        desired_generation: int,
        recovery_state: str,
        primary_reason_code: str,
        current_attempt: int = 0,
        next_delay_seconds: float = 0.0,
        transaction_id: str | None = None,
        last_recovered_utc: str | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if recovery_state not in RECOVERY_STATES:
            raise ValueError("invalid recovery state")
        if primary_reason_code not in RECOVERY_REASON_CODES:
            raise ValueError("invalid recovery reason")
        latch = (
            _read(self.paths.active_latch)
            if self.paths.active_latch.exists()
            else None
        )
        prior = _read(self.paths.status) if self.paths.status.exists() else {}
        value = {
            "schema_version": STATUS_SCHEMA,
            "profile_identity": self.profile_identity,
            "desired_state": desired_state,
            "desired_state_generation": desired_generation,
            "recovery_state": recovery_state,
            "primary_reason_code": primary_reason_code,
            "current_attempt": current_attempt,
            "attempts_in_window": self._attempts_in_window(),
            "next_delay_seconds": next_delay_seconds,
            "fail_closed_latched": latch is not None,
            "fail_closed_transaction_id": (
                latch.get("fail_closed_transaction_id") if latch else None
            ),
            "last_recovery_transaction_id": transaction_id
            or prior.get("last_recovery_transaction_id"),
            "last_recovered_utc": last_recovered_utc
            or prior.get("last_recovered_utc"),
            "observed_identities": dict(observation or {}),
            "updated_utc": utc_now(),
        }
        _atomic_write(self.paths.status, value)
        return value

    def initialize(
        self,
        *,
        desired_state: str,
        desired_generation: int,
        observation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.paths.active_latch.exists():
            state = "FAIL_CLOSED"
            reason = "FAIL_CLOSED_LATCHED"
        elif desired_state == "STOPPED":
            state = "STOPPED"
            reason = "DESIRED_STATE_STOPPED"
        else:
            state = "IDLE"
            reason = "RECOVERY_STABLE"
        return self._write_status(
            desired_state=desired_state,
            desired_generation=desired_generation,
            recovery_state=state,
            primary_reason_code=reason,
            observation=observation,
        )

    def begin(
        self,
        *,
        reason_code: str,
        desired_state: str,
        desired_generation: int,
        observation: Mapping[str, Any],
        selected_action: str,
    ) -> RecoveryAttempt | None:
        if reason_code not in RECOVERY_REASON_CODES:
            raise ValueError("invalid recovery reason")
        if desired_state == "STOPPED":
            self._write_status(
                desired_state=desired_state,
                desired_generation=desired_generation,
                recovery_state="STOPPED",
                primary_reason_code="DESIRED_STATE_STOPPED",
                observation=observation,
            )
            return None
        if not self.policy.automatic_recovery_enabled:
            self._write_status(
                desired_state=desired_state,
                desired_generation=desired_generation,
                recovery_state="DETECTED",
                primary_reason_code="AUTOMATIC_RECOVERY_DISABLED",
                observation=observation,
            )
            return None
        if self.paths.active_latch.exists():
            self._write_status(
                desired_state=desired_state,
                desired_generation=desired_generation,
                recovery_state="FAIL_CLOSED",
                primary_reason_code="FAIL_CLOSED_LATCHED",
                observation=observation,
            )
            return None
        attempts = self._attempts_in_window()
        if attempts >= self.policy.maximum_attempts_in_window:
            transaction_id = self._new_transaction_id()
            self._latch(transaction_id, attempts)
            self._write_status(
                desired_state=desired_state,
                desired_generation=desired_generation,
                recovery_state="FAIL_CLOSED",
                primary_reason_code="RECOVERY_LOOP_DETECTED",
                current_attempt=attempts,
                transaction_id=transaction_id,
                observation=observation,
            )
            return None
        ordinal = attempts + 1
        delay = self.policy.delay_for(ordinal)
        transaction_id = self._new_transaction_id()
        detected = utc_now()
        transaction = {
            "schema_version": TRANSACTION_SCHEMA,
            "recovery_transaction_id": transaction_id,
            "profile_identity": self.profile_identity,
            "desired_state_generation": desired_generation,
            "detected_utc": detected,
            "completed_utc": None,
            "primary_reason_code": reason_code,
            "trigger_observation": dict(observation),
            "attempt_ordinal": ordinal,
            "delay_applied_seconds": delay,
            "selected_action": selected_action,
            "controller_reconciliation_results": [],
            "controller_start_stop_results": [],
            "pre_identities": dict(observation),
            "post_identities": {},
            "readiness_transitions": [
                {
                    "state": "DETECTED",
                    "timestamp_utc": detected,
                    "reason_code": reason_code,
                }
            ],
            "authenticated_proof_request_id": None,
            "outcome": None,
            "error": None,
        }
        _exclusive_write(self.paths.transaction(transaction_id), transaction)
        _append_event(
            self.paths.history,
            {
                "schema_version": EVENT_SCHEMA,
                "event": "ATTEMPT",
                "epoch_seconds": time.time(),
                "timestamp_utc": detected,
                "profile_identity": self.profile_identity,
                "recovery_transaction_id": transaction_id,
                "primary_reason_code": reason_code,
                "attempt_ordinal": ordinal,
            },
        )
        self._write_status(
            desired_state=desired_state,
            desired_generation=desired_generation,
            recovery_state="DETECTED",
            primary_reason_code=reason_code,
            current_attempt=ordinal,
            next_delay_seconds=delay,
            transaction_id=transaction_id,
            observation=observation,
        )
        return RecoveryAttempt(transaction_id, reason_code, ordinal, delay)

    def fail_closed_now(
        self,
        *,
        reason_code: str,
        desired_state: str,
        desired_generation: int,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if reason_code not in {"OWNERSHIP_UNCERTAIN", "ENDPOINT_CONFLICT"}:
            raise ValueError("immediate fail-closed reason is invalid")
        transaction_id = self._new_transaction_id()
        latch = self._latch(
            transaction_id,
            max(1, self._attempts_in_window()),
        )
        latch["primary_reason_code"] = reason_code
        _atomic_write(self.paths.active_latch, latch)
        return self._write_status(
            desired_state=desired_state,
            desired_generation=desired_generation,
            recovery_state="FAIL_CLOSED",
            primary_reason_code=reason_code,
            transaction_id=transaction_id,
            observation=observation,
        )

    @staticmethod
    def _new_transaction_id() -> str:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"rc-{stamp}-{secrets.token_hex(6)}"

    def transition(
        self,
        attempt: RecoveryAttempt,
        *,
        desired_state: str,
        desired_generation: int,
        recovery_state: str,
        observation: Mapping[str, Any] | None = None,
        controller_result: Mapping[str, Any] | None = None,
    ) -> None:
        path = self.paths.transaction(attempt.recovery_transaction_id)
        transaction = _read(path)
        transaction["readiness_transitions"].append(
            {
                "state": recovery_state,
                "timestamp_utc": utc_now(),
                "reason_code": attempt.primary_reason_code,
            }
        )
        if controller_result is not None:
            if recovery_state == "RECONCILING":
                transaction["controller_reconciliation_results"].append(
                    dict(controller_result)
                )
            else:
                transaction["controller_start_stop_results"].append(
                    dict(controller_result)
                )
        if observation is not None:
            transaction["post_identities"] = dict(observation)
        _atomic_write(path, transaction)
        self._write_status(
            desired_state=desired_state,
            desired_generation=desired_generation,
            recovery_state=recovery_state,
            primary_reason_code=attempt.primary_reason_code,
            current_attempt=attempt.attempt_ordinal,
            next_delay_seconds=(
                attempt.delay_seconds if recovery_state == "DELAYING" else 0.0
            ),
            transaction_id=attempt.recovery_transaction_id,
            observation=observation,
        )

    def complete(
        self,
        attempt: RecoveryAttempt,
        *,
        desired_state: str,
        desired_generation: int,
        outcome: str,
        observation: Mapping[str, Any] | None = None,
        error_category: str | None = None,
        proof_request_id: str | None = None,
    ) -> dict[str, Any]:
        if outcome not in OUTCOMES:
            raise ValueError("invalid recovery outcome")
        path = self.paths.transaction(attempt.recovery_transaction_id)
        transaction = _read(path)
        completed = utc_now()
        transaction["completed_utc"] = completed
        transaction["post_identities"] = dict(observation or {})
        transaction["authenticated_proof_request_id"] = proof_request_id
        transaction["outcome"] = outcome
        transaction["error"] = (
            {"category": str(error_category)[:256]}
            if error_category is not None
            else None
        )
        _atomic_write(path, transaction)
        attempts = self._attempts_in_window()
        recovery_state = "RECOVERED" if outcome == "RECOVERED" else "DETECTED"
        reason_code = (
            attempt.primary_reason_code
            if outcome == "RECOVERED"
            else "RECOVERY_ATTEMPT_FAILED"
        )
        if outcome == "CANCELLED_BY_STOPPED":
            recovery_state = "STOPPED"
            reason_code = "DESIRED_STATE_STOPPED"
        elif outcome in {"FAILED", "FAIL_CLOSED"} and (
            attempts >= self.policy.maximum_attempts_in_window
        ):
            self._latch(attempt.recovery_transaction_id, attempts)
            recovery_state = "FAIL_CLOSED"
            reason_code = "RECOVERY_LOOP_DETECTED"
            transaction["outcome"] = "FAIL_CLOSED"
            _atomic_write(path, transaction)
        return self._write_status(
            desired_state=desired_state,
            desired_generation=desired_generation,
            recovery_state=recovery_state,
            primary_reason_code=reason_code,
            current_attempt=attempt.attempt_ordinal,
            transaction_id=attempt.recovery_transaction_id,
            last_recovered_utc=completed if outcome == "RECOVERED" else None,
            observation=observation,
        )

    def healthy_tick(
        self,
        *,
        desired_state: str,
        desired_generation: int,
        observation: Mapping[str, Any],
        ready: bool,
    ) -> None:
        if desired_state == "STOPPED":
            self._ready_since_monotonic = None
            self._write_status(
                desired_state=desired_state,
                desired_generation=desired_generation,
                recovery_state="STOPPED",
                primary_reason_code="DESIRED_STATE_STOPPED",
                observation=observation,
            )
            return
        if not ready or self.paths.active_latch.exists():
            self._ready_since_monotonic = None
            return
        if self._ready_since_monotonic is None:
            self._ready_since_monotonic = time.monotonic()
            return
        if (
            time.monotonic() - self._ready_since_monotonic
            < self.policy.stable_reset_seconds
        ):
            return
        _append_event(
            self.paths.history,
            {
                "schema_version": EVENT_SCHEMA,
                "event": "STABLE_RESET",
                "epoch_seconds": time.time(),
                "timestamp_utc": utc_now(),
                "profile_identity": self.profile_identity,
                "primary_reason_code": "RECOVERY_STABLE",
            },
        )
        self._write_status(
            desired_state=desired_state,
            desired_generation=desired_generation,
            recovery_state="IDLE",
            primary_reason_code="RECOVERY_STABLE",
            observation=observation,
        )
        self._ready_since_monotonic = time.monotonic()

    def reset_fail_closed(
        self,
        *,
        desired_state: str,
        desired_generation: int,
        owned_runtime_absent: bool,
        listeners_absent: bool,
    ) -> dict[str, Any]:
        if desired_state != "STOPPED":
            raise RecoveryError(
                "DESIRED_STATE_STOPPED",
                "reset-recovery requires desired state STOPPED",
            )
        if not owned_runtime_absent or not listeners_absent:
            raise RecoveryError(
                "OWNERSHIP_UNCERTAIN",
                "reset-recovery requires a clean inactive runtime",
            )
        if not self.paths.active_latch.exists():
            self._write_status(
                desired_state=desired_state,
                desired_generation=desired_generation,
                recovery_state="STOPPED",
                primary_reason_code="FAIL_CLOSED_RESET",
            )
            return {"reset": False, "reason_code": "NO_ACTIVE_LATCH"}
        latch = _read(self.paths.active_latch)
        if latch.get("profile_identity") != self.profile_identity:
            raise RecoveryError(
                "OWNERSHIP_UNCERTAIN",
                "fail-closed latch belongs to another profile",
            )
        reset_id = "reset-" + secrets.token_hex(12)
        reset = {
            **latch,
            "schema_version": LATCH_SCHEMA,
            "reset_id": reset_id,
            "reset_utc": utc_now(),
            "reset_reason_code": "FAIL_CLOSED_RESET",
        }
        _exclusive_write(self.paths.reset_record(reset_id), reset)
        current = _read(self.paths.active_latch)
        if current != latch:
            raise RecoveryError(
                "OWNERSHIP_UNCERTAIN",
                "fail-closed latch changed before reset",
            )
        self.paths.active_latch.unlink()
        _fsync_directory(self.paths.fail_closed_dir)
        _append_event(
            self.paths.history,
            {
                "schema_version": EVENT_SCHEMA,
                "event": "FAIL_CLOSED_RESET",
                "epoch_seconds": time.time(),
                "timestamp_utc": reset["reset_utc"],
                "profile_identity": self.profile_identity,
                "reset_id": reset_id,
            },
        )
        self._write_status(
            desired_state=desired_state,
            desired_generation=desired_generation,
            recovery_state="STOPPED",
            primary_reason_code="FAIL_CLOSED_RESET",
        )
        return {
            "reset": True,
            "reason_code": "FAIL_CLOSED_RESET",
            "reset_id": reset_id,
            "history_preserved": True,
        }

    def public_status(self) -> dict[str, Any] | None:
        return _read(self.paths.status) if self.paths.status.exists() else None


def observation_identity(value: Mapping[str, Any]) -> str:
    """Return a non-secret identity for a coherent recovery observation."""

    return "sha256:" + hashlib.sha256(_canonical(dict(value))).hexdigest()
