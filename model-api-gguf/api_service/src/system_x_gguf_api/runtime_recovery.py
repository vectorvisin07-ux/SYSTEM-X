"""Bounded API-lifespan recovery for the owned router and warm model."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Any

from .backend import BackendCoordinator, BackendError
from .warm_model import WarmModelCoordinator


STATUS_SCHEMA = "system-x.api-runtime-recovery-status.v1"
TRANSACTION_SCHEMA = "system-x.api-runtime-recovery-transaction.v1"
LATCH_SCHEMA = "system-x.api-runtime-recovery-fail-closed.v1"
LOGGER = logging.getLogger("uvicorn.error")
MAX_RECORD_BYTES = 1_048_576


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
        raise BackendError("runtime recovery directory is unsafe")


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
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
        payload = _canonical(value)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short API recovery record write")
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


def _append(path: Path, value: dict[str, Any]) -> None:
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
        payload = _canonical(value)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short API recovery history write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_desired_state(path_value: str, profile_identity: str) -> tuple[str, int]:
    path = Path(path_value)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > MAX_RECORD_BYTES
    ):
        raise BackendError("desired-state record identity is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BackendError("desired-state record is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != "system-x.service-desired-state.v1"
        or value.get("profile_identity") != profile_identity
        or value.get("desired_state") not in {"RUNNING", "STOPPED"}
        or type(value.get("generation")) is not int
        or value["generation"] < 1
    ):
        raise BackendError("desired-state authority does not validate")
    return str(value["desired_state"]), int(value["generation"])


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryAttempt:
    transaction_id: str
    reason_code: str
    ordinal: int
    delay_seconds: float


def _profile_latch_path(recovery_root: Path, profile_identity: str | None) -> Path:
    """Keep fail-closed latch ownership isolated from another profile."""

    base = recovery_root / "fail-closed/api-runtime.json"
    if not profile_identity:
        return base
    if not base.exists():
        return base
    try:
        metadata = base.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAX_RECORD_BYTES
        ):
            return base
        value = json.loads(base.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return base
    if isinstance(value, dict) and value.get("profile_identity") == profile_identity:
        return base
    digest = hashlib.sha256(profile_identity.encode("utf-8")).hexdigest()
    return base.with_name(f"api-runtime-{digest}.json")


class RuntimeRecoveryCoordinator:
    """Own one shutdown-fenced router/model recovery loop per API lifespan."""

    def __init__(
        self,
        settings: Any,
        backend: BackendCoordinator,
        warm_model: WarmModelCoordinator,
        *,
        recovery_root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.warm_model = warm_model
        if recovery_root is None:
            branch_root = Path(__file__).resolve(strict=True).parents[3]
            recovery_root = (
                branch_root / "RUNTIME/service_control/recovery"
            )
        else:
            recovery_root = recovery_root.resolve(strict=False)
        self._status_path = recovery_root / "status/api-runtime.json"
        self._transaction_root = recovery_root / "transactions"
        self._history_path = recovery_root / "history/api-runtime.jsonl"
        self._latch_path = _profile_latch_path(
            recovery_root, settings.service_control_profile_identity
        )
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._shutdown_started = False
        self._attempt_epochs: deque[float] = deque()
        self._stable_since: float | None = None
        self._status: dict[str, Any] = {
            "schema_version": STATUS_SCHEMA,
            "profile_identity": settings.service_control_profile_identity,
            "recovery_state": "IDLE",
            "primary_reason_code": "RECOVERY_STABLE",
            "current_attempt": 0,
            "attempts_in_window": 0,
            "last_recovery_transaction_id": None,
            "last_recovered_utc": None,
            "fail_closed_latched": False,
            "updated_utc": utc_now(),
        }

    @property
    def public_status(self) -> dict[str, Any]:
        return dict(self._status)

    def _write_status(
        self,
        state: str,
        reason: str,
        *,
        attempt: RuntimeRecoveryAttempt | None = None,
    ) -> None:
        self._status = {
            **self._status,
            "recovery_state": state,
            "primary_reason_code": reason,
            "current_attempt": attempt.ordinal if attempt else 0,
            "attempts_in_window": len(self._attempt_epochs),
            "last_recovery_transaction_id": (
                attempt.transaction_id
                if attempt is not None
                else self._status.get("last_recovery_transaction_id")
            ),
            "fail_closed_latched": self._latch_path.exists(),
            "updated_utc": utc_now(),
        }
        _atomic_write(self._status_path, self._status)

    def _desired(self) -> tuple[str, int]:
        identity = self.settings.service_control_profile_identity
        path = self.settings.service_control_desired_state_path
        if not self.settings.automatic_recovery_enabled:
            return "STOPPED", 0
        if not isinstance(identity, str) or not isinstance(path, str):
            raise BackendError("runtime recovery authority is not configured")
        return _read_desired_state(path, identity)

    def _recovery_permitted(self) -> bool:
        if self._shutdown_started or self._stop_event.is_set():
            return False
        desired, _generation = self._desired()
        return desired == "RUNNING"

    def _prune_attempts(self) -> None:
        cutoff = (
            time.time()
            - float(self.settings.recovery_attempt_window_seconds)
        )
        while self._attempt_epochs and self._attempt_epochs[0] < cutoff:
            self._attempt_epochs.popleft()

    def _begin(self, reason_code: str) -> RuntimeRecoveryAttempt | None:
        self._prune_attempts()
        if not self._recovery_permitted():
            self._write_status("STOPPED", "DESIRED_STATE_STOPPED")
            return None
        if self._latch_path.exists():
            self._write_status("FAIL_CLOSED", "FAIL_CLOSED_LATCHED")
            return None
        maximum = int(self.settings.recovery_maximum_attempts_in_window)
        if len(self._attempt_epochs) >= maximum:
            latch = {
                "schema_version": LATCH_SCHEMA,
                "profile_identity": (
                    self.settings.service_control_profile_identity
                ),
                "reason_code": "RECOVERY_LOOP_DETECTED",
                "attempts_in_window": len(self._attempt_epochs),
                "maximum_attempts_in_window": maximum,
                "latched_utc": utc_now(),
            }
            _atomic_write(self._latch_path, latch)
            self._write_status("FAIL_CLOSED", "RECOVERY_LOOP_DETECTED")
            return None
        ordinal = len(self._attempt_epochs) + 1
        delay = min(
            float(self.settings.recovery_delay_maximum_seconds),
            float(self.settings.recovery_delay_initial_seconds)
            * float(self.settings.recovery_delay_multiplier)
            ** (ordinal - 1),
        )
        if not math.isfinite(delay):
            raise BackendError("runtime recovery delay is non-finite")
        stamp = dt.datetime.now(dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        attempt = RuntimeRecoveryAttempt(
            f"ar-{stamp}-{secrets.token_hex(6)}",
            reason_code,
            ordinal,
            delay,
        )
        self._attempt_epochs.append(time.time())
        transaction = {
            "schema_version": TRANSACTION_SCHEMA,
            "recovery_transaction_id": attempt.transaction_id,
            "profile_identity": self.settings.service_control_profile_identity,
            "desired_state_generation": self._desired()[1],
            "detected_utc": utc_now(),
            "completed_utc": None,
            "primary_reason_code": reason_code,
            "attempt_ordinal": ordinal,
            "delay_applied_seconds": delay,
            "selected_action": (
                "CONTROLLER_ROUTER_RESTART"
                if reason_code
                in {"ROUTER_PROCESS_LOST", "PRIVATE_LISTENER_LOST"}
                else "EXACT_DEFAULT_TARGET_RELOAD"
            ),
            "pre_router_identity": (
                {
                    "transaction_id": self.backend.identity.transaction_id,
                    "pid": self.backend.identity.pid,
                    "pgid": self.backend.identity.pgid,
                    "sid": self.backend.identity.sid,
                    "process_start_identity": (
                        self.backend.identity.process_start_identity
                    ),
                }
                if self.backend.identity is not None
                else None
            ),
            "pre_warm_identity": (
                self.warm_model.status.identity.public_dict()
                if self.warm_model.status.identity is not None
                else None
            ),
            "post_router_identity": None,
            "post_warm_identity": None,
            "readiness_transitions": [
                {"state": "DETECTED", "timestamp_utc": utc_now()}
            ],
            "outcome": None,
            "error_category": None,
        }
        _atomic_write(
            self._transaction_root / f"{attempt.transaction_id}.json",
            transaction,
        )
        _append(
            self._history_path,
            {
                "schema_version": TRANSACTION_SCHEMA,
                "event": "ATTEMPT",
                "timestamp_utc": utc_now(),
                "epoch_seconds": self._attempt_epochs[-1],
                "recovery_transaction_id": attempt.transaction_id,
                "reason_code": reason_code,
                "attempt_ordinal": ordinal,
            },
        )
        self._write_status("DETECTED", reason_code, attempt=attempt)
        return attempt

    def _transition(
        self, attempt: RuntimeRecoveryAttempt, state: str
    ) -> None:
        path = self._transaction_root / f"{attempt.transaction_id}.json"
        transaction = json.loads(path.read_text(encoding="utf-8"))
        transaction["readiness_transitions"].append(
            {"state": state, "timestamp_utc": utc_now()}
        )
        _atomic_write(path, transaction)
        self._write_status(state, attempt.reason_code, attempt=attempt)

    def _complete(
        self,
        attempt: RuntimeRecoveryAttempt,
        *,
        recovered: bool,
        error_category: str | None = None,
    ) -> None:
        path = self._transaction_root / f"{attempt.transaction_id}.json"
        transaction = json.loads(path.read_text(encoding="utf-8"))
        transaction["completed_utc"] = utc_now()
        transaction["post_router_identity"] = (
            {
                "transaction_id": self.backend.identity.transaction_id,
                "pid": self.backend.identity.pid,
                "pgid": self.backend.identity.pgid,
                "sid": self.backend.identity.sid,
                "process_start_identity": (
                    self.backend.identity.process_start_identity
                ),
            }
            if self.backend.identity is not None
            else None
        )
        transaction["post_warm_identity"] = (
            self.warm_model.status.identity.public_dict()
            if self.warm_model.status.identity is not None
            else None
        )
        transaction["outcome"] = "RECOVERED" if recovered else "FAILED"
        transaction["error_category"] = error_category
        _atomic_write(path, transaction)
        if recovered:
            self._status["last_recovered_utc"] = transaction[
                "completed_utc"
            ]
            self._write_status(
                "RECOVERED", attempt.reason_code, attempt=attempt
            )
            self._stable_since = time.monotonic()
        else:
            maximum = int(
                self.settings.recovery_maximum_attempts_in_window
            )
            if len(self._attempt_epochs) >= maximum:
                _atomic_write(
                    self._latch_path,
                    {
                        "schema_version": LATCH_SCHEMA,
                        "profile_identity": (
                            self.settings.service_control_profile_identity
                        ),
                        "reason_code": "RECOVERY_LOOP_DETECTED",
                        "attempts_in_window": len(self._attempt_epochs),
                        "maximum_attempts_in_window": maximum,
                        "last_failed_transaction_id": (
                            attempt.transaction_id
                        ),
                        "latched_utc": utc_now(),
                    },
                )
                self._write_status(
                    "FAIL_CLOSED",
                    "RECOVERY_LOOP_DETECTED",
                    attempt=attempt,
                )
            else:
                self._write_status(
                    "DETECTED",
                    "RECOVERY_ATTEMPT_FAILED",
                    attempt=attempt,
                )

    async def _delay(self, attempt: RuntimeRecoveryAttempt) -> bool:
        self._transition(attempt, "DELAYING")
        deadline = time.monotonic() + attempt.delay_seconds
        while time.monotonic() < deadline:
            if not self._recovery_permitted():
                return False
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=min(0.05, deadline - time.monotonic()),
                )
            except TimeoutError:
                pass
        return self._recovery_permitted()

    async def _restart_router_with_endpoint_reuse_wait(self) -> None:
        """Keep transient endpoint reuse inside one recovery attempt."""

        deadline = time.monotonic() + min(
            180.0,
            max(
                1.0,
                float(self.settings.private_backend_model_timeout_seconds),
            ),
        )
        while True:
            try:
                await self.backend.recover_router()
                return
            except BackendError as exc:
                message = str(exc)
                if not any(
                    reason in message
                    for reason in ("ENDPOINT_IN_USE", "ENDPOINT_UNAVAILABLE")
                ):
                    raise
                if not self._recovery_permitted():
                    raise BackendError(
                        "router endpoint reuse wait cancelled by STOPPED"
                    ) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendError(
                        "router endpoint reuse wait exceeded its bound"
                    ) from exc
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=min(
                            remaining,
                            max(
                                0.05,
                                float(
                                    self.settings
                                    .private_backend_poll_interval_seconds
                                ),
                            ),
                        ),
                    )
                except TimeoutError:
                    continue
                raise BackendError(
                    "router endpoint reuse wait fenced by shutdown"
                ) from exc

    async def _reload_exact_warm_target_with_transient_wait(
        self,
    ) -> Any:
        """Retry transient adoption without spending another loop attempt."""

        deadline = time.monotonic() + min(
            180.0,
            max(
                1.0,
                float(self.settings.private_backend_model_timeout_seconds),
            ),
        )
        while True:
            try:
                return await self.warm_model.recover_current_target()
            except BackendError as exc:
                message = str(exc)
                if any(
                    terminal in message.lower()
                    for terminal in (
                        "ownership",
                        "another model",
                        "conflict",
                        "fenced",
                        "shutdown",
                        "stopped",
                    )
                ):
                    raise
                if not self._recovery_permitted():
                    raise BackendError(
                        "warm-target recovery cancelled by STOPPED"
                    ) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendError(
                        "warm-target transient wait exceeded its bound"
                    ) from exc
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=min(
                            remaining,
                            max(
                                0.05,
                                float(
                                    self.settings
                                    .private_backend_poll_interval_seconds
                                ),
                            ),
                        ),
                    )
                except TimeoutError:
                    continue
                raise BackendError(
                    "warm-target transient wait fenced by shutdown"
                ) from exc

    async def _recover_router(self, reason_code: str) -> None:
        attempt = self._begin(reason_code)
        if attempt is None:
            return
        try:
            if not await self._delay(attempt):
                self._complete(
                    attempt,
                    recovered=False,
                    error_category="DESIRED_STATE_STOPPED",
                )
                return
            self._transition(attempt, "RESTARTING_ROUTER")
            await self._restart_router_with_endpoint_reuse_wait()
            if not self._recovery_permitted():
                raise BackendError("router recovery cancelled by STOPPED")
            self._transition(attempt, "RELOADING_MODEL")
            warm = (
                await self._reload_exact_warm_target_with_transient_wait()
            )
            if warm.service_readiness_state != "READY":
                raise BackendError("warm model did not recover after router")
            self._transition(attempt, "VERIFYING")
            self._complete(attempt, recovered=True)
        except BaseException as exc:
            self._complete(
                attempt,
                recovered=False,
                error_category=(
                    f"{type(exc).__name__}:{str(exc)}"[:256]
                ),
            )
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _recover_model(self, reason_code: str) -> None:
        attempt = self._begin(reason_code)
        if attempt is None:
            return
        try:
            if not await self._delay(attempt):
                self._complete(
                    attempt,
                    recovered=False,
                    error_category="DESIRED_STATE_STOPPED",
                )
                return
            self._transition(attempt, "RELOADING_MODEL")
            warm = (
                await self._reload_exact_warm_target_with_transient_wait()
            )
            if warm.service_readiness_state != "READY":
                raise BackendError("warm model recovery did not reach READY")
            self._transition(attempt, "VERIFYING")
            self._complete(attempt, recovered=True)
        except BaseException as exc:
            self._complete(
                attempt,
                recovered=False,
                error_category=(
                    f"{type(exc).__name__}:{str(exc)}"[:256]
                ),
            )
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _expected_model(self, warm: Any) -> bool:
        if warm.identity is not None:
            return True
        registry = getattr(self.warm_model, "registry", None)
        if registry is None:
            return False
        try:
            summary = await registry.public_summary()
        except Exception:
            return False
        return bool(
            summary.registry_status == "ready"
            and summary.default_alias_ready
            and summary.default_alias_model_id is not None
        )

    def _stable_without_expected_model(self, reason_code: str) -> None:
        if self._stable_since is None:
            self._stable_since = time.monotonic()
        if (
            self._status["recovery_state"] != "IDLE"
            or self._status["primary_reason_code"] != reason_code
            or self._status["current_attempt"] != 0
        ):
            self._write_status("IDLE", reason_code)

    async def observe_once(self) -> None:
        async with self._lock:
            if not self._recovery_permitted():
                self._write_status("STOPPED", "DESIRED_STATE_STOPPED")
                return
            status = await self.backend.controller.status()
            if not status.ok or status.exit_status != 0:
                reason = (
                    "PRIVATE_LISTENER_LOST"
                    if status.reason_code == "PRIVATE_LISTENER_LOST"
                    else "ROUTER_PROCESS_LOST"
                )
                if status.reason_code in {
                    "ENDPOINT_CONFLICT",
                    "OWNERSHIP_UNCERTAIN",
                }:
                    _atomic_write(
                        self._latch_path,
                        {
                            "schema_version": LATCH_SCHEMA,
                            "profile_identity": (
                                self.settings.service_control_profile_identity
                            ),
                            "reason_code": status.reason_code,
                            "latched_utc": utc_now(),
                        },
                    )
                    self._write_status("FAIL_CLOSED", status.reason_code)
                    return
                await self._recover_router(reason)
                return
            if not self.backend._status_matches_identity(status.data):
                await self._recover_router("ROUTER_PROCESS_LOST")
                return
            warm = await self.warm_model.observe_once()
            if warm.service_readiness_state in {
                "WAITING_FOR_MODEL",
                "MODEL_CANDIDATE_LOADING",
            }:
                self._stable_without_expected_model(
                    "NO_READY_MODEL"
                    if warm.service_readiness_state == "WAITING_FOR_MODEL"
                    else "MODEL_CANDIDATE_LOADING"
                )
                return
            if warm.service_readiness_state != "READY":
                if not await self._expected_model(warm):
                    self._stable_without_expected_model(
                        "NO_EXPECTED_MODEL"
                    )
                    return
                reason = (
                    "MODEL_CHILD_LOST"
                    if warm.identity is not None
                    else "WARM_MODEL_HEALTH_LOST"
                )
                await self._recover_model(reason)
                return
            if self._stable_since is None:
                self._stable_since = time.monotonic()
            elif (
                time.monotonic() - self._stable_since
                >= float(self.settings.recovery_stable_reset_seconds)
            ):
                self._attempt_epochs.clear()
                self._write_status("IDLE", "RECOVERY_STABLE")
                self._stable_since = time.monotonic()
            elif self._status["recovery_state"] not in {
                "IDLE",
                "RECOVERED",
            }:
                self._write_status("IDLE", "RECOVERY_STABLE")

    async def _observe_loop(self) -> None:
        interval = max(
            0.05,
            min(
                float(self.settings.private_backend_poll_interval_seconds),
                5.0,
            ),
        )
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval
                )
            except TimeoutError:
                try:
                    await self.observe_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOGGER.error(
                        "system_x_runtime_recovery_observer_failed %s",
                        type(exc).__name__,
                    )

    async def startup(self) -> None:
        if not self.settings.automatic_recovery_enabled:
            self._write_status(
                "IDLE", "AUTOMATIC_RECOVERY_DISABLED"
            )
            return
        if not self._recovery_permitted():
            self._write_status("STOPPED", "DESIRED_STATE_STOPPED")
            return
        self._write_status("IDLE", "RECOVERY_STABLE")
        if self._task is not None:
            raise RuntimeError("runtime recovery observer is already running")
        self._task = asyncio.create_task(
            self._observe_loop(), name="system-x-runtime-recovery"
        )
        await asyncio.sleep(0)

    async def shutdown(self) -> None:
        self._shutdown_started = True
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._write_status("STOPPED", "DESIRED_STATE_STOPPED")
