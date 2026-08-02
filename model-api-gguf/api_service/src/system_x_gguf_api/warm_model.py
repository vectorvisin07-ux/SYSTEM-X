"""API-lifespan ownership of the configured default model's warm intent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import datetime as dt
import json
import logging
from typing import Any, Literal

from .backend import BackendError, WarmBackendObservation
from .errors import SystemXError
from .model_catalogue import ModelCatalogue, ModelSnapshot
from .model_lifecycle import (
    ModelLifecycleEvidence,
    ModelServiceState,
    resolve_model_service_state,
)
from .model_registry import ModelRegistry


ServiceReadinessState = Literal[
    "SUPERVISOR_STARTING",
    "API_STARTING",
    "BACKEND_STARTING",
    "MODEL_LOADING",
    "WAITING_FOR_MODEL",
    "MODEL_CANDIDATE_LOADING",
    "READY",
    "DEGRADED",
    "STOPPED",
    "FAIL_CLOSED",
]
LOGGER = logging.getLogger("uvicorn.error")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class WarmIdentity:
    requested_alias: str
    resolved_public_model_id: str
    artifact_version_id: str
    registry_generation: int
    capability_manifest_identity: str
    router_transaction_id: str
    model_child_pid: int
    model_child_start_identity: str
    model_child_parent: int
    model_child_process_group: int
    model_child_session: int
    warm_since_utc: str
    last_verified_utc: str
    health_state: str
    router_model_id: str

    def public_dict(self) -> dict[str, Any]:
        """Return only non-secret, non-path warm ownership evidence."""

        return {
            "requested_alias": self.requested_alias,
            "resolved_public_model_id": self.resolved_public_model_id,
            "artifact_version_id": self.artifact_version_id,
            "registry_generation": self.registry_generation,
            "capability_manifest_identity": (
                self.capability_manifest_identity
            ),
            "router_transaction_id": self.router_transaction_id,
            "model_child_pid": self.model_child_pid,
            "model_child_start_identity": self.model_child_start_identity,
            "model_child_parent": self.model_child_parent,
            "model_child_process_group": self.model_child_process_group,
            "model_child_session": self.model_child_session,
            "warm_since_utc": self.warm_since_utc,
            "last_verified_utc": self.last_verified_utc,
            "health_state": self.health_state,
        }


@dataclass(frozen=True, slots=True)
class WarmStatus:
    service_readiness_state: ServiceReadinessState
    default_alias: str
    identity: WarmIdentity | None
    reason_code: str | None
    last_transition_utc: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "service_readiness_state": self.service_readiness_state,
            "default_alias": self.default_alias,
            "resolved_public_model_id": (
                self.identity.resolved_public_model_id
                if self.identity is not None
                else None
            ),
            "artifact_version_id": (
                self.identity.artifact_version_id
                if self.identity is not None
                else None
            ),
            "registry_generation": (
                self.identity.registry_generation
                if self.identity is not None
                else None
            ),
            "reason_code": self.reason_code,
            "warm_identity": (
                self.identity.public_dict()
                if self.identity is not None
                else None
            ),
        }


def full_readiness(
    warm: WarmStatus,
    backend: Any,
    registry: Any,
    *,
    authentication_ready: bool,
    recovery_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve model state while keeping service and inference readiness distinct."""

    identity = warm.identity
    recovery_usable = bool(
        recovery_status is None
        or recovery_status.get("recovery_state") in {"IDLE", "RECOVERED"}
    )
    predicates = {
        "warm_ready": warm.service_readiness_state == "READY",
        "warm_identity_present": identity is not None,
        "backend_status_ready": backend.status == "router_ready",
        "backend_process_running": backend.process_running is True,
        "backend_control_plane_ready": backend.control_plane_ready is True,
        "one_model_loaded": backend.loaded_model_count == 1,
        "model_ready": backend.model_ready is True,
        "registry_ready": registry.registry_status == "ready",
        "registry_has_ready_model": registry.ready_model_count >= 1,
        "authentication_ready": authentication_ready is True,
        "private_health_ready": bool(
            identity is not None and identity.health_state == "ready"
        ),
        "recovery_usable": recovery_usable,
    }
    default_alias_model_id = getattr(
        registry, "default_alias_model_id", None
    )
    resolved_model_id = (
        str(default_alias_model_id)
        if default_alias_model_id is not None
        else identity.resolved_public_model_id
        if identity is not None
        else None
    )
    resolved_alias = warm.default_alias if resolved_model_id is not None else None
    default_target_ready = bool(
        getattr(registry, "default_alias_ready", False)
        or identity is not None
    )
    lifecycle = resolve_model_service_state(
        ModelLifecycleEvidence(
            desired_state=(
                "STOPPED"
                if warm.service_readiness_state == "STOPPED"
                else "RUNNING"
            ),
            fail_closed_latch=bool(
                recovery_status is not None
                and recovery_status.get("recovery_state") == "FAIL_CLOSED"
            ),
            control_plane_operational=bool(
                predicates["backend_status_ready"]
                and predicates["backend_process_running"]
                and predicates["backend_control_plane_ready"]
                and predicates["authentication_ready"]
                and predicates["recovery_usable"]
            ),
            registry_available=predicates["registry_ready"],
            configured_default_alias=warm.default_alias,
            resolved_default_alias=resolved_alias,
            resolved_public_model_id=resolved_model_id,
            default_target_ready=default_target_ready,
            warm_identity_present=identity is not None,
            exact_target_warm_healthy=all(predicates.values()),
            ready_public_model_count=max(
                0, int(getattr(registry, "ready_model_count", 0))
            ),
            candidate_model_count=max(
                0, int(getattr(registry, "candidate_model_count", 0))
            ),
        )
    )
    state = lifecycle.state.value
    ready = lifecycle.inference_ready
    service_available = lifecycle.service_available
    reason_code = None if ready else lifecycle.reason_code
    if recovery_status is not None and not recovery_usable:
        reason_code = str(
            recovery_status.get(
                "primary_reason_code", "runtime_recovery_active"
            )
        )
    service_status = (
        "stopped"
        if lifecycle.state is ModelServiceState.STOPPED
        else "ready"
        if service_available
        else "degraded"
    )
    return {
        "service_readiness_state": state,
        "model_service_state": state,
        "service_available": service_available,
        "inference_ready": ready,
        "ready": ready,
        "service_status": service_status,
        "reason_code": reason_code,
        "default_alias": warm.default_alias,
        "configured_default_alias": warm.default_alias,
        "resolved_default_alias": resolved_alias,
        "resolved_public_model_id": resolved_model_id,
        "artifact_version_id": (
            identity.artifact_version_id if identity else None
        ),
        "warm_identity": identity.public_dict() if identity else None,
        "predicates": predicates,
    }


class WarmModelCoordinator:
    """Resolve, warm, retain, verify, and observe one authoritative alias."""

    def __init__(
        self,
        settings: Any,
        catalogue: ModelCatalogue,
        registry: ModelRegistry,
        backend: Any,
    ) -> None:
        self.settings = settings
        self.catalogue = catalogue
        self.registry = registry
        self.backend = backend
        self.default_alias = str(settings.registry_default_alias)
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._observer_task: asyncio.Task[None] | None = None
        self._status = WarmStatus(
            "BACKEND_STARTING",
            self.default_alias,
            None,
            "backend_starting",
            utc_now(),
        )

    @property
    def status(self) -> WarmStatus:
        return self._status

    @staticmethod
    def _same_target(
        identity: WarmIdentity | None,
        snapshot: ModelSnapshot,
    ) -> bool:
        return bool(
            identity is not None
            and identity.resolved_public_model_id
            == snapshot.public_model_id
            and identity.artifact_version_id == snapshot.bundle_id
            and identity.router_model_id == snapshot.router_model_id
            and identity.capability_manifest_identity
            == snapshot.capability_manifest_identity
        )

    def _transition(
        self,
        state: ServiceReadinessState,
        reason_code: str | None,
        *,
        identity: WarmIdentity | None = None,
    ) -> None:
        self._status = WarmStatus(
            state,
            self.default_alias,
            self._status.identity if identity is None else identity,
            reason_code,
            utc_now(),
        )

    def _degrade(self, reason_code: str) -> WarmStatus:
        self._transition("DEGRADED", reason_code)
        return self._status

    def _idle_without_model(
        self, state: ServiceReadinessState, reason_code: str
    ) -> WarmStatus:
        self._status = WarmStatus(
            state,
            self.default_alias,
            None,
            reason_code,
            utc_now(),
        )
        return self._status

    @staticmethod
    def _resolution_reason(exc: SystemXError) -> str:
        if exc.code == "system_x_model_not_found":
            return "default_alias_missing"
        if exc.code == "system_x_model_unavailable":
            return "default_target_not_ready"
        return "registry_unavailable"

    @staticmethod
    def _identity_from(
        alias: str,
        snapshot: ModelSnapshot,
        observation: WarmBackendObservation,
        *,
        prior: WarmIdentity | None,
    ) -> WarmIdentity:
        now = utc_now()
        child = observation.model_child
        warm_since = (
            prior.warm_since_utc
            if prior is not None
            and prior.resolved_public_model_id == snapshot.public_model_id
            and prior.artifact_version_id == snapshot.bundle_id
            and prior.model_child_pid == child.pid
            and prior.model_child_start_identity
            == child.process_start_identity
            else now
        )
        return WarmIdentity(
            requested_alias=alias,
            resolved_public_model_id=snapshot.public_model_id,
            artifact_version_id=snapshot.bundle_id,
            registry_generation=snapshot.registry_generation,
            capability_manifest_identity=(
                snapshot.capability_manifest_identity
            ),
            router_transaction_id=(
                observation.router_identity.transaction_id
            ),
            model_child_pid=child.pid,
            model_child_start_identity=child.process_start_identity,
            model_child_parent=child.ppid,
            model_child_process_group=child.pgid,
            model_child_session=child.sid,
            router_model_id=snapshot.router_model_id,
            warm_since_utc=warm_since,
            last_verified_utc=now,
            health_state="ready",
        )

    def _log_ready(
        self,
        identity: WarmIdentity,
        observation: WarmBackendObservation,
        *,
        target_changed: bool,
    ) -> None:
        LOGGER.info(
            "system_x_warm_model %s",
            json.dumps(
                {
                    "event": (
                        "warm_target_changed"
                        if target_changed
                        else "warm_target_verified"
                    ),
                    "requested_alias": identity.requested_alias,
                    "resolved_public_model_id": (
                        identity.resolved_public_model_id
                    ),
                    "artifact_version_id": identity.artifact_version_id,
                    "registry_generation": identity.registry_generation,
                    "router_transaction_id": (
                        identity.router_transaction_id
                    ),
                    "model_child_pid": identity.model_child_pid,
                    "model_child_start_identity": (
                        identity.model_child_start_identity
                    ),
                    "load_performed": observation.load_performed,
                    "unload_performed": observation.unload_performed,
                    "health_state": identity.health_state,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    async def observe_once(self) -> WarmStatus:
        """Observe authoritative registry state and verify or adopt it once."""

        async with self._lock:
            if self._status.service_readiness_state == "STOPPED":
                return self._status
            try:
                summary = await self.registry.public_summary()
            except Exception:
                return self._degrade("registry_unavailable")
            if summary.registry_status != "ready":
                return self._degrade(
                    f"registry_{summary.registry_status}"
                )
            try:
                snapshot = await self.catalogue.resolve(self.default_alias)
            except SystemXError as exc:
                if exc.code == "system_x_no_ready_model":
                    if self._status.identity is not None:
                        return self._degrade("expected_default_target_lost")
                    if summary.default_alias_model_id is not None:
                        return self._degrade("default_target_not_ready")
                    if summary.ready_model_count:
                        return self._degrade(
                            "ready_model_without_default_alias"
                        )
                    if summary.candidate_model_count:
                        return self._idle_without_model(
                            "MODEL_CANDIDATE_LOADING",
                            "MODEL_CANDIDATE_LOADING",
                        )
                    return self._idle_without_model(
                        "WAITING_FOR_MODEL", "NO_READY_MODEL"
                    )
                return self._degrade(self._resolution_reason(exc))
            if (
                snapshot.registration_state != "READY"
                or not snapshot.capability_manifest_identity
            ):
                return self._degrade("default_target_not_warmable")

            prior = self._status.identity
            same_target = self._same_target(prior, snapshot)
            self._transition(
                "MODEL_LOADING",
                "model_health_verification"
                if same_target
                else "model_loading",
            )
            try:
                if same_target:
                    observation = await self.backend.verify_warm_model(
                        snapshot.router_model_id,
                        lambda: self.catalogue.verify(snapshot),
                    )
                else:
                    observation = await self.backend.ensure_warm_model(
                        snapshot.router_model_id,
                        (
                            prior.router_model_id
                            if prior is not None
                            else None
                        ),
                        lambda: self.catalogue.verify(snapshot),
                    )
                if not await self.catalogue.verify(snapshot):
                    return self._degrade("default_target_changed_during_warm")
            except BackendError:
                return self._degrade("default_target_private_health_failed")
            except Exception:
                return self._degrade("default_target_warm_failed")

            identity = self._identity_from(
                self.default_alias,
                snapshot,
                observation,
                prior=prior if same_target else None,
            )
            self._transition("READY", None, identity=identity)
            self._log_ready(
                identity,
                observation,
                target_changed=not same_target,
            )
            return self._status

    async def recover_current_target(self) -> WarmStatus:
        """Reload the exact current READY alias target after owned loss."""

        async with self._lock:
            if self._status.service_readiness_state == "STOPPED":
                raise BackendError("warm recovery is fenced by shutdown")
            try:
                summary = await self.registry.public_summary()
                if summary.registry_status != "ready":
                    raise BackendError("registry is not ready for warm recovery")
                snapshot = await self.catalogue.resolve(self.default_alias)
                if (
                    snapshot.registration_state != "READY"
                    or not snapshot.capability_manifest_identity
                ):
                    raise BackendError(
                        "default target is not READY for warm recovery"
                    )
                prior = self._status.identity
                self._transition(
                    "MODEL_LOADING", "model_child_recovery_loading"
                )
                observation = await self.backend.ensure_warm_model(
                    snapshot.router_model_id,
                    prior.router_model_id if prior is not None else None,
                    lambda: self.catalogue.verify(snapshot),
                )
                if not await self.catalogue.verify(snapshot):
                    raise BackendError(
                        "default target changed during warm recovery"
                    )
                identity = self._identity_from(
                    self.default_alias,
                    snapshot,
                    observation,
                    prior=(
                        prior
                        if self._same_target(prior, snapshot)
                        else None
                    ),
                )
                self._transition("READY", None, identity=identity)
                self._log_ready(
                    identity,
                    observation,
                    target_changed=not self._same_target(prior, snapshot),
                )
                return self._status
            except BackendError:
                self._degrade("default_target_recovery_failed")
                raise
            except Exception as exc:
                self._degrade("default_target_recovery_failed")
                raise BackendError(
                    "default target recovery failed"
                ) from exc

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
                except Exception:
                    self._degrade("warm_observer_failed")

    async def startup(self, *, start_observer: bool = True) -> WarmStatus:
        if self.settings.startup_model_policy != "always_warm":
            raise RuntimeError("startup model policy must be always_warm")
        if not self.settings.registry_enabled:
            raise RuntimeError("always_warm requires the model registry")
        if not self.settings.private_backend_enabled:
            raise RuntimeError("always_warm requires the private backend")
        self._transition("MODEL_LOADING", "startup_model_loading")
        await self.observe_once()
        if start_observer:
            if self._observer_task is not None:
                raise RuntimeError("warm-model observer is already running")
            self._observer_task = asyncio.create_task(
                self._observe_loop(), name="system-x-warm-model-observer"
            )
            await asyncio.sleep(0)
        return self._status

    async def shutdown(self) -> WarmStatus:
        self._stop_event.set()
        if self._observer_task is not None:
            try:
                await asyncio.wait_for(self._observer_task, timeout=5.0)
            except TimeoutError:
                self._observer_task.cancel()
                await asyncio.gather(
                    self._observer_task, return_exceptions=True
                )
            self._observer_task = None
        identity = self._status.identity
        if identity is not None:
            await self.backend.release_warm_intent(identity.router_model_id)
            identity = replace(
                identity,
                last_verified_utc=utc_now(),
                health_state="stopped",
            )
        self._status = WarmStatus(
            "STOPPED",
            self.default_alias,
            identity,
            "controlled_shutdown",
            utc_now(),
        )
        return self._status
