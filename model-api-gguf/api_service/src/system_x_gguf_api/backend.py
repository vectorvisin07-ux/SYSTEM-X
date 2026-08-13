"""Lifespan coordinator for the private branch-owned llama-server router."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from .controller_client import (
    BranchControllerClient,
    ControllerClientError,
    ControllerResult,
)
from .router_client import RouterClient, RouterModelList, RouterObservation
from .settings import ServiceSettings


class BackendError(RuntimeError):
    """A bounded private-backend lifecycle or policy failure."""


class BackendModelConflict(BackendError):
    """A different private model is active and switching is prohibited."""


class BackendModelUnavailable(BackendError):
    """The resolved private model cannot be used by the current router."""


class BackendSnapshotConflict(BackendError):
    """The registry mapping changed after public request resolution."""


@dataclass(frozen=True)
class RouterIdentity:
    transaction_id: str
    pid: int
    pgid: int
    sid: int
    process_start_identity: str


@dataclass(frozen=True)
class PublicBackendState:
    status: str
    process_running: bool
    control_plane_ready: bool
    loaded_model_count: int
    maximum_loaded_models: int
    model_ready: bool


@dataclass(frozen=True)
class ModelPropertiesProbe:
    model_id: str
    props: RouterObservation
    loaded_before_probe: bool
    registry_owned_load: bool
    load_observation: RouterObservation | None
    unload_observation: RouterObservation | None
    final_status: str


@dataclass(frozen=True)
class ValidatedInventoryRefresh:
    inventory: RouterModelList
    unloaded_model_ids: tuple[str, ...]


@dataclass(frozen=True)
class InferenceBackendLease:
    router: RouterClient
    router_identity: RouterIdentity
    router_model_id: str
    loaded_by_request: bool
    model_status: str


@dataclass(frozen=True)
class ModelChildIdentity:
    pid: int
    process_start_identity: str
    ppid: int
    pgid: int
    sid: int


@dataclass(frozen=True)
class WarmBackendObservation:
    router_identity: RouterIdentity
    router_model_id: str
    model_status: str
    load_performed: bool
    unload_performed: bool
    private_health_ready: bool
    model_child: ModelChildIdentity


ProcessGroupObserver = Callable[
    [RouterIdentity], tuple[ModelChildIdentity, ...]
]


def _owned_process_group_members(
    identity: RouterIdentity,
) -> tuple[ModelChildIdentity, ...]:
    """Read bounded child identities without exposing argv or environments."""

    members: list[ModelChildIdentity] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="ascii")
            close = raw.rfind(")")
            fields = raw[close + 2 :].split()
            if (
                len(fields) < 20
                or fields[0] == "Z"
                or int(fields[2]) != identity.pgid
                or int(entry.name) == identity.pid
            ):
                continue
            members.append(
                ModelChildIdentity(
                    pid=int(entry.name),
                    process_start_identity=(
                        f"procfs-start-ticks:{fields[19]}"
                    ),
                    ppid=int(fields[1]),
                    pgid=int(fields[2]),
                    sid=int(fields[3]),
                )
            )
        except (
            FileNotFoundError,
            IndexError,
            OSError,
            PermissionError,
            ValueError,
        ):
            continue
    return tuple(sorted(members, key=lambda member: member.pid))


def _exact_owned_model_child(
    identity: RouterIdentity,
    members: tuple[ModelChildIdentity, ...],
) -> ModelChildIdentity:
    direct = tuple(
        member
        for member in members
        if member.ppid == identity.pid
        and member.pgid == identity.pgid
        and member.sid == identity.sid
    )
    if len(direct) != 1 or len(members) != 1:
        raise BackendModelUnavailable(
            "exact owned model-child identity is unavailable"
        )
    child = direct[0]
    if child.pid == identity.pid or child.pid <= 0:
        raise BackendModelUnavailable("model-child identity is invalid")
    return child


def _require_controller_success(
    result: ControllerResult, operation: str
) -> dict[str, Any]:
    if (
        not result.ok
        or result.exit_status != 0
        or result.operation != operation
        or result.stderr
    ):
        raise BackendError(
            f"branch controller {operation} failed: {result.reason_code}"
        )
    return result.data


class BackendCoordinator:
    """Own one private router transaction for one API-service lifespan."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        process_group_observer: ProcessGroupObserver | None = None,
    ) -> None:
        self.settings = settings
        self.controller = BranchControllerClient(
            settings.private_backend_start_timeout_seconds
        )
        self.router: RouterClient | None = None
        self.identity: RouterIdentity | None = None
        self._router_ready = False
        self._loaded_by_transaction: set[str] = set()
        self._warm_model_id: str | None = None
        self._model_operation_lock = asyncio.Lock()
        self._process_group_observer = (
            process_group_observer or _owned_process_group_members
        )

    def _validate_plan(self, data: dict[str, Any]) -> None:
        endpoint = data.get("private_endpoint")
        resolved = data.get("resolved_paths")
        if not isinstance(endpoint, dict) or not isinstance(resolved, dict):
            raise BackendError("router plan omitted required structured fields")
        binary = resolved.get("binary_path")
        models_dir = data.get("models_dir")
        router_cache = data.get("router_cache")
        if not all(isinstance(value, str) and value for value in (binary, models_dir, router_cache)):
            raise BackendError("router plan paths were invalid")
        expected_argv = [
            binary,
            "--host",
            self.settings.private_backend_host,
            "--port",
            str(self.settings.private_backend_port),
            "--models-dir",
            models_dir,
            "--models-max",
            str(self.settings.private_backend_models_max),
            "--no-models-autoload",
            "--jinja",
            "--n-gpu-layers",
            "auto",
            "--fit",
            "on",
            "--offline",
            "--no-webui",
            "--no-agent",
            "--no-ui-mcp-proxy",
            "--cors-origins",
            "localhost",
            "--no-cors-credentials",
        ]
        required = {
            "launch_mode": "router",
            "models_max": 1,
            "models_autoload": False,
            "jinja": True,
            "offline": True,
            "private_endpoint": {
                "host": self.settings.private_backend_host,
                "port": self.settings.private_backend_port,
            },
            "environment_overrides": {"LLAMA_CACHE": router_cache},
            "argv": expected_argv,
        }
        for key, expected in required.items():
            if data.get(key) != expected:
                raise BackendError(f"router plan policy mismatch: {key}")

    @staticmethod
    def _parse_identity(data: dict[str, Any]) -> RouterIdentity:
        transaction_id = data.get("transaction_id")
        process_start_identity = data.get("process_start_identity")
        pid = data.get("pid")
        pgid = data.get("pgid")
        sid = data.get("sid")
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
            or not isinstance(process_start_identity, str)
            or not process_start_identity
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(pgid, int)
            or pgid != pid
            or not isinstance(sid, int)
            or sid != pid
            or data.get("launch_mode") != "router"
        ):
            raise BackendError("router start returned an invalid process identity")
        return RouterIdentity(transaction_id, pid, pgid, sid, process_start_identity)

    def _status_matches_identity(self, data: dict[str, Any]) -> bool:
        identity = self.identity
        return bool(
            identity
            and data.get("active") is True
            and data.get("active_state_consistent") is True
            and data.get("transaction_id") == identity.transaction_id
            and data.get("pid") == identity.pid
            and data.get("pgid") == identity.pgid
            and data.get("sid") == identity.sid
            and data.get("launch_mode") == "router"
        )

    async def _wait_for_router_models(self) -> RouterModelList:
        if self.router is None:
            raise BackendError("router HTTP client is unavailable")
        deadline = (
            time.monotonic() + self.settings.private_backend_start_timeout_seconds
        )
        last: RouterModelList | None = None
        while time.monotonic() < deadline:
            last = await self.router.list_models()
            if last.valid and last.observation.status_code == 200:
                return last
            await asyncio.sleep(self.settings.private_backend_poll_interval_seconds)
        detail = (
            last.observation.error
            if last is not None
            else "no router observation"
        )
        raise BackendError(f"private router readiness timed out: {detail}")

    async def _startup_prestate(self) -> dict[str, Any]:
        """Reconcile only a controller-owned stale inactive prestate."""

        result = await self.controller.status()
        if (
            (
                not result.ok
                or result.exit_status != 0
                or result.operation != "status"
                or result.stderr
            )
            and result.operation == "status"
            and result.reason_code == "ACTIVE_STATE_INCONSISTENT"
            and not result.stderr
        ):
            reconciled = _require_controller_success(
                await self.controller.reconcile(), "reconcile"
            )
            if (
                reconciled.get("active") is not False
                or reconciled.get("active_state_consistent") is not True
            ):
                raise BackendError(
                    "branch controller stale prestate reconciliation "
                    "did not reach inactive-consistent"
                )
            result = await self.controller.status()
        status = _require_controller_success(result, "status")
        if status.get("active") is not False or status.get(
            "active_state_consistent"
        ) is not True:
            raise BackendError(
                "branch controller prestate is not inactive-consistent"
            )
        return status

    async def startup(self) -> None:
        if not self.settings.private_backend_enabled:
            return
        started = False
        try:
            await self._startup_prestate()
            plan = _require_controller_success(
                await self.controller.plan_router(
                    self.settings.private_backend_host,
                    self.settings.private_backend_port,
                    self.settings.private_backend_models_max,
                ),
                "plan",
            )
            self._validate_plan(plan)
            started_data = _require_controller_success(
                await self.controller.start_router(
                    self.settings.private_backend_host,
                    self.settings.private_backend_port,
                    self.settings.private_backend_models_max,
                ),
                "start",
            )
            self.identity = self._parse_identity(started_data)
            started = True
            self.router = RouterClient(
                self.settings.private_backend_host,
                self.settings.private_backend_port,
                self.settings.private_backend_model_timeout_seconds,
                inference_timeout_seconds=(
                    self.settings.private_backend_inference_timeout_seconds
                ),
            )
            await self._wait_for_router_models()
            self._router_ready = True
        except BaseException:
            if started:
                await self._stop_exact_router()
            raise

    async def _shutdown_router_status(self) -> dict[str, Any]:
        """Read or conservatively reconcile router state during API teardown."""

        try:
            status = await self.controller.status()
        except ControllerClientError:
            status = None
        if status is not None:
            if status.ok and status.exit_status == 0:
                return status.data
            if status.reason_code not in {
                "ACTIVE_STATE_INCONSISTENT",
                "PRIVATE_LISTENER_LOST",
            }:
                return _require_controller_success(status, "status")
        return _require_controller_success(
            await self.controller.reconcile(), "reconcile"
        )

    def _shutdown_router_identity_matches(self, data: dict[str, Any]) -> bool:
        identity = self.identity
        return bool(
            identity
            and data.get("active") is True
            and data.get("transaction_id") == identity.transaction_id
            and data.get("pid") == identity.pid
            and data.get("pgid") == identity.pgid
            and data.get("sid") == identity.sid
            and data.get("launch_mode") == "router"
        )

    async def _stop_exact_router(self) -> None:
        close_error: BaseException | None = None
        if self.router is not None:
            try:
                await self.router.aclose()
            except BaseException as exc:
                close_error = exc
            self.router = None
        status = await self._shutdown_router_status()
        if status.get("active") is True:
            if not self._shutdown_router_identity_matches(status):
                raise BackendError("active router ownership did not match lifespan identity")
            stopped = _require_controller_success(
                await self.controller.stop(), "stop"
            )
            if (
                stopped.get("owned_group_absent") is not True
                or stopped.get("active_pid_record_removed") is not True
                or stopped.get("active_lock_removed") is not True
            ):
                raise BackendError("branch controller stop proof was incomplete")
        final_status = await self._shutdown_router_status()
        if (
            final_status.get("active") is not False
            or final_status.get("active_state_consistent") is not True
        ):
            raise BackendError("branch controller did not reach inactive state")
        self.identity = None
        self._router_ready = False
        self._loaded_by_transaction.clear()
        self._warm_model_id = None
        if close_error is not None:
            raise BackendError("router HTTP client close failed") from close_error

    async def recover_router(self) -> RouterIdentity:
        """Reconcile and restart only the exact controller-owned router."""

        async with self._model_operation_lock:
            reconciled = await self.controller.reconcile()
            if not reconciled.ok or reconciled.exit_status != 0:
                raise BackendError(
                    "branch controller reconcile failed: "
                    f"{reconciled.reason_code}"
                )
            data = reconciled.data
            if data.get("active") is True:
                identity = self.identity
                if identity is None or any(
                    data.get(name) != expected
                    for name, expected in (
                        ("transaction_id", identity.transaction_id),
                        ("pid", identity.pid),
                        ("pgid", identity.pgid),
                        ("sid", identity.sid),
                        (
                            "process_start_identity",
                            identity.process_start_identity,
                        ),
                    )
                ):
                    raise BackendError(
                        "live router reconciliation identity is uncertain"
                    )
                stopped = _require_controller_success(
                    await self.controller.stop(), "stop"
                )
                if (
                    stopped.get("owned_group_absent") is not True
                    or stopped.get("active_pid_record_removed") is not True
                    or stopped.get("active_lock_removed") is not True
                ):
                    raise BackendError(
                        "router recovery stop proof was incomplete"
                    )
            close_error: BaseException | None = None
            if self.router is not None:
                try:
                    await self.router.aclose()
                except BaseException as exc:
                    close_error = exc
            self.router = None
            self.identity = None
            self._router_ready = False
            self._loaded_by_transaction.clear()
            self._warm_model_id = None
            if close_error is not None:
                raise BackendError(
                    "stale router client close failed during recovery"
                ) from close_error
            await self.startup()
            if self.identity is None or not self._router_ready:
                raise BackendError("router recovery did not establish identity")
            return self.identity

    async def load_model(self, model_id: str) -> RouterObservation:
        if not self._router_ready or self.router is None:
            raise BackendError("private router is not ready")
        result = await self.router.load_model(model_id)
        if (
            result.status_code == 200
            and isinstance(result.json_value, dict)
            and result.json_value.get("success") is True
        ):
            self._loaded_by_transaction.add(model_id)
        return result

    async def unload_model(self, model_id: str) -> RouterObservation:
        if not self._router_ready or self.router is None:
            raise BackendError("private router is not ready")
        result = await self.router.unload_model(model_id)
        if (
            result.status_code == 200
            and isinstance(result.json_value, dict)
            and result.json_value.get("success") is True
        ):
            self._loaded_by_transaction.discard(model_id)
        return result

    async def refresh_router_inventory(self) -> RouterModelList:
        if not self._router_ready or self.router is None:
            raise BackendError("private router is not ready")
        result = await self.router.list_models(reload=True)
        if (
            not result.valid
            or result.observation.status_code != 200
            or result.observation.error is not None
        ):
            raise BackendError("private router inventory refresh failed")
        return result

    async def current_router_inventory(self) -> RouterModelList:
        if not self._router_ready or self.router is None:
            raise BackendError("private router is not ready")
        result = await self.router.list_models()
        if (
            not result.valid
            or result.observation.status_code != 200
            or result.observation.error is not None
        ):
            raise BackendError("private router inventory read failed")
        return result

    async def active_model_properties(
        self, model_id: str
    ) -> dict[str, Any] | None:
        """Read owned active /props without loading, unloading, or switching."""

        if not self._router_ready or self.router is None:
            return None
        try:
            inventory = await self.current_router_inventory()
            target = next(
                (
                    model
                    for model in inventory.models
                    if model.model_id == model_id
                ),
                None,
            )
            if target is None or target.status not in {"loaded", "sleeping"}:
                return None
            observation = await self.router.get_props(
                model_id, autoload=False
            )
        except (BackendError, RuntimeError):
            return None
        if (
            observation.status_code != 200
            or observation.error is not None
            or not isinstance(observation.json_value, dict)
        ):
            return None
        return observation.json_value

    async def refresh_validated_model_inventory(
        self,
        replacement_model_ids: set[str],
    ) -> ValidatedInventoryRefresh:
        """Unload only locally validated replacements, then refresh once."""

        async with self.serialized_model_operation():
            current = await self.current_router_inventory()
            unloaded: list[str] = []
            for model in sorted(current.models, key=lambda item: item.model_id):
                if (
                    model.model_id not in replacement_model_ids
                    or model.status not in {"loading", "loaded", "sleeping"}
                ):
                    continue
                observation = await self.unload_model(model.model_id)
                if (
                    observation.status_code != 200
                    or observation.error is not None
                    or not isinstance(observation.json_value, dict)
                    or observation.json_value.get("success") is not True
                ):
                    raise BackendError(
                        "validated replacement model unload failed"
                    )
                await self._wait_for_model_status(
                    model.model_id, {"unloaded"}
                )
                unloaded.append(model.model_id)
            refreshed = await self.refresh_router_inventory()
            return ValidatedInventoryRefresh(
                inventory=refreshed,
                unloaded_model_ids=tuple(unloaded),
            )

    @asynccontextmanager
    async def serialized_model_operation(self) -> AsyncIterator[None]:
        async with self._model_operation_lock:
            yield

    async def _assert_router_identity(self) -> RouterIdentity:
        identity = self.identity
        if (
            not self._router_ready
            or self.router is None
            or identity is None
        ):
            raise BackendError("private router is not ready")
        try:
            status = _require_controller_success(
                await self.controller.status(), "status"
            )
        except ControllerClientError as exc:
            raise BackendError("private router ownership check failed") from exc
        if not self._status_matches_identity(status):
            raise BackendError("private router ownership identity changed")
        return identity

    async def _private_model_health(
        self,
        model_id: str,
        identity: RouterIdentity,
    ) -> ModelChildIdentity:
        router = self.router
        if router is None:
            raise BackendError("private router client is unavailable")
        health = await router.health()
        props = await router.get_props(model_id, autoload=False)
        if (
            health.status_code != 200
            or health.error is not None
            or not isinstance(health.json_value, dict)
            or props.status_code != 200
            or props.error is not None
            or not isinstance(props.json_value, dict)
        ):
            raise BackendModelUnavailable(
                "private loaded-model health is unavailable"
            )
        members = self._process_group_observer(identity)
        return _exact_owned_model_child(identity, members)

    async def _warm_model_observation(
        self,
        model_id: str,
        prior_warm_model_id: str | None,
        snapshot_verifier: Callable[[], Awaitable[bool]],
        *,
        permit_load: bool,
    ) -> WarmBackendObservation:
        async with self.serialized_model_operation():
            if not await snapshot_verifier():
                raise BackendSnapshotConflict(
                    "registry model snapshot changed before warm verification"
                )
            identity = await self._assert_router_identity()
            inventory = await self.current_router_inventory()
            matches = [
                model for model in inventory.models if model.model_id == model_id
            ]
            if len(matches) != 1:
                raise BackendModelUnavailable(
                    "warm target is absent or duplicated in router inventory"
                )
            active_states = {"loading", "loaded", "sleeping"}
            foreign = [
                model
                for model in inventory.models
                if model.model_id != model_id
                and model.status in active_states
            ]
            unload_performed = False
            if foreign:
                if (
                    not permit_load
                    or prior_warm_model_id is None
                    or {model.model_id for model in foreign}
                    != {prior_warm_model_id}
                ):
                    raise BackendModelConflict(
                        "another model is active and is not the prior warm target"
                    )
                for previous in foreign:
                    observation = await self.unload_model(previous.model_id)
                    if (
                        observation.status_code != 200
                        or observation.error is not None
                        or not isinstance(observation.json_value, dict)
                        or observation.json_value.get("success") is not True
                    ):
                        raise BackendModelUnavailable(
                            "prior warm target unload failed"
                        )
                    await self._wait_for_model_status(
                        previous.model_id, {"unloaded"}
                    )
                    unload_performed = True
                inventory = await self.current_router_inventory()
                matches = [
                    model
                    for model in inventory.models
                    if model.model_id == model_id
                ]
                if len(matches) != 1:
                    raise BackendModelUnavailable(
                        "warm target disappeared after prior-target unload"
                    )

            target = matches[0]
            load_performed = False
            if target.status in {"loaded", "sleeping"}:
                final_status = target.status
            elif target.status == "loading":
                final_status = await self._wait_for_model_status(
                    model_id, {"loaded", "sleeping"}
                )
            elif target.status == "unloaded" and permit_load:
                load = await self.load_model(model_id)
                if (
                    load.status_code != 200
                    or load.error is not None
                    or not isinstance(load.json_value, dict)
                    or load.json_value.get("success") is not True
                ):
                    raise BackendModelUnavailable(
                        "explicit warm-target load failed"
                    )
                load_performed = True
                final_status = await self._wait_for_model_status(
                    model_id, {"loaded", "sleeping"}
                )
            else:
                raise BackendModelUnavailable(
                    "warm target is not in a ready loaded state"
                )

            final_inventory = await self.current_router_inventory()
            final_matches = [
                model
                for model in final_inventory.models
                if model.model_id == model_id
            ]
            if (
                len(final_matches) != 1
                or final_matches[0].status not in {"loaded", "sleeping"}
                or any(
                    model.model_id != model_id
                    and model.status in active_states
                    for model in final_inventory.models
                )
            ):
                raise BackendModelUnavailable(
                    "private warm-model state is inconsistent"
                )
            if not await snapshot_verifier():
                raise BackendSnapshotConflict(
                    "registry model snapshot changed during warming"
                )
            identity = await self._assert_router_identity()
            child = await self._private_model_health(model_id, identity)
            return WarmBackendObservation(
                router_identity=identity,
                router_model_id=model_id,
                model_status=final_status,
                load_performed=load_performed,
                unload_performed=unload_performed,
                private_health_ready=True,
                model_child=child,
            )

    async def ensure_warm_model(
        self,
        model_id: str,
        prior_warm_model_id: str | None,
        snapshot_verifier: Callable[[], Awaitable[bool]],
    ) -> WarmBackendObservation:
        observation = await self._warm_model_observation(
            model_id,
            prior_warm_model_id,
            snapshot_verifier,
            permit_load=True,
        )
        self._warm_model_id = model_id
        return observation

    async def verify_warm_model(
        self,
        model_id: str,
        snapshot_verifier: Callable[[], Awaitable[bool]],
    ) -> WarmBackendObservation:
        if self._warm_model_id != model_id:
            raise BackendModelUnavailable(
                "requested model is not the retained warm target"
            )
        return await self._warm_model_observation(
            model_id,
            model_id,
            snapshot_verifier,
            permit_load=False,
        )

    async def release_warm_intent(self, model_id: str) -> None:
        async with self.serialized_model_operation():
            if self._warm_model_id not in {None, model_id}:
                raise BackendModelConflict(
                    "warm intent belongs to another model"
                )
            self._warm_model_id = None

    async def _unload_for_temporary_switch(
        self, model_id: str, failure_message: str
    ) -> None:
        observation = await self.unload_model(model_id)
        if (
            observation.status_code != 200
            or observation.error is not None
            or not isinstance(observation.json_value, dict)
            or observation.json_value.get("success") is not True
        ):
            raise BackendModelUnavailable(failure_message)
        await self._wait_for_model_status(model_id, {"unloaded"})

    async def _restore_retained_warm_model(self, model_id: str) -> None:
        """Restore one retained warm intent while the model lock is held."""

        inventory = await self.current_router_inventory()
        matches = [
            model for model in inventory.models if model.model_id == model_id
        ]
        if len(matches) != 1:
            raise BackendModelUnavailable(
                "retained warm target disappeared during temporary switch"
            )
        target = matches[0]
        if target.status in {"loaded", "sleeping"}:
            pass
        elif target.status == "loading":
            await self._wait_for_model_status(
                model_id, {"loaded", "sleeping"}
            )
        elif target.status == "unloaded":
            load = await self.load_model(model_id)
            if (
                load.status_code != 200
                or load.error is not None
                or not isinstance(load.json_value, dict)
                or load.json_value.get("success") is not True
            ):
                raise BackendModelUnavailable(
                    "retained warm target restore load failed"
                )
            await self._wait_for_model_status(
                model_id, {"loaded", "sleeping"}
            )
        else:
            raise BackendModelUnavailable(
                "retained warm target entered an unrestorable state"
            )
        identity = await self._assert_router_identity()
        await self._private_model_health(model_id, identity)

    @asynccontextmanager
    async def inference_session(
        self,
        model_id: str,
        snapshot_verifier: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[InferenceBackendLease]:
        """Serialize resolve verification, explicit load/reuse, and inference."""

        async with self.serialized_model_operation():
            if not await snapshot_verifier():
                raise BackendSnapshotConflict(
                    "registry model snapshot changed before inference"
                )
            identity = await self._assert_router_identity()
            inventory = await self.current_router_inventory()
            matches = [
                model for model in inventory.models if model.model_id == model_id
            ]
            if len(matches) != 1:
                raise BackendModelUnavailable(
                    "resolved model is absent or duplicated in router inventory"
                )
            target = matches[0]
            active_states = {"loading", "loaded", "sleeping"}
            foreign = [
                model
                for model in inventory.models
                if model.model_id != model_id
                and model.status in active_states
            ]
            retained_warm_to_restore: str | None = None
            if foreign:
                retained = self._warm_model_id
                if (
                    retained is None
                    or retained == model_id
                    or {model.model_id for model in foreign} != {retained}
                ):
                    raise BackendModelConflict(
                        "another model is active and is not the retained warm target"
                    )
                for previous in foreign:
                    await self._unload_for_temporary_switch(
                        previous.model_id,
                        "retained warm target unload failed before inference",
                    )
                retained_warm_to_restore = retained
                inventory = await self.current_router_inventory()
                matches = [
                    model
                    for model in inventory.models
                    if model.model_id == model_id
                ]
                if len(matches) != 1:
                    raise BackendModelUnavailable(
                        "inference target disappeared during warm switch"
                    )
                target = matches[0]
            loaded_by_request = False
            if target.status in {"loaded", "sleeping"}:
                final_status = target.status
            elif target.status == "loading":
                final_status = await self._wait_for_model_status(
                    model_id, {"loaded", "sleeping"}
                )
            elif target.status == "unloaded":
                load_observation = await self.load_model(model_id)
                if (
                    load_observation.status_code != 200
                    or load_observation.error is not None
                    or not isinstance(load_observation.json_value, dict)
                    or load_observation.json_value.get("success") is not True
                ):
                    raise BackendModelUnavailable(
                        "explicit inference model load failed"
                    )
                loaded_by_request = True
                final_status = await self._wait_for_model_status(
                    model_id, {"loaded", "sleeping"}
                )
            else:
                raise BackendModelUnavailable(
                    "resolved model is not loadable"
                )
            final_inventory = await self.current_router_inventory()
            final_matches = [
                model
                for model in final_inventory.models
                if model.model_id == model_id
            ]
            if (
                len(final_matches) != 1
                or final_matches[0].status not in {"loaded", "sleeping"}
                or any(
                    model.model_id != model_id
                    and model.status in active_states
                    for model in final_inventory.models
                )
            ):
                raise BackendModelUnavailable(
                    "private router model state is inconsistent"
                )
            if not await snapshot_verifier():
                raise BackendSnapshotConflict(
                    "registry model snapshot changed during load"
                )
            identity = await self._assert_router_identity()
            router = self.router
            if router is None:
                raise BackendError("private router client is unavailable")
            completed = False
            try:
                yield InferenceBackendLease(
                    router=router,
                    router_identity=identity,
                    router_model_id=model_id,
                    loaded_by_request=loaded_by_request,
                    model_status=final_status,
                )
                completed = True
            finally:
                if retained_warm_to_restore is not None:
                    post = await self.current_router_inventory()
                    active_target = next(
                        (
                            model
                            for model in post.models
                            if model.model_id == model_id
                            and model.status in active_states
                        ),
                        None,
                    )
                    if active_target is not None:
                        await self._unload_for_temporary_switch(
                            model_id,
                            "temporary inference target unload failed",
                        )
                    await self._restore_retained_warm_model(
                        retained_warm_to_restore
                    )
                if completed or retained_warm_to_restore is not None:
                    await self._assert_router_identity()

    async def _wait_for_model_status(
        self, model_id: str, accepted: set[str]
    ) -> str:
        if self.router is None:
            raise BackendError("router HTTP client is unavailable")
        deadline = (
            time.monotonic() + self.settings.private_backend_model_timeout_seconds
        )
        last_status = "missing"
        while time.monotonic() < deadline:
            inventory = await self.router.list_models()
            if not inventory.valid:
                await asyncio.sleep(
                    self.settings.private_backend_poll_interval_seconds
                )
                continue
            match = next(
                (model for model in inventory.models if model.model_id == model_id),
                None,
            )
            last_status = match.status if match is not None else "missing"
            if last_status in accepted:
                return last_status
            if last_status == "failed":
                raise BackendError("private router model entered failed state")
            await asyncio.sleep(self.settings.private_backend_poll_interval_seconds)
        raise BackendError(
            f"private router model-state wait timed out at {last_status}"
        )

    async def probe_model_properties(
        self, model_id: str
    ) -> ModelPropertiesProbe:
        if not self._router_ready or self.router is None:
            raise BackendError("private router is not ready")
        async with self.serialized_model_operation():
            inventory = await self.current_router_inventory()
            target = next(
                (model for model in inventory.models if model.model_id == model_id),
                None,
            )
            if target is None:
                raise BackendError("capability-probe model is absent from router")
            loaded_before = target.status in {"loaded", "sleeping"}
            active_states = {"loading", "loaded", "sleeping"}
            foreign = [
                model
                for model in inventory.models
                if model.model_id != model_id
                and model.status in active_states
            ]
            retained_warm_to_restore: str | None = None
            if foreign:
                retained = self._warm_model_id
                if (
                    loaded_before
                    or retained is None
                    or retained == model_id
                    or {model.model_id for model in foreign} != {retained}
                ):
                    raise BackendError(
                        "another model is active and is not the retained warm target"
                    )
                for previous in foreign:
                    await self._unload_for_temporary_switch(
                        previous.model_id,
                        "retained warm target unload failed before registry probe",
                    )
                retained_warm_to_restore = retained
            registry_owned_load = False
            load_observation = None
            unload_observation = None
            final_status = target.status
            props_observation: RouterObservation | None = None
            probe_error: BaseException | None = None
            try:
                if not loaded_before:
                    load_observation = await self.load_model(model_id)
                    if (
                        load_observation.status_code != 200
                        or load_observation.error is not None
                        or not isinstance(load_observation.json_value, dict)
                        or load_observation.json_value.get("success") is not True
                    ):
                        raise BackendError("explicit capability-probe load failed")
                    registry_owned_load = True
                    final_status = await self._wait_for_model_status(
                        model_id, {"loaded", "sleeping"}
                    )
                props_observation = await self.router.get_props(
                    model_id, autoload=False
                )
                if (
                    props_observation.status_code != 200
                    or props_observation.error is not None
                    or not isinstance(props_observation.json_value, dict)
                ):
                    raise BackendError("private /props capability read failed")
            except BaseException as exc:
                probe_error = exc
            finally:
                cleanup_error: BaseException | None = None
                try:
                    if registry_owned_load:
                        unload_observation = await self.unload_model(model_id)
                        if (
                            unload_observation.status_code != 200
                            or unload_observation.error is not None
                            or not isinstance(unload_observation.json_value, dict)
                            or unload_observation.json_value.get("success") is not True
                        ):
                            raise BackendError(
                                "registry-owned capability-probe unload failed"
                            )
                        final_status = await self._wait_for_model_status(
                            model_id, {"unloaded"}
                        )
                except BaseException as exc:
                    cleanup_error = exc
                try:
                    if retained_warm_to_restore is not None:
                        await self._restore_retained_warm_model(
                            retained_warm_to_restore
                        )
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                if cleanup_error is not None:
                    raise BackendError(
                        "temporary capability-probe cleanup failed"
                    ) from cleanup_error
            if probe_error is not None:
                raise BackendError("model capability probe failed") from probe_error
            if props_observation is None:
                raise BackendError("model capability probe produced no props")
            return ModelPropertiesProbe(
                model_id=model_id,
                props=props_observation,
                loaded_before_probe=loaded_before,
                registry_owned_load=registry_owned_load,
                load_observation=load_observation,
                unload_observation=unload_observation,
                final_status=final_status,
            )

    async def public_state(self) -> PublicBackendState:
        if not self.settings.private_backend_enabled:
            return PublicBackendState("disabled", False, False, 0, 1, False)
        if not self._router_ready or self.router is None or self.identity is None:
            return PublicBackendState("unavailable", False, False, 0, 1, False)
        try:
            status = _require_controller_success(
                await self.controller.status(), "status"
            )
            models = await self.router.list_models()
        except (BackendError, ControllerClientError):
            return PublicBackendState("unavailable", False, False, 0, 1, False)
        process_running = self._status_matches_identity(status)
        if not process_running or not models.valid:
            return PublicBackendState(
                "unavailable", process_running, False, 0, 1, False
            )
        loaded = sum(
            model.status in {"loaded", "sleeping"}
            for model in models.models
        )
        return PublicBackendState(
            "router_ready", True, True, loaded, 1, loaded == 1
        )

    async def shutdown(self) -> None:
        if not self.settings.private_backend_enabled or self.identity is None:
            return
        unload_error: BaseException | None = None
        if self.router is not None and self._loaded_by_transaction:
            try:
                models = await self.router.list_models()
                states = {
                    model.model_id: model.status
                    for model in models.models
                    if models.valid
                }
                for model_id in sorted(self._loaded_by_transaction):
                    if states.get(model_id) in {"loaded", "loading"}:
                        await self.router.unload_model(model_id)
            except BaseException as exc:
                unload_error = exc
        await self._stop_exact_router()
        if unload_error is not None:
            raise BackendError("model unload failed during shutdown") from unload_error
