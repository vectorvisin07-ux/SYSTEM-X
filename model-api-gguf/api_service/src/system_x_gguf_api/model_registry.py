"""Serialized automatic reconciliation for router-authorized GGUF bundles."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable

from .artifact_inspector import ArtifactInspector, public_model_version_id
from .capability_inspector import (
    build_capability_evidence,
    normalize_router_model,
)
from .model_monitor import RegistryMonitor
from .registry_store import RegistryStore
from .registry_types import (
    ArtifactInspectionError,
    ArtifactPendingStability,
    ModelState,
    RouterModelEvidence,
    canonical_json,
    utc_now,
)


class ModelRegistryError(RuntimeError):
    """A bounded registry lifecycle or reconciliation failure."""


@dataclass(frozen=True)
class RegistryPublicSummary:
    registry_status: str
    registered_model_count: int
    ready_model_count: int
    rejected_artifact_count: int
    registry_generation: int
    last_reconcile_utc: str | None
    candidate_model_count: int = 0
    default_alias_model_id: str | None = None
    default_alias_ready: bool = False


class ModelRegistry:
    """Own one worker, one probe queue, one monitor, and one SQLite store."""

    def __init__(
        self,
        settings: Any,
        backend: Any,
        *,
        model_root: Path | None = None,
        database_path: Path | None = None,
        monitor_factory: Callable[..., RegistryMonitor] = RegistryMonitor,
    ) -> None:
        package_file = Path(__file__).resolve(strict=True)
        branch_root = package_file.parents[3]
        self.settings = settings
        self.backend = backend
        self.enabled = bool(getattr(settings, "registry_enabled", False))
        self.model_root = (
            model_root or branch_root / "MODEL" / "SUPERMODEL"
        ).resolve(strict=False)
        self.database_path = (
            database_path
            or branch_root / "RUNTIME" / "api" / "database" / "model_registry.sqlite3"
        )
        self.store = RegistryStore(
            self.database_path,
            int(settings.registry_database_busy_timeout_milliseconds),
        )
        self.inspector: ArtifactInspector | None = None
        self.monitor_factory = monitor_factory
        self.monitor: RegistryMonitor | None = None
        self._status = "disabled" if not self.enabled else "starting"
        self._shutdown_event = asyncio.Event()
        self._request_event = asyncio.Event()
        self._condition = asyncio.Condition()
        self._requested_sequence = 0
        self._completed_sequence = 0
        self._failed_sequence = 0
        self._pending_reasons: set[str] = set()
        self._last_error_code: str | None = None
        self._reconcile_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._probe_task: asyncio.Task[None] | None = None
        self._probe_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queued_probe_ids: set[str] = set()
        self._router_evidence: dict[str, RouterModelEvidence] = {}
        self._last_outcomes: list[dict[str, Any]] = []
        self._reconcile_run_count = 0
        self._active_reconcile_count = 0
        self._maximum_concurrent_reconciles = 0
        self._router_refresh_count = 0
        self._validated_replacement_unload_count = 0
        self._database_initialization: dict[str, Any] | None = None

    async def mark_degraded(self, reason_code: str) -> None:
        self._status = "degraded"
        self._last_error_code = reason_code[:128]

    async def mark_recovered(self, reason_family: str) -> None:
        """Clear only the transient degradation family that just recovered."""

        if (
            self._last_error_code is not None
            and self._last_error_code.startswith(f"{reason_family}:")
        ):
            self._last_error_code = None
            if self.enabled:
                self._status = "ready"

    async def request_reconcile(self, reason: str, wait: bool = False) -> None:
        if self._shutdown_event.is_set():
            if wait:
                raise ModelRegistryError("registry is shutting down")
            return
        async with self._condition:
            self._requested_sequence += 1
            target = self._requested_sequence
            self._pending_reasons.add(reason[:64])
            self._request_event.set()
            self._condition.notify_all()
        if wait:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._completed_sequence >= target
                )
                if self._failed_sequence >= target:
                    raise ModelRegistryError(
                        f"reconciliation failed: {self._last_error_code}"
                    )

    async def _worker_loop(self) -> None:
        while True:
            await self._request_event.wait()
            async with self._condition:
                if (
                    self._shutdown_event.is_set()
                    and self._completed_sequence >= self._requested_sequence
                ):
                    return
                target = self._requested_sequence
                reasons = sorted(self._pending_reasons)
                self._pending_reasons.clear()
                self._request_event.clear()
            failed = False
            try:
                async with self._reconcile_lock:
                    self._active_reconcile_count += 1
                    self._maximum_concurrent_reconciles = max(
                        self._maximum_concurrent_reconciles,
                        self._active_reconcile_count,
                    )
                    try:
                        await self._full_reconcile(reasons)
                        await self.mark_recovered("reconcile_failure")
                    finally:
                        self._active_reconcile_count -= 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failed = True
                await self.mark_degraded(
                    f"reconcile_failure:{type(exc).__name__}"
                )
            async with self._condition:
                self._completed_sequence = max(self._completed_sequence, target)
                if failed:
                    self._failed_sequence = max(self._failed_sequence, target)
                self._condition.notify_all()
                if self._requested_sequence > target:
                    self._request_event.set()
                elif (
                    self._shutdown_event.is_set()
                    and self._completed_sequence >= self._requested_sequence
                ):
                    self._request_event.set()

    def _candidate_relative_root(self, model: Any) -> str:
        supplied = model.physical_path
        if supplied is None and model.connected_paths:
            supplied = model.connected_paths[0]
        if supplied is not None:
            candidate = Path(supplied)
            if not candidate.is_absolute():
                candidate = self.model_root / candidate
            try:
                relative = candidate.resolve(strict=True).relative_to(self.model_root)
                if len(relative.parts) == 1 and candidate.is_file():
                    return relative.as_posix()
                return relative.parts[0]
            except (OSError, ValueError):
                pass
        direct = self.model_root / f"{model.model_id}.gguf"
        directory = self.model_root / model.model_id
        if os.path.lexists(direct):
            return direct.name
        if os.path.lexists(directory):
            return directory.name
        return str(model.model_id)[:256]

    def _physical_units_sync(self) -> set[str]:
        units: set[str] = set()
        for entry in os.scandir(self.model_root):
            path = Path(entry.path)
            if entry.is_symlink():
                if entry.name.lower().endswith(".gguf"):
                    units.add(entry.name)
                continue
            if entry.is_file(follow_symlinks=False):
                if entry.name.lower().endswith(".gguf"):
                    units.add(entry.name)
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            found = False
            for current, directories, files in os.walk(path, followlinks=False):
                directories[:] = sorted(directories)
                for name in files:
                    if name.lower().endswith(".gguf"):
                        found = True
                        break
                if found:
                    break
            if found:
                units.add(entry.name)
        return units

    async def _full_reconcile(self, reasons: list[str]) -> None:
        self._reconcile_run_count += 1
        physical_units = await asyncio.to_thread(self._physical_units_sync)
        seen_roots = set(physical_units)
        prior_present_roots = await self.store.present_location_roots()
        removed_roots = prior_present_roots - physical_units
        outcomes: list[dict[str, Any]] = []
        locally_validated: dict[str, Any] = {}
        changed_roots: set[str] = set()
        if self.inspector is None:
            raise ModelRegistryError("registry inspector is not initialized")
        for relative_root in sorted(physical_units):
            try:
                cache = await self.store.location_hash_cache(relative_root)
                previous = await self.store.location_record(relative_root)
                bundle = await asyncio.to_thread(
                    self.inspector.inspect_location,
                    relative_root,
                    cache,
                    self._shutdown_event,
                )
                if bundle.relative_root != relative_root:
                    raise ModelRegistryError(
                        "local inspection changed the logical location identity"
                    )
                locally_validated[relative_root] = bundle
                changed = bool(
                    previous is None
                    or previous["present"] != 1
                    or previous["current_bundle_id"] != bundle.bundle_id
                    or previous["physical_manifest_json"]
                    != canonical_json(bundle.physical_manifest)
                )
                if changed:
                    changed_roots.add(relative_root)
                outcomes.append(
                    {
                        "relative_root": relative_root,
                        "outcome": "LOCALLY_VALIDATED",
                        "bundle_file_count": bundle.file_count,
                        "hash_reused_count": bundle.reused_hash_count,
                        "changed": changed,
                    }
                )
            except ArtifactPendingStability as exc:
                outcomes.append(
                    {
                        "relative_root": relative_root,
                        "outcome": ModelState.PENDING_STABILITY.value,
                        "reason_code": exc.reason_code,
                    }
                )
            except ArtifactInspectionError as exc:
                await self.store.record_rejection(
                    relative_root, exc.reason_code, exc.detail
                )
                outcomes.append(
                    {
                        "relative_root": relative_root,
                        "outcome": "INVALID",
                        "reason_code": exc.reason_code,
                    }
                )

        current = await self.backend.current_router_inventory()
        current_router_models = [
            model for model in current.models if model.source == "models_dir"
        ]
        replacement_model_ids = {
            model.model_id
            for model in current_router_models
            if self._candidate_relative_root(model)
            in changed_roots.union(removed_roots)
        }
        if changed_roots or removed_roots:
            refreshed = await self.backend.refresh_validated_model_inventory(
                replacement_model_ids
            )
            inventory = refreshed.inventory
            self._router_refresh_count += 1
            self._validated_replacement_unload_count += len(
                refreshed.unloaded_model_ids
            )
        else:
            inventory = current
        router_models = [
            model for model in inventory.models if model.source == "models_dir"
        ]
        if len({model.model_id for model in router_models}) != len(router_models):
            raise ModelRegistryError("router inventory contains duplicate model IDs")
        models_by_root: dict[str, list[Any]] = {}
        for model in router_models:
            models_by_root.setdefault(
                self._candidate_relative_root(model), []
            ).append(model)
        for relative_root, bundle in sorted(locally_validated.items()):
            matches = models_by_root.get(relative_root, [])
            if len(matches) != 1:
                await self.store.record_rejection(
                    relative_root,
                    "ROUTER_UNRECOGNIZED",
                    {
                        "source": "post_validation_router_correlation",
                        "match_count": len(matches),
                    },
                )
                outcomes.append(
                    {
                        "relative_root": relative_root,
                        "outcome": "INVALID",
                        "reason_code": "ROUTER_UNRECOGNIZED",
                    }
                )
                continue
            router = normalize_router_model(matches[0])
            self._router_evidence[router.router_model_id] = router
            logical_name = (
                Path(relative_root).stem
                if relative_root.lower().endswith(".gguf")
                else relative_root
            )
            existing_model_version_id = (
                await self.store.model_version_for_location_bundle(
                    relative_root, bundle.bundle_id
                )
            )
            model_version_id = (
                existing_model_version_id
                or public_model_version_id(
                    logical_name,
                    bundle.bundle_sha256,
                    location_identity=relative_root,
                )
            )
            registration = await self.store.register_bundle(
                bundle, router, model_version_id
            )
            outcomes.append(
                {
                    "router_model_id": router.router_model_id,
                    "relative_root": relative_root,
                    "outcome": "registered",
                    "model_version_id": model_version_id,
                    "bundle_file_count": bundle.file_count,
                    "hash_reused_count": bundle.reused_hash_count,
                    "changed": registration["changed"],
                }
            )
        await self.store.mark_missing(seen_roots)
        observed = utc_now()
        await self.store.set_last_reconcile(observed)
        self._last_outcomes = outcomes
        if self._probe_task is not None:
            await self._enqueue_pending_capabilities()

    async def _enqueue_pending_capabilities(self) -> None:
        for row in await self.store.models_needing_capability():
            model_version_id = str(row["model_version_id"])
            if model_version_id in self._queued_probe_ids:
                continue
            self._queued_probe_ids.add(model_version_id)
            await self._probe_queue.put(row)

    def _router_evidence_for_probe(
        self, row: dict[str, Any]
    ) -> RouterModelEvidence:
        router_model_id = str(row["router_model_id"])
        current = self._router_evidence.get(router_model_id)
        if current is not None:
            return current
        payload = json.loads(str(row["router_metadata_json"]))
        return RouterModelEvidence(
            router_model_id=router_model_id,
            router_source=str(payload["router_source"]),
            router_status=str(payload["router_status"]),
            display_name=router_model_id,
            physical_path=None,
            connected_paths=(),
            input_modalities=tuple(payload.get("input_modalities", [])),
            output_modalities=tuple(payload.get("output_modalities", [])),
            observed_utc=utc_now(),
            metadata_json=str(row["router_metadata_json"]),
            metadata_sha256="0" * 64,
        )

    async def _probe_loop(self) -> None:
        while True:
            row = await self._probe_queue.get()
            if row.get("_stop") is True:
                self._probe_queue.task_done()
                return
            model_version_id = str(row["model_version_id"])
            transitioned = False
            try:
                await self.store.transition_state(
                    model_version_id,
                    ModelState.PROBING,
                    "capability_probe_started",
                )
                transitioned = True
                probe = await self.backend.probe_model_properties(
                    str(row["router_model_id"])
                )
                capability = build_capability_evidence(
                    model_version_id,
                    str(row["bundle_id"]),
                    self._router_evidence_for_probe(row),
                    probe.props.json_value,
                )
                await self.store.store_capability_ready(
                    capability, self.settings.registry_default_alias
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if transitioned:
                    try:
                        await self.store.transition_state(
                            model_version_id,
                            ModelState.UNAVAILABLE,
                            "capability_probe_unavailable",
                            {"error_type": type(exc).__name__},
                        )
                    except Exception:
                        pass
                await self.mark_degraded(
                    f"capability_probe_failure:{type(exc).__name__}"
                )
            finally:
                self._queued_probe_ids.discard(model_version_id)
                self._probe_queue.task_done()

    async def startup(self) -> dict[str, Any]:
        if not self.enabled:
            return {"registry_status": "disabled"}
        self._status = "starting"
        self.inspector = ArtifactInspector(
            self.model_root,
            int(self.settings.registry_stability_samples),
            float(self.settings.registry_stability_interval_seconds),
        )
        self._database_initialization = await self.store.initialize()
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="system-x-registry-reconcile-worker"
        )
        await self.request_reconcile("startup_initial", True)
        self.monitor = self.monitor_factory(
            model_root=self.model_root,
            debounce_milliseconds=self.settings.registry_watch_debounce_milliseconds,
            reconcile_interval_seconds=self.settings.registry_reconcile_interval_seconds,
            request_reconcile=self.request_reconcile,
            mark_degraded=self.mark_degraded,
            mark_recovered=self.mark_recovered,
        )
        await self.monitor.start_watcher()
        await self.request_reconcile("startup_post_watcher", True)
        await self.monitor.start_periodic()
        self._probe_task = asyncio.create_task(
            self._probe_loop(), name="system-x-registry-capability-probe"
        )
        await self._enqueue_pending_capabilities()
        if self._status != "degraded":
            self._status = "ready"
        return {
            "registry_status": self._status,
            "database": self._database_initialization,
            "reconcile_run_count": self._reconcile_run_count,
            "maximum_concurrent_reconciles": self._maximum_concurrent_reconciles,
            "router_refresh_count": self._router_refresh_count,
            "validated_replacement_unload_count": (
                self._validated_replacement_unload_count
            ),
            "summary": await self.store.summary(
                str(self.settings.registry_default_alias)
            ),
        }

    async def public_summary(self) -> RegistryPublicSummary:
        if not self.enabled:
            return RegistryPublicSummary("disabled", 0, 0, 0, 0, None)
        if not self.store._initialized:
            return RegistryPublicSummary(self._status, 0, 0, 0, 0, None)
        summary = await self.store.summary(
            str(self.settings.registry_default_alias)
        )
        return RegistryPublicSummary(self._status, **summary)

    async def public_model_rows(self) -> dict[str, Any]:
        if not self.enabled or not self.store._initialized:
            raise ModelRegistryError("registry catalogue is unavailable")
        return await self.store.public_model_rows()

    async def resolve_public_model(self, reference: str) -> dict[str, Any]:
        if not self.enabled or not self.store._initialized:
            raise ModelRegistryError("registry resolver is unavailable")
        return await self.store.resolve_public_model(reference)

    async def public_model_snapshot_matches(
        self,
        reference: str,
        expected: dict[str, Any],
    ) -> bool:
        if not self.enabled or not self.store._initialized:
            return False
        return await self.store.public_model_snapshot_matches(reference, expected)

    async def record_runtime_capability(
        self,
        model_version_id: str,
        capability_name: str,
        request_id: str,
        service_transaction_id: str,
        router_transaction_id: str,
        observed_protocol_surfaces: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not self.enabled or not self.store._initialized:
            raise ModelRegistryError("registry capability evidence is unavailable")
        try:
            return await self.store.record_runtime_capability(
                model_version_id,
                capability_name,
                request_id,
                service_transaction_id,
                router_transaction_id,
                observed_protocol_surfaces,
            )
        except Exception as exc:
            raise ModelRegistryError(
                "runtime capability evidence could not be persisted"
            ) from exc

    async def shutdown(self) -> dict[str, Any]:
        if not self.enabled:
            return {"registry_status": "disabled"}
        self._shutdown_event.set()
        monitor_result = (
            await self.monitor.shutdown()
            if self.monitor is not None
            else {
                "watcher_task_done": True,
                "periodic_task_done": True,
            }
        )
        if self._probe_task is not None:
            try:
                await asyncio.wait_for(self._probe_queue.join(), timeout=5.0)
            except TimeoutError:
                self._probe_task.cancel()
            if not self._probe_task.done():
                await self._probe_queue.put({"_stop": True})
            await asyncio.gather(self._probe_task, return_exceptions=True)
            self._probe_task = None
        self._request_event.set()
        if self._worker_task is not None:
            await asyncio.wait_for(self._worker_task, timeout=5.0)
            self._worker_task = None
        checkpoint = await self.store.checkpoint_and_close()
        self._status = "disabled"
        return {
            "monitor": monitor_result,
            "reconcile_worker_done": True,
            "probe_worker_done": True,
            "checkpoint": checkpoint,
            "reconcile_run_count": self._reconcile_run_count,
            "maximum_concurrent_reconciles": self._maximum_concurrent_reconciles,
            "router_refresh_count": self._router_refresh_count,
            "validated_replacement_unload_count": (
                self._validated_replacement_unload_count
            ),
        }
