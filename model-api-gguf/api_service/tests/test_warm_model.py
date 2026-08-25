"""Focused production-code tests for always-warm readiness and promotion."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pydantic import ValidationError

from system_x_gguf_api.application import create_application
from system_x_gguf_api.backend import (
    BackendCoordinator,
    BackendError,
    BackendModelUnavailable,
    ModelChildIdentity,
    PublicBackendState,
    RouterIdentity,
    WarmBackendObservation,
    _exact_owned_model_child,
)
from system_x_gguf_api.controller_client import ControllerResult
from system_x_gguf_api.errors import SystemXError
from system_x_gguf_api.model_catalogue import ModelCatalogue, ModelSnapshot
from system_x_gguf_api.model_registry import (
    ModelRegistry,
    RegistryPublicSummary,
)
from system_x_gguf_api.registry_types import (
    ArtifactBundleEvidence,
    ArtifactFileEvidence,
    BundleKind,
    CapabilityEvidence,
    ModelState,
    PhysicalIdentity,
    RoleHint,
    RouterModelEvidence,
    canonical_json,
    utc_now,
)
from system_x_gguf_api.router_client import (
    RouterModel,
    RouterModelList,
    RouterObservation,
)
from system_x_gguf_api.schemas import HealthResponse
from system_x_gguf_api.settings import ServiceSettings
from system_x_gguf_api.warm_model import (
    WarmIdentity,
    WarmModelCoordinator,
    WarmStatus,
    full_readiness,
)


def settings(**updates: object) -> SimpleNamespace:
    values = {
        "registry_enabled": True,
        "registry_default_alias": "default",
        "registry_database_busy_timeout_milliseconds": 5000,
        "registry_stability_samples": 2,
        "registry_stability_interval_seconds": 0.1,
        "private_backend_enabled": True,
        "private_backend_poll_interval_seconds": 0.05,
        "private_backend_model_timeout_seconds": 1.0,
        "private_backend_start_timeout_seconds": 1.0,
        "private_backend_inference_timeout_seconds": 1.0,
        "private_backend_host": "127.0.0.1",
        "private_backend_port": 32001,
        "private_backend_models_max": 1,
        "startup_model_policy": "always_warm",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def snapshot(
    public_id: str = "sx-ready-a",
    bundle_id: str = "bundle-a",
    router_id: str = "router-a",
    generation: int = 1,
    state: str = "READY",
    capability: str = "a" * 64,
) -> ModelSnapshot:
    return ModelSnapshot(
        requested_reference="default",
        registry_generation=generation,
        public_model_id=public_id,
        bundle_id=bundle_id,
        router_model_id=router_id,
        registration_state=state,
        created_utc="2026-01-02T03:04:05.000006Z",
        aliases=("default",),
        capability_manifest_identity=capability,
        context_bound=4096,
        chat_template_present=True,
        tool_calling_state="not_tested",
        structured_output_state="not_tested",
        parallel_tool_calling_state="not_tested",
        streaming_state="not_tested",
    )


def backend_observation(
    router_model_id: str,
    *,
    load: bool,
    unload: bool = False,
) -> WarmBackendObservation:
    return WarmBackendObservation(
        router_identity=RouterIdentity(
            "router-transaction", 100, 100, 100, "router-start"
        ),
        router_model_id=router_model_id,
        model_status="loaded",
        load_performed=load,
        unload_performed=unload,
        private_health_ready=True,
        model_child=ModelChildIdentity(
            101, "procfs-start-ticks:101", 100, 100, 100
        ),
    )


class FakeRegistry:
    def __init__(
        self,
        status: str = "ready",
        *,
        registered: int = 1,
        ready: int = 1,
        candidate: int = 0,
        default_alias_model_id: str | None = None,
        default_alias_ready: bool = False,
    ) -> None:
        self.status = status
        self.registered = registered
        self.ready = ready
        self.candidate = candidate
        self.default_alias_model_id = default_alias_model_id
        self.default_alias_ready = default_alias_ready

    async def public_summary(self) -> RegistryPublicSummary:
        return RegistryPublicSummary(
            self.status,
            self.registered,
            self.ready,
            0,
            1,
            None,
            self.candidate,
            self.default_alias_model_id,
            self.default_alias_ready,
        )


class FakeCatalogue:
    def __init__(
        self,
        current: ModelSnapshot | SystemXError,
    ) -> None:
        self.current = current

    async def resolve(self, _reference: str) -> ModelSnapshot:
        if isinstance(self.current, SystemXError):
            raise self.current
        return self.current

    async def verify(self, value: ModelSnapshot) -> bool:
        return self.current is value


class FakeWarmBackend:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[str, str | None]] = []
        self.verify_calls: list[str] = []
        self.release_calls: list[str] = []

    async def ensure_warm_model(
        self, model_id: str, prior: str | None, verifier
    ) -> WarmBackendObservation:
        self.ensure_calls.append((model_id, prior))
        if not await verifier():
            raise RuntimeError("snapshot changed")
        return backend_observation(
            model_id,
            load=True,
            unload=prior is not None and prior != model_id,
        )

    async def verify_warm_model(
        self, model_id: str, verifier
    ) -> WarmBackendObservation:
        self.verify_calls.append(model_id)
        if not await verifier():
            raise RuntimeError("snapshot changed")
        return backend_observation(model_id, load=False)

    async def release_warm_intent(self, model_id: str) -> None:
        self.release_calls.append(model_id)


class WarmCoordinatorMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_ready_target_waits_or_reports_first_candidate(self) -> None:
        no_ready = SystemXError(
            503,
            "system_x_no_ready_model",
            "No READY model is currently available",
            retryable=True,
        )
        backend = FakeWarmBackend()
        waiting = WarmModelCoordinator(
            settings(),
            FakeCatalogue(no_ready),
            FakeRegistry(registered=0, ready=0),
            backend,
        )
        waiting_status = await waiting.observe_once()
        self.assertEqual(
            waiting_status.service_readiness_state, "WAITING_FOR_MODEL"
        )
        self.assertEqual(waiting_status.reason_code, "NO_READY_MODEL")
        self.assertIsNone(waiting_status.identity)

        candidate = WarmModelCoordinator(
            settings(),
            FakeCatalogue(no_ready),
            FakeRegistry(ready=0, candidate=1),
            backend,
        )
        candidate_status = await candidate.observe_once()
        self.assertEqual(
            candidate_status.service_readiness_state,
            "MODEL_CANDIDATE_LOADING",
        )
        self.assertEqual(backend.ensure_calls, [])

        expected_registry = FakeRegistry(
            default_alias_model_id="sx-ready-a",
            default_alias_ready=True,
        )
        expected_catalogue = FakeCatalogue(snapshot())
        expected = WarmModelCoordinator(
            settings(), expected_catalogue, expected_registry, backend
        )
        ready_status = await expected.observe_once()
        self.assertEqual(ready_status.service_readiness_state, "READY")
        expected_catalogue.current = no_ready
        expected_registry.ready = 0
        expected_registry.default_alias_model_id = None
        expected_registry.default_alias_ready = False
        lost_status = await expected.observe_once()
        self.assertEqual(lost_status.service_readiness_state, "DEGRADED")
        self.assertEqual(
            lost_status.reason_code, "expected_default_target_lost"
        )
        self.assertIsNotNone(lost_status.identity)

    async def test_policy_alias_state_and_capability_matrix(self) -> None:
        accepted = ServiceSettings(
            public_port=31001,
            private_backend_enabled=True,
            private_backend_port=31002,
            registry_enabled=True,
            startup_model_policy="always_warm",
        )
        self.assertEqual(accepted.startup_model_policy, "always_warm")
        with self.assertRaises(ValidationError):
            ServiceSettings.model_validate(
                {
                    **accepted.model_dump(),
                    "startup_model_policy": "lazy",
                }
            )

        unavailable = (
            "REGISTERED",
            "PROBING",
            "UNAVAILABLE",
            "REPLACED",
            "REMOVED",
        )
        for state in unavailable:
            with self.subTest(state=state):
                backend = FakeWarmBackend()
                coordinator = WarmModelCoordinator(
                    settings(),
                    FakeCatalogue(snapshot(state=state)),
                    FakeRegistry(),
                    backend,
                )
                result = await coordinator.observe_once()
                self.assertEqual(
                    result.service_readiness_state, "DEGRADED"
                )
                self.assertEqual(backend.ensure_calls, [])

        backend = FakeWarmBackend()
        missing_capability = WarmModelCoordinator(
            settings(),
            FakeCatalogue(snapshot(capability="")),
            FakeRegistry(),
            backend,
        )
        result = await missing_capability.observe_once()
        self.assertEqual(result.service_readiness_state, "DEGRADED")
        self.assertEqual(backend.ensure_calls, [])

        for error in (
            SystemXError(
                404,
                "system_x_model_not_found",
                "not found",
            ),
            SystemXError(
                503,
                "system_x_model_unavailable",
                "unavailable",
            ),
            SystemXError(
                503,
                "system_x_backend_unavailable",
                "registry down",
            ),
        ):
            backend = FakeWarmBackend()
            coordinator = WarmModelCoordinator(
                settings(),
                FakeCatalogue(error),
                FakeRegistry(),
                backend,
            )
            result = await coordinator.observe_once()
            self.assertEqual(
                result.service_readiness_state, "DEGRADED"
            )
            self.assertEqual(backend.ensure_calls, [])

    async def test_exact_load_adoption_repeat_and_shutdown(self) -> None:
        target = snapshot()
        catalogue = FakeCatalogue(target)
        backend = FakeWarmBackend()
        coordinator = WarmModelCoordinator(
            settings(), catalogue, FakeRegistry(), backend
        )
        first = await coordinator.startup()
        self.assertEqual(first.service_readiness_state, "READY")
        self.assertEqual(backend.ensure_calls, [("router-a", None)])
        second = await coordinator.observe_once()
        self.assertEqual(second.service_readiness_state, "READY")
        self.assertEqual(backend.ensure_calls, [("router-a", None)])
        self.assertEqual(backend.verify_calls, ["router-a"])
        stopped = await coordinator.shutdown()
        self.assertEqual(stopped.service_readiness_state, "STOPPED")
        self.assertEqual(backend.release_calls, ["router-a"])
        self.assertIsNone(coordinator._observer_task)


class FakeController:
    def __init__(self, identity: RouterIdentity) -> None:
        self.identity = identity

    async def status(self) -> ControllerResult:
        return ControllerResult(
            operation="status",
            ok=True,
            reason_code="OK",
            message="fixture",
            data={
                "active": True,
                "active_state_consistent": True,
                "transaction_id": self.identity.transaction_id,
                "pid": self.identity.pid,
                "pgid": self.identity.pgid,
                "sid": self.identity.sid,
                "launch_mode": "router",
            },
            stderr="",
            exit_status=0,
        )


class FakeRouter:
    def __init__(self, states: dict[str, str]) -> None:
        self.states = states
        self.loads: list[str] = []
        self.unloads: list[str] = []
        self.health_ready = True
        self.recover_health_on_load = False

    def _model(self, model_id: str, state: str) -> RouterModel:
        return RouterModel(
            model_id=model_id,
            status=state,
            upstream_status=state,
            source="models_dir",
            physical_path=None,
            connected_paths=(),
            input_modalities=("text",),
            output_modalities=("text",),
            raw={},
        )

    async def list_models(self, reload: bool = False) -> RouterModelList:
        del reload
        observation = RouterObservation(200, "{}", {}, None)
        return RouterModelList(
            observation,
            tuple(
                self._model(model_id, state)
                for model_id, state in sorted(self.states.items())
            ),
            True,
        )

    async def load_model(self, model_id: str) -> RouterObservation:
        self.loads.append(model_id)
        self.states[model_id] = "loaded"
        if self.recover_health_on_load:
            self.health_ready = True
        return RouterObservation(
            200, '{"success":true}', {"success": True}, None
        )

    async def unload_model(self, model_id: str) -> RouterObservation:
        self.unloads.append(model_id)
        self.states[model_id] = "unloaded"
        return RouterObservation(
            200, '{"success":true}', {"success": True}, None
        )

    async def health(self) -> RouterObservation:
        if not self.health_ready:
            return RouterObservation(503, '{"status":"loading"}', {}, None)
        return RouterObservation(
            200, '{"status":"ok"}', {"status": "ok"}, None
        )

    async def get_props(
        self, model_id: str, autoload: bool = False
    ) -> RouterObservation:
        del autoload
        if not self.health_ready or self.states.get(model_id) not in {
            "loaded",
            "sleeping",
        }:
            return RouterObservation(503, "{}", {}, None)
        return RouterObservation(200, '{"model":"ok"}', {"model": "ok"}, None)


class BackendWarmBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def make_backend(
        self, states: dict[str, str]
    ) -> tuple[BackendCoordinator, FakeRouter]:
        identity = RouterIdentity("router-tx", 100, 100, 100, "start")
        child = ModelChildIdentity(
            101, "procfs-start-ticks:101", 100, 100, 100
        )
        backend = BackendCoordinator(
            settings(),
            process_group_observer=lambda _identity: (child,),
        )
        router = FakeRouter(states)
        backend.router = router
        backend.identity = identity
        backend.controller = FakeController(identity)
        backend._router_ready = True
        return backend, router

    async def test_load_adopt_switch_health_and_retention(self) -> None:
        backend, router = self.make_backend(
            {"router-a": "unloaded", "router-b": "unloaded"}
        )

        async def valid() -> bool:
            return True

        first = await backend.ensure_warm_model(
            "router-a", None, valid
        )
        self.assertTrue(first.load_performed)
        self.assertEqual(router.loads, ["router-a"])
        verified = await backend.verify_warm_model("router-a", valid)
        self.assertFalse(verified.load_performed)
        self.assertEqual(router.loads, ["router-a"])
        switched = await backend.ensure_warm_model(
            "router-b", "router-a", valid
        )
        self.assertTrue(switched.load_performed)
        self.assertTrue(switched.unload_performed)
        self.assertEqual(router.unloads, ["router-a"])
        self.assertEqual(router.loads, ["router-a", "router-b"])
        await asyncio.sleep(0)
        self.assertEqual(router.states["router-b"], "loaded")
        router.health_ready = False
        with self.assertRaises(BackendModelUnavailable):
            await backend.verify_warm_model("router-b", valid)

    async def test_loaded_model_child_loss_forces_exact_reload(self) -> None:
        backend, router = self.make_backend({"router-a": "loaded"})
        router.health_ready = False
        router.recover_health_on_load = True

        async def valid() -> bool:
            return True

        observation = await backend.ensure_warm_model(
            "router-a", "router-a", valid
        )
        self.assertTrue(observation.unload_performed)
        self.assertTrue(observation.load_performed)
        self.assertEqual(router.unloads, ["router-a"])
        self.assertEqual(router.loads, ["router-a"])
        self.assertEqual(router.states["router-a"], "loaded")

    async def test_model_child_identity_fail_closed(self) -> None:
        identity = RouterIdentity("router-tx", 100, 100, 100, "start")
        accepted = _exact_owned_model_child(
            identity,
            (
                ModelChildIdentity(
                    101, "start-101", 100, 100, 100
                ),
            ),
        )
        self.assertEqual(accepted.pid, 101)
        for members in (
            (),
            (
                ModelChildIdentity(
                    101, "start-101", 999, 100, 100
                ),
            ),
            (
                ModelChildIdentity(
                    101, "start-101", 100, 999, 100
                ),
            ),
            (
                ModelChildIdentity(
                    101, "start-101", 100, 100, 999
                ),
            ),
            (
                ModelChildIdentity(101, "one", 100, 100, 100),
                ModelChildIdentity(102, "two", 100, 100, 100),
            ),
        ):
            with self.subTest(members=members):
                with self.assertRaises(BackendModelUnavailable):
                    _exact_owned_model_child(identity, members)

    async def test_temporary_probe_restores_retained_warm_model(self) -> None:
        backend, router = self.make_backend(
            {"router-a": "loaded", "router-b": "unloaded"}
        )
        backend._warm_model_id = "router-a"

        probe = await backend.probe_model_properties("router-b")

        self.assertTrue(probe.registry_owned_load)
        self.assertEqual(router.unloads, ["router-a", "router-b"])
        self.assertEqual(router.loads, ["router-b", "router-a"])
        self.assertEqual(router.states["router-a"], "loaded")
        self.assertEqual(router.states["router-b"], "unloaded")
        self.assertEqual(backend._warm_model_id, "router-a")

    async def test_temporary_inference_restores_retained_warm_model(
        self,
    ) -> None:
        backend, router = self.make_backend(
            {"router-a": "loaded", "router-b": "unloaded"}
        )
        backend._warm_model_id = "router-a"

        async def valid() -> bool:
            return True

        async with backend.inference_session("router-b", valid) as lease:
            self.assertEqual(lease.router_model_id, "router-b")
            self.assertEqual(router.states["router-a"], "unloaded")
            self.assertEqual(router.states["router-b"], "loaded")

        self.assertEqual(router.unloads, ["router-a", "router-b"])
        self.assertEqual(router.loads, ["router-b", "router-a"])
        self.assertEqual(router.states["router-a"], "loaded")
        self.assertEqual(router.states["router-b"], "unloaded")
        self.assertEqual(backend._warm_model_id, "router-a")

    async def test_startup_reconciles_owned_stale_inactive_prestate(
        self,
    ) -> None:
        backend, _router = self.make_backend({})
        calls: list[str] = []

        class StaleController:
            async def status(self) -> ControllerResult:
                calls.append("status")
                if calls.count("status") == 1:
                    return ControllerResult(
                        operation="status",
                        ok=False,
                        reason_code="ACTIVE_STATE_INCONSISTENT",
                        message="stale fixture",
                        data={"active": False},
                        stderr="",
                        exit_status=3,
                    )
                return ControllerResult(
                    operation="status",
                    ok=True,
                    reason_code="OK",
                    message="reconciled fixture",
                    data={
                        "active": False,
                        "active_state_consistent": True,
                    },
                    stderr="",
                    exit_status=0,
                )

            async def reconcile(self) -> ControllerResult:
                calls.append("reconcile")
                return ControllerResult(
                    operation="reconcile",
                    ok=True,
                    reason_code="OK",
                    message="stale records removed",
                    data={
                        "active": False,
                        "active_state_consistent": True,
                        "reconciled": True,
                    },
                    stderr="",
                    exit_status=0,
                )

        backend.controller = StaleController()
        result = await backend._startup_prestate()
        self.assertFalse(result["active"])
        self.assertTrue(result["active_state_consistent"])
        self.assertEqual(calls, ["status", "reconcile"])

    async def test_startup_reconciles_stale_status_without_active_records(
        self,
    ) -> None:
        backend, _router = self.make_backend({})
        calls: list[str] = []

        class StaleStatusController:
            async def status(self) -> ControllerResult:
                calls.append("status")
                if calls.count("status") == 1:
                    return ControllerResult(
                        operation="status",
                        ok=True,
                        reason_code="OK",
                        message="stale STARTED status fixture",
                        data={
                            "active": False,
                            "active_state_consistent": False,
                            "lifecycle_state": "STARTED",
                        },
                        stderr="",
                        exit_status=0,
                    )
                return ControllerResult(
                    operation="status",
                    ok=True,
                    reason_code="OK",
                    message="reconciled fixture",
                    data={
                        "active": False,
                        "active_state_consistent": True,
                    },
                    stderr="",
                    exit_status=0,
                )

            async def reconcile(self) -> ControllerResult:
                calls.append("reconcile")
                return ControllerResult(
                    operation="reconcile",
                    ok=True,
                    reason_code="OK",
                    message="stale status reconciled",
                    data={
                        "active": False,
                        "active_state_consistent": True,
                        "reconciled": False,
                    },
                    stderr="",
                    exit_status=0,
                )

        backend.controller = StaleStatusController()
        result = await backend._startup_prestate()
        self.assertFalse(result["active"])
        self.assertTrue(result["active_state_consistent"])
        self.assertEqual(calls, ["status", "reconcile"])

    async def test_startup_stale_reconcile_remains_ownership_safe(
        self,
    ) -> None:
        backend, _router = self.make_backend({})
        calls: list[str] = []

        class UncertainController:
            async def status(self) -> ControllerResult:
                calls.append("status")
                return ControllerResult(
                    operation="status",
                    ok=False,
                    reason_code="ACTIVE_STATE_INCONSISTENT",
                    message="stale fixture",
                    data={},
                    stderr="",
                    exit_status=3,
                )

            async def reconcile(self) -> ControllerResult:
                calls.append("reconcile")
                return ControllerResult(
                    operation="reconcile",
                    ok=False,
                    reason_code="OWNERSHIP_UNCERTAIN",
                    message="live identity ambiguous",
                    data={},
                    stderr="",
                    exit_status=3,
                )

        backend.controller = UncertainController()
        with self.assertRaisesRegex(
            BackendError,
            "branch controller reconcile failed: OWNERSHIP_UNCERTAIN",
        ):
            await backend._startup_prestate()
        self.assertEqual(calls, ["status", "reconcile"])

    async def test_startup_does_not_reconcile_foreign_endpoint(
        self,
    ) -> None:
        backend, _router = self.make_backend({})
        calls: list[str] = []

        class ForeignController:
            async def status(self) -> ControllerResult:
                calls.append("status")
                return ControllerResult(
                    operation="status",
                    ok=False,
                    reason_code="ENDPOINT_CONFLICT",
                    message="foreign listener",
                    data={},
                    stderr="",
                    exit_status=3,
                )

            async def reconcile(self) -> ControllerResult:
                calls.append("reconcile")
                raise AssertionError("foreign endpoint must not reconcile")

        backend.controller = ForeignController()
        with self.assertRaisesRegex(
            BackendError,
            "branch controller status failed: ENDPOINT_CONFLICT",
        ):
            await backend._startup_prestate()
        self.assertEqual(calls, ["status"])


class ProductionRegistryPromotionTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "models"
        self.database_root = self.root / "database"
        self.model_root.mkdir(mode=0o700)
        self.database_root.mkdir(mode=0o700)
        self.backend = FakeWarmBackend()
        self.registry = ModelRegistry(
            settings(),
            object(),
            model_root=self.model_root,
            database_path=self.database_root / "registry.sqlite3",
        )
        await self.registry.store.initialize()
        self.registry._status = "ready"
        self.catalogue = ModelCatalogue(
            self.registry, object(), object()
        )
        self.coordinator = WarmModelCoordinator(
            settings(),
            self.catalogue,
            self.registry,
            self.backend,
        )

    async def asyncTearDown(self) -> None:
        if self.registry.store._initialized:
            await self.registry.store.checkpoint_and_close()
        self.temporary.cleanup()

    @staticmethod
    def capability(public_id: str, marker: str) -> CapabilityEvidence:
        manifest = {
            "context": {"default_n_ctx": 4096},
            "chat_template": {"present": True},
            "derived_template_capabilities": {
                "tool_calling": "unknown",
                "parallel_tool_calling": "unknown",
            },
            "runtime_generation_tests": {
                "tool_calling": "NOT_TESTED",
                "structured_output": "NOT_TESTED",
                "parallel_tool_calling": "NOT_TESTED",
                "streaming": "NOT_TESTED",
            },
            "evidence_layers": {"runtime_generation": "NOT_TESTED"},
            "marker": marker,
        }
        encoded = canonical_json(manifest)
        return CapabilityEvidence(
            model_version_id=public_id,
            manifest_json=encoded,
            manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            props_payload_sha256=None,
            observed_utc=utc_now(),
        )

    def bundle(
        self,
        public_id: str,
        bundle_marker: str,
        router_id: str,
    ) -> tuple[ArtifactBundleEvidence, RouterModelEvidence, str]:
        bundle_sha = hashlib.sha256(bundle_marker.encode()).hexdigest()
        bundle_id = f"bundle-{bundle_sha}"
        physical = PhysicalIdentity(1, 1, 0o100600, 24, 1)
        item = ArtifactFileEvidence(
            relative_path="model.gguf",
            file_sha256=bundle_sha,
            size_bytes=24,
            gguf_version=3,
            tensor_count=1,
            metadata_kv_count=1,
            role_hint=RoleHint.PRIMARY,
            physical_identity=physical,
        )
        bundle = ArtifactBundleEvidence(
            bundle_root=self.model_root / "slot",
            relative_root="slot",
            bundle_id=bundle_id,
            bundle_sha256=bundle_sha,
            bundle_kind=BundleKind.SINGLE_FILE,
            size_bytes=24,
            files=(item,),
            physical_manifest=(
                {
                    "relative_path": "model.gguf",
                    **physical.as_dict(),
                },
            ),
        )
        metadata_json = canonical_json(
            {
                "router_source": "models_dir",
                "router_status": "unloaded",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            }
        )
        router = RouterModelEvidence(
            router_model_id=router_id,
            router_source="models_dir",
            router_status="unloaded",
            display_name=router_id,
            physical_path=None,
            connected_paths=(),
            input_modalities=("text",),
            output_modalities=("text",),
            observed_utc=utc_now(),
            metadata_json=metadata_json,
            metadata_sha256=hashlib.sha256(
                metadata_json.encode()
            ).hexdigest(),
        )
        return bundle, router, public_id

    async def register_ready(
        self, public_id: str, marker: str, router_id: str
    ) -> None:
        bundle, router, model_id = self.bundle(
            public_id, marker, router_id
        )
        await self.registry.store.register_bundle(
            bundle, router, model_id
        )
        await self.registry.store.transition_state(
            model_id, ModelState.PROBING, "fixture_probing"
        )
        await self.registry.store.store_capability_ready(
            self.capability(model_id, marker), "default"
        )

    async def test_empty_candidate_and_removed_candidate_lifecycle(self) -> None:
        initial = await self.registry.public_summary()
        self.assertEqual(initial.candidate_model_count, 0)
        self.assertIsNone(initial.default_alias_model_id)
        waiting = await self.coordinator.observe_once()
        self.assertEqual(
            waiting.service_readiness_state, "WAITING_FOR_MODEL"
        )

        bundle, router, model_id = self.bundle(
            "sx-first-candidate", "first", "router-first"
        )
        await self.registry.store.register_bundle(bundle, router, model_id)
        registered = await self.registry.public_summary()
        self.assertEqual(registered.candidate_model_count, 1)
        self.assertIsNone(registered.default_alias_model_id)
        loading_registered = await self.coordinator.observe_once()
        self.assertEqual(
            loading_registered.service_readiness_state,
            "MODEL_CANDIDATE_LOADING",
        )
        await self.registry.store.transition_state(
            model_id, ModelState.PROBING, "fixture_probing"
        )
        loading_probing = await self.coordinator.observe_once()
        self.assertEqual(
            loading_probing.service_readiness_state,
            "MODEL_CANDIDATE_LOADING",
        )
        self.assertEqual(self.backend.ensure_calls, [])

        await self.registry.store.mark_missing(set())
        removed = await self.registry.public_summary()
        self.assertEqual(removed.candidate_model_count, 0)
        back_to_waiting = await self.coordinator.observe_once()
        self.assertEqual(
            back_to_waiting.service_readiness_state, "WAITING_FOR_MODEL"
        )
        self.assertEqual(self.backend.ensure_calls, [])

    async def test_ready_gated_promotion_and_invalid_preservation(self) -> None:
        await self.register_ready("sx-version-a", "a", "router-a")
        first = await self.coordinator.observe_once()
        self.assertEqual(first.identity.resolved_public_model_id, "sx-version-a")
        self.assertEqual(self.backend.ensure_calls, [("router-a", None)])

        bundle_b, router_b, model_b = self.bundle(
            "sx-version-b", "b", "router-b"
        )
        registered = await self.registry.store.register_bundle(
            bundle_b, router_b, model_b
        )
        self.assertTrue(registered["changed"])
        alias_registered = await self.registry.store.resolve_public_model(
            "default"
        )
        self.assertEqual(
            alias_registered["model"]["model_version_id"], "sx-version-a"
        )
        await self.coordinator.observe_once()
        self.assertEqual(self.backend.ensure_calls, [("router-a", None)])

        await self.registry.store.transition_state(
            model_b, ModelState.PROBING, "fixture_probing"
        )
        await self.coordinator.observe_once()
        self.assertEqual(self.backend.ensure_calls, [("router-a", None)])

        promotion = await self.registry.store.store_capability_ready(
            self.capability(model_b, "b"), "default"
        )
        self.assertEqual(promotion["promoted_aliases"], ["default"])
        promoted = await self.coordinator.observe_once()
        self.assertEqual(
            promoted.identity.resolved_public_model_id, "sx-version-b"
        )
        self.assertEqual(
            self.backend.ensure_calls,
            [("router-a", None), ("router-b", "router-a")],
        )
        await self.coordinator.observe_once()
        self.assertEqual(
            self.backend.ensure_calls,
            [("router-a", None), ("router-b", "router-a")],
        )

        await self.registry.store.record_rejection(
            "invalid-candidate",
            "INVALID_GGUF",
            {"fixture": "bounded"},
        )
        before = list(self.backend.ensure_calls)
        before_release = list(self.backend.release_calls)
        retained = await self.coordinator.observe_once()
        self.assertEqual(
            retained.identity.resolved_public_model_id, "sx-version-b"
        )
        self.assertEqual(self.backend.ensure_calls, before)
        self.assertEqual(self.backend.release_calls, before_release)
        current = await self.registry.store.resolve_public_model("default")
        self.assertEqual(
            current["model"]["model_version_id"], "sx-version-b"
        )


class ReadinessAndPrivacyTests(unittest.TestCase):
    def identity(self) -> WarmIdentity:
        return WarmIdentity(
            requested_alias="default",
            resolved_public_model_id="sx-ready",
            artifact_version_id="bundle-ready",
            registry_generation=9,
            capability_manifest_identity="b" * 64,
            router_transaction_id="router-tx",
            model_child_pid=101,
            model_child_start_identity="procfs-start-ticks:101",
            model_child_parent=100,
            model_child_process_group=100,
            model_child_session=100,
            warm_since_utc="2026-01-02T03:04:05.000006Z",
            last_verified_utc="2026-01-02T03:04:05.000007Z",
            health_state="ready",
            router_model_id="private-router-model",
        )

    def test_complete_chain_only_and_http_mapping(self) -> None:
        warm = WarmStatus(
            "READY", "default", self.identity(), None, "now"
        )
        backend = PublicBackendState(
            "router_ready", True, True, 1, 1, True
        )
        registry = RegistryPublicSummary(
            "ready", 1, 1, 0, 9, "now"
        )
        view = full_readiness(
            warm, backend, registry, authentication_ready=True
        )
        self.assertTrue(view["ready"])
        self.assertTrue(view["service_available"])
        self.assertTrue(view["inference_ready"])
        self.assertEqual(view["model_service_state"], "READY")
        self.assertEqual(view["service_readiness_state"], "READY")
        self.assertEqual(200 if view["ready"] else 503, 200)

        for changed_backend, changed_registry, authentication in (
            (
                PublicBackendState(
                    "router_ready", True, True, 0, 1, False
                ),
                registry,
                True,
            ),
            (
                PublicBackendState(
                    "unavailable", False, False, 0, 1, False
                ),
                registry,
                True,
            ),
            (backend, RegistryPublicSummary("degraded", 1, 1, 0, 9, "now"), True),
            (backend, registry, False),
        ):
            with self.subTest(
                backend=changed_backend,
                registry=changed_registry,
                authentication=authentication,
            ):
                candidate = full_readiness(
                    warm,
                    changed_backend,
                    changed_registry,
                    authentication_ready=authentication,
                )
                self.assertFalse(candidate["ready"])
                self.assertEqual(
                    candidate["service_readiness_state"], "DEGRADED"
                )
                self.assertEqual(
                    200 if candidate["ready"] else 503, 503
                )

        waiting = full_readiness(
            WarmStatus(
                "WAITING_FOR_MODEL",
                "default",
                None,
                "NO_READY_MODEL",
                "now",
            ),
            PublicBackendState(
                "router_ready", True, True, 0, 1, False
            ),
            RegistryPublicSummary("ready", 0, 0, 0, 10, "now"),
            authentication_ready=True,
        )
        self.assertEqual(
            waiting["model_service_state"], "WAITING_FOR_MODEL"
        )
        self.assertTrue(waiting["service_available"])
        self.assertFalse(waiting["inference_ready"])
        self.assertFalse(waiting["ready"])
        self.assertEqual(waiting["reason_code"], "NO_READY_MODEL")

    def test_health_uses_its_atomic_observation_snapshot(self) -> None:
        source = inspect.getsource(create_application)
        self.assertIn(
            "warm_status = await warm_model.observe_once()",
            source,
        )
        self.assertIn(
            "readiness = full_readiness(\n"
            "            warm_status,",
            source,
        )
        self.assertNotIn(
            "readiness = full_readiness(\n"
            "            warm_model.status,",
            source,
        )

    def test_public_health_schema_and_secret_path_exclusion(self) -> None:
        identity = self.identity()
        public = identity.public_dict()
        encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("private-router-model", encoded)
        self.assertNotIn("/home/", encoded)
        self.assertNotIn("http://127.0.0.1", encoded)
        self.assertNotIn("prompt", encoded)
        response = HealthResponse(
            request_id="sx_req_" + "a" * 32,
            service_name="system-x-gguf-api",
            service_readiness_state="READY",
            model_service_state="READY",
            service_available=True,
            inference_ready=True,
            ready=True,
            service_status="ready",
            contract_version="system-x.gguf-api.native-inference.v1",
            backend_status="router_ready",
            backend_process_running=True,
            backend_control_plane_ready=True,
            loaded_model_count=1,
            model_ready=True,
            environment_name="fixture",
            registry_status="ready",
            registered_model_count=1,
            ready_model_count=1,
            candidate_model_count=0,
            rejected_artifact_count=0,
            registry_generation=9,
            default_alias="default",
            configured_default_alias="default",
            resolved_default_alias="default",
            resolved_public_model_id="sx-ready",
            artifact_version_id="bundle-ready",
            warm_identity=public,
        )
        self.assertTrue(response.ready)


class RouterShutdownReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_reconciles_lost_router_status_before_stopping(self) -> None:
        identity = RouterIdentity("router-tx", 100, 100, 100, "start")
        calls: list[str] = []

        def response(
            operation: str,
            *,
            ok: bool,
            reason_code: str,
            data: dict[str, object],
            exit_status: int,
        ) -> ControllerResult:
            return ControllerResult(
                operation=operation,
                ok=ok,
                reason_code=reason_code,
                message="fixture",
                data=data,
                stderr="",
                exit_status=exit_status,
            )

        class ShutdownController:
            async def status(self) -> ControllerResult:
                calls.append("status")
                if calls.count("status") == 1:
                    return response(
                        "status",
                        ok=False,
                        reason_code="PRIVATE_LISTENER_LOST",
                        data={
                            "active": True,
                            "transaction_id": identity.transaction_id,
                            "pid": identity.pid,
                            "pgid": identity.pgid,
                            "sid": identity.sid,
                            "launch_mode": "router",
                        },
                        exit_status=3,
                    )
                return response(
                    "status",
                    ok=True,
                    reason_code="OK",
                    data={
                        "active": False,
                        "active_state_consistent": True,
                    },
                    exit_status=0,
                )

            async def reconcile(self) -> ControllerResult:
                calls.append("reconcile")
                return response(
                    "reconcile",
                    ok=True,
                    reason_code="PRIVATE_LISTENER_LOST",
                    data={
                        "active": True,
                        "active_state_consistent": False,
                        "transaction_id": identity.transaction_id,
                        "pid": identity.pid,
                        "pgid": identity.pgid,
                        "sid": identity.sid,
                        "launch_mode": "router",
                    },
                    exit_status=0,
                )

            async def stop(self) -> ControllerResult:
                calls.append("stop")
                return response(
                    "stop",
                    ok=True,
                    reason_code="OK",
                    data={
                        "owned_group_absent": True,
                        "active_pid_record_removed": True,
                        "active_lock_removed": True,
                    },
                    exit_status=0,
                )

        backend = object.__new__(BackendCoordinator)
        backend.settings = SimpleNamespace(private_backend_enabled=True)
        backend.controller = ShutdownController()
        backend.router = None
        backend.identity = identity
        backend._router_ready = True
        backend._loaded_by_transaction = set()
        backend._warm_model_id = None

        await backend._stop_exact_router()

        self.assertEqual(calls, ["status", "reconcile", "stop", "status"])
        self.assertIsNone(backend.identity)
        self.assertFalse(backend._router_ready)

    async def test_shutdown_does_not_stop_new_router_owner(self) -> None:
        identity = RouterIdentity("router-tx", 100, 100, 100, "start")
        calls: list[str] = []

        def response(data: dict[str, object]) -> ControllerResult:
            return ControllerResult(
                operation="status",
                ok=True,
                reason_code="OK",
                message="fixture",
                data=data,
                stderr="",
                exit_status=0,
            )

        class ReplacementOwnerController:
            async def status(self) -> ControllerResult:
                calls.append("status")
                return response(
                    {
                        "active": True,
                        "active_state_consistent": True,
                        "transaction_id": "replacement-tx",
                        "pid": 200,
                        "pgid": 200,
                        "sid": 200,
                        "launch_mode": "router",
                    }
                )

            async def stop(self) -> ControllerResult:
                calls.append("stop")
                raise AssertionError("foreign router must not be stopped")

        backend = object.__new__(BackendCoordinator)
        backend.settings = SimpleNamespace(private_backend_enabled=True)
        backend.controller = ReplacementOwnerController()
        backend.router = None
        backend.identity = identity
        backend._router_ready = True
        backend._loaded_by_transaction = {"candidate"}
        backend._warm_model_id = "candidate"

        await backend._stop_exact_router()
