"""Focused probe cancellation liveness regression tests."""
from __future__ import annotations
import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock
from system_x_gguf_api.model_registry import ModelRegistry
from system_x_gguf_api.registry_types import ModelState, canonical_json
class RecordingStore:
    def __init__(self) -> None:
        self.transitions: list[tuple[str, ModelState, str, object]] = []
    async def transition_state(
        self,
        model_version_id: str,
        state: ModelState,
        reason: str,
        detail: object = None,
    ) -> None:
        self.transitions.append((model_version_id, state, reason, detail))
    async def store_capability_ready(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("cancelled probe must not store capability evidence")
class BlockingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()
    async def probe_model_properties(self, router_model_id: str) -> object:
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")
class ProbeCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_probe_terminalizes_unavailable_before_reraise(self) -> None:
        registry = object.__new__(ModelRegistry)
        registry.store = RecordingStore()
        registry.backend = BlockingBackend()
        registry.settings = SimpleNamespace(registry_default_alias="default")
        registry._probe_queue = asyncio.Queue()
        registry._queued_probe_ids = set()
        registry._router_evidence = {}
        registry._status = "ready"
        registry._last_error_code = None
        await registry._probe_queue.put(
            {
                "model_version_id": "sx-cancelled-probe",
                "bundle_id": "bundle-cancelled-probe",
                "router_model_id": "router-cancelled-probe",
                "router_metadata_json": canonical_json(
                    {
                        "router_source": "models_dir",
                        "router_status": "unloaded",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    }
                ),
            }
        )
        task = asyncio.create_task(registry._probe_loop())
        await asyncio.wait_for(registry.backend.started.wait(), timeout=1.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(
            [(row[0], row[1], row[2]) for row in registry.store.transitions],
            [
                ("sx-cancelled-probe", ModelState.PROBING, "capability_probe_started"),
                ("sx-cancelled-probe", ModelState.UNAVAILABLE, "capability_probe_cancelled"),
            ],
        )
        self.assertEqual(registry.store.transitions[-1][3], {"error_type": "CancelledError"})


    async def test_successful_probe_clears_probe_failure_degradation(self) -> None:
        registry = object.__new__(ModelRegistry)
        registry.store = RecordingStore()
        registry.store.store_capability_ready = mock.AsyncMock()
        registry.backend = SimpleNamespace(
            probe_model_properties=mock.AsyncMock(
                return_value=SimpleNamespace(
                    props=SimpleNamespace(json_value={"ok": True})
                )
            )
        )
        registry.settings = SimpleNamespace(registry_default_alias="default")
        registry._probe_queue = asyncio.Queue()
        registry._queued_probe_ids = set()
        registry._router_evidence = {}
        registry._router_evidence_for_probe = lambda row: object()
        registry.mark_recovered = mock.AsyncMock()
        await registry._probe_queue.put(
            {
                "model_version_id": "sx-successful-probe",
                "bundle_id": "bundle-successful-probe",
                "router_model_id": "router-successful-probe",
                "router_metadata_json": canonical_json(
                    {
                        "router_source": "models_dir",
                        "router_status": "unloaded",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    }
                ),
            }
        )
        await registry._probe_queue.put({"_stop": True})
        with mock.patch(
            "system_x_gguf_api.model_registry.build_capability_evidence",
            return_value={"capability": "ready"},
        ):
            await registry._probe_loop()
        registry.mark_recovered.assert_awaited_once_with(
            "capability_probe_failure"
        )
