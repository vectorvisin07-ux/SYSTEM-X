"""Regression coverage for registry removal during backend recovery."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from system_x_gguf_api.backend import BackendError
from system_x_gguf_api.model_registry import ModelRegistry


class RemovalBeforeRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_physical_removal_is_persisted_before_refresh_failure(self) -> None:
        calls: list[str] = []

        class Store:
            async def present_location_roots(self) -> set[str]:
                return {"removed.gguf", "survivor.gguf"}

            async def mark_missing(self, roots: set[str]) -> None:
                calls.append("mark_missing")
                self.marked = set(roots)

            async def location_hash_cache(self, relative_root: str) -> dict[str, object]:
                return {}

            async def location_record(self, relative_root: str) -> None:
                return None

        class Inspector:
            def inspect_location(
                self,
                relative_root: str,
                cache: dict[str, object],
                shutdown_event: asyncio.Event,
            ) -> SimpleNamespace:
                return SimpleNamespace(
                    relative_root=relative_root,
                    bundle_id="bundle-survivor",
                    physical_manifest=(),
                    file_count=1,
                    reused_hash_count=1,
                )

        class Backend:
            async def current_router_inventory(self) -> SimpleNamespace:
                calls.append("current_inventory")
                return SimpleNamespace(models=())

            async def refresh_validated_model_inventory(
                self, replacement_model_ids: set[str]
            ) -> None:
                calls.append("refresh")
                raise BackendError("recovery in progress")

        registry = object.__new__(ModelRegistry)
        registry._reconcile_run_count = 0
        registry._physical_units_sync = lambda: {"survivor.gguf"}
        registry.store = Store()
        registry.backend = Backend()
        registry.inspector = Inspector()
        registry._shutdown_event = asyncio.Event()
        registry._router_evidence = {}
        registry._last_outcomes = []
        registry._probe_task = None
        registry._router_refresh_count = 0
        registry._validated_replacement_unload_count = 0

        with self.assertRaisesRegex(BackendError, "recovery in progress"):
            await registry._full_reconcile(["fixture_recovery"])

        self.assertEqual(calls, ["mark_missing", "current_inventory", "refresh"])
        self.assertEqual(registry.store.marked, {"survivor.gguf"})


if __name__ == "__main__":
    unittest.main()
