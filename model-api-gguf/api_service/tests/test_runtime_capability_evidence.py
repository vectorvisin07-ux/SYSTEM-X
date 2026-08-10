from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from system_x_gguf_api.backend import PublicBackendState
from system_x_gguf_api.model_registry import RegistryPublicSummary
from system_x_gguf_api.registry_store import RegistryStore, RegistryStoreError
from system_x_gguf_api.registry_types import canonical_json
from system_x_gguf_api.warm_model import WarmStatus, full_readiness


MODEL = "model-r3-capability"
NATIVE = "system-x.streaming.v1"
OPENAI = "system-x.openai-streaming.v1"
MESSAGES = "system-x.anthropic-streaming.v1"
SURFACES = (NATIVE, OPENAI, MESSAGES)


def observation(request_id: str, service_id: str, router_id: str) -> dict[str, str]:
    return {
        "request_id": request_id,
        "service_transaction_id": service_id,
        "router_transaction_id": router_id,
        "observed_utc": "2026-08-09T00:00:00.000000Z",
    }


def streaming_evidence(
    values: dict[str, tuple[str, str, str]],
    *,
    state: str = "AVAILABLE",
) -> dict[str, object]:
    observations = {
        surface: observation(*values[surface]) for surface in values
    }
    first = values[next(iter(values))]
    return {
        "state": state,
        "source": "validated_live_inference",
        "model_version_id": MODEL,
        "request_id": first[0],
        "service_transaction_id": first[1],
        "router_transaction_id": first[2],
        "observed_protocol_surfaces": sorted(values),
        "observations": observations,
        "observed_utc": "2026-08-09T00:00:00.000000Z",
    }


class RuntimeCapabilityEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        root.chmod(0o700)
        self.database = root / "registry.sqlite3"
        self.store = RegistryStore(self.database, 2_000)
        await self.store.initialize()
        now = "2026-08-09T00:00:00.000000Z"
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                INSERT INTO artifact_bundles(
                    bundle_id,bundle_sha256,bundle_kind,file_count,size_bytes,
                    manifest_json,first_seen_utc,last_seen_utc
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                ("bundle-r3-capability", "a" * 64, "single_file", 1, 24,
                 "{}", now, now),
            )
            connection.execute(
                """
                INSERT INTO model_versions(
                    model_version_id,bundle_id,router_model_id,router_source,
                    display_name,state,router_metadata_json,
                    router_metadata_sha256,created_utc,updated_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (MODEL, "bundle-r3-capability", "router-r3-capability",
                 "focused-test", MODEL, "READY", "{}", "b" * 64, now, now),
            )
        self._write_manifest(self._base_manifest())

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _base_manifest() -> dict[str, object]:
        return {
            "runtime_generation_tests": {
                "streaming": "NOT_TESTED",
                "tool_calling": "NOT_TESTED",
                "structured_output": "NOT_TESTED",
            },
            "evidence_layers": {"runtime_generation": "NOT_TESTED"},
            "runtime_capability_evidence": {},
        }

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        encoded = canonical_json(manifest)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO capability_manifests(
                    model_version_id,manifest_json,manifest_sha256,
                    props_payload_sha256,observed_utc
                ) VALUES (?,?,?,?,?)
                ON CONFLICT(model_version_id) DO UPDATE SET
                    manifest_json=excluded.manifest_json,
                    manifest_sha256=excluded.manifest_sha256,
                    props_payload_sha256=excluded.props_payload_sha256,
                    observed_utc=excluded.observed_utc
                """,
                (
                    MODEL,
                    encoded,
                    hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                    None,
                    "2026-08-09T00:00:00.000000Z",
                ),
            )

    def _seed_available(
        self,
        values: dict[str, tuple[str, str, str]],
    ) -> None:
        manifest = self._base_manifest()
        manifest["runtime_generation_tests"] = {
            "streaming": "AVAILABLE",
            "tool_calling": "NOT_TESTED",
            "structured_output": "NOT_TESTED",
        }
        manifest["evidence_layers"] = {
            "runtime_generation": {"streaming": "AVAILABLE"}
        }
        manifest["runtime_capability_evidence"] = {
            "streaming": streaming_evidence(values)
        }
        self._write_manifest(manifest)

    def _snapshot(self) -> dict[str, object]:
        with sqlite3.connect(self.database) as connection:
            generation = int(connection.execute(
                "SELECT value FROM registry_metadata "
                "WHERE key='registry_generation'"
            ).fetchone()[0])
            manifest = connection.execute(
                "SELECT manifest_json,manifest_sha256,observed_utc "
                "FROM capability_manifests WHERE model_version_id=?",
                (MODEL,),
            ).fetchone()
            event_count = connection.execute(
                "SELECT COUNT(*) FROM registry_events"
            ).fetchone()[0]
        return {
            "generation": generation,
            "manifest_json": manifest[0],
            "manifest_sha256": manifest[1],
            "manifest_observed_utc": manifest[2],
            "event_count": event_count,
        }

    def _loaded_manifest(self) -> dict[str, object]:
        with sqlite3.connect(self.database) as connection:
            encoded = connection.execute(
                "SELECT manifest_json FROM capability_manifests "
                "WHERE model_version_id=?", (MODEL,)
            ).fetchone()[0]
        return json.loads(encoded)

    async def _record(
        self,
        request_id: str,
        service_id: str,
        router_id: str,
        surfaces: tuple[str, ...],
        *,
        store: RegistryStore | None = None,
    ) -> dict[str, object]:
        return await (store or self.store).record_runtime_capability(
            MODEL,
            "streaming",
            request_id,
            service_id,
            router_id,
            surfaces,
        )

    def _assert_manifest_hash(self, manifest: dict[str, object]) -> None:
        encoded = canonical_json(manifest)
        with sqlite3.connect(self.database) as connection:
            stored = connection.execute(
                "SELECT manifest_sha256 FROM capability_manifests "
                "WHERE model_version_id=?", (MODEL,)
            ).fetchone()[0]
        self.assertEqual(stored, hashlib.sha256(encoded.encode()).hexdigest())

    async def test_first_streaming_promotion_is_one_generation_and_event(self) -> None:
        before = self._snapshot()
        result = await self._record("req-first", "service-first", "router-first",
                                    (NATIVE,))
        after = self._snapshot()
        self.assertTrue(result["changed"])
        self.assertEqual(after["generation"], before["generation"] + 1)
        self.assertEqual(after["event_count"], before["event_count"] + 1)
        manifest = self._loaded_manifest()
        evidence = manifest["runtime_capability_evidence"]["streaming"]
        self.assertEqual(evidence["state"], "AVAILABLE")
        self.assertEqual(evidence["observed_protocol_surfaces"], [NATIVE])
        self.assertEqual(evidence["observations"][NATIVE]["request_id"], "req-first")
        self.assertEqual(result["observed_protocol_surfaces"], [NATIVE])
        self._assert_manifest_hash(manifest)

    async def test_identical_evidence_is_exact_idempotent_noop(self) -> None:
        values = {surface: ("req-old", "service-old", "router-old")
                  for surface in SURFACES}
        self._seed_available(values)
        before = self._snapshot()
        result = await self._record("req-old", "service-old", "router-old", SURFACES)
        self.assertFalse(result["changed"])
        self.assertEqual(self._snapshot(), before)

    async def test_newer_evidence_refreshes_all_requested_surfaces_once(self) -> None:
        values = {surface: ("req-old", "service-old", "router-old")
                  for surface in SURFACES}
        self._seed_available(values)
        before = self._snapshot()
        result = await self._record("req-new", "service-new", "router-new", SURFACES)
        after = self._snapshot()
        self.assertTrue(result["changed"])
        self.assertEqual(after["generation"], before["generation"] + 1)
        self.assertEqual(after["event_count"], before["event_count"] + 1)
        self.assertNotEqual(after["manifest_sha256"], before["manifest_sha256"])
        evidence = self._loaded_manifest()["runtime_capability_evidence"]["streaming"]
        self.assertEqual(evidence["observed_protocol_surfaces"], sorted(SURFACES))
        self.assertEqual(evidence["request_id"], "req-new")
        for surface in SURFACES:
            self.assertEqual(
                evidence["observations"][surface],
                {
                    "request_id": "req-new",
                    "service_transaction_id": "service-new",
                    "router_transaction_id": "router-new",
                    "observed_utc": evidence["observations"][surface]["observed_utc"],
                },
            )
        self.assertEqual(result["observed_protocol_surfaces"], sorted(SURFACES))

    async def test_refreshing_one_surface_preserves_unrelated_observations(self) -> None:
        values = {
            NATIVE: ("req-native-old", "service-native-old", "router-native-old"),
            OPENAI: ("req-openai-old", "service-openai-old", "router-openai-old"),
            MESSAGES: ("req-messages-old", "service-messages-old", "router-messages-old"),
        }
        self._seed_available(values)
        before = self._loaded_manifest()["runtime_capability_evidence"]["streaming"]
        await self._record("req-native-new", "service-native-new", "router-native-new",
                           (NATIVE,))
        after = self._loaded_manifest()["runtime_capability_evidence"]["streaming"]
        self.assertEqual(after["observed_protocol_surfaces"], sorted(SURFACES))
        self.assertEqual(after["observations"][OPENAI], before["observations"][OPENAI])
        self.assertEqual(after["observations"][MESSAGES], before["observations"][MESSAGES])
        self.assertEqual(after["observations"][NATIVE]["request_id"], "req-native-new")

    async def test_new_surface_adds_sorted_union_and_preserves_existing(self) -> None:
        self._seed_available({NATIVE: ("req-native", "service-native", "router-native")})
        await self._record("req-openai", "service-openai", "router-openai", (OPENAI,))
        evidence = self._loaded_manifest()["runtime_capability_evidence"]["streaming"]
        self.assertEqual(evidence["observed_protocol_surfaces"], sorted((NATIVE, OPENAI)))
        self.assertEqual(evidence["observations"][NATIVE]["request_id"], "req-native")
        self.assertEqual(evidence["observations"][OPENAI]["request_id"], "req-openai")

    async def test_invalid_inputs_and_not_ready_model_leave_database_unchanged(self) -> None:
        invalid_calls = (
            ("req", "service", "router", [NATIVE]),
            ("", "service", "router", (NATIVE,)),
            ("req", "", "router", (NATIVE,)),
            ("req", "service", "", (NATIVE,)),
            ("x" * 129, "service", "router", (NATIVE,)),
            ("req", "service", "router", ("invalid.surface",)),
        )
        for request_id, service_id, router_id, surfaces in invalid_calls:
            before = self._snapshot()
            with self.assertRaises(RegistryStoreError):
                await self._record(request_id, service_id, router_id, surfaces)  # type: ignore[arg-type]
            self.assertEqual(self._snapshot(), before)
        before = self._snapshot()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE model_versions SET state='REGISTERED' WHERE model_version_id=?",
                (MODEL,),
            )
        with self.assertRaisesRegex(RegistryStoreError, "not ready"):
            await self._record("req", "service", "router", (NATIVE,))
        after = self._snapshot()
        self.assertEqual(after["generation"], before["generation"])
        self.assertEqual(after["manifest_json"], before["manifest_json"])
        self.assertEqual(after["event_count"], before["event_count"])

    async def test_malformed_existing_evidence_is_rejected_without_mutation(self) -> None:
        self._seed_available({NATIVE: ("req-old", "service-old", "router-old")})
        manifest = self._loaded_manifest()
        manifest["runtime_capability_evidence"]["streaming"]["observations"] = {}
        self._write_manifest(manifest)
        before = self._snapshot()
        with self.assertRaisesRegex(RegistryStoreError, "evidence is invalid"):
            await self._record("req-new", "service-new", "router-new", (NATIVE,))
        self.assertEqual(self._snapshot(), before)

    async def test_transaction_failure_rolls_back_manifest_generation_and_event(self) -> None:
        before = self._snapshot()
        with patch.object(
            RegistryStore,
            "_insert_event",
            side_effect=RuntimeError("forced event failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced event failure"):
                await self._record("req-fail", "service-fail", "router-fail", (NATIVE,))
        self.assertEqual(self._snapshot(), before)

    async def test_concurrent_writes_serialize_without_lost_surface(self) -> None:
        other = RegistryStore(self.database, 2_000)
        await other.initialize()
        first, second = await asyncio.gather(
            self._record("req-native", "service-native", "router-native", (NATIVE,)),
            self._record("req-openai", "service-openai", "router-openai", (OPENAI,), store=other),
        )
        self.assertTrue(first["changed"])
        self.assertTrue(second["changed"])
        snapshot = self._snapshot()
        self.assertEqual(snapshot["generation"], 2)
        self.assertEqual(snapshot["event_count"], 2)
        evidence = self._loaded_manifest()["runtime_capability_evidence"]["streaming"]
        self.assertEqual(evidence["observed_protocol_surfaces"], sorted((NATIVE, OPENAI)))
        self.assertEqual(
            set(evidence["observations"]),
            {NATIVE, OPENAI},
        )

    async def test_integrity_foreign_keys_and_manifest_hash_remain_valid(self) -> None:
        await self._record("req-integrity", "service-integrity", "router-integrity", (MESSAGES,))
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            generation = int(connection.execute(
                "SELECT value FROM registry_metadata WHERE key='registry_generation'"
            ).fetchone()[0])
            event_generations = [row[0] for row in connection.execute(
                "SELECT generation FROM registry_events ORDER BY generation"
            )]
        self.assertEqual(generation, 1)
        self.assertEqual(event_generations, [1])
        self._assert_manifest_hash(self._loaded_manifest())


class HealthContractH1Tests(unittest.TestCase):
    def test_registry_control_demand_loaded_model_remains_publicly_not_ready(self) -> None:
        warm = WarmStatus(
            "WAITING_FOR_MODEL",
            "default",
            None,
            "startup_policy_unloaded",
            "2026-08-09T00:00:00.000000Z",
        )
        backend = PublicBackendState(
            "router_ready", True, True, 1, 1, True
        )
        registry = RegistryPublicSummary(
            "ready", 1, 1, 0, 7, "2026-08-09T00:00:00.000000Z",
            default_alias_model_id="sx-demand-loaded",
            default_alias_ready=True,
        )
        readiness = full_readiness(
            warm,
            backend,
            registry,
            authentication_ready=True,
        )
        self.assertFalse(readiness["service_available"])
        self.assertFalse(readiness["ready"])
        self.assertFalse(readiness["inference_ready"])
        self.assertEqual(readiness["reason_code"], "EXPECTED_MODEL_NOT_READY")
        self.assertEqual(backend.loaded_model_count, 1)
        self.assertTrue(backend.model_ready)


if __name__ == "__main__":
    unittest.main()
