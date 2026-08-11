"""Focused Mini 05.14 automatic model adaptation regression tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from system_x_gguf_api.registry_store import RegistryStore
from system_x_gguf_api.registry_types import (
    ArtifactBundleEvidence,
    ArtifactFileEvidence,
    BundleKind,
    CapabilityEvidence,
    MODEL_ADAPTATION_CONTRACT,
    ModelState,
    PhysicalIdentity,
    REGISTRY_SCHEMA_IDENTITY,
    REGISTRY_SCHEMA_VERSION,
    RoleHint,
    RouterModelEvidence,
    canonical_json,
    utc_now,
)
from system_x_gguf_api.settings import ServiceSettings


class ModelAdaptationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = RegistryStore(self.root / "registry.sqlite3", 1000)
        await self.store.initialize()

    async def asyncTearDown(self) -> None:
        if self.store._initialized:
            await self.store.checkpoint_and_close()
        self.temporary.cleanup()

    def bundle(
        self, marker: str
    ) -> tuple[ArtifactBundleEvidence, RouterModelEvidence]:
        digest = hashlib.sha256(marker.encode()).hexdigest()
        physical = PhysicalIdentity(1, 100 + ord(marker), 0o100600, 24, 1000)
        item = ArtifactFileEvidence(
            relative_path="continuity-fixture.gguf",
            file_sha256=digest,
            size_bytes=24,
            gguf_version=3,
            tensor_count=1,
            metadata_kv_count=1,
            role_hint=RoleHint.PRIMARY,
            physical_identity=physical,
        )
        bundle = ArtifactBundleEvidence(
            bundle_root=self.root / "MODEL" / "SUPERMODEL",
            relative_root="continuity-fixture.gguf",
            bundle_id="bundle-" + digest,
            bundle_sha256=digest,
            bundle_kind=BundleKind.SINGLE_FILE,
            size_bytes=24,
            files=(item,),
            physical_manifest=(
                {"relative_path": "continuity-fixture.gguf",
                 **physical.as_dict()},
            ),
        )
        metadata = canonical_json({
            "router_source": "models_dir",
            "router_status": "unloaded",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        })
        router = RouterModelEvidence(
            router_model_id="router-" + marker,
            router_source="models_dir",
            router_status="unloaded",
            display_name="continuity-fixture",
            physical_path=None,
            connected_paths=(),
            input_modalities=("text",),
            output_modalities=("text",),
            observed_utc=utc_now(),
            metadata_json=metadata,
            metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(),
        )
        return bundle, router

    @staticmethod
    def capability(model: str, marker: str) -> CapabilityEvidence:
        manifest = canonical_json({
            "schema": "system-x.gguf-model-capabilities.v1",
            "model_version_id": model,
            "marker": marker,
            "runtime_generation_tests": {
                "chat": "NOT_TESTED",
                "completion": "NOT_TESTED",
                "responses": "NOT_TESTED",
                "streaming": "NOT_TESTED",
            },
        })
        return CapabilityEvidence(
            model_version_id=model,
            manifest_json=manifest,
            manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
            props_payload_sha256=None,
            observed_utc=utc_now(),
        )

    async def register(self, model: str, marker: str) -> None:
        bundle, router = self.bundle(marker)
        await self.store.register_bundle(bundle, router, model)

    async def stage_pair(self) -> None:
        await self.register("sx-version-a", "a")
        await self.store.transition_state(
            "sx-version-a", ModelState.PROBING, "fixture_probe_started"
        )
        await self.store.store_capability_ready(
            self.capability("sx-version-a", "a"), "default"
        )
        await self.store.create_bound_alias(
            "continuity-current", "sx-version-a", alias_kind="manual"
        )
        await self.register("sx-version-b", "b")
        await self.store.transition_state(
            "sx-version-b", ModelState.PROBING, "fixture_probe_started"
        )

    async def promote_b(self) -> dict[str, object]:
        return await self.store.store_capability_ready(
            self.capability("sx-version-b", "b"), "default"
        )

    async def test_ready_gated_atomic_replacement_rollback_and_idempotence(
        self,
    ) -> None:
        await self.stage_pair()
        alias = await self.store.resolve_public_model("continuity-current")
        self.assertEqual(alias["model"]["model_version_id"], "sx-version-a")
        before = await self.store.snapshot()
        with self.assertRaises(RuntimeError):
            await self.store.store_capability_ready(
                self.capability("sx-version-b", "b"),
                "default",
                inject_failure_before_commit=True,
            )
        after = await self.store.snapshot()
        self.assertEqual(before["registry_metadata"], after["registry_metadata"])
        self.assertEqual(
            before["capability_manifests"], after["capability_manifests"]
        )
        self.assertEqual(
            (await self.store.resolve_public_model("continuity-current"))["model"][
                "model_version_id"
            ],
            "sx-version-a",
        )

        promotion = await self.promote_b()
        self.assertTrue(promotion["changed"])
        final = await self.store.snapshot()
        states = {
            row["model_version_id"]: row["state"]
            for row in final["model_versions"]
            if row["model_version_id"] in {"sx-version-a", "sx-version-b"}
        }
        self.assertEqual(
            states,
            {"sx-version-a": "REPLACED", "sx-version-b": "READY"},
        )
        current = await self.store.resolve_public_model("continuity-current")
        self.assertEqual(
            current["model"]["model_version_id"], "sx-version-b"
        )
        generation = promotion["generation"]
        self.assertEqual(
            {
                row["event_type"]
                for row in final["registry_events"]
                if row["generation"] == generation
            },
            {"replacement_ready", "alias_promoted"},
        )
        self.assertEqual(
            {
                row["alias"]: row["model_version_id"]
                for row in final["aliases"]
            }["continuity-current"],
            "sx-version-b",
        )
        self.assertEqual(
            {
                row["model_version_id"]
                for row in final["capability_manifests"]
            },
            {"sx-version-a", "sx-version-b"},
        )
        duplicate = await self.promote_b()
        self.assertFalse(duplicate["changed"])
        self.assertEqual(duplicate["generation"], generation)

    async def test_invalid_rejection_revival_fresh_probe_and_history_cleanup(
        self,
    ) -> None:
        await self.stage_pair()
        await self.promote_b()
        before = await self.store.public_model_rows()
        rejection = await self.store.record_rejection(
            "continuity-fixture.gguf",
            "GGUF_MAGIC_INVALID",
            {"fixture": "invalid-stage"},
        )
        self.assertTrue(rejection["changed"])
        after = await self.store.public_model_rows()
        self.assertEqual(
            [row["model_version_id"] for row in before["models"]],
            [row["model_version_id"] for row in after["models"]],
        )
        self.assertEqual(
            (await self.store.resolve_public_model("continuity-current"))[
                "model"
            ]["model_version_id"],
            "sx-version-b",
        )

        await self.store.mark_missing(set())
        missing = await self.store.snapshot()
        self.assertEqual(
            next(
                row["state"]
                for row in missing["model_versions"]
                if row["model_version_id"] == "sx-version-b"
            ),
            "REMOVED",
        )
        self.assertIn(
            "sx-version-b",
            {row["model_version_id"] for row in missing["capability_manifests"]},
        )

        bundle, router = self.bundle("b")
        await self.store.register_bundle(bundle, router, "sx-version-b")
        revived = await self.store.snapshot()
        self.assertEqual(
            next(
                row["state"]
                for row in revived["model_versions"]
                if row["model_version_id"] == "sx-version-b"
            ),
            "REGISTERED",
        )
        self.assertNotIn(
            "sx-version-b",
            {row["model_version_id"] for row in revived["capability_manifests"]},
        )
        self.assertIn(
            "sx-version-b",
            {
                row["model_version_id"]
                for row in await self.store.models_needing_capability()
            },
        )
        await self.store.transition_state(
            "sx-version-b", ModelState.PROBING, "fixture_reprobe_started"
        )
        await self.promote_b()
        self.assertEqual(
            (await self.store.resolve_public_model("continuity-current"))[
                "model"
            ]["model_version_id"],
            "sx-version-b",
        )

        await self.store.remove_alias("continuity-current")
        await self.store.mark_missing(set())
        cleaned = await self.store.snapshot()
        self.assertNotIn(
            "continuity-current",
            {row["alias"] for row in cleaned["aliases"]},
        )
        self.assertIn(
            "continuity-fixture.gguf",
            {row["relative_root"] for row in cleaned["artifact_locations"]},
        )
        self.assertEqual(
            {
                row["model_version_id"]
                for row in cleaned["capability_manifests"]
            },
            {"sx-version-a", "sx-version-b"},
        )

    def test_published_contract_version(self) -> None:
        self.assertEqual(ServiceSettings().service_version, "0.14.0")
        self.assertEqual(
            MODEL_ADAPTATION_CONTRACT,
            "system-x.gguf-model-adaptation.v1",
        )
        self.assertEqual(
            REGISTRY_SCHEMA_IDENTITY, "system-x.gguf-model-registry.v1"
        )
        self.assertEqual(REGISTRY_SCHEMA_VERSION, 2)
