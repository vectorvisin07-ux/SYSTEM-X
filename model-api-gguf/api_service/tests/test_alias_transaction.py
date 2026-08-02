"""Focused isolated tests for the branch-owned default-alias transaction."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from system_x_gguf_api.registry_store import (
    AliasTransactionConflict,
    RegistryStore,
)


INCUMBENT = "model-incumbent"
CANDIDATE = "model-candidate"
INCUMBENT_BUNDLE = "bundle-" + "a" * 64
CANDIDATE_BUNDLE = "bundle-" + "b" * 64
INCUMBENT_MANIFEST = "sha256:" + "c" * 64
CANDIDATE_MANIFEST = "sha256:" + "d" * 64
INCUMBENT_ROOT = "incumbent.gguf"
CANDIDATE_ROOT = "candidate.gguf"
TRANSACTION_ID = "promotion-0123456789abcdef"


def controller_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "api_service_controller"
        / "controller.py"
    )
    specification = importlib.util.spec_from_file_location(
        "alias_transaction_controller",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class AliasTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.database = self.root / "registry.sqlite3"
        self.store = RegistryStore(self.database, 1_000)
        await self.store.initialize()
        now = "2026-07-30T00:00:00.000000Z"
        connection = sqlite3.connect(self.database)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            for model, bundle, root, manifest in (
                (
                    INCUMBENT,
                    INCUMBENT_BUNDLE,
                    INCUMBENT_ROOT,
                    INCUMBENT_MANIFEST,
                ),
                (
                    CANDIDATE,
                    CANDIDATE_BUNDLE,
                    CANDIDATE_ROOT,
                    CANDIDATE_MANIFEST,
                ),
            ):
                connection.execute(
                    """
                    INSERT INTO artifact_bundles(
                        bundle_id,bundle_sha256,bundle_kind,file_count,
                        size_bytes,manifest_json,first_seen_utc,last_seen_utc
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        bundle,
                        bundle[7:],
                        "single_file",
                        1,
                        24,
                        "{}",
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO artifact_locations(
                        relative_root,current_bundle_id,present,
                        physical_manifest_json,first_seen_utc,last_seen_utc
                    ) VALUES (?,?,?,?,?,?)
                    """,
                    (root, bundle, 1, "{}", now, now),
                )
                connection.execute(
                    """
                    INSERT INTO model_versions(
                        model_version_id,bundle_id,router_model_id,
                        router_source,display_name,state,router_metadata_json,
                        router_metadata_sha256,created_utc,updated_utc
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        model,
                        bundle,
                        model,
                        "isolated-fixture",
                        model,
                        "READY",
                        "{}",
                        "e" * 64,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO model_version_locations(
                        model_version_id,relative_root,created_utc,updated_utc
                    ) VALUES (?,?,?,?)
                    """,
                    (model, root, now, now),
                )
                connection.execute(
                    """
                    INSERT INTO capability_manifests(
                        model_version_id,manifest_json,manifest_sha256,
                        props_payload_sha256,observed_utc
                    ) VALUES (?,?,?,?,?)
                    """,
                    (model, "{}", manifest[7:], None, now),
                )
            connection.execute(
                """
                INSERT INTO aliases(
                    alias,model_version_id,alias_kind,created_utc,updated_utc
                ) VALUES ('default',?,'default',?,?)
                """,
                (INCUMBENT, now, now),
            )
            connection.execute(
                """
                INSERT INTO alias_bindings(
                    alias,relative_root,promotion_policy,
                    created_utc,updated_utc
                ) VALUES ('default',?,'on_ready_same_location',?,?)
                """,
                (INCUMBENT_ROOT, now, now),
            )
            connection.execute(
                """
                UPDATE registry_metadata
                SET value='7'
                WHERE key='registry_generation'
                """
            )
            connection.commit()
        finally:
            connection.close()

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def promote(self, generation: int = 7) -> dict[str, object]:
        return await self.store.compare_and_swap_default_alias(
            action="promote",
            promotion_transaction_id=TRANSACTION_ID,
            alias="default",
            expected_current_target=INCUMBENT,
            new_target=CANDIDATE,
            expected_registry_generation=generation,
            target_artifact_version_id=CANDIDATE_BUNDLE,
            target_capability_manifest_identity=CANDIDATE_MANIFEST,
            target_relative_root=CANDIDATE_ROOT,
        )

    async def test_atomic_promotion_idempotence_and_rollback(self) -> None:
        promoted = await self.promote()
        self.assertTrue(promoted["changed"])
        self.assertEqual(promoted["new_registry_generation"], 8)
        duplicate = await self.promote()
        self.assertFalse(duplicate["changed"])
        self.assertEqual(
            duplicate["alias_event_identity"],
            promoted["alias_event_identity"],
        )
        snapshot = await self.store.snapshot()
        self.assertEqual(snapshot["aliases"][0]["model_version_id"], CANDIDATE)
        self.assertEqual(
            {row["state"] for row in snapshot["model_versions"]},
            {"READY"},
        )
        rolled_back = await self.store.compare_and_swap_default_alias(
            action="rollback",
            promotion_transaction_id=TRANSACTION_ID,
            alias="default",
            expected_current_target=CANDIDATE,
            new_target=INCUMBENT,
            expected_registry_generation=8,
            target_artifact_version_id=INCUMBENT_BUNDLE,
            target_capability_manifest_identity=INCUMBENT_MANIFEST,
            target_relative_root=INCUMBENT_ROOT,
            promotion_alias_event_identity=str(
                promoted["alias_event_identity"]
            ),
        )
        self.assertTrue(rolled_back["changed"])
        self.assertEqual(rolled_back["new_registry_generation"], 9)
        snapshot = await self.store.snapshot()
        self.assertEqual(snapshot["aliases"][0]["model_version_id"], INCUMBENT)
        self.assertEqual(
            [
                row["event_type"]
                for row in snapshot["registry_events"]
                if row["event_type"].startswith("default_alias_")
            ],
            ["default_alias_promoted", "default_alias_rolled_back"],
        )

    async def test_generation_and_state_conflicts_are_atomic(self) -> None:
        with self.assertRaises(AliasTransactionConflict):
            await self.promote(generation=6)
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE model_versions SET state='UNAVAILABLE' "
                "WHERE model_version_id=?",
                (CANDIDATE,),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(AliasTransactionConflict):
            await self.promote()
        snapshot = await self.store.snapshot()
        self.assertEqual(snapshot["aliases"][0]["model_version_id"], INCUMBENT)
        self.assertEqual(
            snapshot["registry_metadata"][4]["value"],
            "7",
        )
        self.assertFalse(
            any(
                row["event_type"].startswith("default_alias_")
                for row in snapshot["registry_events"]
            )
        )

    async def test_concurrent_exact_request_changes_alias_once(self) -> None:
        first, second = await asyncio.gather(self.promote(), self.promote())
        self.assertEqual(sorted((first["changed"], second["changed"])), [False, True])
        snapshot = await self.store.snapshot()
        events = [
            row
            for row in snapshot["registry_events"]
            if row["event_type"] == "default_alias_promoted"
        ]
        self.assertEqual(len(events), 1)

    async def test_first_model_rollback_atomically_removes_default(self) -> None:
        connection = sqlite3.connect(self.database)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM aliases WHERE alias='default'")
            connection.commit()
        finally:
            connection.close()
        promoted = await self.store.compare_and_swap_default_alias(
            action="promote",
            promotion_transaction_id=TRANSACTION_ID,
            alias="default",
            expected_current_target=None,
            new_target=CANDIDATE,
            expected_registry_generation=7,
            target_artifact_version_id=CANDIDATE_BUNDLE,
            target_capability_manifest_identity=CANDIDATE_MANIFEST,
            target_relative_root=CANDIDATE_ROOT,
        )
        rolled_back = await self.store.compare_and_swap_default_alias(
            action="rollback",
            promotion_transaction_id=TRANSACTION_ID,
            alias="default",
            expected_current_target=CANDIDATE,
            new_target=None,
            expected_registry_generation=8,
            target_artifact_version_id=None,
            target_capability_manifest_identity=None,
            target_relative_root=None,
            promotion_alias_event_identity=str(
                promoted["alias_event_identity"]
            ),
        )
        self.assertTrue(rolled_back["changed"])
        snapshot = await self.store.snapshot()
        self.assertEqual(snapshot["aliases"], [])
        self.assertEqual(snapshot["alias_bindings"], [])
        generation = next(
            row["value"]
            for row in snapshot["registry_metadata"]
            if row["key"] == "registry_generation"
        )
        self.assertEqual(generation, "9")

    async def test_explicit_retirement_clear_is_atomic_and_idempotent(self) -> None:
        cleared = await self.store.compare_and_swap_default_alias(
            action="clear",
            promotion_transaction_id="retirement-0123456789abcdef",
            alias="default",
            expected_current_target=INCUMBENT,
            new_target=None,
            expected_registry_generation=7,
            target_artifact_version_id=None,
            target_capability_manifest_identity=None,
            target_relative_root=None,
        )
        self.assertTrue(cleared["changed"])
        self.assertEqual(cleared["new_registry_generation"], 8)
        duplicate = await self.store.compare_and_swap_default_alias(
            action="clear",
            promotion_transaction_id="retirement-0123456789abcdef",
            alias="default",
            expected_current_target=INCUMBENT,
            new_target=None,
            expected_registry_generation=7,
            target_artifact_version_id=None,
            target_capability_manifest_identity=None,
            target_relative_root=None,
        )
        self.assertFalse(duplicate["changed"])
        self.assertEqual(
            duplicate["alias_event_identity"],
            cleared["alias_event_identity"],
        )
        snapshot = await self.store.snapshot()
        self.assertEqual(snapshot["aliases"], [])
        self.assertEqual(snapshot["alias_bindings"], [])
        events = [
            row for row in snapshot["registry_events"]
            if row["event_type"] == "default_alias_cleared"
        ]
        self.assertEqual(len(events), 1)

    def test_controller_input_is_exact_bounded_and_pathless(self) -> None:
        controller = controller_module()
        request = {
            "schema_version": controller.ALIAS_TRANSACTION_SCHEMA,
            "action": "promote",
            "promotion_transaction_id": TRANSACTION_ID,
            "alias": "default",
            "expected_current_target": INCUMBENT,
            "new_target": CANDIDATE,
            "expected_registry_generation": 7,
            "target_artifact_version_id": CANDIDATE_BUNDLE,
            "target_capability_manifest_identity": CANDIDATE_MANIFEST,
            "target_relative_root": CANDIDATE_ROOT,
            "promotion_alias_event_identity": None,
        }
        encoded = json.dumps(request).encode()
        self.assertEqual(
            controller.decode_alias_transaction_request(encoded),
            request,
        )
        request["registry_database"] = "/caller/controlled.sqlite3"
        with self.assertRaises(controller.ControllerError):
            controller.decode_alias_transaction_request(
                json.dumps(request).encode()
            )
        with self.assertRaises(controller.ControllerError):
            controller.decode_alias_transaction_request(
                b"x" * (controller.ALIAS_TRANSACTION_INPUT_LIMIT + 1)
            )


if __name__ == "__main__":
    unittest.main()
