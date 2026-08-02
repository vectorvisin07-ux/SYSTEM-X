from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from system_x_inspector.capabilities import (
    build_binding,
    build_capability_record,
    publish_binding,
    publish_capability_record,
)
from system_x_inspector.constants import SCHEMA_IDENTITIES
from system_x_inspector.decision import resolve_decision
from system_x_inspector.decision import (
    build_decision_record,
    publish_decision_record,
    validate_decision_record,
)
from system_x_inspector.errors import InspectorError
from system_x_inspector.machine import main
from system_x_inspector.paths import InspectorPaths
from system_x_inspector.records import (
    atomic_create_json,
    atomic_write_json,
    canonical_json_bytes,
    read_json_record,
)
from system_x_inspector.results import utc_now
from system_x_inspector.runtime import decide_transaction


def sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class DecisionEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-decision-", dir="/tmp")
        )
        self.root = self.temporary / "INSPECTOR"
        self.root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.root)
        for path in (
            self.paths.locks,
            self.paths.status,
            self.paths.transactions,
            self.paths.inspection_results,
            self.paths.decision_results,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.branch = self.temporary / "model-api-gguf"
        (self.branch / "llama.cpp/.git/refs/heads").mkdir(
            parents=True, mode=0o700
        )
        (self.branch / "llama.cpp/.git/HEAD").write_text(
            "ref: refs/heads/main\n", encoding="utf-8"
        )
        (self.branch / "llama.cpp/.git/refs/heads/main").write_text(
            "1" * 40 + "\n", encoding="utf-8"
        )
        (self.branch / "fixture").write_bytes(b"x")
        atomic_write_json(
            self.paths.status / "current.json",
            {
                "schema_version": SCHEMA_IDENTITIES["status"],
                "state": "IDLE",
                "reason_code": "OK",
                "updated_utc": utc_now(),
                "inspector_root": str(self.paths.inspector_root),
                "active_transaction_id": None,
                "last_transaction_id": None,
            },
            mode=0o600,
        )
        self.accepted_artifact = sha(b"accepted-artifact")
        self.gguf = self.gguf_record()
        self.native = self.native_record()
        publish_capability_record(self.paths, self.gguf)
        publish_capability_record(self.paths, self.native)
        publish_binding(
            self.paths,
            build_binding(
                self.gguf,
                binding_generation=1,
                updated_utc="2026-01-01T00:00:00Z",
            ),
        )
        publish_binding(
            self.paths,
            build_binding(
                self.native,
                binding_generation=1,
                updated_utc="2026-01-01T00:00:00Z",
            ),
        )
        self.sequence = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def installed_tuple(self) -> dict[str, object]:
        row = {
            "name": "fixture",
            "root": "branch",
            "path": "fixture",
            "byte_count": 1,
            "sha256": sha(b"x"),
        }
        basis = [
            {
                "root": row["root"],
                "path": row["path"],
                "byte_count": row["byte_count"],
                "sha256": row["sha256"],
            }
        ]
        return {
            "source_commit": "1" * 40,
            "accepted_tag": "fixture",
            "clean_worktree_required": True,
            "components": [row],
            "manifests": [
                {
                    "name": "fixture",
                    "identity": sha(canonical_json_bytes(basis)),
                    "file_count": 1,
                    "byte_count": 1,
                    "files": [row],
                }
            ],
            "platform_registration": {
                "schema_version": "fixture.registration.v1",
                "adapter_identity": "fixture.adapter.v1",
                "registered": True,
                "enabled": True,
            },
        }

    def gguf_record(self) -> dict[str, object]:
        return build_capability_record(
            created_utc="2026-01-01T00:00:00Z",
            branch_identity="model-api-gguf",
            supported_physical_format="GGUF",
            availability="AVAILABLE",
            runtime_engine="llama-server",
            installed_tuple=self.installed_tuple(),
            accepted_evidence=[
                {"basename": "accepted.rs", "sha256": sha(b"evidence")}
            ],
            supported_evidence={
                "supported_exact_artifact_identities": [
                    self.accepted_artifact
                ],
                "accepted_format_versions": [3],
                "accepted_architectures": ["qwen35"],
                "accepted_primary_model_types": ["model"],
                "accepted_modalities": ["text"],
                "accepted_tensor_type_evidence": ["BF16", "F32"],
                "accepted_tokenizer_evidence": sha(b"tokens"),
                "accepted_chat_template_evidence": sha(b"template"),
                "accepted_runtime_capabilities": ["generate/chat"],
                "public_model_id": "fixture",
                "accepted_capability_manifest_identity": sha(b"manifest"),
            },
            unsupported_primary_artifact_roles=["adapter"],
            unproven_valid_policy="RUNTIME_SMOKE_REQUIRED",
            reason_code=None,
        )

    def native_record(self) -> dict[str, object]:
        return build_capability_record(
            created_utc="2026-01-01T00:00:00Z",
            branch_identity="model-api-native",
            supported_physical_format="NATIVE",
            availability="UNAVAILABLE",
            runtime_engine="vLLM",
            installed_tuple=None,
            accepted_evidence=[],
            supported_evidence={"supported_artifact_identities": []},
            unsupported_primary_artifact_roles=[],
            unproven_valid_policy=None,
            reason_code="NATIVE_BRANCH_ACCEPTANCE_NOT_CLOSED",
        )

    def write_inspection(
        self,
        terminal: str,
        *,
        artifact_identity: str | None = None,
        model_type: str | None = None,
        misleading_extension: bool = False,
    ) -> str:
        self.sequence += 1
        inspection_id = (
            f"inspection-20260101T000000{self.sequence:06d}Z-"
            f"{self.sequence:016x}"
        )
        transaction_id = f"tx-fixture-{self.sequence}"
        artifact = artifact_identity or sha(f"artifact-{self.sequence}".encode())
        if terminal == "GGUF":
            physical = "GGUF"
            detected = "GGUF"
            actual_model_type = model_type or "model"
            gguf = {
                "architecture": "qwen35",
                "model_type": actual_model_type,
                "tensor_type_histogram": {"BF16": 1, "F32": 1},
                "tokenizer_token_identity": sha(b"tokens"),
                "chat_template_identity": sha(b"template"),
            }
            native = None
            version = 3
            reasons = [
                "INSPECTION_COMPLETE",
                "GGUF_MAGIC_CONFIRMED",
                "GGUF_STRUCTURE_VALID",
            ]
            architectures = ["qwen35"]
            modalities = ["text"]
        elif terminal == "NATIVE":
            physical = "NATIVE"
            detected = "NATIVE"
            actual_model_type = model_type or "bounded"
            gguf = None
            native = {"model_type": actual_model_type}
            version = "safetensors.v1"
            reasons = [
                "INSPECTION_COMPLETE",
                "NATIVE_CONFIG_VALID",
                "NATIVE_WEIGHT_LAYOUT_VALID",
            ]
            architectures = ["BoundedForCausalLM"]
            modalities = ["text"]
        else:
            physical = (
                "unknown"
                if terminal == "UNKNOWN"
                else "contradictory"
                if terminal == "CONTRADICTORY"
                else "GGUF"
            )
            detected = (
                "UNKNOWN"
                if terminal == "UNKNOWN"
                else "MIXED"
                if terminal == "CONTRADICTORY"
                else "GGUF"
            )
            actual_model_type = None
            gguf = None
            native = None
            version = None
            reasons = [
                {
                    "UNKNOWN": "FORMAT_EVIDENCE_UNKNOWN",
                    "CONTRADICTORY": "FORMAT_EVIDENCE_CONTRADICTORY",
                    "CORRUPT": "GGUF_METADATA_INVALID",
                    "INCOMPLETE": "GGUF_HEADER_TRUNCATED",
                }[terminal]
            ]
            if misleading_extension:
                reasons.append("MISLEADING_EXTENSION_IGNORED")
            architectures = []
            modalities = ["unknown"]
        name = "fixture.gguf" if misleading_extension else "fixture"
        identity_value = sha(b"snapshot")
        record = {
            "schema_version": SCHEMA_IDENTITIES["inspection_result"],
            "inspection_id": inspection_id,
            "transaction_id": transaction_id,
            "created_utc": "2026-01-01T00:00:00Z",
            "source": {
                "candidate_name": name,
                "candidate_kind": "regular_file",
                "relative_path": name,
                "realpath": str(self.root / "MODEL-TEST" / name),
                "device": 1,
                "inode": self.sequence,
                "intake_snapshot_identity": identity_value,
                "pre_inspection_snapshot_identity": identity_value,
                "post_inspection_snapshot_identity": identity_value,
            },
            "artifact": {
                "identity": artifact,
                "algorithm": "sha256",
                "byte_count": 1,
                "file_count": 1,
                "content_manifest_identity": artifact,
                "files": [
                    {
                        "relative_path": name,
                        "byte_count": 1,
                        "sha256": artifact.removeprefix("sha256:"),
                    }
                ],
                "file_manifest_truncated": False,
            },
            "classification": {
                "terminal_class": terminal,
                "detected_family": detected,
                "inspection_confidence": "high",
                "reason_codes": reasons,
            },
            "format": {
                "definition_identity": sha(b"definition"),
                "endianness": "little" if terminal == "GGUF" else None,
                "gguf": gguf,
                "native": native,
                "version": version,
            },
            "normalized": {
                "physical_format": physical,
                "format_version": version,
                "model_type": actual_model_type,
                "architectures": architectures,
                "task": "unknown",
                "modalities": modalities,
                "quantization": None,
                "configuration_source": None,
                "tokenizer_source": [],
                "weight_format": None,
                "shard_count": None,
                "artifact_size": 1,
                "artifact_identity": artifact,
                "inspection_confidence": "high",
                "reason_codes": reasons,
            },
            "evidence": [],
            "warnings": [],
        }
        path = self.paths.inspection_results / f"{inspection_id}.json"
        result_identity = atomic_create_json(path, record, mode=0o600)
        atomic_write_json(
            self.paths.transactions / f"{transaction_id}.json",
            {
                "schema_version": SCHEMA_IDENTITIES["transaction"],
                "transaction_id": transaction_id,
                "operation": "inspect",
                "state": "COMPLETED",
                "reason_code": "INSPECTION_COMPLETE",
                "inspection_result_identity": result_identity,
                "inspection_result_path": str(path),
                "artifact_identity": artifact,
                "terminal_class": terminal,
            },
            mode=0o600,
        )
        return inspection_id

    @staticmethod
    def verified(*_args):
        return {"applicable": True, "verified": True, "mismatches": []}

    def run_main(self, *arguments: str) -> tuple[int, dict[str, object]]:
        output = io.StringIO()
        error = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(
            error
        ):
            exit_status = main(
                ["--inspector-root", str(self.root), *arguments]
            )
        self.assertEqual(error.getvalue(), "")
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        return exit_status, json.loads(lines[0])

    def test_exact_classification_precedence(self) -> None:
        cases = [
            (
                self.write_inspection(
                    "GGUF", artifact_identity=self.accepted_artifact
                ),
                "SUPPORTED",
                "model-api-gguf",
                True,
            ),
            (
                self.write_inspection("GGUF"),
                "RUNTIME_SMOKE_REQUIRED",
                None,
                False,
            ),
            (
                self.write_inspection("GGUF", model_type="adapter"),
                "UNSUPPORTED",
                None,
                False,
            ),
            (
                self.write_inspection("NATIVE"),
                "UNAVAILABLE",
                None,
                False,
            ),
        ]
        for inspection_id, result, branch, allowed in cases:
            with self.subTest(result=result):
                outcome = resolve_decision(
                    self.paths,
                    inspection_id,
                    installed_tuple_verifier=self.verified,
                )
                self.assertEqual(
                    outcome["capability"]["capability_result"], result
                )
                self.assertEqual(outcome["selected_branch"], branch)
                self.assertEqual(outcome["handoff_allowed"], allowed)
                self.assertEqual(outcome["spawn_allowed"], allowed)
        self.assertEqual(list(self.paths.decision_results.iterdir()), [])

    def test_invalid_classes_bypass_capability(self) -> None:
        for terminal in (
            "UNKNOWN",
            "CONTRADICTORY",
            "CORRUPT",
            "INCOMPLETE",
        ):
            with self.subTest(terminal=terminal):
                inspection_id = self.write_inspection(
                    terminal,
                    misleading_extension=terminal == "UNKNOWN",
                )
                outcome = resolve_decision(self.paths, inspection_id)
                self.assertFalse(outcome["capability"]["evaluated"])
                self.assertIsNone(
                    outcome["capability"]["capability_result"]
                )
                self.assertFalse(outcome["spawn_allowed"])
                if terminal == "UNKNOWN":
                    self.assertIn(
                        "EXTENSION_ONLY_EVIDENCE_IGNORED",
                        outcome["reason_codes"],
                    )

    def test_multiple_binding_and_tuple_mismatch_are_ambiguous(self) -> None:
        inspection_id = self.write_inspection("GGUF")
        canonical = self.paths.capability_bindings / "model-api-gguf.json"
        duplicate = self.paths.capability_bindings / "duplicate.json"
        duplicate.write_bytes(canonical.read_bytes())
        os.chmod(duplicate, 0o600)
        outcome = resolve_decision(
            self.paths,
            inspection_id,
            installed_tuple_verifier=self.verified,
        )
        self.assertEqual(
            outcome["capability"]["capability_result"], "AMBIGUOUS"
        )
        duplicate.unlink()
        mismatch = resolve_decision(
            self.paths,
            inspection_id,
            installed_tuple_verifier=lambda *_: {
                "applicable": True,
                "verified": False,
                "mismatches": [{"field": "fixture"}],
            },
        )
        self.assertEqual(
            mismatch["capability"]["capability_result"], "AMBIGUOUS"
        )
        self.assertEqual(
            mismatch["reason_code"],
            "CAPABILITY_INSTALLED_TUPLE_MISMATCH",
        )

    def test_tampered_binding_and_record_fail_closed(self) -> None:
        inspection_id = self.write_inspection("GGUF")
        binding_path = (
            self.paths.capability_bindings / "model-api-gguf.json"
        )
        original = binding_path.read_bytes()
        value = json.loads(original)
        value["binding_generation"] = 2
        binding_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        os.chmod(binding_path, 0o600)
        with self.assertRaises(InspectorError) as caught:
            resolve_decision(
                self.paths,
                inspection_id,
                installed_tuple_verifier=self.verified,
            )
        self.assertEqual(
            caught.exception.reason_code, "CAPABILITY_BINDING_INVALID"
        )
        binding_path.write_bytes(original)
        record_path = self.paths.capability_records / (
            f"{self.gguf['capability_record_id']}.json"
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["runtime_engine"] = "changed"
        record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        os.chmod(record_path, 0o600)
        with self.assertRaises(InspectorError):
            resolve_decision(
                self.paths,
                inspection_id,
                installed_tuple_verifier=self.verified,
            )

    def test_decision_basis_is_stable(self) -> None:
        inspection_id = self.write_inspection(
            "GGUF", artifact_identity=self.accepted_artifact
        )
        first = resolve_decision(
            self.paths,
            inspection_id,
            installed_tuple_verifier=self.verified,
        )
        second = resolve_decision(
            self.paths,
            inspection_id,
            installed_tuple_verifier=self.verified,
        )
        self.assertEqual(
            first["decision_basis_identity"],
            second["decision_basis_identity"],
        )

    def test_machine_capabilities_read_only_and_decide_persists(self) -> None:
        before = sorted(self.paths.transactions.iterdir())
        exit_status, inventory = self.run_main("capabilities")
        self.assertEqual(exit_status, 0)
        self.assertTrue(inventory["ok"])
        self.assertEqual(len(inventory["data"]["bindings"]), 2)
        self.assertEqual(sorted(self.paths.transactions.iterdir()), before)
        inspection_id = self.write_inspection(
            "GGUF", artifact_identity=self.accepted_artifact
        )
        exit_status, envelope = self.run_main(
            "decide", "--inspection-id", inspection_id
        )
        self.assertEqual(exit_status, 0)
        self.assertTrue(envelope["ok"])
        self.assertEqual(
            envelope["data"]["capability_result"], "SUPPORTED"
        )
        self.assertEqual(
            envelope["data"]["selected_branch"], "model-api-gguf"
        )
        result = read_json_record(Path(envelope["paths"]["result"]))
        validate_decision_record(result)
        self.assertEqual(result["result_identity"], envelope["data"]["result_identity"])
        transaction = read_json_record(Path(envelope["paths"]["transaction"]))
        self.assertEqual(transaction["state"], "COMPLETED")
        self.assertEqual(
            transaction["decision_result_identity"],
            result["result_identity"],
        )
        status = read_json_record(self.paths.status / "current.json")
        self.assertEqual(status["state"], "IDLE")
        self.assertIsNone(status["active_transaction_id"])
        self.assertFalse((self.paths.locks / "active.json").exists())

    def test_machine_unsafe_input_exit_two_and_internal_exit_seventy(self) -> None:
        exit_status, envelope = self.run_main(
            "decide", "--inspection-id", "../unsafe"
        )
        self.assertEqual(exit_status, 2)
        self.assertFalse(envelope["ok"])
        self.assertEqual(
            envelope["reason_code"], "INSPECTION_RECORD_INVALID"
        )
        transaction = read_json_record(
            self.paths.transactions / f"{envelope['transaction_id']}.json"
        )
        self.assertEqual(transaction["state"], "FAILED")
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )
        inspection_id = self.write_inspection("UNKNOWN")

        def fail(*_args):
            raise RuntimeError("bounded internal fixture")

        with self.assertRaises(InspectorError) as caught:
            decide_transaction(
                self.paths, inspection_id, resolver=fail
            )
        self.assertEqual(caught.exception.exit_status, 70)
        self.assertEqual(caught.exception.reason_code, "INTERNAL_ERROR")
        self.assertEqual(
            read_json_record(self.paths.status / "current.json")["state"],
            "IDLE",
        )
        self.assertFalse((self.paths.locks / "active.json").exists())

    def test_immutable_decision_publication_and_collision(self) -> None:
        inspection_id = self.write_inspection(
            "GGUF", artifact_identity=self.accepted_artifact
        )
        outcome = resolve_decision(
            self.paths,
            inspection_id,
            installed_tuple_verifier=self.verified,
        )
        decision_id = "decision-20260101T000000000000Z-" + "a" * 16
        first = build_decision_record(
            outcome,
            decision_id=decision_id,
            transaction_id="tx-decision-first",
            decision_timestamp_utc="2026-01-01T00:00:00Z",
        )
        path = publish_decision_record(self.paths, first)
        self.assertEqual(publish_decision_record(self.paths, first), path)
        second = build_decision_record(
            outcome,
            decision_id=decision_id,
            transaction_id="tx-decision-second",
            decision_timestamp_utc="2026-01-01T00:00:01Z",
        )
        with self.assertRaises(InspectorError) as caught:
            publish_decision_record(self.paths, second)
        self.assertEqual(
            caught.exception.reason_code, "DECISION_RECORD_COLLISION"
        )
        self.assertEqual(read_json_record(path), first)


if __name__ == "__main__":
    unittest.main()
