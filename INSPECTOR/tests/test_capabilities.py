from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from system_x_inspector.capabilities import (
    build_binding,
    build_capability_record,
    capability_identity,
    initialize_capability_store,
    load_binding,
    load_capability_record,
    publish_binding,
    publish_capability_record,
    validate_binding,
    validate_capability_record,
    verify_installed_tuple,
)
from system_x_inspector.errors import InspectorError
from system_x_inspector.paths import InspectorPaths
from system_x_inspector.records import canonical_json_bytes


def sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class CapabilityStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-capability-", dir="/tmp")
        )
        self.root = self.temporary / "INSPECTOR"
        self.root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.root)
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
        self.component_path = self.branch / "component.bin"
        self.component_path.write_bytes(b"accepted-component")
        self.record = self.gguf_record()

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def component(self) -> dict[str, object]:
        data = self.component_path.read_bytes()
        return {
            "name": "component",
            "root": "branch",
            "path": "component.bin",
            "byte_count": len(data),
            "sha256": sha(data),
        }

    def manifest(self) -> dict[str, object]:
        component = self.component()
        basis = [
            {
                "root": component["root"],
                "path": component["path"],
                "byte_count": component["byte_count"],
                "sha256": component["sha256"],
            }
        ]
        return {
            "name": "fixture_manifest",
            "identity": sha(canonical_json_bytes(basis)),
            "file_count": 1,
            "byte_count": component["byte_count"],
            "files": [component],
        }

    def installed_tuple(self) -> dict[str, object]:
        return {
            "source_commit": "1" * 40,
            "accepted_tag": "fixture-tag",
            "clean_worktree_required": True,
            "components": [self.component()],
            "manifests": [self.manifest()],
            "platform_registration": {
                "schema_version": "fixture.registration.v1",
                "adapter_identity": "fixture.adapter.v1",
                "registered": True,
                "enabled": True,
            },
        }

    def gguf_record(
        self,
        *,
        created: str = "2026-01-01T00:00:00Z",
        exact_artifact_identities: list[str] | None = None,
    ):
        return build_capability_record(
            created_utc=created,
            branch_identity="model-api-gguf",
            supported_physical_format="GGUF",
            availability="AVAILABLE",
            runtime_engine="llama-server",
            installed_tuple=self.installed_tuple(),
            accepted_evidence=[
                {"basename": "accepted.rs", "sha256": sha(b"evidence")}
            ],
            supported_evidence={
                "supported_exact_artifact_identities": (
                    [sha(b"model")]
                    if exact_artifact_identities is None
                    else exact_artifact_identities
                ),
                "accepted_format_versions": [3],
                "accepted_architectures": ["qwen35"],
                "accepted_primary_model_types": ["model"],
                "accepted_modalities": ["text"],
                "accepted_tensor_type_evidence": ["BF16", "F32"],
                "accepted_tokenizer_evidence": sha(b"tokens"),
                "accepted_chat_template_evidence": sha(b"template"),
                "accepted_runtime_capabilities": ["generate/chat"],
                "public_model_id": "fixture-model",
                "accepted_capability_manifest_identity": sha(b"manifest"),
            },
            unsupported_primary_artifact_roles=["adapter"],
            unproven_valid_policy="RUNTIME_SMOKE_REQUIRED",
            reason_code=None,
        )

    def test_store_paths_modes_and_unknown_collision(self) -> None:
        initialize_capability_store(self.paths)
        for path in (
            self.paths.capability_root,
            self.paths.capability_records,
            self.paths.capability_bindings,
        ):
            self.assertTrue(path.is_dir())
            self.assertFalse(path.is_symlink())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        other = self.temporary / "OTHER"
        other.mkdir(mode=0o700)
        other_paths = InspectorPaths.discover(other)
        other_paths.capability_root.mkdir(mode=0o700)
        (other_paths.capability_root / "unknown").write_text(
            "preserve", encoding="utf-8"
        )
        with self.assertRaises(InspectorError):
            initialize_capability_store(other_paths)
        self.assertEqual(
            (other_paths.capability_root / "unknown").read_text(
                encoding="utf-8"
            ),
            "preserve",
        )

    def test_cold_install_allows_no_previously_supported_artifact(self) -> None:
        cold = self.gguf_record(exact_artifact_identities=[])
        self.assertEqual(
            cold["supported_evidence"][
                "supported_exact_artifact_identities"
            ],
            [],
        )
        self.assertEqual(validate_capability_record(cold), cold)

    def test_canonical_identity_and_no_pid_law(self) -> None:
        reordered = json.loads(json.dumps(self.record, sort_keys=True))
        self.assertEqual(
            capability_identity(self.record), capability_identity(reordered)
        )
        other_time = self.gguf_record(created="2030-02-03T04:05:06Z")
        self.assertEqual(
            self.record["capability_record_identity"],
            other_time["capability_record_identity"],
        )
        changed = copy.deepcopy(self.record)
        changed["installed_tuple"]["components"][0]["sha256"] = sha(
            b"changed"
        )
        with self.assertRaises(InspectorError):
            validate_capability_record(changed)
        prohibited = copy.deepcopy(self.record)
        prohibited["installed_tuple"]["current_pid"] = 1234
        with self.assertRaises(InspectorError) as caught:
            validate_capability_record(prohibited)
        self.assertEqual(
            caught.exception.reason_code, "CAPABILITY_RECORD_INVALID"
        )

    def test_immutable_record_and_binding_generation(self) -> None:
        path = publish_capability_record(self.paths, self.record)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(publish_capability_record(self.paths, self.record), path)
        loaded = load_capability_record(
            self.paths, self.record["capability_record_id"]
        )
        self.assertEqual(loaded, self.record)
        first = build_binding(
            self.record,
            binding_generation=1,
            updated_utc="2026-01-01T00:00:00Z",
        )
        binding_path = publish_binding(self.paths, first)
        self.assertEqual(stat.S_IMODE(binding_path.stat().st_mode), 0o600)
        second = build_binding(
            self.record,
            binding_generation=2,
            updated_utc="2026-01-01T00:00:01Z",
        )
        publish_binding(self.paths, second)
        self.assertEqual(
            load_binding(self.paths, "model-api-gguf"), second
        )
        skipped = build_binding(
            self.record,
            binding_generation=4,
            updated_utc="2026-01-01T00:00:02Z",
        )
        with self.assertRaises(InspectorError):
            publish_binding(self.paths, skipped)

    def test_record_and_binding_tamper_rejected(self) -> None:
        record_path = publish_capability_record(self.paths, self.record)
        tampered = json.loads(record_path.read_text(encoding="utf-8"))
        tampered["runtime_engine"] = "changed"
        record_path.write_text(
            json.dumps(tampered) + "\n", encoding="utf-8"
        )
        os.chmod(record_path, 0o600)
        with self.assertRaises(InspectorError):
            load_capability_record(
                self.paths, self.record["capability_record_id"]
            )
        shutil.rmtree(self.paths.capability_root)
        publish_capability_record(self.paths, self.record)
        binding = build_binding(
            self.record,
            binding_generation=1,
            updated_utc="2026-01-01T00:00:00Z",
        )
        path = publish_binding(self.paths, binding)
        tampered_binding = json.loads(path.read_text(encoding="utf-8"))
        tampered_binding["binding_generation"] = 2
        path.write_text(
            json.dumps(tampered_binding) + "\n", encoding="utf-8"
        )
        os.chmod(path, 0o600)
        with self.assertRaises(InspectorError):
            load_binding(self.paths, "model-api-gguf")

    def test_installed_tuple_verification_and_mismatch(self) -> None:
        first = verify_installed_tuple(
            self.paths, self.record, branch_root=self.branch
        )
        self.assertTrue(first["verified"])
        self.component_path.write_bytes(b"changed")
        second = verify_installed_tuple(
            self.paths, self.record, branch_root=self.branch
        )
        self.assertFalse(second["verified"])
        self.assertTrue(second["mismatches"])

    def test_installed_tuple_accepts_authenticated_vendored_source_identity(
        self,
    ) -> None:
        shutil.rmtree(self.branch / "llama.cpp/.git")
        (self.branch / "LLAMA_CPP_SOURCE_IDENTITY.json").write_text(
            json.dumps(
                {
                    "schema": "system-x.llama-cpp-source-identity.v1",
                    "origin": "https://github.com/ggml-org/llama.cpp",
                    "commit": "1" * 40,
                    "build_output_excluded": True,
                    "tracked_file_count": 1,
                    "complete_vendored_manifest_sha256": "2" * 64,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result = verify_installed_tuple(
            self.paths,
            self.record,
            branch_root=self.branch,
            user_config_root=self.temporary / "config",
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["source_commit"], "1" * 40)

    def test_verification_binds_installation_context_and_binding(self) -> None:
        bound_config = self.temporary / "bound-config"
        ambient_config = self.temporary / "ambient-config"
        paths = InspectorPaths.discover(
            self.root,
            explicit_user_config_root=bound_config,
        )
        binding = build_binding(
            self.record,
            binding_generation=1,
            updated_utc="2026-01-01T00:00:00Z",
        )
        publish_capability_record(paths, self.record)
        publish_binding(paths, binding)
        with patch.dict(os.environ, {"HOME": str(ambient_config)}):
            observed = verify_installed_tuple(
                paths,
                self.record,
                branch_root=self.branch,
                binding=binding,
            )
        context = observed["verification_context"]
        self.assertEqual(context["user_config_root"], str(bound_config))
        self.assertEqual(context["binding_generation"], 1)
        self.assertEqual(
            context["capability_record_identity"],
            self.record["capability_record_identity"],
        )
        self.assertEqual(
            context["pre_read_identities"],
            context["post_read_identities"],
        )
        self.assertTrue(observed["verified"])

        changed = build_binding(
            self.record,
            binding_generation=2,
            updated_utc="2026-01-01T00:00:01Z",
        )
        publish_binding(paths, changed)
        with self.assertRaises(InspectorError) as caught:
            verify_installed_tuple(
                paths,
                self.record,
                branch_root=self.branch,
                binding=binding,
            )
        self.assertEqual(
            caught.exception.data["verification_outcome"],
            "binding_context_changed",
        )

    def test_unexpected_component_observer_is_fail_closed_and_typed(self) -> None:
        with patch(
            "system_x_inspector.capabilities._observed_component",
            side_effect=RuntimeError("fixture-internal"),
        ):
            with self.assertRaises(InspectorError) as caught:
                verify_installed_tuple(
                    self.paths,
                    self.record,
                    branch_root=self.branch,
                )
        self.assertEqual(
            caught.exception.data["verification_outcome"],
            "unexpected_exception",
        )
        self.assertEqual(
            caught.exception.data["exception_class"],
            "builtins.RuntimeError",
        )

    def test_binding_identity_tamper(self) -> None:
        value = build_binding(
            self.record,
            binding_generation=1,
            updated_utc="2026-01-01T00:00:00Z",
        )
        value["binding_identity"] = sha(b"tampered")
        with self.assertRaises(InspectorError) as caught:
            validate_binding(value)
        self.assertEqual(
            caught.exception.reason_code, "CAPABILITY_BINDING_INVALID"
        )


if __name__ == "__main__":
    unittest.main()
