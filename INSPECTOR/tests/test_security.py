from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

from system_x_inspector.constants import (
    OPERATIONS,
    REASON_CODES,
    SCHEMA_IDENTITIES,
)


class SecurityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parent.parent
        cls.package = cls.root / "system_x_inspector"
        cls.sources = sorted(cls.package.glob("*.py"))
        cls.schemas = sorted((cls.root / "schemas").glob("*.json"))

    def test_all_python_and_json_files_parse(self) -> None:
        for path in sorted(
            [
                *self.sources,
                *(self.root / "tests").glob("*.py"),
            ]
        ):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in self.schemas:
            with self.subTest(path=path.name):
                self.assertIsInstance(
                    json.loads(path.read_text(encoding="utf-8")), dict
                )

    def test_schema_and_source_identities_match(self) -> None:
        observed = {
            json.loads(path.read_text(encoding="utf-8"))["$id"]
            for path in self.schemas
        }
        self.assertEqual(
            observed,
            {
                SCHEMA_IDENTITIES["configuration"],
                SCHEMA_IDENTITIES["machine_result"],
                SCHEMA_IDENTITIES["automatic_intake_result"],
                SCHEMA_IDENTITIES["automatic_intake_basis"],
                SCHEMA_IDENTITIES["status"],
                SCHEMA_IDENTITIES["transaction"],
                SCHEMA_IDENTITIES["intake_candidate"],
                SCHEMA_IDENTITIES["inspection_result"],
                SCHEMA_IDENTITIES["branch_capability"],
                SCHEMA_IDENTITIES["capability_binding"],
                SCHEMA_IDENTITIES["branch_decision"],
                SCHEMA_IDENTITIES["handoff_result"],
                SCHEMA_IDENTITIES["service_publication"],
                SCHEMA_IDENTITIES["gguf_qualification_result"],
                SCHEMA_IDENTITIES["gguf_promotion_result"],
                SCHEMA_IDENTITIES["gguf_retirement_result"],
                SCHEMA_IDENTITIES["gguf_deployment_result"],
                SCHEMA_IDENTITIES["api_connection_receipt"],
            },
        )
        machine_schema = json.loads(
            (self.root / "schemas/machine-result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        schema_operations = set(
            machine_schema["properties"]["operation"]["enum"]
        )
        self.assertEqual(schema_operations - {"unknown"}, set(OPERATIONS))

    def test_import_graph_is_standard_library_only(self) -> None:
        imported = set()
        for path in self.sources:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    if node.module:
                        imported.add(node.module.split(".", 1)[0])
        allowed = set(sys.stdlib_module_names) | {"system_x_inspector"}
        self.assertFalse(imported - allowed)
        self.assertFalse(
            imported
            & {
                "torch",
                "transformers",
                "vllm",
                "system_x_gguf_api",
                "branch_controller",
                "api_service_controller",
            }
        )

    def test_no_network_process_control_or_server_code(self) -> None:
        forbidden_imports = {
            "subprocess",
            "socket",
            "http",
            "urllib",
            "requests",
            "ftplib",
            "asyncio",
        }
        forbidden_calls = {
            "Popen",
            "run",
            "call",
            "fork",
            "forkpty",
            "kill",
            "killpg",
            "system",
            "execv",
            "execve",
            "spawnv",
            "socket",
            "bind",
            "listen",
            "connect",
        }
        for path in self.sources:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = set()
            called = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    if node.module:
                        imported.add(node.module.split(".", 1)[0])
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called.add(node.func.attr)
            with self.subTest(path=path.name):
                if path.name in {
                    "service_publication.py",
                    "qualification.py",
                    "promotion.py",
                    "retirement.py",
                }:
                    if path.name == "service_publication.py":
                        permitted_imports = {"http"}
                        permitted_calls = {"connect"}
                    elif path.name == "qualification.py":
                        permitted_imports = {
                            "http",
                            "subprocess",
                            "urllib",
                        }
                        permitted_calls = {"connect", "run"}
                    elif path.name == "promotion.py":
                        permitted_imports = {"subprocess"}
                        permitted_calls = {"connect", "run"}
                    else:
                        permitted_imports = {"http", "subprocess"}
                        permitted_calls = {"connect", "run"}
                    self.assertFalse(
                        imported
                        & (
                            forbidden_imports
                            - permitted_imports
                        )
                    )
                    self.assertFalse(
                        called & (forbidden_calls - permitted_calls)
                    )
                else:
                    self.assertFalse(imported & forbidden_imports)
                    self.assertFalse(called & forbidden_calls)

    def test_no_hardcoded_host_gpu_port_or_model_identity(self) -> None:
        forbidden_literals = (
            "/home/user",
            "POWER-HOUSE",
            "Ubuntu",
            "WSL",
            "56259",
            "54037",
            "CUDA",
            "NVIDIA",
        )
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in self.sources
        )
        for value in forbidden_literals:
            with self.subTest(value=value):
                self.assertNotIn(value, combined)

    def test_intake_source_has_no_content_reader_or_format_parser(self) -> None:
        path = self.package / "intake.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden_calls = {
            "open",
            "read",
            "read1",
            "read_bytes",
            "read_text",
        }
        observed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    observed.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    observed.add(node.func.attr)
        self.assertFalse(observed & forbidden_calls)
        combined = path.read_text(encoding="utf-8")
        for value in (
            "magic",
            "model_type",
            "architecture",
            "quantization",
            "tokenizer",
            "safetensors",
            "pytorch",
        ):
            self.assertNotIn(value, combined.casefold())

    def test_capability_and_decision_surfaces_are_authorization_only(self) -> None:
        selected = [
            self.package / "capabilities.py",
            self.package / "decision.py",
        ]
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in selected
        ).casefold()
        for value in (
            "subprocess",
            "socket",
            "service start",
            "service stop",
            "model transfer",
            "registry write",
            "gpu probe",
            "perform inference",
        ):
            self.assertNotIn(value, combined)
        expected = {
            "GGUF_ACCEPTED_CAPABILITY_MATCH",
            "GGUF_RUNTIME_SMOKE_REQUIRED",
            "GGUF_REQUIREMENT_UNSUPPORTED",
            "NATIVE_BRANCH_UNAVAILABLE",
            "EXTENSION_ONLY_EVIDENCE_IGNORED",
        }
        self.assertTrue(expected <= REASON_CODES)
        decision_schema = json.loads(
            (self.root / "schemas/branch-decision.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            {
                "selected_branch",
                "handoff_allowed",
                "spawn_allowed",
            }
            <= set(decision_schema["properties"])
        )

    def test_handoff_surface_is_filesystem_only_and_closed(self) -> None:
        path = self.package / "handoff.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module:
                    imported.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
        self.assertFalse(
            imported
            & {
                "sqlite3",
                "subprocess",
                "socket",
                "http",
                "urllib",
                "requests",
                "system_x_gguf_api",
                "model_registry",
                "registry_store",
            }
        )
        self.assertFalse(
            called
            & {
                "Popen",
                "run",
                "call",
                "kill",
                "killpg",
                "system",
                "connect",
                "request",
            }
        )
        folded = source.casefold()
        for value in (
            "model_registry.sqlite3",
            "system_x_gguf_api",
            "registry_store",
        ):
            with self.subTest(value=value):
                self.assertNotIn(value, folded)
        expected_reasons = {
            "HANDOFF_COMPLETE",
            "HANDOFF_DECISION_NOT_FOUND",
            "HANDOFF_DECISION_INVALID",
            "HANDOFF_DECISION_NOT_SUPPORTED",
            "HANDOFF_BRANCH_NOT_SELECTED",
            "HANDOFF_NOT_AUTHORIZED",
            "HANDOFF_DECISION_STALE",
            "HANDOFF_CAPABILITY_BINDING_INVALID",
            "HANDOFF_INSTALLED_TUPLE_MISMATCH",
            "HANDOFF_SOURCE_NOT_FOUND",
            "HANDOFF_SOURCE_INVALID",
            "HANDOFF_SOURCE_SYMLINK",
            "HANDOFF_SOURCE_HARDLINK_REJECTED",
            "HANDOFF_SOURCE_IDENTITY_MISMATCH",
            "HANDOFF_SOURCE_CHANGED",
            "HANDOFF_TARGET_NAME_INVALID",
            "HANDOFF_TARGET_COLLISION",
            "HANDOFF_REGISTRY_LOCATION_COLLISION",
            "HANDOFF_STAGING_INVALID",
            "HANDOFF_STAGING_COLLISION",
            "HANDOFF_INSUFFICIENT_STORAGE",
            "HANDOFF_COPY_FAILED",
            "HANDOFF_STAGED_IDENTITY_MISMATCH",
            "HANDOFF_PUBLICATION_CONFLICT",
            "HANDOFF_PUBLICATION_FAILED",
            "HANDOFF_RESULT_COLLISION",
            "HANDOFF_INTERNAL_ERROR",
        }
        self.assertTrue(expected_reasons <= REASON_CODES)
        status_schema = json.loads(
            (self.root / "schemas/status.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            {
                "VALIDATING_HANDOFF",
                "STAGING_ARTIFACT",
                "VERIFYING_STAGED_ARTIFACT",
                "PUBLISHING_ARTIFACT",
            }
            <= set(status_schema["properties"]["state"]["enum"])
        )


if __name__ == "__main__":
    unittest.main()
