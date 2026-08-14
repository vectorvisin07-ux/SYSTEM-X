from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import sys
import unittest
from pathlib import Path


class SourceGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parent.parent
        cls.package_sources = sorted(
            (cls.root / "system_x_inspector").glob("*.py")
        )
        cls.all_sources = sorted(
            [
                *cls.package_sources,
                *(cls.root / "schemas").glob("*.json"),
                *(cls.root / "tests").glob("*.py"),
            ],
            key=lambda path: path.relative_to(cls.root).as_posix(),
        )

    def test_environment_has_no_third_party_distributions(self) -> None:
        distributions = sorted(
            distribution.metadata.get("Name")
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        )
        self.assertEqual(distributions, [])
        self.assertTrue(sys.flags.no_user_site)

    def test_package_imports_are_standard_library_only(self) -> None:
        imported: set[str] = set()
        for path in self.package_sources:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    imported.add(node.module.split(".", 1)[0])
        allowed = set(sys.stdlib_module_names) | {"system_x_inspector"}
        self.assertFalse(imported - allowed)
        self.assertFalse(
            imported
            & {
                "torch",
                "transformers",
                "vllm",
                "safetensors",
                "requests",
                "system_x_gguf_api",
            }
        )

    def test_no_deserialization_or_model_execution_calls(self) -> None:
        forbidden_qualified_calls = {
            "pickle.load",
            "pickle.loads",
            "torch.load",
            "torch.jit.load",
        }
        forbidden_attributes = {
            "from_pretrained",
            "load_model",
            "generate",
            "inference",
        }
        observed_qualified: set[str] = set()
        observed_attributes: set[str] = set()
        for path in self.package_sources:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    observed_attributes.add(node.func.attr)
                    if isinstance(node.func.value, ast.Name):
                        observed_qualified.add(
                            f"{node.func.value.id}.{node.func.attr}"
                        )
                    elif (
                        isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                    ):
                        observed_qualified.add(
                            f"{node.func.value.value.id}."
                            f"{node.func.value.attr}.{node.func.attr}"
                        )
        self.assertFalse(observed_qualified & forbidden_qualified_calls)
        self.assertFalse(observed_attributes & forbidden_attributes)

    def test_json_schemas_are_closed_and_parseable(self) -> None:
        for path in sorted((self.root / "schemas").glob("*.json")):
            with self.subTest(path=path.name):
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsInstance(value, dict)
                self.assertFalse(value["additionalProperties"])
        inspection = json.loads(
            (self.root / "schemas/inspection-result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        closed = {
            "source",
            "artifact",
            "classification",
            "format",
            "sourceDescriptor",
            "ggufSummary",
            "nativeSummary",
            "normalized",
            "evidence",
        }
        self.assertTrue(
            all(
                inspection["$defs"][name]["additionalProperties"] is False
                for name in closed
            )
        )

    def test_source_freeze_manifest_is_deterministic(self) -> None:
        self.assertTrue((self.root / "environment.lock.json").is_file())
        self.assertTrue(all(path.is_file() for path in self.all_sources))
        manifest = [
            {
                "path": path.relative_to(self.root).as_posix(),
                "byte_count": len(path.read_bytes()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in self.all_sources
        ]
        encoded = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        identity = "sha256:" + hashlib.sha256(encoded).hexdigest()
        repeated = "sha256:" + hashlib.sha256(encoded).hexdigest()
        self.assertEqual(identity, repeated)
        self.assertRegex(identity, r"^sha256:[0-9a-f]{64}$")
        print(
            json.dumps(
                {
                    "source_freeze_file_count": len(manifest),
                    "source_freeze_byte_count": sum(
                        item["byte_count"] for item in manifest
                    ),
                    "source_freeze_identity": identity,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
