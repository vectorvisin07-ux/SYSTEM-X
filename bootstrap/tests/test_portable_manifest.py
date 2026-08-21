from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BOOTSTRAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOTSTRAP))

from system_x_bootstrap.portable_manifest import (  # noqa: E402
    MANIFEST_PATH,
    ManifestError,
    build_manifest,
    validate_manifest,
)


class PortableManifestTests(unittest.TestCase):
    def make_tree(self) -> tuple[Path, list[dict[str, str]]]:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "bootstrap").mkdir()
        (root / "README.md").write_text("portable\n", encoding="utf-8")
        (root / "bootstrap" / "run.py").write_text("print('ok')\n", encoding="utf-8")
        (root / MANIFEST_PATH).write_text("{}\n", encoding="utf-8")
        records = [
            {"path": MANIFEST_PATH, "git_mode": "100644", "git_blob": "a" * 40},
            {"path": "README.md", "git_mode": "100644", "git_blob": "b" * 40},
            {"path": "bootstrap/run.py", "git_mode": "100755", "git_blob": "c" * 40},
        ]
        (root / "bootstrap" / "run.py").chmod(0o755)
        return root, records

    def test_full_tree_build_and_validation_excludes_self(self) -> None:
        root, records = self.make_tree()
        manifest = build_manifest(root, records)
        self.assertEqual(manifest["semantics"], "FULL_TREE")
        self.assertEqual(manifest["source_root"], ".")
        self.assertEqual(manifest["entry_count"], 2)
        self.assertEqual([entry["path"] for entry in manifest["entries"]], ["README.md", "bootstrap/run.py"])
        self.assertEqual(validate_manifest(root, manifest, records)["status"], "PASS")

    def test_rejects_self_entry(self) -> None:
        root, records = self.make_tree()
        manifest = build_manifest(root, records)
        manifest["entries"].append({"path": MANIFEST_PATH})
        manifest["entries"].sort(key=lambda entry: entry["path"])
        with self.assertRaises(ManifestError):
            validate_manifest(root, manifest, records)

    def test_rejects_traversal(self) -> None:
        root, records = self.make_tree()
        manifest = build_manifest(root, records)
        manifest["entries"][0]["path"] = "../outside"
        with self.assertRaises(ManifestError):
            validate_manifest(root, manifest, records)

    def test_rejects_member_digest_drift(self) -> None:
        root, records = self.make_tree()
        manifest = build_manifest(root, records)
        manifest["entries"][0]["sha256"] = "0" * 64
        with self.assertRaises(ManifestError):
            validate_manifest(root, manifest, records)

    def test_json_round_trip_preserves_explicit_contract(self) -> None:
        root, records = self.make_tree()
        manifest = build_manifest(root, records)
        encoded = json.dumps(manifest, sort_keys=True)
        self.assertEqual(json.loads(encoded)["self_exclusion"]["path"], MANIFEST_PATH)


if __name__ == "__main__":
    unittest.main()
