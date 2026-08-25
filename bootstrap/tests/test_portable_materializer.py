from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BOOTSTRAP = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(BOOTSTRAP))

from system_x_bootstrap import portable_materializer  # noqa: E402
from system_x_bootstrap.portable_materializer import (  # noqa: E402
    MANIFEST_PATH,
    MaterializationError,
    materialize_portable_tree,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def _umask(value: int):
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


class PortableMaterializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "source"
        self.root.mkdir()
        self._create_source()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_git(self, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(self.root), "--no-optional-locks", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _create_source(self) -> None:
        (self.root / "model-api-gguf" / "llama.cpp").mkdir(parents=True)
        files = {
            "ordinary.txt": (b"base ordinary\n", 0o664),
            "script.sh": (b"#!/bin/sh\nexit 0\n", 0o775),
            "private.key": (b"owner only\n", 0o600),
            "model-api-gguf/llama.cpp/README.md": (b"vendor source\n", 0o664),
        }
        for relative, (data, mode) in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(mode)
        (self.root / MANIFEST_PATH).write_text('{"entries":[]}\n', encoding="utf-8")
        self._run_git("init", "-q")
        self._run_git("config", "user.email", "materializer-test@example.invalid")
        self._run_git("config", "user.name", "Materializer Test")
        self._run_git("config", "core.sharedRepository", "group")
        self._run_git("add", ".")
        self._run_git("commit", "-qm", "base")

        (self.root / "ordinary.txt").write_bytes(b"overlay ordinary\n")
        (self.root / "ordinary.txt").chmod(0o664)
        (self.root / "overlay.txt").write_bytes(b"candidate overlay\n")
        (self.root / "overlay.txt").chmod(0o664)

        mode_by_path = {
            "ordinary.txt": 0o644,
            "script.sh": 0o755,
            "private.key": 0o600,
            "model-api-gguf/llama.cpp/README.md": 0o644,
            "overlay.txt": 0o644,
        }
        entries = []
        for relative, mode in sorted(mode_by_path.items()):
            data = (self.root / relative).read_bytes()
            entries.append(
                {
                    "path": relative,
                    "bytes": len(data),
                    "sha256": _sha256(data),
                    "portable_mode": format(mode, "#06o"),
                }
            )
        manifest = {
            "schema_version": "system-x.portable-tree-manifest.v4",
            "semantics": "FULL_TREE",
            "source_root": ".",
            "manifest_path": MANIFEST_PATH,
            "entries": entries,
            "entry_count": len(entries),
            "tracked_regular_file_count": len(entries) + 1,
        }
        (self.root / MANIFEST_PATH).write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.candidate_map = self.root.parent / "candidate-map.json"
        candidate_entries = []
        for relative in ("ordinary.txt", "overlay.txt", MANIFEST_PATH):
            path = self.root / relative
            data = path.read_bytes()
            candidate_entries.append(
                {
                    "path": relative,
                    "bytes": len(data),
                    "sha256": _sha256(data),
                    "portable_mode": format(
                        0o644 if relative != "script.sh" else 0o755,
                        "#06o",
                    ),
                }
            )
        self.candidate_map.write_text(
            json.dumps({"paths": candidate_entries}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _archive_mode(self, relative: str) -> int:
        archive = subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "--no-optional-locks",
                "archive",
                "--format=tar",
                "HEAD",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as reader:
            return next(item for item in reader if item.name == relative).mode & 0o777

    def _materialize(self, name: str, mask: int) -> tuple[Path, dict[str, object]]:
        destination = self.root.parent / name
        with _umask(mask):
            result = materialize_portable_tree(
                self.root,
                destination,
                self.candidate_map,
            )
        return destination, result

    def _mode_map(self, destination: Path) -> dict[str, tuple[int, int, str, int, int]]:
        manifest = json.loads((self.root / MANIFEST_PATH).read_text(encoding="utf-8"))
        result = {}
        for entry in manifest["entries"]:
            path = destination / entry["path"]
            data = path.read_bytes()
            details = path.stat()
            result[entry["path"]] = (
                len(data),
                details.st_mode & 0o777,
                _sha256(data),
                details.st_uid,
                details.st_gid,
            )
        return result

    def test_real_archive_overlay_and_exact_modes(self) -> None:
        with _umask(0o0002):
            archive_mode = self._archive_mode("ordinary.txt")
        self.assertEqual(archive_mode, 0o664)
        destination, result = self._materialize("materialized", 0o0002)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["content_mismatch_count"], 0)
        self.assertEqual(result["mode_mismatch_count"], 0)
        self.assertEqual(result["object_type_mismatch_count"], 0)
        self.assertEqual(result["containment_error_count"], 0)
        expected = {
            "ordinary.txt": 0o644,
            "script.sh": 0o755,
            "private.key": 0o600,
            "model-api-gguf/llama.cpp/README.md": 0o644,
            "overlay.txt": 0o644,
        }
        observed = self._mode_map(destination)
        for relative, mode in expected.items():
            self.assertEqual(observed[relative][1], mode)
        source_owner = (self.root.stat().st_uid, self.root.stat().st_gid)
        for _, (_, _, _, uid, gid) in observed.items():
            self.assertEqual((uid, gid), source_owner)

    def test_umask_independence(self) -> None:
        results = []
        for index, mask in enumerate((0o0000, 0o0002, 0o0022, 0o0077)):
            destination, result = self._materialize(f"mask-{index}", mask)
            results.append(self._mode_map(destination))
            self.assertEqual(result["mode_mismatch_count"], 0)
        self.assertTrue(all(value == results[0] for value in results[1:]))

    def test_rejects_candidate_symlink_hardlink_and_special_file(self) -> None:
        overlay = self.root / "overlay.txt"
        overlay.unlink()
        overlay.symlink_to(self.root / "ordinary.txt")
        with self.assertRaises(MaterializationError):
            self._materialize("symlink", 0o0002)

        overlay.unlink()
        os.link(self.root / "ordinary.txt", overlay)
        with self.assertRaises(MaterializationError):
            self._materialize("hardlink", 0o0002)

        overlay.unlink()
        os.mkfifo(overlay)
        with self.assertRaises(MaterializationError):
            self._materialize("special", 0o0002)

    def test_mode_failure_has_exact_diagnostic_and_no_publication(self) -> None:
        with mock.patch.object(portable_materializer.os, "fchmod", lambda fd, mode: None):
            with self.assertRaises(MaterializationError) as caught:
                self._materialize("mode-failure", 0o0002)
        diagnostic = json.loads(str(caught.exception))
        self.assertEqual(diagnostic["reason_code"], "MATERIALIZED_MODE_MISMATCH")
        self.assertEqual(diagnostic["expected_mode"], "0o0644")
        self.assertFalse((self.root.parent / "mode-failure").exists())


if __name__ == "__main__":
    unittest.main()
