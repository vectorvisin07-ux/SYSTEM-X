from __future__ import annotations

import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path

from system_x_inspector.errors import InspectorError
from system_x_inspector.intake import list_intake, validate_intake
from system_x_inspector.paths import InspectorPaths


class IntakeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(
            tempfile.mkdtemp(prefix="inspector-intake-", dir="/tmp")
        )
        self.root = self.temporary / "INSPECTOR"
        self.root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.root)
        self.paths.intake_root.mkdir(mode=0o700)
        self.bounds = {
            "maximum_directory_depth": 8,
            "maximum_entry_count": 128,
            "maximum_relative_path_bytes": 1024,
            "maximum_component_bytes": 128,
        }
        self.sockets: list[socket.socket] = []

    def tearDown(self) -> None:
        for item in self.sockets:
            item.close()
        shutil.rmtree(self.temporary)

    def assert_reason(self, expected: str, callable_value: object) -> None:
        with self.assertRaises(InspectorError) as caught:
            callable_value()
        self.assertEqual(caught.exception.reason_code, expected)

    def test_regular_file_metadata_snapshot_is_stable(self) -> None:
        marker = "PACKET-CONTENT-MUST-NOT-ENTER-METADATA"
        (self.paths.intake_root / "candidate.bin").write_text(
            marker, encoding="utf-8"
        )
        first = validate_intake(
            self.paths, self.bounds, "candidate.bin"
        )
        second = validate_intake(
            self.paths, self.bounds, "candidate.bin"
        )
        self.assertEqual(first, second)
        self.assertEqual(first["root_type"], "regular_file")
        self.assertEqual(first["entry_count"], 1)
        self.assertEqual(
            first["format_classification"], "not_performed"
        )
        self.assertEqual(first["snapshot_basis"], "filesystem_metadata_only")
        self.assertNotIn(marker, str(first))

    def test_regular_directory_bundle(self) -> None:
        bundle = self.paths.intake_root / "bundle"
        (bundle / "nested").mkdir(parents=True)
        (bundle / "alpha.bin").write_bytes(b"alpha")
        (bundle / "nested" / "beta.bin").write_bytes(b"beta")
        result = validate_intake(self.paths, self.bounds, "bundle")
        self.assertEqual(result["root_type"], "regular_directory")
        self.assertEqual(result["entry_count"], 4)
        self.assertEqual(
            [item["relative_path"] for item in result["metadata_manifest"]],
            ["bundle", "bundle/alpha.bin", "bundle/nested", "bundle/nested/beta.bin"],
        )

    def test_implicit_single_and_explicit_multiple_selection(self) -> None:
        (self.paths.intake_root / "one").write_bytes(b"1")
        implicit = validate_intake(self.paths, self.bounds)
        self.assertEqual(implicit["target_name"], "one")
        (self.paths.intake_root / "two").write_bytes(b"2")
        listing = list_intake(self.paths)
        self.assertEqual(listing["candidate_count"], 2)
        self.assertTrue(listing["explicit_selection_required"])
        self.assert_reason(
            "INTAKE_MULTIPLE_CANDIDATES",
            lambda: validate_intake(self.paths, self.bounds),
        )
        explicit = validate_intake(self.paths, self.bounds, "two")
        self.assertEqual(explicit["target_name"], "two")

    def test_empty_intake_rejected(self) -> None:
        self.assert_reason(
            "INTAKE_EMPTY",
            lambda: validate_intake(self.paths, self.bounds),
        )

    def test_root_and_descendant_symlinks_rejected(self) -> None:
        target = self.paths.intake_root / ".packet-target"
        target.write_bytes(b"x")
        root_link = self.paths.intake_root / "root-link"
        root_link.symlink_to(target)
        self.assert_reason(
            "INTAKE_TARGET_SYMLINK",
            lambda: validate_intake(self.paths, self.bounds, "root-link"),
        )
        bundle = self.paths.intake_root / "bundle"
        bundle.mkdir()
        (bundle / "descendant-link").symlink_to(target)
        self.assert_reason(
            "INTAKE_DESCENDANT_SYMLINK",
            lambda: validate_intake(self.paths, self.bounds, "bundle"),
        )

    def test_fifo_and_unix_socket_rejected(self) -> None:
        fifo = self.paths.intake_root / "candidate-fifo"
        os.mkfifo(fifo, 0o600)
        self.assert_reason(
            "INTAKE_SPECIAL_FILE",
            lambda: validate_intake(
                self.paths, self.bounds, "candidate-fifo"
            ),
        )
        socket_path = self.paths.intake_root / "candidate-socket"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen(1)
        self.sockets.append(listener)
        self.assert_reason(
            "INTAKE_SPECIAL_FILE",
            lambda: validate_intake(
                self.paths, self.bounds, "candidate-socket"
            ),
        )

    def test_path_and_branch_escape_rejected(self) -> None:
        for name in ("../escape", "nested/escape", "..", ".hidden"):
            with self.subTest(name=name):
                self.assert_reason(
                    "INTAKE_TARGET_INVALID",
                    lambda name=name: validate_intake(
                        self.paths, self.bounds, name
                    ),
                )
        self.assert_reason(
            "INTAKE_TARGET_OUTSIDE_ROOT",
            lambda: validate_intake(self.paths, self.bounds, "/tmp/escape"),
        )
        self.assert_reason(
            "INTAKE_BRANCH_PATH_REJECTED",
            lambda: validate_intake(
                self.paths,
                self.bounds,
                "/protected/model-api-gguf/artifact",
            ),
        )

    def test_directory_depth_bound(self) -> None:
        target = self.paths.intake_root / "depth"
        (target / "one" / "two").mkdir(parents=True)
        bounds = {**self.bounds, "maximum_directory_depth": 1}
        self.assert_reason(
            "INTAKE_DEPTH_EXCEEDED",
            lambda: validate_intake(self.paths, bounds, "depth"),
        )

    def test_entry_count_bound(self) -> None:
        target = self.paths.intake_root / "entries"
        target.mkdir()
        (target / "one").write_bytes(b"1")
        (target / "two").write_bytes(b"2")
        bounds = {**self.bounds, "maximum_entry_count": 2}
        self.assert_reason(
            "INTAKE_ENTRY_COUNT_EXCEEDED",
            lambda: validate_intake(self.paths, bounds, "entries"),
        )

    def test_path_length_and_component_bounds(self) -> None:
        (self.paths.intake_root / "lengthy").write_bytes(b"x")
        component_bounds = {
            **self.bounds,
            "maximum_component_bytes": 4,
        }
        self.assert_reason(
            "INTAKE_PATH_LENGTH_EXCEEDED",
            lambda: validate_intake(
                self.paths, component_bounds, "lengthy"
            ),
        )
        target = self.paths.intake_root / "path"
        target.mkdir()
        (target / "component").write_bytes(b"x")
        relative_bounds = {
            **self.bounds,
            "maximum_relative_path_bytes": 8,
        }
        self.assert_reason(
            "INTAKE_PATH_LENGTH_EXCEEDED",
            lambda: validate_intake(
                self.paths, relative_bounds, "path"
            ),
        )

    def test_metadata_change_during_validation_rejected(self) -> None:
        target = self.paths.intake_root / "changing"
        target.write_bytes(b"unchanged-content")

        def mutate_metadata(path: Path) -> None:
            details = path.stat()
            os.utime(
                path,
                ns=(details.st_atime_ns, details.st_mtime_ns + 1_000_000),
            )

        self.assert_reason(
            "INTAKE_CHANGED_DURING_VALIDATION",
            lambda: validate_intake(
                self.paths,
                self.bounds,
                "changing",
                stability_hook=mutate_metadata,
            ),
        )

    def test_hidden_packet_names_are_not_candidates(self) -> None:
        (self.paths.intake_root / ".packet-hidden").write_bytes(b"x")
        self.assertEqual(list_intake(self.paths)["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
