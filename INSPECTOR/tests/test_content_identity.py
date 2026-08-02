from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from system_x_inspector import content_identity
from system_x_inspector.content_identity import (
    identify_artifact,
    identify_directory_bundle,
    identify_regular_file,
)
from system_x_inspector.errors import InspectorError


class ContentIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def assert_reason(self, reason: str, function: object) -> None:
        with self.assertRaises(InspectorError) as caught:
            function()
        self.assertEqual(caught.exception.reason_code, reason)

    def test_regular_file_repeated_identity_and_changed_byte(self) -> None:
        path = self.temporary / "artifact.bin"
        path.write_bytes(b"abcdefgh")
        first = identify_regular_file(path, chunk_bytes=3)
        second = identify_artifact(path, chunk_bytes=2)
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(first.byte_count, 8)
        self.assertEqual(first.file_count, 1)
        self.assertEqual(
            first.pre_inspection_snapshot_identity,
            first.post_inspection_snapshot_identity,
        )
        path.write_bytes(b"abcdefgH")
        changed = identify_regular_file(path, chunk_bytes=3)
        self.assertNotEqual(first.identity, changed.identity)

    def test_bundle_identity_is_path_and_enumeration_independent(self) -> None:
        left = self.temporary / "left"
        right = self.temporary / "right"
        left.mkdir()
        right.mkdir()
        (left / "b").write_bytes(b"two")
        (left / "a").write_bytes(b"one")
        (right / "a").write_bytes(b"one")
        (right / "b").write_bytes(b"two")
        one = identify_directory_bundle(left, chunk_bytes=2)
        two = identify_directory_bundle(right, chunk_bytes=1)
        self.assertEqual(one.identity, two.identity)
        self.assertEqual(
            [item["relative_path"] for item in one.files], ["a", "b"]
        )
        self.assertEqual(one.byte_count, 6)
        self.assertEqual(one.file_count, 2)

    def test_root_and_descendant_symlinks_are_rejected(self) -> None:
        target = self.temporary / "target"
        target.write_bytes(b"x")
        link = self.temporary / "link"
        link.symlink_to(target)
        self.assert_reason(
            "ARTIFACT_READ_FAILED", lambda: identify_artifact(link)
        )
        bundle = self.temporary / "bundle"
        bundle.mkdir()
        (bundle / "link").symlink_to(target)
        self.assert_reason(
            "ARTIFACT_READ_FAILED",
            lambda: identify_directory_bundle(bundle),
        )

    def test_path_inode_replacement_during_hash_is_rejected(self) -> None:
        path = self.temporary / "artifact"
        path.write_bytes(b"abcdefgh")
        replaced = False

        def hook(current: Path, total: int) -> None:
            nonlocal replaced
            if not replaced:
                replaced = True
                original = current.with_name("original")
                current.rename(original)
                current.write_bytes(b"abcdefgh")

        self.assert_reason(
            "ARTIFACT_CHANGED_DURING_INSPECTION",
            lambda: identify_regular_file(
                path, chunk_bytes=2, progress_hook=hook
            ),
        )

    def test_truncation_and_size_change_are_rejected(self) -> None:
        path = self.temporary / "artifact"
        path.write_bytes(b"abcdefgh")
        changed = False

        def hook(current: Path, total: int) -> None:
            nonlocal changed
            if not changed:
                changed = True
                os.truncate(current, 3)

        self.assert_reason(
            "ARTIFACT_CHANGED_DURING_INSPECTION",
            lambda: identify_regular_file(
                path, chunk_bytes=2, progress_hook=hook
            ),
        )

    def test_mtime_or_ctime_change_is_rejected(self) -> None:
        path = self.temporary / "artifact"
        path.write_bytes(b"abcdefgh")
        changed = False

        def hook(current: Path, total: int) -> None:
            nonlocal changed
            if not changed:
                changed = True
                current.chmod(0o600)

        self.assert_reason(
            "ARTIFACT_CHANGED_DURING_INSPECTION",
            lambda: identify_regular_file(
                path, chunk_bytes=2, progress_hook=hook
            ),
        )

    def test_directory_concurrent_mutation_is_rejected(self) -> None:
        root = self.temporary / "bundle"
        root.mkdir()
        path = root / "a"
        path.write_bytes(b"abcdefgh")
        changed = False

        def hook(current: Path, total: int) -> None:
            nonlocal changed
            if not changed:
                changed = True
                (root / "new").write_bytes(b"x")

        self.assert_reason(
            "ARTIFACT_CHANGED_DURING_INSPECTION",
            lambda: identify_directory_bundle(
                root, chunk_bytes=2, progress_hook=hook
            ),
        )

    def test_arithmetic_overflow_fails_closed(self) -> None:
        path = self.temporary / "artifact"
        path.write_bytes(b"abcdefgh")
        with mock.patch.object(content_identity, "MAX_U64", 4):
            self.assert_reason(
                "ARTIFACT_IDENTITY_FAILED",
                lambda: identify_regular_file(path, chunk_bytes=3),
            )

    def test_stream_reads_never_exceed_configured_chunk(self) -> None:
        path = self.temporary / "artifact"
        path.write_bytes(b"x" * 100_000)
        observed: list[int] = []
        real_read = os.read

        def bounded_read(descriptor: int, amount: int) -> bytes:
            observed.append(amount)
            return real_read(descriptor, amount)

        with mock.patch(
            "system_x_inspector.content_identity.os.read",
            side_effect=bounded_read,
        ):
            result = identify_regular_file(path, chunk_bytes=4096)
        self.assertEqual(result.byte_count, 100_000)
        self.assertTrue(observed)
        self.assertLessEqual(max(observed), 4096)


if __name__ == "__main__":
    unittest.main()
