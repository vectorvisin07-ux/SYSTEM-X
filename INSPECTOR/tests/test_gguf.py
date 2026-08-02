from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from system_x_inspector.gguf import (
    FORMAT_DEFINITION_IDENTITY,
    GGUFIssue,
    GGML_TYPE_GEOMETRY,
    inspect_gguf,
)


def string(value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def metadata_value(type_code: int, value: object) -> bytes:
    if type_code == 8:
        assert isinstance(value, (str, bytes))
        return string(value)
    if type_code == 7:
        assert isinstance(value, int)
        return struct.pack("<B", value)
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 10: "Q", 11: "q", 12: "d"}
    return struct.pack("<" + formats[type_code], value)


def build_gguf(
    *,
    version: int = 3,
    metadata: list[tuple[str | bytes, int, object]] | None = None,
    tensors: list[tuple[str, list[int], int, int]] | None = None,
    data: bytes | None = None,
) -> bytes:
    if metadata is None:
        metadata = [
            ("general.architecture", 8, "test"),
            ("general.alignment", 4, 32),
            ("general.file_type", 4, 0),
            ("general.quantization_version", 4, 2),
        ]
    tensors = tensors if tensors is not None else [
        ("weight", [2], 0, 0)
    ]
    output = bytearray(b"GGUF")
    output += struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, type_code, value in metadata:
        output += string(key)
        output += struct.pack("<I", type_code)
        output += metadata_value(type_code, value)
    for name, dimensions, type_code, offset in tensors:
        output += string(name)
        output += struct.pack("<I", len(dimensions))
        output += b"".join(struct.pack("<Q", item) for item in dimensions)
        output += struct.pack("<IQ", type_code, offset)
    if tensors:
        output += b"\0" * ((-len(output)) % 32)
    if data is None:
        end = 0
        for _, dimensions, type_code, offset in tensors:
            _, block, size, _ = GGML_TYPE_GEOMETRY[type_code]
            count = 1
            for dimension in dimensions:
                count *= dimension
            end = max(end, offset + count // block * size)
        data = b"\0" * end
    output += data
    return bytes(output)


class GGUFTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def write(self, value: bytes, name: str = "artifact") -> Path:
        path = self.temporary / name
        path.write_bytes(value)
        return path

    def assert_issue(
        self, value: bytes, category: str, reason: str
    ) -> None:
        with self.assertRaises(GGUFIssue) as caught:
            inspect_gguf(self.write(value))
        self.assertEqual(caught.exception.category, category)
        self.assertEqual(caught.exception.reason_code, reason)

    def test_valid_structure_and_bounded_normalized_evidence(self) -> None:
        metadata = [
            ("general.architecture", 8, "test"),
            ("general.alignment", 4, 32),
            ("general.file_type", 4, 1),
            ("general.quantization_version", 4, 2),
            ("tokenizer.ggml.tokens", 8, "bounded"),
            ("tokenizer.chat_template", 8, "chat"),
        ]
        result = inspect_gguf(self.write(build_gguf(metadata=metadata)))
        assert result is not None
        self.assertEqual(result.version, 3)
        self.assertEqual(result.architecture, "test")
        self.assertEqual(result.model_type, "test")
        self.assertEqual(result.tensor_count, 1)
        self.assertEqual(result.tensor_type_histogram, {"F32": 1})
        self.assertTrue(result.tokenizer_metadata_present)
        self.assertTrue(result.chat_template_present)
        self.assertEqual(result.tensor_payload_bytes_read, 0)
        self.assertRegex(
            FORMAT_DEFINITION_IDENTITY, r"^sha256:[0-9a-f]{64}$"
        )
        self.assertEqual(
            result.format_definition_identity, FORMAT_DEFINITION_IDENTITY
        )
        encoded = json.dumps(result.as_dict(), sort_keys=True)
        self.assertNotIn("bounded\"],", encoded)

    def test_bad_magic_is_not_detected(self) -> None:
        self.assertIsNone(inspect_gguf(self.write(b"random bytes")))

    def test_unknown_version_is_unknown_not_corrupt(self) -> None:
        self.assert_issue(
            build_gguf(version=99),
            "unknown",
            "GGUF_VERSION_UNSUPPORTED",
        )

    def test_header_metadata_and_tensor_table_truncation(self) -> None:
        self.assert_issue(
            b"GGUF\x03\x00",
            "incomplete",
            "GGUF_HEADER_TRUNCATED",
        )
        complete = build_gguf()
        metadata_end = 24 + len(string("general.architecture")) + 4
        self.assert_issue(
            complete[:metadata_end],
            "incomplete",
            "GGUF_METADATA_TRUNCATED",
        )
        tensor_value = build_gguf(
            metadata=[("general.architecture", 8, "test")],
            tensors=[("w", [2], 0, 0)],
        )
        tensor_start = tensor_value.find(string("w"), 24)
        self.assert_issue(
            tensor_value[: tensor_start + len(string("w")) + 2],
            "incomplete",
            "GGUF_TENSOR_TABLE_TRUNCATED",
        )

    def test_duplicate_key_invalid_type_utf8_and_bool(self) -> None:
        duplicate = [
            ("general.architecture", 8, "a"),
            ("general.architecture", 8, "b"),
        ]
        self.assert_issue(
            build_gguf(metadata=duplicate, tensors=[]),
            "corrupt",
            "GGUF_METADATA_INVALID",
        )
        value = bytearray(build_gguf())
        first_type_offset = 24 + len(string("general.architecture"))
        value[first_type_offset:first_type_offset + 4] = struct.pack("<I", 99)
        self.assert_issue(
            bytes(value), "corrupt", "GGUF_METADATA_INVALID"
        )
        self.assert_issue(
            build_gguf(
                metadata=[("general.architecture", 8, b"\xff")],
                tensors=[],
            ),
            "corrupt",
            "GGUF_METADATA_INVALID",
        )
        self.assert_issue(
            build_gguf(
                metadata=[
                    ("general.architecture", 8, "a"),
                    ("flag", 7, 2),
                ],
                tensors=[],
            ),
            "corrupt",
            "GGUF_METADATA_INVALID",
        )

    def test_invalid_dimensions_and_unknown_tensor_type(self) -> None:
        self.assert_issue(
            build_gguf(tensors=[("w", [0], 0, 0)]),
            "corrupt",
            "GGUF_TENSOR_DESCRIPTOR_INVALID",
        )
        value = bytearray(build_gguf())
        # The final 12 descriptor bytes before aligned padding are type+offset.
        descriptor_end = len(value.rstrip(b"\0"))
        # Build a no-data file to make the descriptor location deterministic.
        raw = bytearray(build_gguf(tensors=[("w", [2], 0, 0)], data=b""))
        marker = string("w") + struct.pack("<IQ", 1, 2)
        position = raw.find(marker)
        type_position = position + len(marker)
        raw[type_position:type_position + 4] = struct.pack("<I", 99)
        self.assert_issue(
            bytes(raw), "corrupt", "GGUF_TENSOR_TYPE_UNKNOWN"
        )

    def test_misaligned_overlapping_and_out_of_bounds_ranges(self) -> None:
        self.assert_issue(
            build_gguf(tensors=[("w", [2], 0, 1)], data=b"\0" * 16),
            "corrupt",
            "GGUF_TENSOR_RANGE_INVALID",
        )
        self.assert_issue(
            build_gguf(
                tensors=[
                    ("a", [2], 0, 0),
                    ("b", [2], 0, 0),
                ],
                data=b"\0" * 16,
            ),
            "corrupt",
            "GGUF_TENSOR_RANGE_INVALID",
        )
        self.assert_issue(
            build_gguf(tensors=[("w", [2], 0, 0)], data=b"\0" * 4),
            "incomplete",
            "GGUF_TENSOR_DATA_TRUNCATED",
        )


if __name__ == "__main__":
    unittest.main()
