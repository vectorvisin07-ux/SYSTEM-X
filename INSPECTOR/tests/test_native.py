from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from system_x_inspector.native import NativeIssue, inspect_native


def write_config(root: Path, **extra: object) -> None:
    value = {"model_type": "bounded", "architectures": ["BoundedForCausalLM"]}
    value.update(extra)
    (root / "config.json").write_text(
        json.dumps(value, separators=(",", ":")), encoding="utf-8"
    )


def safetensors_bytes(
    entries: list[tuple[str, str, list[int], bytes]]
) -> bytes:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, data in entries:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


def write_safe(
    root: Path,
    name: str,
    entries: list[tuple[str, str, list[int], bytes]],
) -> None:
    (root / name).write_bytes(safetensors_bytes(entries))


class NativeTest(unittest.TestCase):
    def expect_issue(
        self, root: Path, category: str, reason_code: str
    ) -> NativeIssue:
        with self.assertRaises(NativeIssue) as raised:
            inspect_native(root)
        self.assertEqual(raised.exception.category, category)
        self.assertEqual(raised.exception.reason_code, reason_code)
        return raised.exception

    def test_single_file_safetensors_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, quantization_config={"quant_method": "bounded_q"})
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            write_safe(root, "model.safetensors", [("w", "F32", [2], b"\0" * 8)])
            result = inspect_native(root)
            self.assertEqual(result.weight_format, "safetensors")
            self.assertEqual(result.shard_count, 1)
            self.assertEqual(result.tensor_count, 1)
            self.assertEqual(result.tensor_bytes, 8)
            self.assertEqual(
                result.quantization, {"quant_method": "bounded_q"}
            )
            self.assertEqual(result.tokenizer_source, ("tokenizer.json",))
            self.assertFalse(result.payload_deserialized)
            self.assertFalse(result.code_executed)

    def test_two_shard_safetensors_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            write_safe(root, "model-00001-of-00002.safetensors", [("a", "F16", [2], b"a" * 4)])
            write_safe(root, "model-00002-of-00002.safetensors", [("b", "U8", [2], b"b" * 2)])
            index = {
                "metadata": {"total_size": 6},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps(index, separators=(",", ":")), encoding="utf-8"
            )
            result = inspect_native(root)
            self.assertEqual((result.shard_count, result.tensor_count), (2, 2))
            self.assertEqual(result.tensor_bytes, 6)
            self.assertEqual(len(result.evidence), 2)

    def test_missing_and_malformed_config_and_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_safe(root, "model.safetensors", [("w", "U8", [1], b"x")])
            self.expect_issue(root, "INCOMPLETE", "NATIVE_CONFIG_MISSING")
            (root / "config.json").write_text(
                '{"model_type":"a","model_type":"b"}', encoding="utf-8"
            )
            self.expect_issue(root, "CORRUPT", "NATIVE_CONFIG_INVALID")
            write_config(root)
            (root / "model.safetensors").unlink()
            self.expect_issue(root, "INCOMPLETE", "NATIVE_WEIGHT_MISSING")

    def test_missing_shard_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {"a": "missing.safetensors"}}),
                encoding="utf-8",
            )
            self.expect_issue(root, "INCOMPLETE", "NATIVE_SHARD_MISSING")
            index_path.write_text(
                json.dumps({"weight_map": {"a": "../escape.safetensors"}}),
                encoding="utf-8",
            )
            self.expect_issue(root, "CORRUPT", "NATIVE_SHARD_INDEX_INVALID")

    def test_duplicate_tensor_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            write_safe(root, "a.safetensors", [("dup", "U8", [1], b"a")])
            write_safe(root, "b.safetensors", [("dup", "U8", [1], b"b")])
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "dup": "a.safetensors",
                            "other": "b.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.expect_issue(
                root, "CORRUPT", "NATIVE_SHARD_DUPLICATE_TENSOR"
            )

    def test_invalid_header_overlap_and_total_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            path = root / "model.safetensors"
            path.write_bytes(struct.pack("<Q", 500) + b"{}")
            self.expect_issue(
                root, "CORRUPT", "NATIVE_SAFETENSORS_HEADER_INVALID"
            )
            header = {
                "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
            }
            encoded = json.dumps(header).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"x" * 6)
            self.expect_issue(
                root, "CORRUPT", "NATIVE_SAFETENSORS_RANGE_INVALID"
            )
            valid = safetensors_bytes([("a", "U8", [1], b"x")])
            path.write_bytes(valid + b"trailing")
            self.expect_issue(
                root, "CORRUPT", "NATIVE_SAFETENSORS_RANGE_INVALID"
            )

    def test_multiple_shards_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            write_safe(root, "a.safetensors", [("a", "U8", [1], b"a")])
            write_safe(root, "b.safetensors", [("b", "U8", [1], b"b")])
            self.expect_issue(
                root, "INCOMPLETE", "NATIVE_SHARD_INDEX_MISSING"
            )

    def test_pytorch_is_structural_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            marker = root / "must-not-exist"
            with zipfile.ZipFile(root / "pytorch_model.bin", "w") as archive:
                archive.writestr(
                    "archive/data.pkl",
                    b"payload that must never be unpickled: " + str(marker).encode(),
                )
                archive.writestr("archive/data/0", b"tensor bytes")
            result = inspect_native(root)
            self.assertEqual(result.weight_format, "pytorch_zip_structural_only")
            self.assertIn("PYTORCH_PAYLOAD_NOT_DESERIALIZED", result.reason_codes)
            self.assertFalse(marker.exists())

    def test_remote_code_is_evidence_only_and_never_imported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            write_config(
                root,
                auto_map={"AutoModel": "modeling_evil.EvilModel"},
            )
            (root / "modeling_evil.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
                encoding="utf-8",
            )
            write_safe(root, "model.safetensors", [("w", "U8", [1], b"x")])
            result = inspect_native(root)
            self.assertIn("NATIVE_REMOTE_CODE_DECLARED", result.warnings)
            self.assertFalse(marker.exists())
            self.assertFalse(result.code_executed)


if __name__ == "__main__":
    unittest.main()
