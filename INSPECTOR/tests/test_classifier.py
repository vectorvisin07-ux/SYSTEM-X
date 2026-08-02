from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from system_x_inspector.classifier import (
    NORMALIZED_FIELDS,
    classify_artifact,
)
from tests.test_gguf import build_gguf
from tests.test_native import write_config, write_safe


IDENTITY = "sha256:" + "a" * 64


class ClassifierTest(unittest.TestCase):
    def classify(self, path: Path):
        return classify_artifact(
            path, artifact_identity=IDENTITY, artifact_size=None
        )

    def test_all_six_terminal_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root / "not-authoritative.bin"
            gguf.write_bytes(build_gguf())
            self.assertEqual(self.classify(gguf).terminal_class, "GGUF")

            native = root / "native"
            native.mkdir()
            write_config(native)
            write_safe(native, "model.safetensors", [("w", "U8", [1], b"x")])
            self.assertEqual(self.classify(native).terminal_class, "NATIVE")

            unknown = root / "misleading.gguf"
            unknown.write_bytes(b"not a physical model")
            self.assertEqual(self.classify(unknown).terminal_class, "UNKNOWN")

            corrupt = root / "corrupt"
            corrupt.write_bytes(
                build_gguf(
                    metadata=[
                        ("general.architecture", 8, "one"),
                        ("general.architecture", 8, "two"),
                    ]
                )
            )
            self.assertEqual(self.classify(corrupt).terminal_class, "CORRUPT")

            incomplete = root / "incomplete"
            incomplete.write_bytes(b"GGUF\x03\x00")
            self.assertEqual(
                self.classify(incomplete).terminal_class, "INCOMPLETE"
            )

            contradictory = root / "contradictory"
            contradictory.mkdir()
            write_config(contradictory)
            write_safe(
                contradictory,
                "model.safetensors",
                [("w", "U8", [1], b"x")],
            )
            (contradictory / "other.data").write_bytes(build_gguf())
            self.assertEqual(
                self.classify(contradictory).terminal_class, "CONTRADICTORY"
            )

    def test_reason_order_is_deterministic_and_fields_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.gguf"
            path.write_bytes(b"not gguf")
            first = self.classify(path)
            second = self.classify(path)
            self.assertEqual(first.as_dict(), second.as_dict())
            self.assertEqual(tuple(first.normalized), NORMALIZED_FIELDS)
            self.assertEqual(
                first.normalized["reason_codes"],
                ["FORMAT_EVIDENCE_UNKNOWN", "MISLEADING_EXTENSION_IGNORED"],
            )

    def test_filename_has_no_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "plain.txt"
            valid.write_bytes(build_gguf())
            self.assertEqual(self.classify(valid).terminal_class, "GGUF")
            false_name = root / "claimed.gguf"
            false_name.write_bytes(b"plain")
            result = self.classify(false_name)
            self.assertEqual(result.terminal_class, "UNKNOWN")
            self.assertIn(
                "MISLEADING_EXTENSION_IGNORED",
                result.normalized["reason_codes"],
            )

    def test_task_and_modality_inference_is_conservative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ambiguous = root / "ambiguous"
            ambiguous.mkdir()
            write_config(
                ambiguous,
                architectures=["BoundedModel"],
                model_type="bounded",
            )
            write_safe(
                ambiguous, "model.safetensors", [("w", "U8", [1], b"x")]
            )
            value = self.classify(ambiguous).normalized
            self.assertEqual(value["task"], "unknown")
            self.assertEqual(value["modalities"], ["unknown"])

            causal = root / "causal"
            causal.mkdir()
            write_config(causal, architectures=["BoundedForCausalLM"])
            write_safe(causal, "model.safetensors", [("w", "U8", [1], b"x")])
            value = self.classify(causal).normalized
            self.assertEqual(value["task"], "causal_language_modeling")
            self.assertEqual(value["modalities"], ["text"])

            multimodal = root / "multimodal"
            multimodal.mkdir()
            write_config(
                multimodal,
                architectures=["LlavaForConditionalGeneration"],
            )
            write_safe(
                multimodal, "model.safetensors", [("w", "U8", [1], b"x")]
            )
            value = self.classify(multimodal).normalized
            self.assertEqual(value["task"], "multimodal_generation")
            self.assertEqual(value["modalities"], ["text", "image"])

    def test_alternative_native_weights_are_not_contradictory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root)
            write_safe(root, "model.safetensors", [("w", "U8", [1], b"x")])
            with zipfile.ZipFile(root / "pytorch_model.bin", "w") as archive:
                archive.writestr("archive/data.pkl", b"never deserialize")
            result = self.classify(root)
            self.assertEqual(result.terminal_class, "NATIVE")
            self.assertEqual(
                result.normalized["weight_format"],
                "multiple_alternative_representations",
            )
            self.assertIn(
                "NATIVE_MULTIPLE_WEIGHT_REPRESENTATIONS",
                result.normalized["reason_codes"],
            )


if __name__ == "__main__":
    unittest.main()
