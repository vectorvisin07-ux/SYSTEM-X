"""Documentation and CLI consistency checks for the portable product surface."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSPECTOR_ROOT = ROOT / "INSPECTOR"
BOOTSTRAP_ROOT = ROOT / "bootstrap"


class ReleaseDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.installation = (
            ROOT / "docs" / "INSTALLATION.md"
        ).read_text(encoding="utf-8")
        cls.build_runtime = (
            ROOT / "docs" / "BUILD_AND_RUNTIME.md"
        ).read_text(encoding="utf-8")
        cls.troubleshooting = (
            ROOT / "docs" / "TROUBLESHOOTING.md"
        ).read_text(encoding="utf-8")
        cls.layout = (
            ROOT / "docs" / "REPOSITORY_LAYOUT.md"
        ).read_text(encoding="utf-8")
        cls.private_state = (
            ROOT / "docs" / "GENERATED_PRIVATE_AND_MODEL_STATE.md"
        ).read_text(encoding="utf-8")

    def test_root_describes_the_complete_automatic_flow(self) -> None:
        required = (
            "reconstruct --authorize",
            "WAITING_FOR_MODEL",
            "INSPECTOR/MODEL-TEST",
            "exactly one",
            ".gguf",
            "Do not type the model name",
            "INSPECTOR/RUNTIME/status/api-connection.json",
            "deploy-gguf",
            "llama.cpp",
            "llama-server",
            "READY",
            "show-connection",
            "default",
            "/system/v1/health",
            "/system/v1/models",
            "/v1/messages",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.readme)
        order = (
            "reconstruct --authorize",
            "WAITING_FOR_MODEL",
            "INSPECTOR/MODEL-TEST",
            "READY",
            "INSPECTOR/RUNTIME/status/api-connection.json",
            "show-connection",
            "default",
        )
        positions = [self.readme.index(value) for value in order]
        self.assertEqual(positions, sorted(positions))

    def test_connected_documents_agree_on_automatic_contract(self) -> None:
        combined = "\n".join(
            (
                self.installation,
                self.build_runtime,
                self.troubleshooting,
                self.layout,
                self.private_state,
            )
        )
        for value in (
            "INSPECTOR/MODEL-TEST",
            "WAITING_FOR_MODEL",
            "READY",
            "llama.cpp",
            "llama-server",
            "show-connection",
        ):
            self.assertIn(value, combined)
        self.assertNotIn("vLLM", self.installation)
        self.assertNotIn("vLLM", self.build_runtime)
        self.assertNotIn("vLLM", self.troubleshooting)
        self.assertNotIn("vLLM", self.private_state)

    def test_bootstrap_parser_matches_documented_read_only_and_reconstruct_commands(self) -> None:
        sys.path.insert(0, str(BOOTSTRAP_ROOT))
        from system_x_bootstrap.cli import parser

        reconstruct = parser().parse_args(["reconstruct", "--authorize"])
        self.assertEqual(reconstruct.operation, "reconstruct")
        self.assertTrue(reconstruct.authorize)
        verify = parser().parse_args(
            ["verify", "--level", "waiting-for-model"]
        )
        self.assertEqual(verify.operation, "verify")
        self.assertEqual(verify.level, "waiting-for-model")

    def test_inspector_cli_has_zero_argument_read_only_receipt_command(self) -> None:
        sys.path.insert(0, str(INSPECTOR_ROOT))
        from system_x_inspector.machine import build_parser

        arguments = build_parser().parse_args(["show-connection"])
        self.assertEqual(arguments.operation, "show-connection")
        self.assertIsNone(arguments.inspector_root)

    def test_implementation_matches_the_documented_boundaries(self) -> None:
        automatic = (
            ROOT / "INSPECTOR" / "system_x_inspector" / "automatic_intake.py"
        ).read_text(encoding="utf-8")
        receipt = (
            ROOT / "INSPECTOR" / "system_x_inspector" / "connection_receipt.py"
        ).read_text(encoding="utf-8")
        application = (
            ROOT
            / "model-api-gguf"
            / "api_service"
            / "src"
            / "system_x_gguf_api"
            / "application.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"contract": "INSPECTOR/MODEL-TEST"', automatic)
        self.assertIn('"deployment_mode": "install-first"', automatic)
        self.assertIn("NOOP_READY_MODEL_PRESENT", automatic)
        self.assertIn("AUTOMATIC_COPY_IN_PROGRESS", automatic)
        self.assertIn("def show_connection", receipt)
        self.assertIn('"RAW API KEY:"', receipt)
        self.assertIn('"/system/v1/health"', application)
        self.assertIn('"/v1/messages"', application)


if __name__ == "__main__":
    unittest.main()
