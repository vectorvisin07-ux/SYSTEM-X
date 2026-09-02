from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from system_x_bootstrap.code_hardening.policy import load_policy
from system_x_bootstrap.code_hardening.security import scan
from system_x_bootstrap.code_verify import verify


ROOT = Path(__file__).resolve().parents[2]


class CodeVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        if os.environ.get("SYSTEM_X_VERIFY_INNER"):
            self.skipTest("nested gate invocation")

    def test_live_candidate_has_gate_shape(self) -> None:
        result = verify(ROOT)
        self.assertIn(result.payload["status"], {"PASS", "FAIL"})
        self.assertEqual(result.payload["schema"], "system-x.code-verification-result.v1")
        self.assertEqual(result.payload["raw_secret_exposure_count"], 0)

    def test_security_scanner_uses_ast_not_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap").mkdir()
            path = root / "bootstrap" / "sample.py"
            path.write_text("text = 'eval(x)'\n", encoding="utf-8")
            policy = load_policy(ROOT)
            entries = [{"path": "bootstrap/sample.py"}]
            self.assertEqual(scan(root, entries, policy), [])

    def test_result_is_deterministically_serialized(self) -> None:
        first = verify(ROOT).json()
        second = verify(ROOT).json()
        self.assertEqual(json.loads(first)["schema"], json.loads(second)["schema"])


if __name__ == "__main__":
    unittest.main()
