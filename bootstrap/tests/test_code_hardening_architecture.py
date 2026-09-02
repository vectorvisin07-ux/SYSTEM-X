from __future__ import annotations
import ast
from pathlib import Path
import unittest
from system_x_bootstrap.code_hardening.architecture import scan
from system_x_bootstrap.code_hardening.policy import load_policy

class ArchitectureTests(unittest.TestCase):
    def test_forbidden_import_is_detected(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with self.subTest(kind="ast"):
            self.assertEqual(scan(root, [{"path": "bootstrap/system_x_bootstrap/code_hardening/architecture.py"}], load_policy(root)), [])

if __name__ == "__main__":
    unittest.main()
