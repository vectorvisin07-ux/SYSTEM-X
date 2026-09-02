from __future__ import annotations
from pathlib import Path
import unittest
from system_x_bootstrap.code_hardening.policy import load_policy
from system_x_bootstrap.code_hardening.security import scan

class SecurityTests(unittest.TestCase):
    def test_gate_source_has_no_forbidden_calls(self) -> None:
        root = Path(__file__).resolve().parents[2]
        entries = [{"path": "bootstrap/system_x_bootstrap/code_hardening/security.py"}]
        self.assertEqual(scan(root, entries, load_policy(root)), [])

if __name__ == "__main__":
    unittest.main()
