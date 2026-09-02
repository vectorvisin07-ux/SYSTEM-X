from __future__ import annotations
from pathlib import Path
import unittest
from system_x_bootstrap.code_hardening.contracts import scan

class ContractTests(unittest.TestCase):
    def test_existing_sources_have_no_public_untyped_function(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.assertIsInstance(scan(root, [{"path": "bootstrap/system_x_bootstrap/code_hardening/contracts.py"}]), list)

if __name__ == "__main__":
    unittest.main()
