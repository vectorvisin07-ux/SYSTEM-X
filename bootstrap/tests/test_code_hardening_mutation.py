from __future__ import annotations
from pathlib import Path
import unittest
from system_x_bootstrap.code_hardening.mutation_gate import run

class MutationTests(unittest.TestCase):
    def test_targeted_mutation_gate_is_bounded(self) -> None:
        result = run(Path(__file__).resolve().parents[2])
        self.assertTrue(result["executed"])
        self.assertEqual(result["critical_survivors"], 0)

if __name__ == "__main__":
    unittest.main()
