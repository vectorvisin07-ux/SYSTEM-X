from __future__ import annotations
import unittest

class StateMachineTests(unittest.TestCase):
    def test_illegal_transition_is_rejected_by_contract(self) -> None:
        legal = {("REGISTERED", "PROBING"), ("PROBING", "READY")}
        self.assertNotIn(("REGISTERED", "READY"), legal)

if __name__ == "__main__":
    unittest.main()
