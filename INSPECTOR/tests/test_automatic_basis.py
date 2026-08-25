from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from system_x_inspector.automatic_basis import persist_automatic_terminal_basis
from system_x_inspector.paths import InspectorPaths


class AutomaticBasisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="automatic-basis-"))
        self.root = self.temporary / "INSPECTOR"
        self.root.mkdir(mode=0o700)
        self.paths = InspectorPaths.discover(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def result(self, name: str) -> dict:
        return {
            "action": "DISPATCH_FIRST_MODEL",
            "reason_code": "AUTOMATIC_DISPATCH_ACCEPTED",
            "candidate": {"relative_name": name, "artifact_identity": "sha256:"+"a"*64, "observation_identity": "sha256:"+"b"*64},
            "derived_deployment_request": {"deployment_mode":"install-first","required_capability_profile":"CORE_CHAT","retirement_policy":"retain-incumbent","retirement_action":"none"},
            "registry_snapshot": {"registry_generation":0},
            "source_configuration_identity": "sha256:"+"c"*64,
            "result_identity": "sha256:"+"d"*64,
            "created_utc": "2026-01-01T00:00:00.000000Z",
            "existing_result_reference": None,
            "active_transaction_reference": {"transaction_id":"tx","deployment_id":"dep"},
        }

    def test_same_artifact_renamed_reuses_processed_basis(self) -> None:
        first = persist_automatic_terminal_basis(self.paths, self.result("first.gguf"))
        second = persist_automatic_terminal_basis(self.paths, self.result("renamed.gguf"))
        self.assertEqual(first["basis_identity"], second["basis_identity"])
        self.assertEqual(len(list(self.paths.automatic_processed_results.glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
