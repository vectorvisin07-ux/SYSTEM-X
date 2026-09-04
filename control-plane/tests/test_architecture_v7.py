import asyncio
import tempfile
import unittest
from pathlib import Path

from system_x_architecture.domain import ErrorCategory, ModelId, Result, ServiceState, SystemXError, ReasonCode
from system_x_architecture.infrastructure import AtomicFileRepository, StructuredTaskOwner
from system_x_architecture.verify import report


class ArchitectureV7Tests(unittest.TestCase):
    def test_strict_result_and_identity(self):
        self.assertEqual(str(ModelId("m")), "m")
        with self.assertRaises(ValueError):
            ModelId(" m")
        self.assertEqual(Result.ok(ServiceState.RUNNING).value, ServiceState.RUNNING)
        with self.assertRaises(ValueError):
            Result(value="x", error=SystemXError(ErrorCategory.STATE, ReasonCode("BAD"), "bad", "bad"))

    def test_atomic_repository_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = AtomicFileRepository(Path(directory))
            self.assertIsNone(repo.load("status").value)
            self.assertEqual(repo.store("status", b"ok", 0).value, 1)
            self.assertEqual(repo.load("status").value, b"ok")

    def test_task_owner_waits(self):
        seen: list[str] = []
        async def child() -> None:
            seen.append("started")
        asyncio.run(StructuredTaskOwner().run(child()))
        self.assertEqual(seen, ["started"])

    def test_gate(self):
        result = report(Path(__file__).resolve().parents[2])
        self.assertEqual(result["status"], "PASS")
