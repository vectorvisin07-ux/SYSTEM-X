import asyncio
from pathlib import Path
import signal
import unittest
from unittest.mock import AsyncMock, patch

from system_x_gguf_api import controller_client


VALID_STATUS = (
    b'{"data":{},"message":"ok","ok":true,'
    b'"operation":"status","reason_code":"OK",'
    b'"schema_version":"system-x.gguf-branch-controller.v1"}\n'
)


class _Process:
    pid = 4242
    returncode = 0

    async def communicate(self):
        return VALID_STATUS, b""

    async def wait(self):
        return self.returncode


class _TimeoutProcess:
    pid = 5252
    returncode = None

    async def communicate(self):
        raise TimeoutError()

    async def wait(self):
        self.returncode = -signal.SIGKILL
        return self.returncode


class BranchControllerProcessIsolationTests(unittest.IsolatedAsyncioTestCase):
    def _client(self):
        client = object.__new__(controller_client.BranchControllerClient)
        client._operation_timeout_seconds = 1.0
        client._controller_path = Path("/system-x/branch_controller/controller.py")
        return client

    async def test_controller_child_starts_in_its_own_process_group(self):
        process = _Process()
        with patch.object(
            controller_client.asyncio,
            "create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ) as create:
            result = await self._client().status()

        self.assertTrue(result.ok)
        self.assertIsNone(result.data.get("missing"))
        self.assertTrue(create.call_args.kwargs["start_new_session"])

    async def test_timeout_kills_only_the_isolated_controller_group(self):
        process = _TimeoutProcess()
        with patch.object(
            controller_client.asyncio,
            "create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ), patch.object(controller_client.os, "killpg") as killpg:
            with self.assertRaises(controller_client.ControllerClientError):
                await self._client().status()

        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
