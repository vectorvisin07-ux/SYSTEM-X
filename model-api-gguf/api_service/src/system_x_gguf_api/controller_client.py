"""Internal structured client for the branch lifecycle controller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


CONTROLLER_SCHEMA = "system-x.gguf-branch-controller.v1"
MAX_CONTROLLER_OUTPUT_BYTES = 1024 * 1024


class ControllerClientError(RuntimeError):
    """A bounded controller invocation or response failure."""


@dataclass(frozen=True)
class ControllerResult:
    operation: str
    ok: bool
    reason_code: str
    message: str
    data: dict[str, Any]
    stderr: str
    exit_status: int


def derive_branch_controller_path() -> Path:
    package_file = Path(__file__).resolve(strict=True)
    branch_root = package_file.parents[3]
    controller = branch_root / "branch_controller" / "controller.py"
    info = controller.lstat()
    if not controller.is_file() or controller.is_symlink():
        raise ControllerClientError("branch controller is not a direct regular file")
    if info.st_size <= 0:
        raise ControllerClientError("branch controller is empty")
    return controller.resolve(strict=True)


class BranchControllerClient:
    """Invoke only the branch controller's bounded lifecycle operations."""

    def __init__(self, operation_timeout_seconds: float) -> None:
        if operation_timeout_seconds <= 0 or operation_timeout_seconds > 180:
            raise ValueError("controller operation timeout is out of bounds")
        self._operation_timeout_seconds = float(operation_timeout_seconds)
        self._controller_path = derive_branch_controller_path()

    async def _invoke(self, operation: str, arguments: list[str]) -> ControllerResult:
        if operation not in {"plan", "start", "status", "stop", "reconcile"}:
            raise ValueError("unsupported controller operation")
        argv = [sys.executable, str(self._controller_path), operation, *arguments]
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self._controller_path.parent.parent),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._operation_timeout_seconds
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ControllerClientError(
                f"branch controller {operation} timed out"
            ) from exc
        if (
            len(stdout) > MAX_CONTROLLER_OUTPUT_BYTES
            or len(stderr) > MAX_CONTROLLER_OUTPUT_BYTES
        ):
            raise ControllerClientError("branch controller output exceeded bound")
        stdout_text = stdout.decode("utf-8", errors="strict")
        stderr_text = stderr.decode("utf-8", errors="replace")
        lines = stdout_text.splitlines()
        if len(lines) != 1:
            raise ControllerClientError(
                "branch controller stdout was not exactly one JSON record"
            )
        try:
            payload = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ControllerClientError(
                "branch controller stdout was malformed JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ControllerClientError("branch controller response was not an object")
        if payload.get("schema_version") != CONTROLLER_SCHEMA:
            raise ControllerClientError("branch controller schema did not match")
        if payload.get("operation") != operation:
            raise ControllerClientError("branch controller operation did not match")
        if not isinstance(payload.get("ok"), bool):
            raise ControllerClientError("branch controller ok field was invalid")
        if not isinstance(payload.get("data"), dict):
            raise ControllerClientError("branch controller data field was invalid")
        if not isinstance(payload.get("reason_code"), str):
            raise ControllerClientError("branch controller reason code was invalid")
        if not isinstance(payload.get("message"), str):
            raise ControllerClientError("branch controller message was invalid")
        return ControllerResult(
            operation=operation,
            ok=payload["ok"],
            reason_code=payload["reason_code"],
            message=payload["message"],
            data=payload["data"],
            stderr=stderr_text,
            exit_status=process.returncode,
        )

    async def plan_router(
        self, host: str, port: int, models_max: int
    ) -> ControllerResult:
        return await self._invoke(
            "plan",
            [
                "--router",
                "--host",
                host,
                "--port",
                str(port),
                "--models-max",
                str(models_max),
            ],
        )

    async def start_router(
        self, host: str, port: int, models_max: int
    ) -> ControllerResult:
        return await self._invoke(
            "start",
            [
                "--router",
                "--host",
                host,
                "--port",
                str(port),
                "--models-max",
                str(models_max),
            ],
        )

    async def status(self) -> ControllerResult:
        return await self._invoke("status", [])

    async def stop(self) -> ControllerResult:
        return await self._invoke("stop", [])

    async def reconcile(self) -> ControllerResult:
        return await self._invoke("reconcile", [])
