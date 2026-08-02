"""Bounded subprocess execution used only after an operation is invoked."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .errors import BootstrapError, ErrorCode


MAX_CAPTURE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class Runner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 60,
    ) -> CommandResult: ...


class SubprocessRunner:
    """Shell-free subprocess runner with a closed environment overlay."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 60,
    ) -> CommandResult:
        if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
            raise BootstrapError(ErrorCode.PRECONDITION_FAILED, "invalid command vector")
        command_env = os.environ.copy()
        command_env.update({
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        })
        if env:
            command_env.update(env)
        try:
            completed = subprocess.run(
                list(argv),
                cwd=cwd,
                env=command_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BootstrapError(
                ErrorCode.EXTERNAL_COMMAND_FAILED,
                "external command could not be executed",
                context={"program": argv[0], "error_type": type(exc).__name__},
            ) from exc
        if len(completed.stdout) > MAX_CAPTURE_BYTES or len(completed.stderr) > MAX_CAPTURE_BYTES:
            raise BootstrapError(
                ErrorCode.EXTERNAL_COMMAND_FAILED,
                "external command exceeded the bounded capture envelope",
                context={"program": argv[0]},
            )
        return CommandResult(
            tuple(argv),
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )


def require_success(result: CommandResult, *, purpose: str) -> CommandResult:
    if result.returncode != 0:
        raise BootstrapError(
            ErrorCode.EXTERNAL_COMMAND_FAILED,
            purpose,
            context={
                "program": result.argv[0],
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-1000:],
            },
        )
    return result
