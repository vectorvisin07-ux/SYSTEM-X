"""Small reusable infrastructure adapters for atomic state and task ownership."""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from .domain import ErrorCategory, ReasonCode, Result, SystemXError
from .ports import ConfigurationSnapshot


def _error(reason: str, message: str) -> SystemXError:
    return SystemXError(ErrorCategory.PERSISTENCE, ReasonCode(reason), message, message)


class AtomicFileRepository:
    """One owner for generation-checked, fsync-before-replace state writes."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> Result[bytes | None]:
        path = self._path(key)
        try:
            return Result.ok(path.read_bytes() if path.is_file() else None)
        except OSError as exc:
            return Result.fail(_error("PERSISTENCE_READ", str(exc)))

    def store(self, key: str, payload: bytes, expected_generation: int) -> Result[int]:
        path = self._path(key)
        generation = expected_generation + 1
        temporary = path.with_name(f".{path.name}.{generation}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(self._root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return Result.ok(generation)
        except (OSError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return Result.fail(_error("PERSISTENCE_WRITE", str(exc)))

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.state"


class StructuredTaskOwner:
    """Owns every child task and waits for cancellation before returning."""

    async def run(self, *coroutines: object) -> None:
        async with asyncio.TaskGroup() as group:
            for coroutine in coroutines:
                if not asyncio.iscoroutine(coroutine):
                    raise TypeError("task owner accepts coroutine objects only")
                group.create_task(coroutine)


def load_configuration(root: Path) -> ConfigurationSnapshot:
    resolved = root.resolve(strict=True)
    return ConfigurationSnapshot(resolved, "http://127.0.0.1:56259", "default")

