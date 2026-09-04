"""Abstract application ports.  Concrete platform work stays in adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .domain import ModelId, Result, StatusSnapshot


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    root: Path
    public_origin: str
    model_alias: str


class StatusPort(Protocol):
    def read(self) -> Result[StatusSnapshot]: ...


class PersistencePort(Protocol):
    def load(self, key: str) -> Result[bytes | None]: ...
    def store(self, key: str, payload: bytes, expected_generation: int) -> Result[int]: ...


class ChatPort(Protocol):
    def complete(self, model: ModelId, prompt: str) -> Result[str]: ...

