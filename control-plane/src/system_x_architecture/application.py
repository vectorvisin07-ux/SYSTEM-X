"""Use cases depend on ports and return typed domain results."""
from __future__ import annotations

from .domain import Result, StatusSnapshot
from .ports import StatusPort


class SystemService:
    def __init__(self, status: StatusPort) -> None:
        self._status = status

    def status(self) -> Result[StatusSnapshot]:
        return self._status.read()

