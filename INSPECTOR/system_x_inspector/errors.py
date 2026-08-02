"""Bounded domain errors for the machine interface."""

from __future__ import annotations

from typing import Any


class InspectorError(Exception):
    """A deterministic domain rejection."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        exit_status: int = 2,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.data = data or {}
        self.exit_status = exit_status
