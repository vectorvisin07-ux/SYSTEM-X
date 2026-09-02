"""Deterministic machine and human report model."""

from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class VerificationReport:
    payload: dict[str, object]

    @property
    def ok(self) -> bool:
        return self.payload["status"] == "PASS"

    def json(self) -> str:
        return json.dumps(self.payload, sort_keys=True, separators=(",", ":"))

    def human(self) -> str:
        return "System X verify-code: PASS" if self.ok else "System X verify-code: FAIL (see owner-only report)"
