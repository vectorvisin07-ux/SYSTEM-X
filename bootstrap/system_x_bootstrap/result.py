"""Secret-safe machine result envelope."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .errors import BootstrapError, ErrorCode


RESULT_SCHEMA = "system-x.bootstrap.result.v1"
RESULT_STATUSES = frozenset({"ok", "blocked", "failed-clean", "fail-closed"})
_FORBIDDEN_KEYS = re.compile(r"(?:password|passwd|secret|token|raw[_-]?key|private[_-]?key|pepper)", re.I)
_REDACTED = "[REDACTED]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def redact(value: Any, *, key: str = "") -> Any:
    """Recursively remove fields that could expose credential material."""

    if key and _FORBIDDEN_KEYS.search(key):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, bytes):
        return _REDACTED
    if isinstance(value, str) and ("-----BEGIN " in value or len(value) > 8192):
        return _REDACTED
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class MachineResult:
    operation: str
    status: str
    state: str
    changed: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    errors: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    receipt_id: str = field(default_factory=lambda: secrets.token_hex(16))
    recorded_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.operation or self.status not in RESULT_STATUSES or not self.state:
            raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "invalid machine-result envelope")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "version": 1,
            "receipt_id": self.receipt_id,
            "recorded_utc": self.recorded_utc,
            "operation": self.operation,
            "status": self.status,
            "state": self.state,
            "changed": self.changed,
            "details": redact(dict(self.details)),
            "errors": redact([dict(item) for item in self.errors]),
            "warnings": redact(list(self.warnings)),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def from_error(cls, operation: str, state: str, error: BootstrapError) -> "MachineResult":
        status = "blocked" if error.code in {
            ErrorCode.AUTHORIZATION_REQUIRED,
            ErrorCode.PRECONDITION_FAILED,
        } else "fail-closed"
        return cls(operation=operation, status=status, state=state, errors=(error.as_dict(),))
