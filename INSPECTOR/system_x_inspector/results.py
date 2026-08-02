"""Machine-result envelope construction."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .constants import SCHEMA_IDENTITIES


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def machine_result(
    *,
    operation: str,
    ok: bool,
    reason_code: str,
    message: str,
    inspector_root: Path,
    transaction_id: str | None = None,
    data: dict[str, Any] | None = None,
    paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_IDENTITIES["machine_result"],
        "operation": operation,
        "ok": ok,
        "reason_code": reason_code,
        "message": message,
        "timestamp_utc": utc_now(),
        "inspector_root": str(inspector_root),
        "transaction_id": transaction_id,
        "data": data or {},
        "paths": paths or {},
    }
