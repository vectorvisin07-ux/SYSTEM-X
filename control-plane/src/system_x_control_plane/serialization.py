from __future__ import annotations
import hashlib, json
from typing import Any
def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
def require_object(value: Any, label: str = "value") -> dict[str, Any]:
    if not isinstance(value, dict): raise ValueError(f"{label} must be an object")
    return value
