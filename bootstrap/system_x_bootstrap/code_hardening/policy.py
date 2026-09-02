"""Strict, versioned policy for the V2 source gate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Policy:
    schema: str
    python: str
    source_roots: tuple[str, ...]
    forbidden_imports: tuple[str, ...]
    forbidden_calls: tuple[str, ...]
    max_file_bytes: int


def load_policy(root: Path) -> Policy:
    path = root / "bootstrap" / "configuration" / "code-hardening-policy.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "python", "source_roots", "forbidden_imports", "forbidden_calls", "max_file_bytes"}
    if set(value) != required or value["schema"] != "system-x.code-hardening-policy.v1":
        raise ValueError("invalid code-hardening policy")
    if not isinstance(value["source_roots"], list) or not all(isinstance(x, str) for x in value["source_roots"]):
        raise ValueError("invalid source roots")
    return Policy(value["schema"], value["python"], tuple(value["source_roots"]), tuple(value["forbidden_imports"]), tuple(value["forbidden_calls"]), int(value["max_file_bytes"]))
