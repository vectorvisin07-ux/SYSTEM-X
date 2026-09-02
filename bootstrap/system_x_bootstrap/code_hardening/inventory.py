"""Authenticated source inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXCLUDED = {".git", "RUNTIME", "MODEL", "build", ".venv", "__pycache__", "llama.cpp"}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, policy) -> list[dict[str, object]]:
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or any(part in EXCLUDED for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if not any(relative == prefix or relative.startswith(prefix + "/") for prefix in policy.source_roots):
            continue
        if path.stat().st_size > policy.max_file_bytes:
            raise ValueError(f"source file exceeds policy limit: {relative}")
        owner = "TEST" if "/tests/" in relative else "DOCUMENTATION" if path.suffix in {".md", ".json"} else "BOOTSTRAP_FRONT_DOOR" if relative.startswith("bootstrap/") else "INSPECTOR" if relative.startswith("INSPECTOR/") else "AUTOMATIC_COORDINATOR"
        result.append({"path": relative, "bytes": path.stat().st_size, "mode": path.stat().st_mode & 0o777, "sha256": _digest(path), "owner": owner})
    return result
