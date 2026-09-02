"""AST architecture and import-boundary gate."""

from __future__ import annotations

import ast
from pathlib import Path


def scan(root: Path, entries: list[dict[str, object]], policy) -> list[dict[str, str]]:
    findings = []
    for entry in entries:
        path = root / str(entry["path"])
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = str(entry["path"])
        if "/tests/" in relative:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                registry_owner = (relative == "bootstrap/system_x_bootstrap/runtime.py" or relative.startswith("INSPECTOR/system_x_inspector/") or relative.startswith("model-api-gguf/api_service/src/system_x_gguf_api/"))
                if name in policy.forbidden_imports and not registry_owner:
                    findings.append({"id": "ARCH-IMPORT", "path": relative, "line": str(node.lineno), "detail": name})
    return findings
