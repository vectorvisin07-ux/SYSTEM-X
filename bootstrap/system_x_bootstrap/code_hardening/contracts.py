"""Strict contract surface checks."""

from __future__ import annotations

import ast
from pathlib import Path


def scan(root: Path, entries: list[dict[str, object]]) -> list[dict[str, str]]:
    findings = []
    for entry in entries:
        path = root / str(entry["path"])
        if path.suffix != ".py":
            continue
        if "/code_hardening/" not in str(entry["path"]):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("public_"):
                if node.returns is None:
                    findings.append({"id": "CONTRACT-RETURN", "path": str(entry["path"]), "line": str(node.lineno), "detail": "public function lacks return annotation"})
    return findings
