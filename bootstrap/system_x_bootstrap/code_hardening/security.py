"""Classifying AST security scanner; strings and comments are not calls."""

from __future__ import annotations

import ast
from pathlib import Path


def scan(root: Path, entries: list[dict[str, object]], policy) -> list[dict[str, str]]:
    findings = []
    for entry in entries:
        path = root / str(entry["path"])
        if path.suffix != ".py":
            continue
        if "/tests/" in str(entry["path"]):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in policy.forbidden_calls:
                    findings.append({"id": "SECURITY-CALL", "path": str(entry["path"]), "line": str(node.lineno), "detail": name})
            if isinstance(node, ast.Assert) and "/code_hardening/" in str(entry["path"]):
                findings.append({"id": "SECURITY-ASSERT", "path": str(entry["path"]), "line": str(node.lineno), "detail": "runtime assert"})
    return findings
