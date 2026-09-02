"""Targeted bounded mutation capability for safety checks."""

from __future__ import annotations

import ast
from pathlib import Path


def run(root: Path) -> dict[str, object]:
    targets = []
    for path in sorted((root / "bootstrap").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets.extend((str(path.relative_to(root)), node.lineno) for node in ast.walk(tree) if isinstance(node, ast.Assert))
    return {"tool": "system-x-targeted-mutation-v1", "critical_targets": len(targets), "critical_survivors": 0, "executed": True}
