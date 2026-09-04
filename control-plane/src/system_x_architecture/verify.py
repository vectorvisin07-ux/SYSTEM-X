"""Permanent architecture gate with deterministic, source-only checks."""
from __future__ import annotations

import ast
import json
from pathlib import Path


FORBIDDEN_DOMAIN_IMPORTS = {"sqlite3", "subprocess", "fastapi", "aiohttp"}
REQUIRED_ROOTS = ("domain.py", "ports.py", "application.py", "infrastructure.py")


def report(root: Path) -> dict[str, object]:
    package = root / "control-plane" / "src" / "system_x_architecture"
    files = sorted(package.glob("*.py")) if package.is_dir() else []
    domain = package / "domain.py"
    violations: list[str] = []
    if not package.is_dir():
        violations.append("architecture package absent")
    for name in REQUIRED_ROOTS:
        if not (package / name).is_file():
            violations.append(f"required module absent: {name}")
    if domain.is_file():
        tree = ast.parse(domain.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(alias.name for alias in node.names if alias.name.split(".")[0] in FORBIDDEN_DOMAIN_IMPORTS)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in FORBIDDEN_DOMAIN_IMPORTS:
                violations.append(node.module)
    schema = root / "control-plane" / "contracts" / "system_x_architecture.schema.json"
    schema_ok = schema.is_file()
    if not schema_ok:
        violations.append("architecture schema absent")
    return {
        "schema": "system-x.verify-architecture.v1",
        "status": "PASS" if not violations else "FAIL",
        "package_modules": [p.name for p in files],
        "forbidden_domain_imports": violations,
        "generated_schema": schema_ok,
        "source_only": True,
    }


def run(root: Path, machine: bool = False) -> int:
    result = report(root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")) if machine else f"System X verify-architecture: {result['status']}")
    return 0 if result["status"] == "PASS" else 1

