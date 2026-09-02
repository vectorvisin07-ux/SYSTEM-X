from __future__ import annotations
from pathlib import Path
from .core import backup

def main(root: Path, action: str, machine: bool = False) -> int:
    result = backup(root, action); print(__import__('json').dumps(result.payload(), sort_keys=True) if machine else f"System X backup/{action}: {result.status}"); return 0 if result.status == "PASS" else 1
