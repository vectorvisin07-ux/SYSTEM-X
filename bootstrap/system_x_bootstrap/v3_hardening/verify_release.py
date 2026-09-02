from __future__ import annotations
from pathlib import Path
from .core import verify_release

def main(root: Path, machine: bool = False) -> int:
    result = verify_release(root); print(__import__('json').dumps(result.payload(), sort_keys=True) if machine else f"System X verify-release: {result.status}"); return 0 if result.status == "PASS" else 1
