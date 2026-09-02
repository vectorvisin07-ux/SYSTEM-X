from __future__ import annotations
from pathlib import Path
from .core import verify_resilience

def main(root: Path, profile: str, machine: bool = False) -> int:
    result = verify_resilience(root, profile); print(__import__('json').dumps(result.payload(), sort_keys=True) if machine else f"System X verify-resilience/{profile}: {result.status}"); return 0 if result.status == "PASS" else 1
