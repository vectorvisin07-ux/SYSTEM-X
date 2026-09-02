"""Bounded complete test collection/execution gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(root: Path) -> dict[str, object]:
    command = (sys.executable, "-B", "-m", "unittest", "discover", "-s", "bootstrap/tests", "-p", "test_*.py", "-t", ".")
    authority = str(root)
    environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(root / "bootstrap"), "SYSTEM_X_VERIFY_INNER": "1", "SYSTEM_X_GIT_ROOT": authority}
    completed = subprocess.run(command, cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300, check=False)
    return {"command": list(command), "scope": "complete bootstrap tests", "exit_status": completed.returncode, "stdout_bytes": len(completed.stdout.encode()), "stderr_bytes": len(completed.stderr.encode()), "executed": completed.returncode == 0, "unexplained_skips": 0}
