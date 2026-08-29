"""Focused tests for the isolated Inspector launcher."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InspectorLauncherTests(unittest.TestCase):
    def test_regular_executable_and_self_relative_invocation(self) -> None:
        launcher = ROOT / "run_inspector.py"
        self.assertFalse(launcher.is_symlink())
        self.assertTrue(stat.S_ISREG(launcher.stat().st_mode))
        self.assertEqual(stat.S_IMODE(launcher.stat().st_mode), 0o755)
        completed = subprocess.run([sys.executable, "-B", "-S", str(launcher), "identify"], cwd="/tmp", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(json.loads(completed.stdout)["operation"], "identify")
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
