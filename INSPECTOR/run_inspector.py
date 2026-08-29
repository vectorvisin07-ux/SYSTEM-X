#!/usr/bin/env python3.14
"""Self-relative Inspector launcher for isolated product invocations."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve(strict=True).parent
sys.path.insert(0, str(ROOT))

from system_x_inspector.machine import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
