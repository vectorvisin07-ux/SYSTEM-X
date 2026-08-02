#!/usr/bin/env python3
"""Self-relative System X bootstrap launcher."""

from __future__ import annotations

from pathlib import Path
import sys


BOOTSTRAP_ROOT = Path(__file__).resolve(strict=True).parent
sys.path.insert(0, str(BOOTSTRAP_ROOT))

from system_x_bootstrap.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
