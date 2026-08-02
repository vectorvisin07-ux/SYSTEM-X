"""Portable, standard-library-only System X bootstrap."""

from __future__ import annotations

from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, discover_repository_root
from .result import MachineResult

__all__ = [
    "BOOTSTRAP_VERSION",
    "BootstrapError",
    "ErrorCode",
    "MachineResult",
    "RepositoryPaths",
    "discover_repository_root",
]

BOOTSTRAP_VERSION = "1.0.0"
