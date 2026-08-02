"""Explicit bounded registry for platform-service adapters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .contract import (
    ADAPTER_IDENTITY,
    AdapterError,
    PlatformServiceAdapter,
)

if TYPE_CHECKING:
    from .foreground import ForegroundProcessHostAdapter
    from .linux_systemd_user import LinuxSystemdUserServiceAdapter


LINUX_SYSTEMD_USER_ADAPTER_IDENTITY = (
    "system-x.linux-systemd-user-service.v1"
)

_BUILT_IN_IDENTITIES = frozenset(
    (ADAPTER_IDENTITY, LINUX_SYSTEMD_USER_ADAPTER_IDENTITY)
)


def available_adapter_identities() -> tuple[str, ...]:
    return tuple(sorted(_BUILT_IN_IDENTITIES))


def create_adapter(
    adapter_identity: str,
    adapter_runtime_root: Path,
) -> PlatformServiceAdapter:
    """Instantiate only an exact built-in identity.

    No user value is interpreted as a module name, path, command, or entry
    point.  The sole local import occurs only after exact identity selection.
    """

    if adapter_identity not in _BUILT_IN_IDENTITIES:
        raise AdapterError(
            "ADAPTER_NOT_SUPPORTED",
            "the requested adapter identity is not registered",
            data={
                "requested_adapter_identity": str(adapter_identity)[:256],
                "available_adapter_identities": (
                    available_adapter_identities()
                ),
            },
        )
    if adapter_identity == ADAPTER_IDENTITY:
        from .foreground import ForegroundProcessHostAdapter

        return ForegroundProcessHostAdapter(adapter_runtime_root)
    from .linux_systemd_user import LinuxSystemdUserServiceAdapter

    return LinuxSystemdUserServiceAdapter(adapter_runtime_root)
