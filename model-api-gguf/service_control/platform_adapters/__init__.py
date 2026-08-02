"""Portable platform-service adapter contracts for System X.

Importing this package performs no registration, process, listener, or
service-manager action.
"""

from .contract import (
    ACTIVATION_METHOD,
    ADAPTER_IDENTITY,
    ADAPTER_VERSION,
    MANIFEST_SCHEMA,
    OPERATIONS,
    PLATFORM_FAMILY,
    REQUIRED_HOST_CAPABILITIES,
    RESULT_SCHEMA,
    STATUS_SCHEMA,
    AdapterError,
    PlatformServiceAdapter,
    compute_configuration_identity,
    result_envelope,
)
from .registry import (
    LINUX_SYSTEMD_USER_ADAPTER_IDENTITY,
    available_adapter_identities,
    create_adapter,
)

__all__ = [
    "ACTIVATION_METHOD",
    "ADAPTER_IDENTITY",
    "ADAPTER_VERSION",
    "MANIFEST_SCHEMA",
    "LINUX_SYSTEMD_USER_ADAPTER_IDENTITY",
    "OPERATIONS",
    "PLATFORM_FAMILY",
    "REQUIRED_HOST_CAPABILITIES",
    "RESULT_SCHEMA",
    "STATUS_SCHEMA",
    "AdapterError",
    "PlatformServiceAdapter",
    "available_adapter_identities",
    "compute_configuration_identity",
    "create_adapter",
    "result_envelope",
]
