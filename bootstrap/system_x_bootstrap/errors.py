"""Closed error vocabulary for the portable bootstrap."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping


class ErrorCode(StrEnum):
    AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"
    BOOTSTRAP_ENVIRONMENT_LOCK_MISSING = "BOOTSTRAP_ENVIRONMENT_LOCK_MISSING"
    BUILD_COLLISION = "BUILD_COLLISION"
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"
    CONFIGURATION_MISSING = "CONFIGURATION_MISSING"
    CREDENTIAL_COLLISION = "CREDENTIAL_COLLISION"
    EXTERNAL_COMMAND_FAILED = "EXTERNAL_COMMAND_FAILED"
    HOST_UNSUPPORTED = "HOST_UNSUPPORTED"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    LOCK_HELD = "LOCK_HELD"
    PATH_UNSAFE = "PATH_UNSAFE"
    PLAN_MISMATCH = "PLAN_MISMATCH"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    RUNTIME_COLLISION = "RUNTIME_COLLISION"
    SECRET_POLICY_VIOLATION = "SECRET_POLICY_VIOLATION"
    SUBMODULE_MISMATCH = "SUBMODULE_MISMATCH"
    TRANSACTION_RECOVERY_REQUIRED = "TRANSACTION_RECOVERY_REQUIRED"
    UNKNOWN_STATE = "UNKNOWN_STATE"


class BootstrapError(RuntimeError):
    """Expected fail-closed bootstrap error with non-secret context."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = dict(context or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": self.context,
        }
