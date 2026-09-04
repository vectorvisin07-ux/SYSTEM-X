"""Pure System X domain values.  This module has no platform imports."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar


class DomainError(ValueError):
    """Raised only when a domain value cannot be constructed."""


@dataclass(frozen=True, slots=True)
class _Identity:
    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value.strip() != self.value or any(ord(c) < 32 for c in self.value):
            raise DomainError("identity must be non-empty and printable")

    def __str__(self) -> str:
        return self.value


class ModelId(_Identity):
    pass


class DeploymentId(_Identity):
    pass


class OperationId(_Identity):
    pass


class RequestId(_Identity):
    pass


class TransactionId(_Identity):
    pass


class PublicModelAlias(_Identity):
    pass


class ServiceName(_Identity):
    pass


class ReasonCode(_Identity):
    pass


class InstallationState(StrEnum):
    CLONED = "CLONED"
    INSTALLED = "INSTALLED"


class ServiceState(StrEnum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"


class ReadinessState(StrEnum):
    WAITING = "WAITING"
    READY = "READY"
    DEGRADED = "DEGRADED"


class ModelState(StrEnum):
    ABSENT = "ABSENT"
    READY = "READY"
    UNKNOWN = "UNKNOWN"


class ConnectionState(StrEnum):
    NOT_READY = "NOT_READY"
    READY = "READY"
    STALE = "STALE"


class OperationState(StrEnum):
    REQUESTED = "REQUESTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ErrorCategory(StrEnum):
    CONFIGURATION = "ConfigurationError"
    CONTRACT = "ContractError"
    STATE = "StateError"
    CONFLICT = "ConflictError"
    OWNERSHIP = "OwnershipError"
    PERSISTENCE = "PersistenceError"
    PROCESS = "ProcessError"
    TIMEOUT = "TimeoutError"
    RECOVERY = "RecoveryError"


@dataclass(frozen=True, slots=True)
class SystemXError:
    category: ErrorCategory
    reason: ReasonCode
    public_message: str
    operator_message: str


T = TypeVar("T")
_UNSET = object()


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """Exactly one of value and error is present."""

    value: T | None | object = _UNSET
    error: SystemXError | None = None

    def __post_init__(self) -> None:
        if (self.value is _UNSET) == (self.error is None):
            raise DomainError("Result must contain exactly one of value or error")

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(value=value)

    @classmethod
    def fail(cls, error: SystemXError) -> "Result[T]":
        return cls(error=error)


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    installation: InstallationState
    service: ServiceState
    readiness: ReadinessState
    model: ModelState
    connection: ConnectionState
    model_id: ModelId | None
