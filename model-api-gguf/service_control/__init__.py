"""Stable operating-profile and desired-state primitives for System X.

This package is configuration-only.  Importing it does not start, stop, or
inspect any service.
"""

from .operating_profile import (
    DEFAULT_DESIRED_STATE_PATH,
    DEFAULT_PROFILE_PATH,
    DESIRED_STATE_SCHEMA,
    OPERATING_PROFILE_SCHEMA,
    RESULT_SCHEMA,
    DesiredState,
    Endpoint,
    OperatingProfile,
    ServiceControlError,
    initialize_desired_state,
    load_desired_state,
    load_operating_profile,
    set_desired_state,
)
from .supervisor import (
    BRANCH_CONTROLLER_SCHEMA,
    RESULT_SCHEMA as SUPERVISOR_RESULT_SCHEMA,
    STATUS_SCHEMA,
    TRANSACTION_SCHEMA,
    ControllerAdapter,
    ForegroundSupervisor,
    SupervisorError,
    SupervisorPaths,
    administrative_status,
    administrative_stop,
    administrative_reset_recovery,
)
from .recovery import (
    RECOVERY_REASON_CODES,
    RECOVERY_STATES,
    RecoveryAttempt,
    RecoveryError,
    RecoveryPolicy,
    RecoveryStore,
)

__all__ = [
    "DEFAULT_DESIRED_STATE_PATH",
    "DEFAULT_PROFILE_PATH",
    "DESIRED_STATE_SCHEMA",
    "OPERATING_PROFILE_SCHEMA",
    "RESULT_SCHEMA",
    "DesiredState",
    "Endpoint",
    "OperatingProfile",
    "ServiceControlError",
    "initialize_desired_state",
    "load_desired_state",
    "load_operating_profile",
    "set_desired_state",
    "BRANCH_CONTROLLER_SCHEMA",
    "SUPERVISOR_RESULT_SCHEMA",
    "STATUS_SCHEMA",
    "TRANSACTION_SCHEMA",
    "ControllerAdapter",
    "ForegroundSupervisor",
    "SupervisorError",
    "SupervisorPaths",
    "administrative_status",
    "administrative_stop",
    "administrative_reset_recovery",
    "RECOVERY_REASON_CODES",
    "RECOVERY_STATES",
    "RecoveryAttempt",
    "RecoveryError",
    "RecoveryPolicy",
    "RecoveryStore",
]
