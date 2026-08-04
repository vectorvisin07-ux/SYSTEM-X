#!/usr/bin/env python3
"""Validate System X operating profiles and atomically store desired state.

This operating-profile module deliberately contains no process, listener, model,
or platform service-manager behavior.  It is a standard-library-only configuration
component intended for later supervision code.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, NoReturn, Sequence
import unicodedata


OPERATING_PROFILE_SCHEMA = "system-x.service-operating-profile.v1"
DESIRED_STATE_SCHEMA = "system-x.service-desired-state.v1"
RESULT_SCHEMA = "system-x.service-control-result.v1"

STARTUP_MODEL_POLICY = "always_warm"
DESIRED_STATES = frozenset(("RUNNING", "STOPPED"))

MAX_ALIAS_CHARACTERS = 128
MAX_GRACEFUL_TIMEOUT_SECONDS = 3_600.0
MAX_RECOVERY_DELAY_SECONDS = 86_400.0
MAX_RECOVERY_MULTIPLIER = 16.0
DEFAULT_MAXIMUM_ATTEMPTS_IN_WINDOW = 3
DEFAULT_ATTEMPT_WINDOW_SECONDS = 60.0
DEFAULT_STABLE_RESET_SECONDS = 30.0
MAX_JSON_BYTES = 1_048_576

SERVICE_CONTROL_DIR = Path(__file__).resolve().parent
BRANCH_ROOT = SERVICE_CONTROL_DIR.parent
DEFAULT_RUNTIME_DIR = BRANCH_ROOT / "RUNTIME" / "service_control"
DEFAULT_PROFILE_PATH = DEFAULT_RUNTIME_DIR / "operating-profile.json"
DEFAULT_DESIRED_STATE_PATH = DEFAULT_RUNTIME_DIR / "desired-state.json"

_PROFILE_REQUIRED_FIELDS = frozenset(
    (
        "schema_version",
        "public_endpoint",
        "private_router_endpoint",
        "default_model_alias",
        "startup_model_policy",
        "automatic_recovery_enabled",
        "graceful_shutdown",
        "recovery_delay",
    )
)
_PROFILE_FIELDS = _PROFILE_REQUIRED_FIELDS | frozenset(("recovery_loop",))
_ENDPOINT_FIELDS = frozenset(("host", "port"))
_GRACEFUL_FIELDS = frozenset(("enabled", "timeout_seconds"))
_RECOVERY_FIELDS = frozenset(
    ("initial_seconds", "maximum_seconds", "multiplier")
)
_RECOVERY_LOOP_FIELDS = frozenset(
    (
        "maximum_attempts_in_window",
        "attempt_window_seconds",
        "stable_reset_seconds",
    )
)
_DESIRED_FIELDS = frozenset(
    (
        "schema_version",
        "profile_identity",
        "desired_state",
        "generation",
        "updated_utc",
    )
)
_IDENTITY_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_UTC_PATTERN = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z"
)


class ServiceControlError(ValueError):
    """A stable, machine-readable domain failure."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def as_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}


@dataclass(frozen=True)
class OperatingProfile:
    public_endpoint: Endpoint
    private_router_endpoint: Endpoint
    default_model_alias: str
    startup_model_policy: str
    automatic_recovery_enabled: bool
    graceful_shutdown_enabled: bool
    graceful_shutdown_timeout_seconds: float
    recovery_delay_initial_seconds: float
    recovery_delay_maximum_seconds: float
    recovery_delay_multiplier: float
    recovery_maximum_attempts_in_window: int
    recovery_attempt_window_seconds: float
    recovery_stable_reset_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPERATING_PROFILE_SCHEMA,
            "public_endpoint": self.public_endpoint.as_dict(),
            "private_router_endpoint": self.private_router_endpoint.as_dict(),
            "default_model_alias": self.default_model_alias,
            "startup_model_policy": self.startup_model_policy,
            "automatic_recovery_enabled": self.automatic_recovery_enabled,
            "graceful_shutdown": {
                "enabled": self.graceful_shutdown_enabled,
                "timeout_seconds": self.graceful_shutdown_timeout_seconds,
            },
            "recovery_delay": {
                "initial_seconds": self.recovery_delay_initial_seconds,
                "maximum_seconds": self.recovery_delay_maximum_seconds,
                "multiplier": self.recovery_delay_multiplier,
            },
            "recovery_loop": {
                "maximum_attempts_in_window": (
                    self.recovery_maximum_attempts_in_window
                ),
                "attempt_window_seconds": (
                    self.recovery_attempt_window_seconds
                ),
                "stable_reset_seconds": self.recovery_stable_reset_seconds,
            },
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.as_dict())

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


@dataclass(frozen=True)
class DesiredState:
    profile_identity: str
    desired_state: str
    generation: int
    updated_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DESIRED_STATE_SCHEMA,
            "profile_identity": self.profile_identity,
            "desired_state": self.desired_state,
            "generation": self.generation,
            "updated_utc": self.updated_utc,
        }


def _fail(reason_code: str, message: str) -> NoReturn:
    raise ServiceControlError(reason_code, message)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        _fail("serialization_failed", f"JSON serialization failed: {exc}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail("invalid_type", f"{label} must be a JSON object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], required: frozenset[str], label: str
) -> None:
    present = frozenset(value)
    unknown = sorted(present - required)
    missing = sorted(required - present)
    if unknown:
        _fail("unknown_field", f"{label} has unknown fields: {unknown}")
    if missing:
        _fail("missing_field", f"{label} is missing fields: {missing}")


def _require_boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail("invalid_boolean", f"{label} must be a Boolean")
    return value


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool,
) -> float:
    if type(value) not in (int, float):
        _fail("invalid_number", f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        _fail("non_finite_number", f"{label} must be finite")
    below = number < minimum if minimum_inclusive else number <= minimum
    if below or number > maximum:
        relation = ">=" if minimum_inclusive else ">"
        _fail(
            "number_out_of_range",
            f"{label} must be {relation} {minimum} and <= {maximum}",
        )
    return number


def _validate_endpoint(value: Any, label: str) -> Endpoint:
    endpoint = _require_mapping(value, label)
    _require_exact_fields(endpoint, _ENDPOINT_FIELDS, label)
    host = endpoint["host"]
    if not isinstance(host, str):
        _fail("invalid_endpoint_host", f"{label}.host must be a string")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        _fail(
            "invalid_endpoint_host",
            f"{label}.host must be a numeric loopback address",
        )
    if not address.is_loopback:
        _fail(
            "non_loopback_endpoint",
            f"{label}.host must be a numeric loopback address",
        )
    port = endpoint["port"]
    if type(port) is not int or not 1 <= port <= 65_535:
        _fail(
            "invalid_endpoint_port",
            f"{label}.port must be an integer in 1..65535",
        )
    return Endpoint(host=address.compressed, port=port)


def _validate_alias(value: Any) -> str:
    if not isinstance(value, str):
        _fail("invalid_model_alias", "default_model_alias must be a string")
    if not value or len(value) > MAX_ALIAS_CHARACTERS:
        _fail(
            "invalid_model_alias",
            f"default_model_alias must contain 1..{MAX_ALIAS_CHARACTERS} characters",
        )
    if value != value.strip():
        _fail(
            "invalid_model_alias",
            "default_model_alias must not contain surrounding whitespace",
        )
    if "/" in value or "\\" in value:
        _fail(
            "invalid_model_alias",
            "default_model_alias must not contain a path separator",
        )
    if any(unicodedata.category(char).startswith("C") for char in value):
        _fail(
            "invalid_model_alias",
            "default_model_alias must not contain control characters",
        )
    return value


def validate_operating_profile(value: Any) -> OperatingProfile:
    """Validate and normalize a decoded operating-profile object."""

    profile = _require_mapping(value, "operating profile")
    present = frozenset(profile)
    unknown = sorted(present - _PROFILE_FIELDS)
    missing = sorted(_PROFILE_REQUIRED_FIELDS - present)
    if unknown:
        _fail(
            "unknown_field",
            f"operating profile has unknown fields: {unknown}",
        )
    if missing:
        _fail(
            "missing_field",
            f"operating profile is missing fields: {missing}",
        )
    if profile["schema_version"] != OPERATING_PROFILE_SCHEMA:
        _fail(
            "unsupported_profile_schema",
            f"schema_version must be {OPERATING_PROFILE_SCHEMA}",
        )

    public_endpoint = _validate_endpoint(
        profile["public_endpoint"], "public_endpoint"
    )
    private_endpoint = _validate_endpoint(
        profile["private_router_endpoint"], "private_router_endpoint"
    )
    if public_endpoint == private_endpoint:
        _fail(
            "duplicate_endpoint",
            "public_endpoint and private_router_endpoint must be distinct",
        )

    startup_policy = profile["startup_model_policy"]
    if startup_policy != STARTUP_MODEL_POLICY:
        _fail(
            "unsupported_startup_model_policy",
            f"startup_model_policy must be {STARTUP_MODEL_POLICY}",
        )

    graceful = _require_mapping(
        profile["graceful_shutdown"], "graceful_shutdown"
    )
    _require_exact_fields(graceful, _GRACEFUL_FIELDS, "graceful_shutdown")

    recovery = _require_mapping(profile["recovery_delay"], "recovery_delay")
    _require_exact_fields(recovery, _RECOVERY_FIELDS, "recovery_delay")
    recovery_loop_value = profile.get(
        "recovery_loop",
        {
            "maximum_attempts_in_window": (
                DEFAULT_MAXIMUM_ATTEMPTS_IN_WINDOW
            ),
            "attempt_window_seconds": DEFAULT_ATTEMPT_WINDOW_SECONDS,
            "stable_reset_seconds": DEFAULT_STABLE_RESET_SECONDS,
        },
    )
    recovery_loop = _require_mapping(recovery_loop_value, "recovery_loop")
    _require_exact_fields(
        recovery_loop, _RECOVERY_LOOP_FIELDS, "recovery_loop"
    )
    maximum_attempts = recovery_loop["maximum_attempts_in_window"]
    if (
        type(maximum_attempts) is not int
        or not 1 <= maximum_attempts <= 16
    ):
        _fail(
            "invalid_recovery_loop",
            "recovery_loop.maximum_attempts_in_window must be an integer in 1..16",
        )

    initial_seconds = _finite_number(
        recovery["initial_seconds"],
        "recovery_delay.initial_seconds",
        minimum=0.0,
        maximum=MAX_RECOVERY_DELAY_SECONDS,
        minimum_inclusive=True,
    )
    maximum_seconds = _finite_number(
        recovery["maximum_seconds"],
        "recovery_delay.maximum_seconds",
        minimum=0.0,
        maximum=MAX_RECOVERY_DELAY_SECONDS,
        minimum_inclusive=False,
    )
    if initial_seconds > maximum_seconds:
        _fail(
            "invalid_recovery_delay_relation",
            "recovery_delay.initial_seconds must not exceed maximum_seconds",
        )

    return OperatingProfile(
        public_endpoint=public_endpoint,
        private_router_endpoint=private_endpoint,
        default_model_alias=_validate_alias(profile["default_model_alias"]),
        startup_model_policy=startup_policy,
        automatic_recovery_enabled=_require_boolean(
            profile["automatic_recovery_enabled"],
            "automatic_recovery_enabled",
        ),
        graceful_shutdown_enabled=_require_boolean(
            graceful["enabled"], "graceful_shutdown.enabled"
        ),
        graceful_shutdown_timeout_seconds=_finite_number(
            graceful["timeout_seconds"],
            "graceful_shutdown.timeout_seconds",
            minimum=0.0,
            maximum=MAX_GRACEFUL_TIMEOUT_SECONDS,
            minimum_inclusive=False,
        ),
        recovery_delay_initial_seconds=initial_seconds,
        recovery_delay_maximum_seconds=maximum_seconds,
        recovery_delay_multiplier=_finite_number(
            recovery["multiplier"],
            "recovery_delay.multiplier",
            minimum=1.0,
            maximum=MAX_RECOVERY_MULTIPLIER,
            minimum_inclusive=True,
        ),
        recovery_maximum_attempts_in_window=maximum_attempts,
        recovery_attempt_window_seconds=_finite_number(
            recovery_loop["attempt_window_seconds"],
            "recovery_loop.attempt_window_seconds",
            minimum=1.0,
            maximum=3_600.0,
            minimum_inclusive=True,
        ),
        recovery_stable_reset_seconds=_finite_number(
            recovery_loop["stable_reset_seconds"],
            "recovery_loop.stable_reset_seconds",
            minimum=1.0,
            maximum=3_600.0,
            minimum_inclusive=True,
        ),
    )


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    chain = list(reversed(absolute.parents)) + [absolute]
    for component in chain:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail("path_inspection_failed", f"{label} cannot be inspected: {exc}")
        if stat.S_ISLNK(metadata.st_mode):
            _fail("symlink_rejected", f"{label} contains a symlink component")


def _read_regular_file(path: Path, label: str) -> bytes:
    path = Path(path)
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        _fail("missing_file", f"{label} does not exist")
    except OSError as exc:
        if getattr(exc, "errno", None) == getattr(os, "ELOOP", object()):
            _fail("symlink_rejected", f"{label} must not be a symlink")
        _fail("file_open_failed", f"{label} cannot be opened: {exc}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail("not_regular_file", f"{label} must be a regular file")
        if metadata.st_size > MAX_JSON_BYTES:
            _fail("file_too_large", f"{label} exceeds {MAX_JSON_BYTES} bytes")
        chunks: list[bytes] = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_JSON_BYTES:
            _fail("file_too_large", f"{label} exceeds {MAX_JSON_BYTES} bytes")
        return data
    finally:
        os.close(descriptor)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Any:
    data = _read_regular_file(path, label)
    if b"\x00" in data:
        _fail("nul_byte_rejected", f"{label} contains a NUL byte")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail("invalid_utf8", f"{label} is not valid UTF-8: {exc}")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: _fail(
                "non_finite_number", f"non-finite JSON number rejected: {token}"
            ),
        )
    except ServiceControlError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        _fail("invalid_json", f"{label} is not valid JSON: {exc}")


def load_operating_profile(path: Path | str = DEFAULT_PROFILE_PATH) -> OperatingProfile:
    """Load, validate, normalize, and identify an operating profile."""

    return validate_operating_profile(
        _load_json(Path(path), "operating profile file")
    )


def _validate_profile_identity(value: Any) -> str:
    if not isinstance(value, str) or _IDENTITY_PATTERN.fullmatch(value) is None:
        _fail(
            "invalid_profile_identity",
            "profile_identity must be sha256 plus 64 lowercase hexadecimal digits",
        )
    return value


def _validate_desired_state_name(value: Any) -> str:
    if value not in DESIRED_STATES:
        _fail("invalid_desired_state", "desired_state must be RUNNING or STOPPED")
    return value


def _validate_updated_utc(value: Any) -> str:
    if not isinstance(value, str) or _UTC_PATTERN.fullmatch(value) is None:
        _fail(
            "invalid_updated_utc",
            "updated_utc must be normalized UTC with six fractional digits",
        )
    try:
        parsed = _datetime.datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    except ValueError:
        _fail("invalid_updated_utc", "updated_utc is not a valid UTC timestamp")
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        _fail("invalid_updated_utc", "updated_utc is not normalized")
    return value


def validate_desired_state(
    value: Any, expected_profile_identity: str | None = None
) -> DesiredState:
    """Validate a decoded desired-state object and its optional profile binding."""

    state_value = _require_mapping(value, "desired state")
    _require_exact_fields(state_value, _DESIRED_FIELDS, "desired state")
    if state_value["schema_version"] != DESIRED_STATE_SCHEMA:
        _fail(
            "unsupported_desired_state_schema",
            f"schema_version must be {DESIRED_STATE_SCHEMA}",
        )
    profile_identity = _validate_profile_identity(
        state_value["profile_identity"]
    )
    if (
        expected_profile_identity is not None
        and profile_identity != expected_profile_identity
    ):
        _fail(
            "profile_identity_mismatch",
            "desired state is bound to a different operating profile",
        )
    generation = state_value["generation"]
    if type(generation) is not int or generation < 1:
        _fail("invalid_generation", "generation must be a positive integer")
    return DesiredState(
        profile_identity=profile_identity,
        desired_state=_validate_desired_state_name(
            state_value["desired_state"]
        ),
        generation=generation,
        updated_utc=_validate_updated_utc(state_value["updated_utc"]),
    )


def load_desired_state(
    path: Path | str = DEFAULT_DESIRED_STATE_PATH,
    expected_profile_identity: str | None = None,
) -> DesiredState:
    """Load a complete desired-state file and enforce its profile binding."""

    return validate_desired_state(
        _load_json(Path(path), "desired-state file"),
        expected_profile_identity=expected_profile_identity,
    )


def _normalized_utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    parent = path.parent
    _reject_symlink_components(parent, "desired-state directory")
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        _fail(
            "state_directory_failed",
            f"desired-state directory cannot be created: {exc}",
        )
    _reject_symlink_components(parent, "desired-state directory")
    try:
        parent_metadata = parent.stat()
    except OSError as exc:
        _fail(
            "state_directory_failed",
            f"desired-state directory cannot be inspected: {exc}",
        )
    if not stat.S_ISDIR(parent_metadata.st_mode):
        _fail(
            "state_directory_failed",
            "desired-state parent must be a directory",
        )

    existing_mode: int | None = None
    try:
        target_metadata = path.lstat()
    except FileNotFoundError:
        target_metadata = None
    except OSError as exc:
        _fail("path_inspection_failed", f"desired-state target failed: {exc}")
    if target_metadata is not None:
        if stat.S_ISLNK(target_metadata.st_mode):
            _fail("symlink_rejected", "desired-state target must not be a symlink")
        if not stat.S_ISREG(target_metadata.st_mode):
            _fail(
                "not_regular_file",
                "desired-state target must be a regular file or absent",
            )
        existing_mode = stat.S_IMODE(target_metadata.st_mode)

    payload = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor: int | None = None
    owned_temporary_path: str | None = None
    try:
        try:
            descriptor, owned_temporary_path = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=parent,
            )
            final_mode = (
                0o600 if existing_mode is None else existing_mode & 0o600
            )
            os.fchmod(descriptor, final_mode)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while persisting desired state")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(owned_temporary_path, path)
            owned_temporary_path = None
        except ServiceControlError:
            raise
        except OSError as exc:
            _fail(
                "atomic_state_write_failed",
                f"atomic desired-state write failed: {exc}",
            )

        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        try:
            directory_descriptor = os.open(parent, directory_flags)
        except OSError as exc:
            _fail(
                "directory_sync_failed",
                f"desired-state directory open failed after replace: {exc}",
            )
        try:
            try:
                os.fsync(directory_descriptor)
            except OSError as exc:
                _fail(
                    "directory_sync_failed",
                    f"desired-state directory sync failed after replace: {exc}",
                )
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if owned_temporary_path is not None:
            try:
                os.unlink(owned_temporary_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def initialize_desired_state(
    profile: OperatingProfile,
    path: Path | str = DEFAULT_DESIRED_STATE_PATH,
    desired_state: str = "STOPPED",
    *,
    updated_utc: str | None = None,
) -> DesiredState:
    """Create generation one, failing if any target already exists."""

    state_path = Path(path)
    _validate_desired_state_name(desired_state)
    _reject_symlink_components(state_path, "desired-state target")
    try:
        state_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        _fail("path_inspection_failed", f"desired-state target failed: {exc}")
    else:
        _fail(
            "desired_state_already_exists",
            "desired-state target already exists",
        )
    state = DesiredState(
        profile_identity=profile.identity,
        desired_state=desired_state,
        generation=1,
        updated_utc=_validate_updated_utc(
            updated_utc if updated_utc is not None else _normalized_utc_now()
        ),
    )
    _atomic_write_json(state_path, state.as_dict())
    return state


def set_desired_state(
    profile: OperatingProfile,
    desired_state: str,
    path: Path | str = DEFAULT_DESIRED_STATE_PATH,
    *,
    expected_generation: int | None = None,
    updated_utc: str | None = None,
) -> DesiredState:
    """Atomically update a profile-bound desired state."""

    state_path = Path(path)
    requested_state = _validate_desired_state_name(desired_state)
    current = load_desired_state(
        state_path, expected_profile_identity=profile.identity
    )
    if expected_generation is not None:
        if type(expected_generation) is not int or expected_generation < 1:
            _fail(
                "invalid_expected_generation",
                "expected_generation must be a positive integer",
            )
        if current.generation != expected_generation:
            _fail(
                "stale_expected_generation",
                "expected_generation does not match the current generation",
            )
    next_state = DesiredState(
        profile_identity=profile.identity,
        desired_state=requested_state,
        generation=current.generation + 1,
        updated_utc=_validate_updated_utc(
            updated_utc if updated_utc is not None else _normalized_utc_now()
        ),
    )
    _atomic_write_json(state_path, next_state.as_dict())
    return next_state


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ServiceControlError("invalid_arguments", message)


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        "--profile-path",
        dest="profile_path",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
    )


def _add_state_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-path",
        "--desired-state-path",
        dest="state_path",
        type=Path,
        default=DEFAULT_DESIRED_STATE_PATH,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="system-x-service-control")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    for name in ("validate-profile", "show-profile"):
        operation_parser = subparsers.add_parser(name)
        _add_profile_argument(operation_parser)

    initialize_parser = subparsers.add_parser("initialize-desired-state")
    _add_profile_argument(initialize_parser)
    _add_state_argument(initialize_parser)
    initialize_parser.add_argument(
        "--state",
        "--desired-state",
        dest="requested_state",
        choices=sorted(DESIRED_STATES),
        default="STOPPED",
    )

    show_state_parser = subparsers.add_parser("show-desired-state")
    _add_profile_argument(show_state_parser)
    _add_state_argument(show_state_parser)

    set_state_parser = subparsers.add_parser("set-desired-state")
    _add_profile_argument(set_state_parser)
    _add_state_argument(set_state_parser)
    set_state_parser.add_argument(
        "--state",
        "--desired-state",
        dest="requested_state",
        choices=sorted(DESIRED_STATES),
        required=True,
    )
    set_state_parser.add_argument("--expected-generation", type=int)
    return parser


def _resolved(path: Path) -> str:
    return os.path.abspath(os.fspath(path))


def _result(
    *,
    operation: str,
    ok: bool,
    reason_code: str,
    message: str,
    profile_path: Path | None,
    state_path: Path | None,
    profile: OperatingProfile | None = None,
    state: DesiredState | None = None,
    include_profile: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "operation": operation,
        "ok": ok,
        "reason_code": reason_code,
        "message": message,
        "profile_identity": profile.identity if profile is not None else None,
        "profile": (
            profile.as_dict()
            if include_profile and profile is not None
            else None
        ),
        "desired_state": state.desired_state if state is not None else None,
        "generation": state.generation if state is not None else None,
        "updated_utc": state.updated_utc if state is not None else None,
        "resolved_paths": {
            "profile": _resolved(profile_path) if profile_path is not None else None,
            "desired_state": (
                _resolved(state_path) if state_path is not None else None
            ),
        },
    }


def execute_operation(arguments: argparse.Namespace) -> dict[str, Any]:
    operation = arguments.operation
    profile_path: Path = arguments.profile_path
    state_path: Path | None = getattr(arguments, "state_path", None)
    profile = load_operating_profile(profile_path)

    if operation == "validate-profile":
        return _result(
            operation=operation,
            ok=True,
            reason_code="ok",
            message="operating profile is valid",
            profile_path=profile_path,
            state_path=None,
            profile=profile,
        )
    if operation == "show-profile":
        return _result(
            operation=operation,
            ok=True,
            reason_code="ok",
            message="operating profile loaded",
            profile_path=profile_path,
            state_path=None,
            profile=profile,
            include_profile=True,
        )
    if operation == "initialize-desired-state":
        state = initialize_desired_state(
            profile,
            state_path,
            arguments.requested_state,
        )
    elif operation == "show-desired-state":
        state = load_desired_state(
            state_path, expected_profile_identity=profile.identity
        )
    elif operation == "set-desired-state":
        state = set_desired_state(
            profile,
            arguments.requested_state,
            state_path,
            expected_generation=arguments.expected_generation,
        )
    else:
        _fail("unknown_operation", f"unsupported operation: {operation}")
    return _result(
        operation=operation,
        ok=True,
        reason_code="ok",
        message="desired state operation completed",
        profile_path=profile_path,
        state_path=state_path,
        profile=profile,
        state=state,
    )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    operation = values[0] if values else "argument-error"
    profile_path: Path | None = None
    state_path: Path | None = None
    try:
        arguments = build_argument_parser().parse_args(values)
        operation = arguments.operation
        profile_path = arguments.profile_path
        state_path = getattr(arguments, "state_path", None)
        result = execute_operation(arguments)
        exit_code = 0
    except ServiceControlError as exc:
        result = _result(
            operation=operation,
            ok=False,
            reason_code=exc.reason_code,
            message=exc.message,
            profile_path=profile_path,
            state_path=state_path,
        )
        exit_code = 2
    except Exception as exc:  # defensive machine-contract boundary
        result = _result(
            operation=operation,
            ok=False,
            reason_code="unexpected_error",
            message=f"{type(exc).__name__}: {exc}",
            profile_path=profile_path,
            state_path=state_path,
        )
        print(
            f"system-x-service-control unexpected error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        exit_code = 70
    print(_canonical_json(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
