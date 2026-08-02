"""Closed configuration validation without third-party dependencies."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import SAFETY_MAXIMA, SCHEMA_IDENTITIES
from .errors import InspectorError
from .paths import InspectorPaths


TOP_LEVEL_FIELDS = {
    "schema_version",
    "intake_root",
    "runtime_root",
    "intake_bounds",
    "record_policy",
    "result_roots",
}
BOUND_FIELDS = set(SAFETY_MAXIMA)
POLICY_FIELDS = {
    "status_file_mode",
    "transaction_file_mode",
    "log_file_mode",
}
RESULT_FIELDS = {"inspection", "decision", "handoff", "publication"}


@dataclass(frozen=True)
class ValidatedConfiguration:
    values: dict[str, Any]
    identity: str


def _object(
    value: object, expected_fields: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InspectorError("CONFIG_INVALID", f"{label} must be an object")
    if set(value) != expected_fields:
        raise InspectorError(
            "CONFIG_INVALID", f"{label} fields are incomplete or unknown"
        )
    return value


def _canonical_path(value: object, expected: Path, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise InspectorError("CONFIG_INVALID", f"{label} must be a path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise InspectorError("CONFIG_INVALID", f"{label} must be absolute")
    if candidate.resolve(strict=False) != expected.resolve(strict=False):
        raise InspectorError(
            "CONFIG_INVALID", f"{label} does not match the Inspector layout"
        )
    return str(expected.resolve(strict=False))


def validate_configuration_values(
    value: object, paths: InspectorPaths
) -> ValidatedConfiguration:
    document = _object(value, TOP_LEVEL_FIELDS, "configuration")
    if document["schema_version"] != SCHEMA_IDENTITIES["configuration"]:
        raise InspectorError(
            "CONFIG_INVALID", "configuration schema identity is invalid"
        )
    bounds = _object(document["intake_bounds"], BOUND_FIELDS, "intake_bounds")
    canonical_bounds: dict[str, int] = {}
    for name, maximum in SAFETY_MAXIMA.items():
        item = bounds[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise InspectorError(
                "CONFIG_INVALID", f"{name} must be a positive integer"
            )
        if item > maximum:
            raise InspectorError(
                "CONFIG_INVALID", f"{name} exceeds the source safety maximum"
            )
        canonical_bounds[name] = item
    if (
        canonical_bounds["maximum_component_bytes"]
        > canonical_bounds["maximum_relative_path_bytes"]
    ):
        raise InspectorError(
            "CONFIG_INVALID",
            "maximum_component_bytes exceeds maximum_relative_path_bytes",
        )
    policy = _object(document["record_policy"], POLICY_FIELDS, "record_policy")
    canonical_policy: dict[str, str] = {}
    for name in sorted(POLICY_FIELDS):
        if policy[name] != "0600":
            raise InspectorError(
                "CONFIG_INVALID", f"{name} must preserve private mode 0600"
            )
        canonical_policy[name] = "0600"
    roots = _object(document["result_roots"], RESULT_FIELDS, "result_roots")
    canonical = {
        "schema_version": SCHEMA_IDENTITIES["configuration"],
        "intake_root": _canonical_path(
            document["intake_root"], paths.intake_root, "intake_root"
        ),
        "runtime_root": _canonical_path(
            document["runtime_root"], paths.runtime_root, "runtime_root"
        ),
        "intake_bounds": canonical_bounds,
        "record_policy": canonical_policy,
        "result_roots": {
            "inspection": _canonical_path(
                roots["inspection"],
                paths.inspection_results,
                "result_roots.inspection",
            ),
            "decision": _canonical_path(
                roots["decision"],
                paths.decision_results,
                "result_roots.decision",
            ),
            "handoff": _canonical_path(
                roots["handoff"],
                paths.handoff_results,
                "result_roots.handoff",
            ),
            "publication": _canonical_path(
                roots["publication"],
                paths.publication_results,
                "result_roots.publication",
            ),
        },
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ValidatedConfiguration(
        values=canonical,
        identity="sha256:" + hashlib.sha256(encoded).hexdigest(),
    )


def load_configuration(
    path: Path,
    paths: InspectorPaths,
    *,
    allow_external_packet_fixture: bool = False,
) -> ValidatedConfiguration:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError("CONFIG_INVALID", "configuration does not exist") from error
    if stat.S_ISLNK(details.st_mode):
        raise InspectorError(
            "CONFIG_SYMLINK_REJECTED", "configuration symlink rejected"
        )
    if not stat.S_ISREG(details.st_mode):
        raise InspectorError(
            "CONFIG_INVALID", "configuration must be a regular file"
        )
    resolved = path.resolve(strict=True)
    if (
        not allow_external_packet_fixture
        and not resolved.is_relative_to(paths.inspector_root)
    ):
        raise InspectorError(
            "CONFIG_INVALID", "configuration is outside the Inspector root"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspectorError(
            "CONFIG_INVALID", "configuration is not valid UTF-8 JSON"
        ) from error
    return validate_configuration_values(value, paths)
