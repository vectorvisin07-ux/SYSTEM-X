"""Bounded metadata-only intake validation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

from .constants import FORMAT_CLASSIFICATION, SCHEMA_IDENTITIES
from .errors import InspectorError
from .paths import InspectorPaths


RESERVED_NAMES = frozenset(
    {
        "RUNTIME",
        "results",
        "staging",
        "tmp",
        "locks",
        "status",
        "transactions",
        "logs",
    }
)
PROTECTED_NAME_MARKERS = (
    "model-api-gguf",
    "model-api-native",
    "system-x-chat",
    "OPENCLAW",
)


def _root_ready(paths: InspectorPaths) -> None:
    try:
        details = paths.intake_root.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "LAYOUT_INVALID", "MODEL-TEST intake root does not exist"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise InspectorError(
            "LAYOUT_INVALID",
            "MODEL-TEST intake root is not a regular non-symlink directory",
        )
    if paths.intake_root.resolve(strict=True).parent != paths.inspector_root:
        raise InspectorError(
            "LAYOUT_INVALID", "MODEL-TEST is not a direct Inspector child"
        )


def _visible_name(name: str) -> bool:
    return bool(name) and not name.startswith(".") and name not in RESERVED_NAMES


def _entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "regular_directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def _basic_metadata(path: Path, *, relative_name: str) -> dict[str, Any]:
    details = path.lstat()
    return {
        "relative_name": relative_name,
        "root_type": _entry_type(details.st_mode),
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": f"{stat.S_IMODE(details.st_mode):04o}",
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
    }


def list_intake(paths: InspectorPaths) -> dict[str, Any]:
    _root_ready(paths)
    candidates = []
    with os.scandir(paths.intake_root) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if not _visible_name(entry.name):
                continue
            candidates.append(
                _basic_metadata(
                    Path(entry.path), relative_name=entry.name
                )
            )
    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "explicit_selection_required": len(candidates) > 1,
        "candidate_content_reads": 0,
    }


def select_target(paths: InspectorPaths, target_name: str | None) -> str:
    if target_name is None:
        candidates = [
            item["relative_name"] for item in list_intake(paths)["candidates"]
        ]
        if not candidates:
            raise InspectorError("INTAKE_EMPTY", "intake root is empty")
        if len(candidates) > 1:
            raise InspectorError(
                "INTAKE_MULTIPLE_CANDIDATES",
                "multiple candidates require explicit selection",
                data={"candidate_count": len(candidates)},
            )
        return candidates[0]
    if not isinstance(target_name, str) or not target_name:
        raise InspectorError(
            "INTAKE_TARGET_INVALID", "target name must be non-empty"
        )
    if any(marker in target_name for marker in PROTECTED_NAME_MARKERS):
        raise InspectorError(
            "INTAKE_BRANCH_PATH_REJECTED",
            "protected branch or workspace path rejected",
        )
    candidate = Path(target_name)
    if candidate.is_absolute():
        raise InspectorError(
            "INTAKE_TARGET_OUTSIDE_ROOT", "absolute target path rejected"
        )
    if (
        target_name in {".", ".."}
        or "/" in target_name
        or "\\" in target_name
        or not _visible_name(target_name)
    ):
        raise InspectorError(
            "INTAKE_TARGET_INVALID",
            "target must identify one visible direct child",
        )
    return target_name


def _metadata(path: Path, relative_path: str) -> dict[str, Any]:
    details = path.lstat()
    return {
        "relative_path": relative_path,
        "entry_type": _entry_type(details.st_mode),
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": f"{stat.S_IMODE(details.st_mode):04o}",
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
    }


def _check_path_bounds(
    relative: str,
    *,
    depth: int,
    bounds: dict[str, int],
) -> None:
    if depth > bounds["maximum_directory_depth"]:
        raise InspectorError(
            "INTAKE_DEPTH_EXCEEDED", "directory depth bound exceeded"
        )
    encoded = relative.encode("utf-8")
    if len(encoded) > bounds["maximum_relative_path_bytes"]:
        raise InspectorError(
            "INTAKE_PATH_LENGTH_EXCEEDED",
            "relative path byte bound exceeded",
        )
    for component in Path(relative).parts:
        if len(component.encode("utf-8")) > bounds["maximum_component_bytes"]:
            raise InspectorError(
                "INTAKE_PATH_LENGTH_EXCEEDED",
                "path component byte bound exceeded",
            )


def _protected_roots(paths: InspectorPaths) -> tuple[Path, ...]:
    system_x_root = paths.inspector_root.parent
    systems_root = system_x_root.parent
    return (
        system_x_root / "model-api-gguf",
        system_x_root / "model-api-native",
        systems_root / "system-x-chat",
    )


def _ensure_entry_contained(path: Path, paths: InspectorPaths) -> None:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(paths.intake_root.resolve(strict=True)):
        raise InspectorError(
            "INTAKE_TARGET_OUTSIDE_ROOT", "intake entry escaped MODEL-TEST"
        )
    if any(
        resolved.is_relative_to(protected.resolve(strict=False))
        for protected in _protected_roots(paths)
    ):
        raise InspectorError(
            "INTAKE_BRANCH_PATH_REJECTED",
            "intake entry resolves into a protected product root",
        )


def _append_manifest_entry(
    manifest: list[dict[str, Any]],
    path: Path,
    relative: str,
    *,
    depth: int,
    bounds: dict[str, int],
    paths: InspectorPaths,
    root: bool,
) -> None:
    _check_path_bounds(relative, depth=depth, bounds=bounds)
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "INTAKE_CHANGED_DURING_VALIDATION",
            "intake entry disappeared during validation",
        ) from error
    if stat.S_ISLNK(details.st_mode):
        raise InspectorError(
            "INTAKE_TARGET_SYMLINK" if root else "INTAKE_DESCENDANT_SYMLINK",
            "intake symlink rejected",
        )
    if not (stat.S_ISREG(details.st_mode) or stat.S_ISDIR(details.st_mode)):
        raise InspectorError(
            "INTAKE_SPECIAL_FILE", "intake special file rejected"
        )
    if os.path.ismount(path) and path != paths.intake_root:
        raise InspectorError(
            "INTAKE_TARGET_OUTSIDE_ROOT", "mounted intake substitution rejected"
        )
    _ensure_entry_contained(path, paths)
    manifest.append(_metadata(path, relative))
    if len(manifest) > bounds["maximum_entry_count"]:
        raise InspectorError(
            "INTAKE_ENTRY_COUNT_EXCEEDED", "entry count bound exceeded"
        )


def _snapshot(
    target: Path,
    target_name: str,
    *,
    paths: InspectorPaths,
    bounds: dict[str, int],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    _append_manifest_entry(
        manifest,
        target,
        target_name,
        depth=0,
        bounds=bounds,
        paths=paths,
        root=True,
    )
    if manifest[0]["entry_type"] == "regular_file":
        if not os.access(target, os.R_OK, effective_ids=True):
            raise InspectorError(
                "INTAKE_TARGET_INVALID",
                "intake file is not readable for metadata validation",
            )
        return manifest
    pending: list[tuple[Path, str, int]] = [(target, target_name, 0)]
    while pending:
        directory, prefix, directory_depth = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda item: item.name)
        except FileNotFoundError as error:
            raise InspectorError(
                "INTAKE_CHANGED_DURING_VALIDATION",
                "intake directory disappeared during validation",
            ) from error
        next_directories = []
        for entry in children:
            relative = (
                f"{prefix}/{entry.name}" if prefix else entry.name
            )
            path = Path(entry.path)
            depth = directory_depth + 1
            _append_manifest_entry(
                manifest,
                path,
                relative,
                depth=depth,
                bounds=bounds,
                paths=paths,
                root=False,
            )
            if manifest[-1]["entry_type"] == "regular_directory":
                next_directories.append((path, relative, depth))
        pending.extend(reversed(next_directories))
    return manifest


def _snapshot_identity(manifest: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_intake(
    paths: InspectorPaths,
    bounds: dict[str, int],
    target_name: str | None = None,
    *,
    stability_hook: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    _root_ready(paths)
    selected = select_target(paths, target_name)
    target = paths.intake_root / selected
    try:
        target.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "INTAKE_TARGET_INVALID", "selected intake target does not exist"
        ) from error
    first = _snapshot(
        target, selected, paths=paths, bounds=bounds
    )
    if stability_hook is not None:
        stability_hook(target)
    second = _snapshot(
        target, selected, paths=paths, bounds=bounds
    )
    if first != second:
        raise InspectorError(
            "INTAKE_CHANGED_DURING_VALIDATION",
            "intake metadata changed during validation",
        )
    identity = _snapshot_identity(first)
    return {
        "schema_version": SCHEMA_IDENTITIES["intake_candidate"],
        "target_name": selected,
        "root_type": first[0]["entry_type"],
        "metadata_manifest": first,
        "entry_count": len(first),
        "aggregate_declared_file_bytes": sum(
            item["size"]
            for item in first
            if item["entry_type"] == "regular_file"
        ),
        "intake_snapshot_identity": identity,
        "snapshot_basis": "filesystem_metadata_only",
        "format_classification": FORMAT_CLASSIFICATION,
    }
