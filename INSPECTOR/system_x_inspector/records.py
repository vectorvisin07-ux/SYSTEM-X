"""Atomic deterministic foundation record persistence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any

from .errors import InspectorError


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: object, *, mode: int = 0o600) -> str:
    parent = path.parent
    parent_details = parent.lstat()
    if stat.S_ISLNK(parent_details.st_mode) or not stat.S_ISDIR(
        parent_details.st_mode
    ):
        raise InspectorError(
            "LAYOUT_INVALID", "record parent is not a regular directory"
        )
    if path.exists() or path.is_symlink():
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise InspectorError(
                "LAYOUT_INVALID", "record target has an unsafe physical type"
            )
    data = canonical_json_bytes(value)
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        mode,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        os.chmod(path, mode)
        fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def atomic_create_json(path: Path, value: object, *, mode: int = 0o600) -> str:
    """Publish an immutable JSON record atomically without overwriting."""

    parent = path.parent
    try:
        parent_details = parent.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "LAYOUT_INVALID", "record parent does not exist"
        ) from error
    if stat.S_ISLNK(parent_details.st_mode) or not stat.S_ISDIR(
        parent_details.st_mode
    ):
        raise InspectorError(
            "LAYOUT_INVALID", "record parent is not a regular directory"
        )
    data = canonical_json_bytes(value)
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write while creating immutable record")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise InspectorError(
                "INSPECTION_RECORD_COLLISION",
                "inspection result identity already exists",
            ) from error
        fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_directory(parent)
    details = path.lstat()
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or stat.S_IMODE(details.st_mode) != mode
        or details.st_nlink != 1
    ):
        raise InspectorError(
            "INSPECTION_INTERNAL_ERROR",
            "immutable inspection result failed physical verification",
            exit_status=70,
        )
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_json_record(path: Path) -> dict[str, Any]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise InspectorError(
            "LAYOUT_INVALID", "record is not a regular non-symlink file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspectorError("LAYOUT_INVALID", "record JSON is invalid") from error
    if not isinstance(value, dict):
        raise InspectorError("LAYOUT_INVALID", "record must be an object")
    return value


def record_identity(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
