"""Streaming complete artifact identity with physical source revalidation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import InspectorError


DEFAULT_HASH_CHUNK_BYTES = 8 * 1024 * 1024
MAX_U64 = (1 << 64) - 1
ProgressHook = Callable[[Path, int], None]


@dataclass(frozen=True)
class ArtifactIdentity:
    identity: str
    byte_count: int
    file_count: int
    content_manifest_identity: str
    files: tuple[dict[str, Any], ...]
    pre_inspection_snapshot_identity: str
    post_inspection_snapshot_identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "algorithm": "sha256",
            "byte_count": self.byte_count,
            "file_count": self.file_count,
            "content_manifest_identity": self.content_manifest_identity,
            "files": [dict(item) for item in self.files],
            "pre_inspection_snapshot_identity": (
                self.pre_inspection_snapshot_identity
            ),
            "post_inspection_snapshot_identity": (
                self.post_inspection_snapshot_identity
            ),
        }


def _checked_add(left: int, right: int) -> int:
    if left < 0 or right < 0 or left > MAX_U64 - right:
        raise InspectorError(
            "ARTIFACT_IDENTITY_FAILED",
            "artifact byte-count arithmetic overflow",
        )
    return left + right


def _stat_identity(details: os.stat_result) -> dict[str, Any]:
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "mode": details.st_mode,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def _snapshot_identity(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _changed(message: str) -> InspectorError:
    return InspectorError("ARTIFACT_CHANGED_DURING_INSPECTION", message)


def _regular_lstat(path: Path) -> os.stat_result:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _changed("artifact path disappeared during inspection") from error
    if stat.S_ISLNK(details.st_mode):
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact symlink rejected"
        )
    if not stat.S_ISREG(details.st_mode):
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact is not a regular file"
        )
    return details


def _hash_regular(
    path: Path,
    *,
    relative_path: str,
    chunk_bytes: int,
    progress_hook: ProgressHook | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        isinstance(chunk_bytes, bool)
        or not isinstance(chunk_bytes, int)
        or chunk_bytes <= 0
    ):
        raise InspectorError(
            "ARTIFACT_IDENTITY_FAILED", "hash chunk size is invalid"
        )
    pre_path = _regular_lstat(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        reason = (
            "ARTIFACT_READ_FAILED"
            if error.errno in {errno.ELOOP, errno.EACCES, errno.EPERM}
            else "ARTIFACT_CHANGED_DURING_INSPECTION"
        )
        raise InspectorError(reason, "artifact could not be opened safely") from error
    hasher = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(pre_path) != _stat_identity(opened):
            raise _changed("artifact path and opened descriptor differ")
        while True:
            try:
                chunk = os.read(descriptor, chunk_bytes)
            except OSError as error:
                raise InspectorError(
                    "ARTIFACT_READ_FAILED", "artifact read failed"
                ) from error
            if not chunk:
                break
            total = _checked_add(total, len(chunk))
            hasher.update(chunk)
            if progress_hook is not None:
                progress_hook(path, total)
        post_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        post_path = _regular_lstat(path)
    except InspectorError as error:
        if error.reason_code == "ARTIFACT_READ_FAILED":
            raise _changed("artifact path type changed during inspection") from error
        raise
    expected = _stat_identity(opened)
    if (
        _stat_identity(post_descriptor) != expected
        or _stat_identity(post_path) != expected
        or total != opened.st_size
    ):
        raise _changed("artifact identity or timestamps changed while hashing")
    digest = hasher.hexdigest()
    entry = {
        "relative_path": relative_path,
        "byte_count": total,
        "sha256": digest,
    }
    return entry, _stat_identity(opened), _stat_identity(post_path)


def _entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "regular_directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def _directory_snapshot(
    root: Path,
    *,
    maximum_entries: int,
    maximum_depth: int,
) -> list[dict[str, Any]]:
    try:
        root_details = root.lstat()
    except FileNotFoundError as error:
        raise _changed("artifact directory disappeared") from error
    if stat.S_ISLNK(root_details.st_mode):
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact directory symlink rejected"
        )
    if not stat.S_ISDIR(root_details.st_mode):
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact root is not a directory"
        )
    root_real = root.resolve(strict=True)
    result = [
        {
            "relative_path": ".",
            "entry_type": "regular_directory",
            **_stat_identity(root_details),
        }
    ]
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda item: item.name)
        except FileNotFoundError as error:
            raise _changed("artifact directory changed during traversal") from error
        next_directories: list[tuple[Path, int]] = []
        for entry in children:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                details = path.lstat()
            except FileNotFoundError as error:
                raise _changed("artifact entry disappeared") from error
            kind = _entry_type(details.st_mode)
            if kind == "symlink":
                raise InspectorError(
                    "ARTIFACT_READ_FAILED",
                    "artifact descendant symlink rejected",
                )
            if kind not in {"regular_file", "regular_directory"}:
                raise InspectorError(
                    "ARTIFACT_READ_FAILED",
                    "artifact special file rejected",
                )
            child_depth = depth + 1
            if child_depth > maximum_depth:
                raise InspectorError(
                    "ARTIFACT_IDENTITY_FAILED",
                    "artifact directory depth bound exceeded",
                )
            if not path.resolve(strict=True).is_relative_to(root_real):
                raise InspectorError(
                    "ARTIFACT_READ_FAILED", "artifact path escaped its root"
                )
            result.append(
                {
                    "relative_path": relative,
                    "entry_type": kind,
                    **_stat_identity(details),
                }
            )
            if len(result) > maximum_entries:
                raise InspectorError(
                    "ARTIFACT_IDENTITY_FAILED",
                    "artifact entry-count bound exceeded",
                )
            if kind == "regular_directory":
                next_directories.append((path, child_depth))
        pending.extend(reversed(next_directories))
    return sorted(result, key=lambda item: item["relative_path"])


def identify_regular_file(
    path: Path,
    *,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    progress_hook: ProgressHook | None = None,
) -> ArtifactIdentity:
    entry, pre, post = _hash_regular(
        path,
        relative_path=path.name,
        chunk_bytes=chunk_bytes,
        progress_hook=progress_hook,
    )
    identity = "sha256:" + entry["sha256"]
    return ArtifactIdentity(
        identity=identity,
        byte_count=entry["byte_count"],
        file_count=1,
        content_manifest_identity=identity,
        files=(entry,),
        pre_inspection_snapshot_identity=_snapshot_identity(pre),
        post_inspection_snapshot_identity=_snapshot_identity(post),
    )


def identify_directory_bundle(
    root: Path,
    *,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    maximum_entries: int = 100_000,
    maximum_depth: int = 64,
    progress_hook: ProgressHook | None = None,
) -> ArtifactIdentity:
    before = _directory_snapshot(
        root,
        maximum_entries=maximum_entries,
        maximum_depth=maximum_depth,
    )
    files = [
        item for item in before if item["entry_type"] == "regular_file"
    ]
    manifest: list[dict[str, Any]] = []
    total = 0
    for item in files:
        relative = item["relative_path"]
        entry, _, _ = _hash_regular(
            root / relative,
            relative_path=relative,
            chunk_bytes=chunk_bytes,
            progress_hook=progress_hook,
        )
        manifest.append(entry)
        total = _checked_add(total, entry["byte_count"])
    after = _directory_snapshot(
        root,
        maximum_entries=maximum_entries,
        maximum_depth=maximum_depth,
    )
    if before != after:
        raise _changed("artifact directory changed while hashing")
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    identity = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return ArtifactIdentity(
        identity=identity,
        byte_count=total,
        file_count=len(manifest),
        content_manifest_identity=identity,
        files=tuple(manifest),
        pre_inspection_snapshot_identity=_snapshot_identity(before),
        post_inspection_snapshot_identity=_snapshot_identity(after),
    )


def identify_artifact(
    path: Path,
    *,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    maximum_entries: int = 100_000,
    maximum_depth: int = 64,
    progress_hook: ProgressHook | None = None,
) -> ArtifactIdentity:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact path does not exist"
        ) from error
    if stat.S_ISLNK(details.st_mode):
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact symlink rejected"
        )
    if stat.S_ISREG(details.st_mode):
        return identify_regular_file(
            path,
            chunk_bytes=chunk_bytes,
            progress_hook=progress_hook,
        )
    if stat.S_ISDIR(details.st_mode):
        return identify_directory_bundle(
            path,
            chunk_bytes=chunk_bytes,
            maximum_entries=maximum_entries,
            maximum_depth=maximum_depth,
            progress_hook=progress_hook,
        )
    raise InspectorError(
        "ARTIFACT_READ_FAILED", "artifact physical type is unsupported"
    )


def artifact_source_snapshot(
    path: Path,
    *,
    maximum_entries: int = 100_000,
    maximum_depth: int = 64,
) -> str:
    """Return a metadata-only identity for whole-transaction revalidation."""

    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _changed("artifact disappeared during inspection") from error
    if stat.S_ISLNK(details.st_mode):
        raise InspectorError("ARTIFACT_READ_FAILED", "artifact symlink rejected")
    if stat.S_ISREG(details.st_mode):
        value: object = {
            "relative_path": path.name,
            "entry_type": "regular_file",
            **_stat_identity(details),
        }
    elif stat.S_ISDIR(details.st_mode):
        value = _directory_snapshot(
            path,
            maximum_entries=maximum_entries,
            maximum_depth=maximum_depth,
        )
    else:
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact physical type is unsupported"
        )
    return _snapshot_identity(value)
