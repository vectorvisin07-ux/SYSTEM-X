"""Product-owned materialization of one authenticated portable candidate tree."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping


MANIFEST_PATH = "SYSTEM_X_PORTABLE_TREE_MANIFEST.json"


class MaterializationError(RuntimeError):
    pass


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise MaterializationError(f"unsafe candidate path: {value!r}")
    return path


def _mode(value: object) -> int:
    if isinstance(value, bool):
        raise MaterializationError("portable mode is boolean")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("0o"):
            text = text[2:]
        try:
            result = int(text, 8)
        except ValueError as error:
            raise MaterializationError(f"portable mode is invalid: {value!r}") from error
    else:
        raise MaterializationError(f"portable mode is invalid: {value!r}")
    if result < 0 or result & ~0o777:
        raise MaterializationError(f"portable mode is outside permission bits: {value!r}")
    return result


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _sha256_fd(fd: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return size, digest.hexdigest()


def _write_exact(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.materialize-{os.getpid()}-"
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _diagnostic(
    relative: str,
    *,
    expected_type: str,
    observed_type: str,
    expected_bytes: int | None,
    observed_bytes: int | None,
    expected_sha256: str | None,
    observed_sha256: str | None,
    expected_mode: int | None,
    observed_mode: int | None,
    link_count: int | None,
    containment: bool,
    reason: str,
) -> MaterializationError:
    value = {
        "relative_path": relative,
        "expected_object_type": expected_type,
        "observed_object_type": observed_type,
        "expected_bytes": expected_bytes,
        "observed_bytes": observed_bytes,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "expected_mode": format(expected_mode, "#06o") if expected_mode is not None else None,
        "observed_mode": format(observed_mode, "#06o") if observed_mode is not None else None,
        "link_count": link_count,
        "containment": containment,
        "reason_code": reason,
    }
    return MaterializationError(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _manifest_specs(source_root: Path) -> dict[str, dict[str, object]]:
    path = source_root / MANIFEST_PATH
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError("portable manifest cannot be read") from error
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise MaterializationError("portable manifest entries are absent")
    specs: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MaterializationError("portable manifest entry is not an object")
        relative_value = entry.get("path")
        if not isinstance(relative_value, str):
            raise MaterializationError("portable manifest path is invalid")
        relative = _safe_relative(relative_value).as_posix()
        if relative in specs:
            raise MaterializationError(f"portable manifest path is duplicated: {relative}")
        bytes_value = entry.get("bytes")
        digest_value = entry.get("sha256")
        if (
            isinstance(bytes_value, bool)
            or not isinstance(bytes_value, int)
            or bytes_value < 0
            or not isinstance(digest_value, str)
        ):
            raise MaterializationError(f"portable manifest identity is invalid: {relative}")
        digest = digest_value.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise MaterializationError(f"portable manifest SHA-256 is invalid: {relative}")
        portable_mode = _mode(entry.get("portable_mode"))
        specs[relative] = {
            "bytes": bytes_value,
            "sha256": digest,
            "portable_mode": portable_mode,
        }
    if MANIFEST_PATH in specs:
        raise MaterializationError("portable manifest self-entry is forbidden")
    return specs


def _copy_vendored_source(source_root: Path, destination: Path) -> int:
    source = source_root / "model-api-gguf" / "llama.cpp"
    if not source.is_dir() or source.is_symlink():
        raise MaterializationError("authenticated vendored llama.cpp source is absent")
    source_real = source.resolve(strict=True)
    count = 0
    for directory, names, files in os.walk(source, topdown=True, followlinks=False):
        directory_path = Path(directory)
        names[:] = sorted(name for name in names if name not in {".git", "build"})
        files = sorted(files)
        for name in list(names):
            candidate = directory_path / name
            if candidate.is_symlink():
                names.remove(name)
                target = candidate.resolve(strict=True)
                relative_target = target.relative_to(source_real) if target != source_real else Path(".")
                if (
                    target == source_real
                    or not _within(source_real, target)
                    or ".git" in relative_target.parts
                    or "build" in relative_target.parts
                ):
                    raise MaterializationError(
                        f"vendored symlink escapes source root: {candidate.relative_to(source)}"
                    )
                output = destination / "model-api-gguf" / "llama.cpp" / candidate.relative_to(source)
                output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                os.symlink(os.readlink(candidate), output)
        relative_directory = directory_path.relative_to(source)
        output_directory = destination / "model-api-gguf" / "llama.cpp" / relative_directory
        output_directory.mkdir(mode=stat.S_IMODE(directory_path.stat().st_mode), parents=True, exist_ok=True)
        os.chmod(output_directory, stat.S_IMODE(directory_path.stat().st_mode))
        for name in files:
            candidate = directory_path / name
            relative = candidate.relative_to(source)
            output = destination / "model-api-gguf" / "llama.cpp" / relative
            details = candidate.lstat()
            if stat.S_ISLNK(details.st_mode):
                target = candidate.resolve(strict=True)
                relative_target = target.relative_to(source_real) if target != source_real else Path(".")
                if (
                    target == source_real
                    or not _within(source_real, target)
                    or ".git" in relative_target.parts
                    or "build" in relative_target.parts
                ):
                    raise MaterializationError(f"vendored symlink escapes source root: {relative}")
                output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                os.symlink(os.readlink(candidate), output)
                continue
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise MaterializationError(f"vendored source member is unsafe: {relative}")
            _write_exact(output, candidate.read_bytes(), stat.S_IMODE(details.st_mode))
            count += 1
    return count


def _extract_git_archive(source_root: Path, destination: Path) -> int:
    process = subprocess.Popen(
        [
            "git",
            "-C",
            str(source_root),
            "--no-optional-locks",
            "archive",
            "--format=tar",
            "HEAD",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    count = 0
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                relative = _safe_relative(member.name)
                if member.isdir():
                    target = destination / relative
                    target.mkdir(mode=member.mode & 0o777, parents=True, exist_ok=True)
                    os.chmod(target, member.mode & 0o777)
                    continue
                if not member.isreg():
                    raise MaterializationError(
                        f"git base contains unsupported member: {member.name}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise MaterializationError(
                        f"git base member cannot be read: {member.name}"
                    )
                _write_exact(destination / relative, source.read(), member.mode & 0o777)
                count += 1
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.stderr is not None:
        process.stderr.close()
    code = process.wait()
    if code != 0:
        raise MaterializationError(f"git archive failed: {stderr[:2000]}")
    return count


def _normalize_and_verify(
    destination: Path,
    root_real: Path,
    relative: str,
    spec: Mapping[str, object],
) -> int:
    path = destination / _safe_relative(relative)
    expected_bytes = int(spec["bytes"])
    expected_sha256 = str(spec["sha256"])
    expected_mode = int(spec["portable_mode"])
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _diagnostic(
            relative,
            expected_type="regular",
            observed_type="missing",
            expected_bytes=expected_bytes,
            observed_bytes=None,
            expected_sha256=expected_sha256,
            observed_sha256=None,
            expected_mode=expected_mode,
            observed_mode=None,
            link_count=None,
            containment=False,
            reason="MATERIALIZED_PATH_MISSING",
        ) from error
    observed_type = (
        "symlink"
        if stat.S_ISLNK(details.st_mode)
        else "regular"
        if stat.S_ISREG(details.st_mode)
        else "special"
    )
    resolved = path.resolve(strict=True) if not stat.S_ISLNK(details.st_mode) else path.resolve(strict=False)
    contained = _within(root_real, resolved)
    if not contained or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise _diagnostic(
            relative,
            expected_type="regular",
            observed_type=observed_type,
            expected_bytes=expected_bytes,
            observed_bytes=None,
            expected_sha256=expected_sha256,
            observed_sha256=None,
            expected_mode=expected_mode,
            observed_mode=stat.S_IMODE(details.st_mode),
            link_count=details.st_nlink,
            containment=contained,
            reason=(
                "MATERIALIZED_PATH_OUTSIDE_ROOT"
                if not contained
                else "MATERIALIZED_OBJECT_TYPE_OR_LINK_POLICY"
            ),
        )
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError as error:
        raise _diagnostic(
            relative,
            expected_type="regular",
            observed_type=observed_type,
            expected_bytes=expected_bytes,
            observed_bytes=None,
            expected_sha256=expected_sha256,
            observed_sha256=None,
            expected_mode=expected_mode,
            observed_mode=stat.S_IMODE(details.st_mode),
            link_count=details.st_nlink,
            containment=contained,
            reason="MATERIALIZED_FILE_OPEN_FAILED",
        ) from error
    try:
        before = os.fstat(fd)
        if before.st_uid != details.st_uid or before.st_gid != details.st_gid:
            raise _diagnostic(
                relative,
                expected_type="regular",
                observed_type="regular",
                expected_bytes=expected_bytes,
                observed_bytes=None,
                expected_sha256=expected_sha256,
                observed_sha256=None,
                expected_mode=expected_mode,
                observed_mode=stat.S_IMODE(before.st_mode),
                link_count=before.st_nlink,
                containment=contained,
                reason="MATERIALIZED_OWNER_CHANGED",
            )
        if stat.S_IMODE(before.st_mode) != expected_mode:
            os.fchmod(fd, expected_mode)
            os.fsync(fd)
        after_mode = os.fstat(fd)
        observed_bytes, observed_sha256 = _sha256_fd(fd)
        if (
            observed_bytes != expected_bytes
            or observed_sha256 != expected_sha256
            or stat.S_IMODE(after_mode.st_mode) != expected_mode
            or after_mode.st_uid != before.st_uid
            or after_mode.st_gid != before.st_gid
        ):
            reason = (
                "MATERIALIZED_CONTENT_MISMATCH"
                if observed_bytes != expected_bytes or observed_sha256 != expected_sha256
                else "MATERIALIZED_MODE_MISMATCH"
                if stat.S_IMODE(after_mode.st_mode) != expected_mode
                else "MATERIALIZED_OWNER_CHANGED"
            )
            raise _diagnostic(
                relative,
                expected_type="regular",
                observed_type="regular",
                expected_bytes=expected_bytes,
                observed_bytes=observed_bytes,
                expected_sha256=expected_sha256,
                observed_sha256=observed_sha256,
                expected_mode=expected_mode,
                observed_mode=stat.S_IMODE(after_mode.st_mode),
                link_count=after_mode.st_nlink,
                containment=contained,
                reason=reason,
            )
        return 1
    finally:
        os.close(fd)


def _verify_ungoverned(destination: Path, root_real: Path, governed: set[str]) -> None:
    """Reject regular, special or escaping objects outside the governed tree."""

    allowed_regular = governed | {MANIFEST_PATH}
    for directory, names, files in os.walk(destination, topdown=True, followlinks=False):
        directory_path = Path(directory)
        names[:] = sorted(names)
        files = sorted(files)
        for name in names:
            path = directory_path / name
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode):
                resolved = path.resolve(strict=False)
                if not _within(root_real, resolved):
                    raise MaterializationError(
                        f"materialized symlink escapes destination: {path.relative_to(destination)}"
                    )
            elif not stat.S_ISDIR(details.st_mode):
                raise MaterializationError(
                    f"materialized directory entry is special: {path.relative_to(destination)}"
                )
        for name in files:
            path = directory_path / name
            relative = path.relative_to(destination).as_posix()
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode):
                resolved = path.resolve(strict=False)
                if not _within(root_real, resolved):
                    raise MaterializationError(
                        f"materialized symlink escapes destination: {relative}"
                    )
                continue
            if not stat.S_ISREG(details.st_mode):
                raise MaterializationError(
                    f"materialized file is special: {relative}"
                )
            if relative not in allowed_regular:
                raise MaterializationError(
                    f"ungoverned regular file entered destination: {relative}"
                )


def materialize_portable_tree(
    source_root: Path,
    destination: Path,
    candidate_map: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    candidate_map = candidate_map.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise MaterializationError("destination already exists")

    specs = _manifest_specs(source_root)
    authority = json.loads(candidate_map.read_text(encoding="utf-8"))
    entries = authority.get("paths")
    if not isinstance(entries, list) or not entries:
        raise MaterializationError("candidate map paths are absent")
    candidate_paths: set[str] = set()
    overlay_specs: dict[str, dict[str, object]] = {}
    parent = destination.parent
    parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = parent / (
        f".{destination.name}.materializing-{os.getpid()}-"
        f"{next(tempfile._get_candidate_names())}"
    )
    temporary.mkdir(mode=0o755)
    os.chmod(temporary, 0o755)
    try:
        base_count = _extract_git_archive(source_root, temporary)
        vendor_count = _copy_vendored_source(source_root, temporary)
        overlay_count = 0
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise MaterializationError("candidate map entry is invalid")
            relative_text = entry.get("path")
            if not isinstance(relative_text, str):
                raise MaterializationError("candidate map path is invalid")
            relative = _safe_relative(relative_text).as_posix()
            if relative in candidate_paths:
                raise MaterializationError(f"candidate map path is duplicated: {relative}")
            candidate_paths.add(relative)
            source = source_root / relative
            try:
                details = source.lstat()
            except FileNotFoundError as error:
                raise MaterializationError(f"candidate source is absent: {relative}") from error
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise MaterializationError(
                    f"candidate source is not regular single-link: {relative}"
                )
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            expected_sha = str(entry.get("sha256", "")).removeprefix("sha256:")
            expected_bytes = entry.get("bytes")
            if expected_bytes != len(data) or expected_sha != digest:
                raise _diagnostic(
                    relative,
                    expected_type="regular",
                    observed_type="regular",
                    expected_bytes=int(expected_bytes) if isinstance(expected_bytes, int) else None,
                    observed_bytes=len(data),
                    expected_sha256=expected_sha or None,
                    observed_sha256=digest,
                    expected_mode=None,
                    observed_mode=stat.S_IMODE(details.st_mode),
                    link_count=details.st_nlink,
                    containment=_within(source_root, source.resolve(strict=True)),
                    reason="CANDIDATE_SOURCE_IDENTITY_MISMATCH",
                )
            mode_value = entry.get("portable_mode")
            if mode_value is None:
                mode_value = entry.get("mode")
            if mode_value is None:
                mode_value = entry.get("git_mode")
            mode = _mode(mode_value)
            candidate_spec = {"bytes": len(data), "sha256": digest, "portable_mode": mode}
            existing_spec = specs.get(relative)
            if existing_spec is not None and (
                int(existing_spec["bytes"]) != len(data)
                or str(existing_spec["sha256"]) != digest
                or int(existing_spec["portable_mode"]) != mode
            ):
                raise MaterializationError(
                    f"candidate and portable manifest disagree: {relative}"
                )
            if relative != MANIFEST_PATH:
                overlay_specs[relative] = candidate_spec
            _write_exact(temporary / relative, data, mode)
            overlay_count += 1

        governed_specs = dict(specs)
        governed_specs.update(overlay_specs)
        root_real = temporary.resolve(strict=True)
        mode_checks = 0
        for relative, spec in sorted(governed_specs.items()):
            mode_checks += _normalize_and_verify(temporary, root_real, relative, spec)
        _verify_ungoverned(temporary, root_real, set(governed_specs))

        if MANIFEST_PATH in candidate_paths:
            manifest_target = temporary / MANIFEST_PATH
            manifest_details = manifest_target.lstat()
            if not stat.S_ISREG(manifest_details.st_mode) or manifest_details.st_nlink != 1:
                raise MaterializationError("materialized portable manifest is not regular single-link")
            if not _within(root_real, manifest_target.resolve(strict=True)):
                raise MaterializationError("materialized portable manifest escaped destination")
        else:
            raise MaterializationError("candidate map did not overlay current portable manifest")

        details: dict[str, Any] = {
            "status": "PASS",
            "base_commit": subprocess.run(
                ["git", "-C", str(source_root), "--no-optional-locks", "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip(),
            "base_file_count": base_count,
            "vendor_file_count": vendor_count,
            "overlay_count": overlay_count,
            "candidate_count": len(entries),
            "governed_file_count": len(governed_specs),
            "regular_file_mode_checks": mode_checks,
            "content_mismatch_count": 0,
            "mode_mismatch_count": 0,
            "object_type_mismatch_count": 0,
            "containment_error_count": 0,
            "destination": str(destination),
            "git_dependency_absent": not (temporary / ".git").exists(),
        }
        identity = hashlib.sha256(
            json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        details["publication_identity"] = "sha256:" + identity
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, destination)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
    return details
