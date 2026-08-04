"""Self-relative repository discovery and path-containment checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .errors import BootstrapError, ErrorCode


ROOT_MARKERS = (
    "SYSTEM_X_REPOSITORY_MANIFEST.json",
    "SYSTEM_X_NEW_UBUNTU_REQUIREMENTS.json",
    "bootstrap/run_bootstrap.py",
)


def _candidate_ancestors(start: Path) -> Iterable[Path]:
    candidate = start if start.is_dir() else start.parent
    yield candidate
    yield from candidate.parents


def discover_repository_root(start: Path | str | None = None) -> Path:
    """Find System X without depending on the caller's working directory."""

    origin = Path(start) if start is not None else Path(__file__)
    origin = origin.expanduser().resolve(strict=True)
    for candidate in _candidate_ancestors(origin):
        if all((candidate / marker).is_file() for marker in ROOT_MARKERS):
            if not (candidate / "model-api-gguf").is_dir():
                continue
            return candidate
    raise BootstrapError(
        ErrorCode.PRECONDITION_FAILED,
        "System X repository root could not be discovered",
        context={"start": str(origin)},
    )


def validate_relative_path(value: str) -> PurePosixPath:
    """Validate a portable repository-relative path before touching disk."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise BootstrapError(ErrorCode.PATH_UNSAFE, "path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith(("/", "\\")):
        raise BootstrapError(ErrorCode.PATH_UNSAFE, "absolute paths are prohibited")
    if any(part in ("", ".", "..") for part in path.parts):
        raise BootstrapError(ErrorCode.PATH_UNSAFE, "path traversal is prohibited")
    if "\\" in value:
        raise BootstrapError(ErrorCode.PATH_UNSAFE, "portable paths must use forward slashes")
    return path


def resolve_contained(
    root: Path,
    relative: str,
    *,
    allow_missing: bool = True,
    reject_symlinks: bool = True,
) -> Path:
    """Resolve a relative path and reject traversal or a symlink escape."""

    portable = validate_relative_path(relative)
    root_resolved = root.resolve(strict=True)
    candidate = root_resolved.joinpath(*portable.parts)

    cursor = root_resolved
    for part in portable.parts:
        cursor = cursor / part
        if cursor.is_symlink() and reject_symlinks:
            raise BootstrapError(
                ErrorCode.PATH_UNSAFE,
                "symlinks are not accepted for this repository path",
                context={"path": relative},
            )
        if not cursor.exists():
            break

    resolved = candidate.resolve(strict=not allow_missing)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise BootstrapError(
            ErrorCode.PATH_UNSAFE,
            "path resolves outside the repository",
            context={"path": relative},
        ) from exc
    return candidate


@dataclass(frozen=True, slots=True)
class RepositoryPaths:
    root: Path
    bootstrap: Path
    configuration: Path
    schemas: Path
    bootstrap_state: Path
    runtime: Path
    transaction_directory: Path
    transaction_lock: Path
    transaction_status: Path

    @classmethod
    def discover(cls, start: Path | str | None = None) -> "RepositoryPaths":
        root = discover_repository_root(start)
        return cls(
            root=root,
            bootstrap=root / "bootstrap",
            configuration=root / "bootstrap" / "configuration",
            schemas=root / "bootstrap" / "schemas",
            bootstrap_state=root / ".system-x-bootstrap-state",
            runtime=root / "model-api-gguf" / "RUNTIME",
            transaction_directory=root / ".system-x-bootstrap-state" / "transactions",
            transaction_lock=root / ".system-x-bootstrap-state" / "locks" / "system-x-bootstrap.lock",
            transaction_status=root / ".system-x-bootstrap-state" / "status.json",
        )

    def contained(self, relative: str, **kwargs: object) -> Path:
        return resolve_contained(self.root, relative, **kwargs)
