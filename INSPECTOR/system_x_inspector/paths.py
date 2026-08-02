"""Self-relative, contained Inspector path discovery."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InspectorError


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_root(path: Path) -> Path:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "LAYOUT_INVALID", "Inspector root does not exist"
        ) from error
    if stat.S_ISLNK(details.st_mode):
        raise InspectorError("LAYOUT_INVALID", "Inspector root is a symlink")
    if not stat.S_ISDIR(details.st_mode):
        raise InspectorError(
            "LAYOUT_INVALID", "Inspector root is not a directory"
        )
    return path.resolve(strict=True)


@dataclass(frozen=True)
class InspectorPaths:
    inspector_root: Path
    source_root: Path
    schema_root: Path
    environment_lock: Path
    intake_root: Path
    runtime_root: Path
    logs: Path
    locks: Path
    status: Path
    transactions: Path
    inspection_results: Path
    decision_results: Path
    handoff_results: Path
    publication_results: Path
    qualification_results: Path
    promotion_results: Path
    retirement_results: Path
    deployment_results: Path
    staging: Path
    tmp: Path
    capability_root: Path
    capability_records: Path
    capability_bindings: Path

    @classmethod
    def discover(cls, explicit_root: Path | None = None) -> "InspectorPaths":
        source_root = Path(__file__).resolve().parent
        candidate = explicit_root if explicit_root is not None else source_root.parent
        root = _validate_root(Path(candidate))
        values = cls(
            inspector_root=root,
            source_root=source_root,
            schema_root=root / "schemas",
            environment_lock=root / "environment.lock.json",
            intake_root=root / "MODEL-TEST",
            runtime_root=root / "RUNTIME",
            logs=root / "RUNTIME" / "logs",
            locks=root / "RUNTIME" / "locks",
            status=root / "RUNTIME" / "status",
            transactions=root / "RUNTIME" / "transactions",
            inspection_results=root / "RUNTIME" / "results" / "inspection",
            decision_results=root / "RUNTIME" / "results" / "decision",
            handoff_results=root / "RUNTIME" / "results" / "handoff",
            publication_results=root / "RUNTIME" / "results" / "publication",
            qualification_results=(
                root / "RUNTIME" / "results" / "qualification"
            ),
            promotion_results=(
                root / "RUNTIME" / "results" / "promotion"
            ),
            retirement_results=(
                root / "RUNTIME" / "results" / "retirement"
            ),
            deployment_results=(
                root / "RUNTIME" / "results" / "deployment"
            ),
            staging=root / "RUNTIME" / "staging",
            tmp=root / "RUNTIME" / "tmp",
            capability_root=root / "capabilities",
            capability_records=root / "capabilities" / "records",
            capability_bindings=root / "capabilities" / "bindings",
        )
        for path in values.persistent_mapping().values():
            if not _inside(path, root):
                raise InspectorError(
                    "LAYOUT_INVALID",
                    "A derived Inspector path escaped the Inspector root",
                )
        return values

    def persistent_mapping(self) -> dict[str, Path]:
        return {
            key: value
            for key, value in self.as_mapping().items()
            if key != "source_root"
        }

    def as_mapping(self) -> dict[str, Path]:
        return {
            "inspector_root": self.inspector_root,
            "source_root": self.source_root,
            "schema_root": self.schema_root,
            "environment_lock": self.environment_lock,
            "intake_root": self.intake_root,
            "runtime_root": self.runtime_root,
            "logs": self.logs,
            "locks": self.locks,
            "status": self.status,
            "transactions": self.transactions,
            "inspection_results": self.inspection_results,
            "decision_results": self.decision_results,
            "handoff_results": self.handoff_results,
            "publication_results": self.publication_results,
            "qualification_results": self.qualification_results,
            "promotion_results": self.promotion_results,
            "retirement_results": self.retirement_results,
            "deployment_results": self.deployment_results,
            "staging": self.staging,
            "tmp": self.tmp,
            "capability_root": self.capability_root,
            "capability_records": self.capability_records,
            "capability_bindings": self.capability_bindings,
        }

    @property
    def current_connection_status(self) -> Path:
        return self.status / "api-connection.json"

    @property
    def deployment_lock(self) -> Path:
        return self.locks / "deployment-active.json"


@dataclass(frozen=True)
class BranchHandoffPaths:
    """Authenticated self-relative GGUF branch admission paths."""

    system_x_root: Path
    branch_root: Path
    managed_root: Path
    branch_staging_root: Path

    @classmethod
    def discover(
        cls,
        inspector_paths: InspectorPaths,
        explicit_branch_root: Path | None = None,
    ) -> "BranchHandoffPaths":
        system_x_root = inspector_paths.inspector_root.parent
        candidate = (
            Path(explicit_branch_root)
            if explicit_branch_root is not None
            else system_x_root / "model-api-gguf"
        )
        if candidate.parent != system_x_root or candidate.name != "model-api-gguf":
            raise InspectorError(
                "HANDOFF_STAGING_INVALID",
                "GGUF branch is not the authenticated Inspector sibling",
            )
        branch_root = _validate_handoff_directory(
            candidate, "GGUF branch root"
        )
        managed_root = _validate_handoff_directory(
            branch_root / "MODEL" / "SUPERMODEL",
            "GGUF managed model root",
        )
        branch_staging_root = _validate_handoff_directory(
            branch_root / "RUNTIME" / "api" / "replacement-staging",
            "GGUF branch staging root",
        )
        if (
            branch_root.parent != system_x_root
            or not _inside(managed_root, branch_root)
            or not _inside(branch_staging_root, branch_root)
        ):
            raise InspectorError(
                "HANDOFF_STAGING_INVALID",
                "GGUF branch admission paths escaped their authenticated root",
            )
        if managed_root.stat().st_dev != branch_staging_root.stat().st_dev:
            raise InspectorError(
                "HANDOFF_STAGING_INVALID",
                "GGUF managed and staging roots are on different filesystems",
            )
        return cls(
            system_x_root=system_x_root,
            branch_root=branch_root,
            managed_root=managed_root,
            branch_staging_root=branch_staging_root,
        )

    def relative_to_branch(self, path: Path) -> str:
        try:
            return path.relative_to(self.branch_root).as_posix()
        except ValueError as error:
            raise InspectorError(
                "HANDOFF_STAGING_INVALID",
                "GGUF handoff path escaped the authenticated branch",
            ) from error


def _validate_handoff_directory(path: Path, label: str) -> Path:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "HANDOFF_STAGING_INVALID", f"{label} does not exist"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise InspectorError(
            "HANDOFF_STAGING_INVALID",
            f"{label} is not a regular non-symlink directory",
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise InspectorError(
            "HANDOFF_STAGING_INVALID",
            f"{label} contains a symlinked path component",
        )
    return resolved


def physical_state(path: Path, root: Path) -> dict[str, Any]:
    try:
        details = path.lstat()
    except FileNotFoundError:
        if not _inside(path.resolve(strict=False), root):
            return {"path": str(path), "state": "outside_root"}
        return {"path": str(path), "state": "absent"}
    if stat.S_ISLNK(details.st_mode):
        state = "symlink"
    elif not _inside(path.resolve(strict=False), root):
        state = "outside_root"
    elif stat.S_ISDIR(details.st_mode):
        state = "regular_directory"
    elif stat.S_ISREG(details.st_mode):
        state = "regular_file"
    else:
        state = "wrong_type"
    return {
        "path": str(path),
        "state": state,
        "mode": f"{stat.S_IMODE(details.st_mode):04o}",
        "device": details.st_dev,
        "inode": details.st_ino,
        "size": details.st_size,
    }
