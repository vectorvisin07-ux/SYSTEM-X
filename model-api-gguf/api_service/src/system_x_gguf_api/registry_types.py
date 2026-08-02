"""Immutable types shared by the internal automatic model registry."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from enum import StrEnum
import json
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA_IDENTITY = "system-x.gguf-model-registry.v1"
REGISTRY_SCHEMA_VERSION = 2
CAPABILITY_SCHEMA_IDENTITY = "system-x.gguf-model-capabilities.v1"
MODEL_ADAPTATION_CONTRACT = "system-x.gguf-model-adaptation.v1"
BUNDLE_MANIFEST_PREAMBLE = b"system-x.gguf-bundle-manifest.v1\n"


def utc_now() -> str:
    """Return one sortable UTC timestamp."""

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    """Serialize bounded internal evidence deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class BundleKind(StrEnum):
    SINGLE_FILE = "single_file"
    DIRECTORY_BUNDLE = "directory_bundle"


class RoleHint(StrEnum):
    PRIMARY = "primary"
    SHARD = "shard"
    MMPROJ_SIDECAR = "mmproj_sidecar"
    MTP_SIDECAR = "mtp_sidecar"
    OTHER_GGUF = "other_gguf"


class ModelState(StrEnum):
    DISCOVERED = "DISCOVERED"
    PENDING_STABILITY = "PENDING_STABILITY"
    VALIDATING = "VALIDATING"
    REGISTERED = "REGISTERED"
    PROBING = "PROBING"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    REJECTED = "REJECTED"
    REPLACED = "REPLACED"
    REMOVED = "REMOVED"


@dataclass(frozen=True, slots=True)
class PhysicalIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: Any) -> "PhysicalIdentity":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True, slots=True)
class ArtifactFileEvidence:
    relative_path: str
    file_sha256: str
    size_bytes: int
    gguf_version: int
    tensor_count: int
    metadata_kv_count: int
    role_hint: RoleHint
    physical_identity: PhysicalIdentity
    hash_reused: bool = False

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "file_sha256": self.file_sha256,
            "gguf_version": self.gguf_version,
            "tensor_count": self.tensor_count,
            "metadata_kv_count": self.metadata_kv_count,
            "role_hint": self.role_hint.value,
        }


@dataclass(frozen=True, slots=True)
class ArtifactBundleEvidence:
    bundle_root: Path
    relative_root: str
    bundle_id: str
    bundle_sha256: str
    bundle_kind: BundleKind
    size_bytes: int
    files: tuple[ArtifactFileEvidence, ...]
    physical_manifest: tuple[dict[str, Any], ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def reused_hash_count(self) -> int:
        return sum(item.hash_reused for item in self.files)

    def content_manifest(self) -> list[dict[str, Any]]:
        return [item.manifest_dict() for item in self.files]


@dataclass(frozen=True, slots=True)
class RouterModelEvidence:
    router_model_id: str
    router_source: str
    router_status: str
    display_name: str
    physical_path: str | None
    connected_paths: tuple[str, ...]
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    observed_utc: str
    metadata_json: str
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    model_version_id: str
    manifest_json: str
    manifest_sha256: str
    props_payload_sha256: str | None
    observed_utc: str


class ArtifactInspectionError(RuntimeError):
    """A bounded model-domain rejection from physical inspection."""

    def __init__(
        self, reason_code: str, message: str, detail: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.detail = detail or {}


class ArtifactPendingStability(ArtifactInspectionError):
    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__("PENDING_STABILITY", message, detail)


HashCache = dict[str, tuple[PhysicalIdentity, str]]
