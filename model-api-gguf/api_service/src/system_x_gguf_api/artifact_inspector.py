"""Safe physical discovery and content identity for connected GGUF bundles."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import struct
import time
from types import SimpleNamespace
import unicodedata
from typing import Any

from .registry_types import (
    ArtifactBundleEvidence,
    ArtifactFileEvidence,
    ArtifactInspectionError,
    ArtifactPendingStability,
    BUNDLE_MANIFEST_PREAMBLE,
    BundleKind,
    HashCache,
    PhysicalIdentity,
    RoleHint,
)


GGUF_MAGIC = b"GGUF"
SUPPORTED_GGUF_VERSIONS = frozenset({2, 3})
DEFAULT_HASH_CHUNK_SIZE = 8 * 1024 * 1024
PUBLIC_SLUG_MAXIMUM = 48
SHARD_PATTERN = re.compile(
    r"^(?P<base>.+)-(?P<number>[0-9]{5})-of-(?P<total>[0-9]{5})\.gguf$",
    re.IGNORECASE,
)


def public_slug(seed: str) -> str:
    normalized = unicodedata.normalize("NFKC", seed).lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    normalized = normalized[:PUBLIC_SLUG_MAXIMUM].rstrip("-")
    return normalized or "model"


def public_model_version_id(
    seed: str,
    bundle_sha256: str,
    *,
    location_identity: str | None = None,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
        raise ValueError("bundle SHA-256 is invalid")
    location_suffix = ""
    if location_identity is not None:
        normalized_location = unicodedata.normalize("NFKC", location_identity)
        location_digest = hashlib.sha256(
            b"system-x.gguf-location.v1\0"
            + normalized_location.encode("utf-8")
        ).hexdigest()[:8]
        location_suffix = f"-{location_digest}"
    return (
        f"sx-gguf-{public_slug(seed)}"
        f"{location_suffix}-{bundle_sha256[:12]}"
    )


def _shutdown_requested(shutdown_event: Any | None) -> bool:
    return bool(shutdown_event is not None and shutdown_event.is_set())


class ArtifactInspector:
    """Inspect one router-recognized model as one arbitrary-sized bundle."""

    def __init__(
        self,
        model_root: Path,
        stability_samples: int,
        stability_interval_seconds: float,
        hash_chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    ) -> None:
        root_info = model_root.lstat()
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise ValueError("model root must be a direct directory")
        self.model_root = model_root.resolve(strict=True)
        if not 2 <= stability_samples <= 10:
            raise ValueError("stability sample count is out of bounds")
        if not 0.0 <= stability_interval_seconds <= 10.0:
            raise ValueError("stability interval is out of bounds")
        if hash_chunk_size < 24:
            raise ValueError("hash chunk size is too small")
        self.stability_samples = stability_samples
        self.stability_interval_seconds = stability_interval_seconds
        self.hash_chunk_size = hash_chunk_size

    def _safe_contained_path(self, supplied: str | Path) -> Path:
        candidate = Path(supplied)
        if not candidate.is_absolute():
            candidate = self.model_root / candidate
        if not os.path.lexists(candidate):
            raise ArtifactInspectionError(
                "ARTIFACT_MISSING", "router-referenced artifact is missing"
            )
        if stat.S_ISLNK(candidate.lstat().st_mode):
            raise ArtifactInspectionError(
                "SYMLINK_REJECTED", "router-referenced artifact is a symlink"
            )
        try:
            canonical = candidate.resolve(strict=True)
            relative = canonical.relative_to(self.model_root)
        except (OSError, ValueError) as exc:
            raise ArtifactInspectionError(
                "PATH_ESCAPE", "router-referenced artifact escapes managed root"
            ) from exc
        current = self.model_root
        for component in relative.parts:
            current = current / component
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ArtifactInspectionError(
                    "SYMLINK_REJECTED", "artifact path contains a symlink"
                )
        return canonical

    def _fallback_router_path(self, model_id: str) -> Path:
        direct = self.model_root / f"{model_id}.gguf"
        directory = self.model_root / model_id
        candidates = [
            path
            for path in (direct, directory)
            if os.path.lexists(path) and not path.is_symlink()
        ]
        if len(candidates) != 1:
            raise ArtifactInspectionError(
                "ROUTER_PATH_UNRESOLVED",
                "router model ID did not resolve to exactly one managed artifact",
                {"candidate_count": len(candidates)},
            )
        return self._safe_contained_path(candidates[0])

    def inspect_location(
        self,
        relative_root: str,
        hash_cache: HashCache | None = None,
        shutdown_event: Any | None = None,
    ) -> ArtifactBundleEvidence:
        """Validate one physical top-level unit before any router refresh."""

        relative = Path(relative_root)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.parts[0] in {"", ".", ".."}
        ):
            raise ArtifactInspectionError(
                "LOCATION_INVALID",
                "physical candidate is not one top-level managed unit",
            )
        root = self._safe_contained_path(relative)
        if root.is_file():
            primary = root
        elif root.is_dir():
            members = self._directory_members(root)
            if not members:
                raise ArtifactInspectionError(
                    "NO_GGUF_MEMBERS",
                    "bundle contains no regular GGUF member",
                )
            preferred = [
                member
                for member in members
                if "mmproj" not in member.name.lower()
                and "mtp" not in member.name.lower()
            ]
            primary = (preferred or list(members))[0]
        else:
            raise ArtifactInspectionError(
                "ARTIFACT_NOT_REGULAR",
                "managed unit is neither a regular GGUF nor a direct directory",
            )
        reference = SimpleNamespace(
            model_id=relative.stem,
            physical_path=str(primary),
            connected_paths=(),
        )
        return self.inspect(reference, hash_cache, shutdown_event)

    def _resolve_bundle(
        self, router_model: Any
    ) -> tuple[Path, Path, tuple[Path, ...]]:
        supplied_paths: list[Path] = []
        physical_path = getattr(router_model, "physical_path", None)
        if physical_path:
            supplied_paths.append(self._safe_contained_path(physical_path))
        for value in getattr(router_model, "connected_paths", ()) or ():
            path = self._safe_contained_path(value)
            if path not in supplied_paths:
                supplied_paths.append(path)
        if not supplied_paths:
            supplied_paths.append(
                self._fallback_router_path(str(router_model.model_id))
            )
        primary = supplied_paths[0]
        relative = primary.relative_to(self.model_root)
        if len(relative.parts) == 1 and primary.is_file():
            bundle_root = primary
        else:
            bundle_root = self.model_root / relative.parts[0]
            if not bundle_root.is_dir() or bundle_root.is_symlink():
                raise ArtifactInspectionError(
                    "BUNDLE_ROOT_INVALID",
                    "top-level bundle root is not a direct directory",
                )
            for connected in supplied_paths:
                try:
                    connected.relative_to(bundle_root)
                except ValueError as exc:
                    raise ArtifactInspectionError(
                        "CONNECTED_PATH_OUTSIDE_BUNDLE",
                        "router connected path crosses top-level bundle boundary",
                    ) from exc
        return bundle_root, primary, tuple(supplied_paths)

    def _directory_members(self, root: Path) -> tuple[Path, ...]:
        members: list[Path] = []

        def visit(directory: Path) -> None:
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name)
            except OSError as exc:
                raise ArtifactInspectionError(
                    "BUNDLE_ENUMERATION_FAILED",
                    "managed bundle could not be enumerated",
                    {"error_type": type(exc).__name__},
                ) from exc
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise ArtifactInspectionError(
                        "SYMLINK_REJECTED",
                        "managed bundle contains a symlink",
                        {"relative_path": path.relative_to(root).as_posix()},
                    )
                if entry.is_dir(follow_symlinks=False):
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    if entry.name.lower().endswith(".gguf"):
                        members.append(path)
                elif entry.name.lower().endswith(".gguf"):
                    raise ArtifactInspectionError(
                        "SPECIAL_OBJECT_REJECTED",
                        "GGUF-named bundle member is not a regular file",
                    )

        visit(root)
        return tuple(sorted(members, key=lambda path: path.relative_to(root).as_posix()))

    def _members_for_bundle(
        self, root: Path, connected_paths: tuple[Path, ...]
    ) -> tuple[Path, ...]:
        if root.is_file():
            members = {
                path
                for path in connected_paths
                if path.name.lower().endswith(".gguf")
            }
            members.add(root)
            for path in members:
                if path.parent != self.model_root:
                    raise ArtifactInspectionError(
                        "CONNECTED_PATH_OUTSIDE_BUNDLE",
                        "direct-file connected sidecar is outside managed root",
                    )
            result = tuple(sorted(members, key=lambda path: path.name))
        else:
            result = self._directory_members(root)
        if not result:
            raise ArtifactInspectionError(
                "NO_GGUF_MEMBERS", "bundle contains no regular GGUF member"
            )
        for path in result:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ArtifactInspectionError(
                    "ARTIFACT_NOT_REGULAR", "bundle member is not a direct regular file"
                )
        return result

    @staticmethod
    def _relative_path(root: Path, member: Path) -> str:
        return member.name if root.is_file() else member.relative_to(root).as_posix()

    def _manifest(
        self, root: Path, connected_paths: tuple[Path, ...]
    ) -> tuple[tuple[Path, PhysicalIdentity], ...]:
        members = self._members_for_bundle(root, connected_paths)
        return tuple(
            (path, PhysicalIdentity.from_stat(path.lstat())) for path in members
        )

    def _stable_manifest(
        self,
        root: Path,
        connected_paths: tuple[Path, ...],
        shutdown_event: Any | None,
    ) -> tuple[tuple[Path, PhysicalIdentity], ...]:
        samples = []
        for index in range(self.stability_samples):
            if _shutdown_requested(shutdown_event):
                raise ArtifactPendingStability("inspection cancelled during sampling")
            samples.append(self._manifest(root, connected_paths))
            if index + 1 < self.stability_samples:
                time.sleep(self.stability_interval_seconds)
        baseline = tuple(
            (self._relative_path(root, path), identity)
            for path, identity in samples[0]
        )
        for sample in samples[1:]:
            normalized = tuple(
                (self._relative_path(root, path), identity)
                for path, identity in sample
            )
            if normalized != baseline:
                raise ArtifactPendingStability(
                    "complete bundle manifest changed between stability samples"
                )
        return samples[-1]

    @staticmethod
    def _read_header(value: bytes) -> tuple[int, int, int]:
        if len(value) < 24:
            raise ArtifactInspectionError(
                "GGUF_TRUNCATED", "GGUF member is smaller than its fixed header"
            )
        if value[:4] != GGUF_MAGIC:
            raise ArtifactInspectionError(
                "GGUF_MAGIC_INVALID", "GGUF member magic does not match"
            )
        little_version = struct.unpack_from("<I", value, 4)[0]
        if little_version in SUPPORTED_GGUF_VERSIONS:
            order = "<"
            version = little_version
        else:
            big_version = struct.unpack_from(">I", value, 4)[0]
            if big_version not in SUPPORTED_GGUF_VERSIONS:
                raise ArtifactInspectionError(
                    "GGUF_VERSION_UNSUPPORTED",
                    "GGUF member version is unsupported",
                    {"little_endian_value": little_version},
                )
            order = ">"
            version = big_version
        tensor_count, metadata_kv_count = struct.unpack_from(f"{order}QQ", value, 8)
        if tensor_count > 100_000_000 or metadata_kv_count > 100_000_000:
            raise ArtifactInspectionError(
                "GGUF_HEADER_COUNT_INVALID",
                "GGUF fixed-header count exceeds the bounded reader limit",
            )
        return version, tensor_count, metadata_kv_count

    def _read_member(
        self,
        path: Path,
        expected: PhysicalIdentity,
        cached: tuple[PhysicalIdentity, str] | None,
        shutdown_event: Any | None,
    ) -> tuple[str, int, int, int, bool]:
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ArtifactPendingStability(
                "bundle member could not be opened safely",
                {"error_type": type(exc).__name__},
            ) from exc
        try:
            before = PhysicalIdentity.from_stat(os.fstat(descriptor))
            if before != expected or not stat.S_ISREG(before.mode):
                raise ArtifactPendingStability(
                    "bundle member identity changed before hashing"
                )
            first = os.read(descriptor, min(self.hash_chunk_size, expected.size))
            version, tensor_count, metadata_kv_count = self._read_header(first)
            reused = bool(cached is not None and cached[0] == expected)
            if reused:
                digest_hex = cached[1]
                if not re.fullmatch(r"[0-9a-f]{64}", digest_hex):
                    raise ArtifactInspectionError(
                        "HASH_CACHE_INVALID", "stored file hash is invalid"
                    )
            else:
                digest = hashlib.sha256()
                digest.update(first)
                while block := os.read(descriptor, self.hash_chunk_size):
                    if _shutdown_requested(shutdown_event):
                        raise ArtifactPendingStability(
                            "inspection cancelled during hashing"
                        )
                    digest.update(block)
                digest_hex = digest.hexdigest()
            after = PhysicalIdentity.from_stat(os.fstat(descriptor))
            if before != after:
                raise ArtifactPendingStability(
                    "bundle member identity changed during hashing"
                )
            return digest_hex, version, tensor_count, metadata_kv_count, reused
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_shards(root: Path, members: tuple[Path, ...]) -> None:
        groups: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
        for member in members:
            relative = member.name if root.is_file() else member.relative_to(root).as_posix()
            match = SHARD_PATTERN.match(member.name)
            if match is None:
                continue
            key = (str(Path(relative).parent), match.group("base").lower())
            groups.setdefault(key, []).append(
                (
                    int(match.group("number")),
                    int(match.group("total")),
                    relative,
                )
            )
        for values in groups.values():
            totals = {total for _, total, _ in values}
            if len(totals) != 1:
                raise ArtifactInspectionError(
                    "SHARD_TOTAL_CONTRADICTION",
                    "shard group declares contradictory totals",
                )
            total = next(iter(totals))
            numbers = [number for number, _, _ in values]
            if total < 1 or len(numbers) != len(set(numbers)):
                raise ArtifactInspectionError(
                    "SHARD_DUPLICATE", "shard group contains duplicate numbers"
                )
            if sorted(numbers) != list(range(1, total + 1)):
                raise ArtifactInspectionError(
                    "SHARD_INCOMPLETE",
                    "shard group is not one-based and contiguous through its total",
                    {"declared_total": total, "observed_numbers": sorted(numbers)},
                )

    @staticmethod
    def _role_hint(path: Path, primary: Path) -> RoleHint:
        lower = path.name.lower()
        if "mmproj" in lower:
            return RoleHint.MMPROJ_SIDECAR
        if "mtp" in lower:
            return RoleHint.MTP_SIDECAR
        if SHARD_PATTERN.match(path.name):
            return RoleHint.SHARD
        if path == primary:
            return RoleHint.PRIMARY
        return RoleHint.OTHER_GGUF

    def inspect(
        self,
        router_model: Any,
        hash_cache: HashCache | None = None,
        shutdown_event: Any | None = None,
    ) -> ArtifactBundleEvidence:
        root, primary, connected_paths = self._resolve_bundle(router_model)
        stable = self._stable_manifest(root, connected_paths, shutdown_event)
        members = tuple(path for path, _ in stable)
        if primary not in members:
            raise ArtifactInspectionError(
                "ROUTER_PRIMARY_NOT_IN_BUNDLE",
                "router-recognized primary GGUF is absent from bundle members",
            )
        self._validate_shards(root, members)
        cache = hash_cache or {}
        stable_by_relative = {
            self._relative_path(root, path): identity for path, identity in stable
        }
        complete_cache_usable = set(cache) == set(stable_by_relative) and all(
            cache[relative][0] == identity
            for relative, identity in stable_by_relative.items()
        )
        files = []
        for path, identity in stable:
            relative = self._relative_path(root, path)
            digest, version, tensors, metadata, reused = self._read_member(
                path,
                identity,
                cache.get(relative) if complete_cache_usable else None,
                shutdown_event,
            )
            files.append(
                ArtifactFileEvidence(
                    relative_path=relative,
                    file_sha256=digest,
                    size_bytes=identity.size,
                    gguf_version=version,
                    tensor_count=tensors,
                    metadata_kv_count=metadata,
                    role_hint=self._role_hint(path, primary),
                    physical_identity=identity,
                    hash_reused=reused,
                )
            )
        final_manifest = self._manifest(root, connected_paths)
        if tuple(
            (self._relative_path(root, path), identity)
            for path, identity in final_manifest
        ) != tuple(
            (self._relative_path(root, path), identity) for path, identity in stable
        ):
            raise ArtifactPendingStability(
                "complete bundle manifest changed after hashing"
            )
        ordered = tuple(sorted(files, key=lambda item: item.relative_path))
        if len(ordered) == 1:
            bundle_sha256 = ordered[0].file_sha256
        else:
            digest = hashlib.sha256()
            digest.update(BUNDLE_MANIFEST_PREAMBLE)
            for item in ordered:
                digest.update(item.relative_path.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(item.size_bytes).encode("ascii"))
                digest.update(b"\0")
                digest.update(item.file_sha256.encode("ascii"))
                digest.update(b"\n")
            bundle_sha256 = digest.hexdigest()
        relative_root = root.relative_to(self.model_root).as_posix()
        physical_manifest = tuple(
            {
                "relative_path": item.relative_path,
                **item.physical_identity.as_dict(),
            }
            for item in ordered
        )
        return ArtifactBundleEvidence(
            bundle_root=root,
            relative_root=relative_root,
            bundle_id=f"bundle-{bundle_sha256}",
            bundle_sha256=bundle_sha256,
            bundle_kind=(
                BundleKind.SINGLE_FILE if root.is_file() else BundleKind.DIRECTORY_BUNDLE
            ),
            size_bytes=sum(item.size_bytes for item in ordered),
            files=ordered,
            physical_manifest=physical_manifest,
        )
