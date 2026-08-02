"""Strict, bounded JSON configuration loading with canonical identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import BootstrapError, ErrorCode
from .paths import RepositoryPaths, validate_relative_path


MAX_CONFIGURATION_BYTES = 2 * 1024 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object_without_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapError(
                ErrorCode.CONFIGURATION_INVALID,
                "duplicate JSON object key",
                context={"key": key},
            )
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class LoadedConfiguration:
    name: str
    path: Path
    data: dict[str, Any]
    sha256: str
    identity: str
    version: int


def load_configuration(paths: RepositoryPaths, name: str) -> LoadedConfiguration:
    portable = validate_relative_path(name)
    if len(portable.parts) != 1 or portable.suffix != ".json":
        raise BootstrapError(ErrorCode.PATH_UNSAFE, "configuration name must be one JSON filename")
    path = paths.configuration / portable.name
    if not path.is_file() or path.is_symlink():
        raise BootstrapError(
            ErrorCode.CONFIGURATION_MISSING,
            "required configuration is absent",
            context={"name": portable.name},
        )
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_CONFIGURATION_BYTES or b"\x00" in raw:
        raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "configuration byte envelope is invalid")
    try:
        decoded = raw.decode("utf-8")
        data = json.loads(decoded, object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "configuration is not strict UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise BootstrapError(ErrorCode.CONFIGURATION_INVALID, "configuration root must be an object")
    identity = data.get("identity")
    version = data.get("version")
    if not isinstance(identity, str) or not identity or type(version) is not int or version < 1:
        raise BootstrapError(
            ErrorCode.CONFIGURATION_INVALID,
            "configuration requires a non-empty identity and positive integer version",
            context={"name": portable.name},
        )
    return LoadedConfiguration(
        name=portable.name,
        path=path,
        data=data,
        sha256=hashlib.sha256(raw).hexdigest(),
        identity=identity,
        version=version,
    )


def load_registry(paths: RepositoryPaths, names: Iterable[str]) -> dict[str, LoadedConfiguration]:
    loaded: dict[str, LoadedConfiguration] = {}
    identities: set[str] = set()
    for name in names:
        config = load_configuration(paths, name)
        if config.identity in identities:
            raise BootstrapError(
                ErrorCode.CONFIGURATION_INVALID,
                "configuration identity is duplicated",
                context={"identity": config.identity},
            )
        loaded[config.name] = config
        identities.add(config.identity)
    return loaded
