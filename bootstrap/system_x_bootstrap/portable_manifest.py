"""Build and validate the source-controlled portable candidate-tree manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Mapping


MANIFEST_PATH = "SYSTEM_X_PORTABLE_TREE_MANIFEST.json"
SCHEMA_VERSION = "system-x.portable-tree-manifest.v4"
SEMANTICS = "FULL_TREE"
REGULAR_MODES = frozenset({"100644", "100755"})
ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "TRACKED_CANDIDATE_INPUT",
        "TRACKED_TEST",
        "STATIC_CONTRACT",
        "DOCUMENTATION",
        "VENDORED_SOURCE",
        "DETERMINISTIC_FIXTURE",
    }
)


class ManifestError(ValueError):
    """Raised when a candidate-tree manifest is malformed or incomplete."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(path: str) -> bool:
    candidate = Path(path)
    return bool(path) and not candidate.is_absolute() and ".." not in candidate.parts and candidate.as_posix() == path


def _classification(path: str) -> str:
    lower = path.lower()
    if path.startswith("model-api-gguf/llama.cpp/"):
        return "VENDORED_SOURCE"
    if "/tests/" in ("/" + path) or path.startswith("tests/") or Path(path).name.startswith("test_"):
        return "TRACKED_TEST"
    if lower.endswith((".md", ".rst", ".txt")) or Path(path).name.lower().startswith("readme"):
        return "DOCUMENTATION"
    if (
        "/configuration/" in ("/" + path)
        or lower.endswith((".json", ".toml", ".yaml", ".yml", ".ini", ".lock"))
        or "manifest" in Path(path).name.lower()
    ):
        return "STATIC_CONTRACT"
    if lower.endswith((".gguf", ".bin")):
        return "DETERMINISTIC_FIXTURE"
    return "TRACKED_CANDIDATE_INPUT"


def git_records(root: Path) -> list[dict[str, str]]:
    """Return regular files from the current candidate index without refreshing it."""

    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "-z"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ManifestError("git ls-files failed: " + result.stderr.decode("utf-8", "replace").strip())
    records: list[dict[str, str]] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            header, path_raw = raw.split(b"\t", 1)
            mode, blob, _stage = header.decode("ascii").split()
            path = path_raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ManifestError("invalid git ls-files record") from exc
        if mode in REGULAR_MODES:
            records.append({"path": path, "git_mode": mode, "git_blob": blob})
    return sorted(records, key=lambda row: row["path"])


def _entry(root: Path, record: Mapping[str, str]) -> dict[str, object]:
    path = str(record["path"])
    if not _safe_path(path):
        raise ManifestError("unsafe candidate path: " + path)
    full = root / path
    if not full.is_file() or full.is_symlink():
        raise ManifestError("regular candidate file is absent: " + path)
    data = full.read_bytes()
    mode = str(record["git_mode"])
    if mode not in REGULAR_MODES:
        raise ManifestError("non-regular mode in candidate record: " + path)
    return {
        "bytes": len(data),
        "classification": _classification(path),
        "git_blob": str(record["git_blob"]),
        "git_mode": mode,
        "path": path,
        "portable_mode": format(full.stat().st_mode & 0o777, "#06o"),
        "sha256": _sha256(data),
    }


def build_manifest(root: Path, records: Iterable[Mapping[str, str]] | None = None) -> dict[str, object]:
    """Build a FULL_TREE manifest from the current candidate Git tree."""

    candidate = sorted((dict(row) for row in (records if records is not None else git_records(root))), key=lambda row: row["path"])
    paths = [str(row["path"]) for row in candidate]
    if len(paths) != len(set(paths)):
        raise ManifestError("candidate records contain duplicate paths")
    if paths != sorted(paths):
        raise ManifestError("candidate records are not lexicographically ordered")
    if MANIFEST_PATH not in paths:
        raise ManifestError("candidate tree does not contain the portable manifest")
    entries = [_entry(root, row) for row in candidate if row["path"] != MANIFEST_PATH]
    exclusion = {
        "path": MANIFEST_PATH,
        "reason": "The manifest describes the candidate Git tree and excludes itself to avoid a self-referential digest.",
    }
    return {
        "coverage": {
            "candidate_git_tree": "COMPLETE",
            "constructor_graph": "COMPLETE_CANDIDATE_GIT_TREE_CLOSURE",
            "generated_state": "EXCLUDED",
        },
        "entries": entries,
        "entry_count": len(entries),
        "excluded": [exclusion],
        "manifest_path": MANIFEST_PATH,
        "ordering": "lexicographic-path",
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "self_exclusion": exclusion,
        "source_root": ".",
        "tracked_regular_file_count": len(candidate),
    }


def write_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    path = root / MANIFEST_PATH
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(encoded)


def _expected_records(records: Iterable[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in records:
        path = str(row["path"])
        if str(row["git_mode"]) in REGULAR_MODES:
            if path in result:
                raise ManifestError("duplicate candidate path: " + path)
            result[path] = {"git_mode": str(row["git_mode"]), "git_blob": str(row["git_blob"])}
    return result


def validate_manifest(
    root: Path,
    manifest: Mapping[str, object] | None = None,
    records: Iterable[Mapping[str, str]] | None = None,
) -> dict[str, object]:
    """Validate manifest structure, tree coverage, Git identity, and bytes."""

    path = root / MANIFEST_PATH
    if manifest is None:
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ManifestError("manifest contains NUL bytes")
        manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("schema_version is not the portable-tree schema")
    if manifest.get("semantics") != SEMANTICS:
        raise ManifestError("manifest semantics are not FULL_TREE")
    if manifest.get("source_root") != ".":
        raise ManifestError("source_root must be relative dot")
    if manifest.get("ordering") != "lexicographic-path":
        raise ManifestError("manifest ordering is not lexicographic-path")
    if manifest.get("manifest_path") != MANIFEST_PATH:
        raise ManifestError("manifest_path is incorrect")
    exclusion = manifest.get("self_exclusion")
    if not isinstance(exclusion, Mapping) or exclusion.get("path") != MANIFEST_PATH:
        raise ManifestError("explicit self exclusion is missing")
    excluded = manifest.get("excluded")
    if excluded != [exclusion]:
        raise ManifestError("excluded does not match explicit self exclusion")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ManifestError("entries is not a list")
    paths = [entry.get("path") if isinstance(entry, Mapping) else None for entry in entries]
    if any(not isinstance(path_value, str) or not _safe_path(path_value) for path_value in paths):
        raise ManifestError("manifest contains an unsafe path")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ManifestError("manifest paths are not sorted and unique")
    if MANIFEST_PATH in paths:
        raise ManifestError("manifest self-entry is forbidden")
    expected = _expected_records(records if records is not None else git_records(root))
    expected_without_self = {key: value for key, value in expected.items() if key != MANIFEST_PATH}
    if set(paths) != set(expected_without_self):
        missing = sorted(set(expected_without_self) - set(paths))
        extra = sorted(set(paths) - set(expected_without_self))
        raise ManifestError(f"candidate closure mismatch missing={missing[:5]} extra={extra[:5]}")
    if manifest.get("entry_count") != len(entries) or manifest.get("tracked_regular_file_count") != len(expected):
        raise ManifestError("manifest counts are incorrect")
    errors: list[str] = []
    for entry in entries:
        assert isinstance(entry, Mapping)
        path_value = str(entry["path"])
        row = expected_without_self[path_value]
        if entry.get("git_mode") != row["git_mode"] or entry.get("git_blob") != row["git_blob"]:
            errors.append(path_value + ":git-identity")
        if entry.get("classification") not in ALLOWED_CLASSIFICATIONS:
            errors.append(path_value + ":classification")
        full = root / path_value
        if not full.is_file() or full.is_symlink():
            errors.append(path_value + ":not-regular")
            continue
        data = full.read_bytes()
        if entry.get("bytes") != len(data):
            errors.append(path_value + ":bytes")
        if entry.get("sha256") != _sha256(data):
            errors.append(path_value + ":sha256")
        if entry.get("portable_mode") != format(full.stat().st_mode & 0o777, "#06o"):
            errors.append(path_value + ":mode")
    if errors:
        raise ManifestError("manifest member errors: " + ",".join(errors[:10]))
    return {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "entry_count": len(entries),
        "tracked_regular_file_count": len(expected),
        "self_excluded": True,
        "errors": 0,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.write:
        manifest = build_manifest(root)
        write_manifest(root, manifest)
        print(json.dumps({"status": "WRITTEN", "entry_count": manifest["entry_count"]}, sort_keys=True))
    if args.validate or not args.write:
        print(json.dumps(validate_manifest(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
