"""Bounded, non-executing inspection of native model directories."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any


MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
MAX_INDEX_ENTRIES = 1_000_000
MAX_TENSORS = 1_000_000
MAX_DIMENSIONS = 4
MAX_TENSOR_NAME_BYTES = 4096
MAX_RETAINED_ITEMS = 512
MAX_RETAINED_STRING_BYTES = 4096
MAX_U64 = (1 << 64) - 1

SAFETENSORS_DTYPES = MappingProxyType(
    {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "U16": 2,
        "I16": 2,
        "F16": 2,
        "BF16": 2,
        "U32": 4,
        "I32": 4,
        "F32": 4,
        "U64": 8,
        "I64": 8,
        "F64": 8,
    }
)


def _format_definition_identity() -> str:
    value = {
        "configuration": "duplicate-key-safe-bounded-json",
        "safetensors_header": "little-endian-uint64",
        "safetensors_dtypes": dict(SAFETENSORS_DTYPES),
        "pytorch": "structural-only-no-deserialization",
        "limits": {
            "json": MAX_JSON_BYTES,
            "safetensors_header": MAX_SAFETENSORS_HEADER_BYTES,
            "index_entries": MAX_INDEX_ENTRIES,
            "tensors": MAX_TENSORS,
            "dimensions": MAX_DIMENSIONS,
            "tensor_name": MAX_TENSOR_NAME_BYTES,
        },
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


FORMAT_DEFINITION_IDENTITY = _format_definition_identity()

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
)


class NativeIssue(Exception):
    """A stable native-format terminal classification."""

    def __init__(self, category: str, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.reason_code = reason_code
        self.message = message


class _DuplicateJSONKey(ValueError):
    pass


@dataclass(frozen=True)
class NativeResult:
    model_type: str | None
    architectures: tuple[str, ...]
    quantization: dict[str, Any] | None
    declared_dtype: str | None
    configuration_source: str
    tokenizer_source: tuple[str, ...]
    weight_format: str
    shard_count: int
    tensor_count: int
    tensor_bytes: int
    dtype_histogram: dict[str, int]
    warnings: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    payload_deserialized: bool = False
    code_executed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "quantization": self.quantization,
            "declared_dtype": self.declared_dtype,
            "configuration_source": self.configuration_source,
            "tokenizer_source": list(self.tokenizer_source),
            "weight_format": self.weight_format,
            "shard_count": self.shard_count,
            "tensor_count": self.tensor_count,
            "tensor_bytes": self.tensor_bytes,
            "dtype_histogram": dict(self.dtype_histogram),
            "warnings": list(self.warnings),
            "reason_codes": list(self.reason_codes),
            "evidence": [dict(item) for item in self.evidence],
            "payload_deserialized": self.payload_deserialized,
            "code_executed": self.code_executed,
        }


@dataclass(frozen=True)
class _SafeTensorFile:
    name: str
    tensors: tuple[str, ...]
    tensor_bytes: int
    file_size: int
    dtype_histogram: dict[str, int]


def _issue(category: str, reason: str, message: str) -> NativeIssue:
    return NativeIssue(category, reason, message)


def _checked_product(values: list[int], multiplier: int) -> int:
    total = multiplier
    for value in values:
        if value != 0 and total > MAX_U64 // value:
            raise _issue(
                "CORRUPT",
                "NATIVE_SAFETENSORS_RANGE_INVALID",
                "safetensors tensor size arithmetic overflow",
            )
        total *= value
    return total


def _regular_stat(path: Path, reason_code: str) -> os.stat_result:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise _issue("INCOMPLETE", reason_code, f"missing file: {path.name}") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise _issue(
            "CORRUPT",
            reason_code,
            f"non-regular or symlink file rejected: {path.name}",
        )
    return details


def _open_nofollow(path: Path, reason_code: str) -> tuple[int, os.stat_result]:
    before = _regular_stat(path, reason_code)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        category = "CORRUPT" if error.errno in {
            errno.ELOOP,
            errno.EACCES,
            errno.EPERM,
        } else "INCOMPLETE"
        raise _issue(category, reason_code, f"safe open failed: {path.name}") from error
    opened = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_mode,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    ):
        os.close(descriptor)
        raise _issue(
            "CORRUPT",
            reason_code,
            f"file changed during safe open: {path.name}",
        )
    return descriptor, opened


def _revalidate_open_file(
    path: Path,
    descriptor: int,
    opened: os.stat_result,
    reason_code: str,
) -> None:
    current_fd = os.fstat(descriptor)
    current_path = _regular_stat(path, reason_code)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(opened) != identity(current_fd) or identity(opened) != identity(
        current_path
    ):
        raise _issue(
            "CORRUPT",
            reason_code,
            f"file changed while inspected: {path.name}",
        )


def _read_bounded(path: Path, maximum: int, reason_code: str) -> bytes:
    descriptor, opened = _open_nofollow(path, reason_code)
    try:
        if opened.st_size > maximum:
            raise _issue(
                "CORRUPT",
                reason_code,
                f"bounded metadata file exceeds {maximum} bytes: {path.name}",
            )
        pieces: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            pieces.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise _issue(
                    "CORRUPT",
                    reason_code,
                    f"bounded metadata read exceeded limit: {path.name}",
                )
        _revalidate_open_file(path, descriptor, opened, reason_code)
        return b"".join(pieces)
    finally:
        os.close(descriptor)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, maximum: int, reason_code: str) -> Any:
    raw = _read_bounded(path, maximum, reason_code)
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJSONKey, ValueError) as error:
        raise _issue(
            "CORRUPT",
            reason_code,
            f"invalid duplicate-safe JSON: {path.name}",
        ) from error


def _safe_root(root: Path) -> None:
    try:
        details = root.lstat()
    except FileNotFoundError as error:
        raise _issue(
            "INCOMPLETE", "NATIVE_CONFIG_MISSING", "native directory is missing"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise _issue(
            "CORRUPT",
            "NATIVE_CONFIG_INVALID",
            "native artifact root must be a non-symlink directory",
        )


def _safe_child(root: Path, name: str, reason_code: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or len(PurePosixPath(name).parts) != 1
        or name in {".", ".."}
    ):
        raise _issue(
            "CORRUPT", reason_code, "weight-map path escape rejected"
        )
    child = root / name
    try:
        resolved_parent = child.parent.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise _issue("INCOMPLETE", reason_code, "artifact root changed") from error
    if resolved_parent != root_resolved:
        raise _issue("CORRUPT", reason_code, "weight-map path escaped root")
    return child


def _bounded_quantization(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    retained: dict[str, Any] = {}
    for key in sorted(value):
        item = value[key]
        if (
            isinstance(key, str)
            and len(key.encode("utf-8")) <= 256
            and len(retained) < 64
            and (
                item is None
                or isinstance(item, (bool, int, float))
                or (
                    isinstance(item, str)
                    and len(item.encode("utf-8")) <= MAX_RETAINED_STRING_BYTES
                )
            )
        ):
            retained[key] = item
    return retained or None


def _normalize_config(
    value: Any,
) -> tuple[
    str | None,
    tuple[str, ...],
    dict[str, Any] | None,
    str | None,
    bool,
]:
    if not isinstance(value, dict):
        raise _issue(
            "CORRUPT", "NATIVE_CONFIG_INVALID", "config.json root must be an object"
        )
    model_type = value.get("model_type")
    if model_type is not None and (
        not isinstance(model_type, str) or not model_type.strip()
    ):
        raise _issue(
            "CORRUPT", "NATIVE_CONFIG_INVALID", "model_type must be a string"
        )
    architectures_value = value.get("architectures", [])
    if architectures_value is None:
        architectures_value = []
    if (
        not isinstance(architectures_value, list)
        or len(architectures_value) > MAX_RETAINED_ITEMS
        or any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > MAX_RETAINED_STRING_BYTES
            for item in architectures_value
        )
    ):
        raise _issue(
            "CORRUPT",
            "NATIVE_CONFIG_INVALID",
            "architectures must be a bounded string list",
        )
    quantization = _bounded_quantization(value.get("quantization_config"))
    declared_dtype = value.get("torch_dtype")
    if declared_dtype is not None and (
        not isinstance(declared_dtype, str)
        or not declared_dtype
        or len(declared_dtype.encode("utf-8")) > MAX_RETAINED_STRING_BYTES
    ):
        raise _issue(
            "CORRUPT", "NATIVE_CONFIG_INVALID", "torch_dtype is invalid"
        )
    remote_code = bool(
        value.get("auto_map")
        or value.get("custom_pipelines")
        or value.get("trust_remote_code")
    )
    return (
        model_type,
        tuple(architectures_value),
        quantization,
        declared_dtype,
        remote_code,
    )


def _parse_safetensors(path: Path) -> _SafeTensorFile:
    reason = "NATIVE_SAFETENSORS_HEADER_INVALID"
    descriptor, opened = _open_nofollow(path, reason)
    try:
        prefix = os.pread(descriptor, 8, 0)
        if len(prefix) != 8:
            raise _issue("CORRUPT", reason, "safetensors header length is truncated")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
            raise _issue("CORRUPT", reason, "safetensors header length is invalid")
        data_start = 8 + header_length
        if data_start > opened.st_size:
            raise _issue("CORRUPT", reason, "safetensors header is truncated")
        header_raw = os.pread(descriptor, header_length, 8)
        if len(header_raw) != header_length:
            raise _issue("CORRUPT", reason, "safetensors header is truncated")
        try:
            header = json.loads(
                header_raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_pairs_no_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {token}")
                ),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJSONKey,
            ValueError,
        ) as error:
            raise _issue("CORRUPT", reason, "invalid safetensors header JSON") from error
        if not isinstance(header, dict):
            raise _issue("CORRUPT", reason, "safetensors header must be an object")
        tensors: list[tuple[str, int, int]] = []
        dtype_histogram: dict[str, int] = {}
        for name, descriptor_value in header.items():
            if name == "__metadata__":
                if not isinstance(descriptor_value, dict) or any(
                    not isinstance(key, str) or not isinstance(item, str)
                    for key, item in descriptor_value.items()
                ):
                    raise _issue(
                        "CORRUPT", reason, "safetensors metadata must be string pairs"
                    )
                continue
            if (
                not isinstance(name, str)
                or not name
                or len(name.encode("utf-8")) > MAX_TENSOR_NAME_BYTES
            ):
                raise _issue("CORRUPT", reason, "invalid safetensors tensor name")
            if len(tensors) >= MAX_TENSORS or not isinstance(descriptor_value, dict):
                raise _issue("CORRUPT", reason, "invalid safetensors tensor table")
            dtype = descriptor_value.get("dtype")
            shape = descriptor_value.get("shape")
            offsets = descriptor_value.get("data_offsets")
            if dtype not in SAFETENSORS_DTYPES:
                raise _issue("CORRUPT", reason, "unsupported safetensors dtype")
            if (
                not isinstance(shape, list)
                or len(shape) > MAX_DIMENSIONS
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item < 0
                    for item in shape
                )
            ):
                raise _issue(
                    "CORRUPT",
                    "NATIVE_SAFETENSORS_RANGE_INVALID",
                    "invalid safetensors shape",
                )
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item < 0
                    for item in offsets
                )
                or offsets[0] > offsets[1]
            ):
                raise _issue(
                    "CORRUPT",
                    "NATIVE_SAFETENSORS_RANGE_INVALID",
                    "invalid safetensors data offsets",
                )
            expected = _checked_product(shape, SAFETENSORS_DTYPES[dtype])
            if offsets[1] - offsets[0] != expected:
                raise _issue(
                    "CORRUPT",
                    "NATIVE_SAFETENSORS_RANGE_INVALID",
                    "safetensors tensor byte extent does not match shape",
                )
            tensors.append((name, offsets[0], offsets[1]))
            dtype_histogram[dtype] = dtype_histogram.get(dtype, 0) + 1
        if not tensors:
            raise _issue("CORRUPT", reason, "safetensors contains no tensors")
        ordered = sorted(tensors, key=lambda item: (item[1], item[2], item[0]))
        previous_end = 0
        for _name, start, end in ordered:
            if start < previous_end:
                raise _issue(
                    "CORRUPT",
                    "NATIVE_SAFETENSORS_RANGE_INVALID",
                    "overlapping safetensors tensor ranges",
                )
            previous_end = end
        payload_size = opened.st_size - data_start
        if previous_end != payload_size:
            raise _issue(
                "CORRUPT",
                "NATIVE_SAFETENSORS_RANGE_INVALID",
                "safetensors total payload size mismatch",
            )
        _revalidate_open_file(path, descriptor, opened, reason)
        return _SafeTensorFile(
            name=path.name,
            tensors=tuple(item[0] for item in tensors),
            tensor_bytes=sum(item[2] - item[1] for item in tensors),
            file_size=opened.st_size,
            dtype_histogram=dict(sorted(dtype_histogram.items())),
        )
    finally:
        os.close(descriptor)


def _weight_map(index: Any) -> tuple[dict[str, str], int | None]:
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise _issue(
            "CORRUPT",
            "NATIVE_SHARD_INDEX_INVALID",
            "native shard index lacks a weight_map object",
        )
    mapping = index["weight_map"]
    if not mapping or len(mapping) > MAX_INDEX_ENTRIES:
        raise _issue(
            "CORRUPT",
            "NATIVE_SHARD_INDEX_INVALID",
            "native shard weight_map size is invalid",
        )
    if any(
        not isinstance(tensor, str)
        or not tensor
        or len(tensor.encode("utf-8")) > MAX_TENSOR_NAME_BYTES
        or not isinstance(shard, str)
        or not shard
        for tensor, shard in mapping.items()
    ):
        raise _issue(
            "CORRUPT",
            "NATIVE_SHARD_INDEX_INVALID",
            "native shard weight_map entry is invalid",
        )
    total_size = None
    metadata = index.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise _issue(
                "CORRUPT",
                "NATIVE_SHARD_INDEX_INVALID",
                "native shard index metadata is invalid",
            )
        candidate = metadata.get("total_size")
        if candidate is not None:
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
                raise _issue(
                    "CORRUPT",
                    "NATIVE_SHARD_INDEX_INVALID",
                    "native shard total_size is invalid",
                )
            total_size = candidate
    return dict(mapping), total_size


def _validate_safetensors_index(
    root: Path, index_path: Path
) -> tuple[
    str,
    int,
    int,
    int,
    tuple[dict[str, Any], ...],
    dict[str, int],
]:
    index = _load_json(
        index_path, MAX_JSON_BYTES, "NATIVE_SHARD_INDEX_INVALID"
    )
    mapping, declared_total = _weight_map(index)
    shard_names = sorted(set(mapping.values()))
    shard_results: list[_SafeTensorFile] = []
    seen: set[str] = set()
    for shard_name in shard_names:
        shard_path = _safe_child(root, shard_name, "NATIVE_SHARD_INDEX_INVALID")
        if not shard_name.endswith(".safetensors"):
            raise _issue(
                "CORRUPT",
                "NATIVE_SHARD_INDEX_INVALID",
                "safetensors index references a non-safetensors shard",
            )
        try:
            result = _parse_safetensors(shard_path)
        except NativeIssue as error:
            if error.category == "INCOMPLETE":
                raise _issue(
                    "INCOMPLETE", "NATIVE_SHARD_MISSING", f"missing shard: {shard_name}"
                ) from error
            raise
        duplicates = seen.intersection(result.tensors)
        if duplicates:
            raise _issue(
                "CORRUPT",
                "NATIVE_SHARD_DUPLICATE_TENSOR",
                "tensor occurs in more than one safetensors shard",
            )
        seen.update(result.tensors)
        shard_results.append(result)
    if set(mapping) != seen or any(
        tensor not in seen or mapping[tensor] != shard.name
        for shard in shard_results
        for tensor in shard.tensors
    ):
        raise _issue(
            "CORRUPT",
            "NATIVE_SHARD_INDEX_INVALID",
            "safetensors index and physical shard tensors disagree",
        )
    total = sum(item.tensor_bytes for item in shard_results)
    if declared_total is not None and declared_total != total:
        raise _issue(
            "CORRUPT",
            "NATIVE_SHARD_INDEX_INVALID",
            "safetensors index total_size mismatch",
        )
    evidence = tuple(
        {
            "source": item.name,
            "tensor_count": len(item.tensors),
            "tensor_bytes": item.tensor_bytes,
        }
        for item in shard_results[:MAX_RETAINED_ITEMS]
    )
    histogram: dict[str, int] = {}
    for item in shard_results:
        for dtype, count in item.dtype_histogram.items():
            histogram[dtype] = histogram.get(dtype, 0) + count
    return (
        "safetensors",
        len(shard_results),
        len(seen),
        total,
        evidence,
        dict(sorted(histogram.items())),
    )


def _validate_pytorch_zip(path: Path) -> tuple[int, int]:
    reason = "NATIVE_WEIGHT_FORMAT_UNKNOWN"
    descriptor, opened = _open_nofollow(path, reason)
    try:
        duplicate = os.dup(descriptor)
        with os.fdopen(duplicate, "rb", closefd=True) as stream:
            if not zipfile.is_zipfile(stream):
                raise _issue(
                    "UNKNOWN",
                    reason,
                    "PyTorch weight is not a recognized structural ZIP container",
                )
            stream.seek(0)
            try:
                with zipfile.ZipFile(stream, "r") as archive:
                    infos = archive.infolist()
                    if not infos or len(infos) > MAX_INDEX_ENTRIES:
                        raise _issue(
                            "CORRUPT", reason, "PyTorch ZIP member count is invalid"
                        )
                    has_pickle = False
                    for info in infos:
                        parts = PurePosixPath(info.filename).parts
                        if (
                            PurePosixPath(info.filename).is_absolute()
                            or ".." in parts
                            or "\\" in info.filename
                            or info.flag_bits & 0x1
                        ):
                            raise _issue(
                                "CORRUPT",
                                reason,
                                "unsafe PyTorch ZIP member rejected",
                            )
                        if info.filename.endswith("data.pkl"):
                            has_pickle = True
                    if not has_pickle:
                        raise _issue(
                            "UNKNOWN",
                            reason,
                            "PyTorch ZIP lacks structural data.pkl evidence",
                        )
            except zipfile.BadZipFile as error:
                raise _issue("CORRUPT", reason, "invalid PyTorch ZIP structure") from error
        _revalidate_open_file(path, descriptor, opened, reason)
        return len(infos), opened.st_size
    finally:
        os.close(descriptor)


def _validate_pytorch_index(
    root: Path, index_path: Path
) -> tuple[
    str,
    int,
    int,
    int,
    tuple[dict[str, Any], ...],
    dict[str, int],
]:
    index = _load_json(
        index_path, MAX_JSON_BYTES, "NATIVE_SHARD_INDEX_INVALID"
    )
    mapping, declared_total = _weight_map(index)
    shard_names = sorted(set(mapping.values()))
    evidence: list[dict[str, Any]] = []
    total = 0
    for name in shard_names:
        path = _safe_child(root, name, "NATIVE_SHARD_INDEX_INVALID")
        if not name.endswith((".bin", ".pt", ".pth")):
            raise _issue(
                "CORRUPT",
                "NATIVE_SHARD_INDEX_INVALID",
                "PyTorch index references an unknown shard suffix",
            )
        try:
            member_count, size = _validate_pytorch_zip(path)
        except NativeIssue as error:
            if error.category == "INCOMPLETE":
                raise _issue(
                    "INCOMPLETE", "NATIVE_SHARD_MISSING", f"missing shard: {name}"
                ) from error
            raise
        total += size
        evidence.append(
            {"source": name, "zip_member_count": member_count, "file_bytes": size}
        )
    if declared_total is not None and declared_total != total:
        raise _issue(
            "CORRUPT",
            "NATIVE_SHARD_INDEX_INVALID",
            "PyTorch index total_size mismatch",
        )
    return (
        "pytorch_zip_structural_only",
        len(shard_names),
        len(mapping),
        total,
        tuple(evidence[:MAX_RETAINED_ITEMS]),
        {},
    )


def _direct_regular_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    with os.scandir(root) as entries:
        for entry in entries:
            path = Path(entry.path)
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode):
                raise _issue(
                    "CORRUPT",
                    "NATIVE_CONFIG_INVALID",
                    f"native descendant symlink rejected: {entry.name}",
                )
            if stat.S_ISREG(details.st_mode):
                paths.append(path)
            elif not stat.S_ISDIR(details.st_mode):
                raise _issue(
                    "CORRUPT",
                    "NATIVE_CONFIG_INVALID",
                    f"native special file rejected: {entry.name}",
                )
    return sorted(paths, key=lambda item: item.name)


def inspect_native(root: Path) -> NativeResult:
    """Inspect a native directory without importing code or loading tensors."""

    _safe_root(root)
    files = _direct_regular_files(root)
    by_name = {path.name: path for path in files}
    config_path = by_name.get("config.json")
    if config_path is None:
        raise _issue(
            "INCOMPLETE", "NATIVE_CONFIG_MISSING", "config.json is missing"
        )
    config = _load_json(config_path, MAX_JSON_BYTES, "NATIVE_CONFIG_INVALID")
    (
        model_type,
        architectures,
        quantization,
        declared_dtype,
        declared_remote,
    ) = _normalize_config(config)

    warnings: list[str] = []
    reason_codes = ["NATIVE_CONFIG_VALID"]
    python_sources = [path.name for path in files if path.suffix == ".py"]
    if declared_remote or python_sources:
        warnings.append("NATIVE_REMOTE_CODE_DECLARED")
        reason_codes.append("NATIVE_REMOTE_CODE_DECLARED")

    tokenizer_sources = tuple(
        name for name in TOKENIZER_FILES if name in by_name
    )
    if not tokenizer_sources:
        reason_codes.append("NATIVE_TOKENIZER_NOT_FOUND")

    safetensors = [
        path for path in files if path.name.endswith(".safetensors")
    ]
    safetensors_indexes = [
        path for path in files if path.name.endswith(".safetensors.index.json")
    ]
    pytorch = [
        path for path in files if path.suffix in {".bin", ".pt", ".pth"}
    ]
    pytorch_indexes = [
        path
        for path in files
        if path.name.endswith((".bin.index.json", ".pt.index.json", ".pth.index.json"))
    ]

    if len(safetensors_indexes) > 1 or len(pytorch_indexes) > 1:
        raise _issue(
            "CORRUPT",
            "NATIVE_SHARD_INDEX_INVALID",
            "multiple indexes for one native weight representation",
        )

    if safetensors_indexes:
        result = _validate_safetensors_index(root, safetensors_indexes[0])
        referenced = set(
            _weight_map(
                _load_json(
                    safetensors_indexes[0],
                    MAX_JSON_BYTES,
                    "NATIVE_SHARD_INDEX_INVALID",
                )
            )[0].values()
        )
        if any(path.name not in referenced for path in safetensors):
            warnings.append("NATIVE_MULTIPLE_WEIGHT_REPRESENTATIONS")
    elif len(safetensors) == 1:
        item = _parse_safetensors(safetensors[0])
        result = (
            "safetensors",
            1,
            len(item.tensors),
            item.tensor_bytes,
            (
                {
                    "source": item.name,
                    "tensor_count": len(item.tensors),
                    "tensor_bytes": item.tensor_bytes,
                },
            ),
            item.dtype_histogram,
        )
    elif len(safetensors) > 1:
        raise _issue(
            "INCOMPLETE",
            "NATIVE_SHARD_INDEX_MISSING",
            "multiple safetensors shards require an index",
        )
    elif pytorch_indexes:
        result = _validate_pytorch_index(root, pytorch_indexes[0])
    elif len(pytorch) == 1:
        members, file_size = _validate_pytorch_zip(pytorch[0])
        result = (
            "pytorch_zip_structural_only",
            1,
            0,
            file_size,
            (
                {
                    "source": pytorch[0].name,
                    "zip_member_count": members,
                    "file_bytes": file_size,
                },
            ),
            {},
        )
    elif len(pytorch) > 1:
        raise _issue(
            "INCOMPLETE",
            "NATIVE_SHARD_INDEX_MISSING",
            "multiple PyTorch shards require an index",
        )
    else:
        raise _issue(
            "INCOMPLETE", "NATIVE_WEIGHT_MISSING", "native weights are missing"
        )

    pytorch_alternative = False
    if safetensors and pytorch_indexes:
        _validate_pytorch_index(root, pytorch_indexes[0])
        pytorch_alternative = True
    elif safetensors and len(pytorch) == 1:
        try:
            _validate_pytorch_zip(pytorch[0])
        except NativeIssue as issue:
            if issue.category != "UNKNOWN":
                raise
        else:
            pytorch_alternative = True
    if pytorch_alternative:
        warnings.append("NATIVE_MULTIPLE_WEIGHT_REPRESENTATIONS")
        result = (
            "multiple_alternative_representations",
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
        )
    if result[0].startswith("pytorch") or pytorch_alternative:
        reason_codes.extend(
            ["NATIVE_PYTORCH_STRUCTURAL_ONLY", "PYTORCH_PAYLOAD_NOT_DESERIALIZED"]
        )
    reason_codes.append("NATIVE_WEIGHT_LAYOUT_VALID")
    for warning in warnings:
        if warning not in reason_codes:
            reason_codes.append(warning)
    return NativeResult(
        model_type=model_type,
        architectures=architectures,
        quantization=quantization,
        declared_dtype=declared_dtype,
        configuration_source="config.json",
        tokenizer_source=tokenizer_sources,
        weight_format=result[0],
        shard_count=result[1],
        tensor_count=result[2],
        tensor_bytes=result[3],
        dtype_histogram=result[5],
        warnings=tuple(dict.fromkeys(warnings)),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        evidence=result[4],
    )
