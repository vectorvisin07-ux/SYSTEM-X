"""Inspector-owned bounded GGUF structural parser.

Tensor descriptors and extents are validated without reading tensor values.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import InspectorError


GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = frozenset({2, 3})
DEFAULT_ALIGNMENT = 32
MAX_METADATA_COUNT = 1_000_000
MAX_METADATA_DEPTH = 4
MAX_ARRAY_ELEMENTS = 10_000_000
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_TENSOR_COUNT = 1_000_000
MAX_TENSOR_DIMENSIONS = 4
MAX_TENSOR_NAME_BYTES = 4096
MAX_RETAINED_METADATA = 512
MAX_RETAINED_STRING_BYTES = 4096
MAX_U64 = (1 << 64) - 1

_VALUE_TYPES = {
    0: ("uint8", "B", 1),
    1: ("int8", "b", 1),
    2: ("uint16", "H", 2),
    3: ("int16", "h", 2),
    4: ("uint32", "I", 4),
    5: ("int32", "i", 4),
    6: ("float32", "f", 4),
    7: ("bool", "B", 1),
    8: ("string", None, None),
    9: ("array", None, None),
    10: ("uint64", "Q", 8),
    11: ("int64", "q", 8),
    12: ("float64", "d", 8),
}
GGUF_VALUE_TYPES = MappingProxyType(_VALUE_TYPES)

_GEOMETRY = {
    0: ("F32", 1, 4, True),
    1: ("F16", 1, 2, True),
    2: ("Q4_0", 32, 18, True),
    3: ("Q4_1", 32, 20, True),
    6: ("Q5_0", 32, 22, True),
    7: ("Q5_1", 32, 24, True),
    8: ("Q8_0", 32, 34, True),
    9: ("Q8_1", 32, 40, True),
    10: ("Q2_K", 256, 84, True),
    11: ("Q3_K", 256, 110, True),
    12: ("Q4_K", 256, 144, True),
    13: ("Q5_K", 256, 176, True),
    14: ("Q6_K", 256, 210, True),
    15: ("Q8_K", 256, 292, True),
    16: ("IQ2_XXS", 256, 66, True),
    17: ("IQ2_XS", 256, 74, True),
    18: ("IQ3_XXS", 256, 98, True),
    19: ("IQ1_S", 256, 50, True),
    20: ("IQ4_NL", 32, 18, True),
    21: ("IQ3_S", 256, 110, True),
    22: ("IQ2_S", 256, 82, True),
    23: ("IQ4_XS", 256, 136, True),
    24: ("I8", 1, 1, True),
    25: ("I16", 1, 2, True),
    26: ("I32", 1, 4, True),
    27: ("I64", 1, 8, True),
    28: ("F64", 1, 8, True),
    29: ("IQ1_M", 256, 56, True),
    30: ("BF16", 1, 2, True),
    34: ("TQ1_0", 256, 54, True),
    35: ("TQ2_0", 256, 66, True),
    39: ("MXFP4", 32, 17, True),
    40: ("NVFP4", 64, 36, True),
    41: ("Q1_0", 128, 18, True),
    42: ("Q2_0", 64, 18, True),
}
GGML_TYPE_GEOMETRY = MappingProxyType(_GEOMETRY)


def _definition_identity() -> str:
    value = {
        "magic": GGUF_MAGIC.hex(),
        "versions": sorted(SUPPORTED_VERSIONS),
        "default_alignment": DEFAULT_ALIGNMENT,
        "value_types": {
            str(key): list(item) for key, item in _VALUE_TYPES.items()
        },
        "tensor_types": {
            str(key): list(item) for key, item in _GEOMETRY.items()
        },
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


FORMAT_DEFINITION_IDENTITY = _definition_identity()


@dataclass(frozen=True)
class GGUFIssue(Exception):
    category: str
    reason_code: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class GGUFResult:
    version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    tensor_data_offset: int
    tensor_data_byte_count: int
    architecture: str
    model_type: str
    general_file_type: int | None
    quantization_version: int | None
    tensor_type_histogram: dict[str, int]
    tokenizer_metadata_present: bool
    tokenizer_token_count: int | None
    tokenizer_token_identity: str | None
    chat_template_present: bool
    chat_template_identity: str | None
    retained_metadata: tuple[dict[str, Any], ...]
    tensor_name_count: int
    tensor_name_identity: str
    format_definition_identity: str
    tensor_payload_bytes_read: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "endianness": "little",
            "tensor_count": self.tensor_count,
            "metadata_count": self.metadata_count,
            "alignment": self.alignment,
            "tensor_data_offset": self.tensor_data_offset,
            "tensor_data_byte_count": self.tensor_data_byte_count,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "general_file_type": self.general_file_type,
            "quantization_version": self.quantization_version,
            "tensor_type_histogram": dict(self.tensor_type_histogram),
            "tokenizer_metadata_present": self.tokenizer_metadata_present,
            "tokenizer_token_count": self.tokenizer_token_count,
            "tokenizer_token_identity": self.tokenizer_token_identity,
            "chat_template_present": self.chat_template_present,
            "chat_template_identity": self.chat_template_identity,
            "retained_metadata": [
                dict(item) for item in self.retained_metadata
            ],
            "tensor_name_count": self.tensor_name_count,
            "tensor_name_identity": self.tensor_name_identity,
            "format_definition_identity": self.format_definition_identity,
            "tensor_payload_bytes_read": self.tensor_payload_bytes_read,
        }


def _checked_add(left: int, right: int, issue: GGUFIssue) -> int:
    if left < 0 or right < 0 or left > MAX_U64 - right:
        raise issue
    return left + right


def _checked_multiply(left: int, right: int, issue: GGUFIssue) -> int:
    if left < 0 or right < 0 or (right and left > MAX_U64 // right):
        raise issue
    return left * right


class _Reader:
    def __init__(self, descriptor: int, size: int) -> None:
        self.descriptor = descriptor
        self.size = size
        self.offset = 0

    def read(self, amount: int, issue: GGUFIssue) -> bytes:
        end = _checked_add(self.offset, amount, issue)
        if end > self.size:
            raise issue
        try:
            value = os.pread(self.descriptor, amount, self.offset)
        except OSError as error:
            raise InspectorError(
                "ARTIFACT_READ_FAILED", "GGUF structure read failed"
            ) from error
        if len(value) != amount:
            raise issue
        self.offset = end
        return value

    def unpack(self, code: str, issue: GGUFIssue) -> Any:
        amount = struct.calcsize("<" + code)
        return struct.unpack("<" + code, self.read(amount, issue))[0]

    def skip(self, amount: int, issue: GGUFIssue) -> None:
        end = _checked_add(self.offset, amount, issue)
        if end > self.size:
            raise issue
        self.offset = end

    def hash_range(self, start: int, end: int) -> str:
        hasher = hashlib.sha256()
        cursor = start
        while cursor < end:
            amount = min(1024 * 1024, end - cursor)
            try:
                chunk = os.pread(self.descriptor, amount, cursor)
            except OSError as error:
                raise InspectorError(
                    "ARTIFACT_READ_FAILED",
                    "GGUF metadata summary read failed",
                ) from error
            if len(chunk) != amount:
                raise InspectorError(
                    "ARTIFACT_CHANGED_DURING_INSPECTION",
                    "GGUF metadata changed during summary",
                )
            hasher.update(chunk)
            cursor += amount
        return "sha256:" + hasher.hexdigest()


def _read_string(
    reader: _Reader,
    issue: GGUFIssue,
    *,
    maximum: int = MAX_STRING_BYTES,
) -> tuple[bytes, str]:
    length = reader.unpack("Q", issue)
    if length > maximum:
        raise GGUFIssue(
            "corrupt",
            "GGUF_METADATA_INVALID",
            "GGUF string length exceeds the safety bound",
        )
    raw = reader.read(length, issue)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GGUFIssue(
            "corrupt",
            "GGUF_METADATA_INVALID",
            "GGUF string is not valid UTF-8",
        ) from error
    return raw, text


def _metadata_issue(message: str) -> GGUFIssue:
    return GGUFIssue("incomplete", "GGUF_METADATA_TRUNCATED", message)


def _parse_value(
    reader: _Reader,
    type_code: int,
    *,
    depth: int,
) -> dict[str, Any]:
    if type_code not in GGUF_VALUE_TYPES:
        raise GGUFIssue(
            "corrupt",
            "GGUF_METADATA_INVALID",
            "GGUF metadata value type is unknown",
        )
    if depth > MAX_METADATA_DEPTH:
        raise GGUFIssue(
            "corrupt",
            "GGUF_METADATA_INVALID",
            "GGUF metadata nesting depth exceeded",
        )
    start = reader.offset
    type_name, code, width = GGUF_VALUE_TYPES[type_code]
    scalar: Any = None
    element_count = 1
    samples: list[Any] = []
    if type_code == 8:
        raw, text = _read_string(reader, _metadata_issue("string truncated"))
        if len(raw) <= MAX_RETAINED_STRING_BYTES:
            scalar = text
    elif type_code == 9:
        element_type = reader.unpack(
            "I", _metadata_issue("array element type truncated")
        )
        if element_type not in GGUF_VALUE_TYPES or element_type == 9:
            raise GGUFIssue(
                "corrupt",
                "GGUF_METADATA_INVALID",
                "GGUF array element type is invalid",
            )
        element_count = reader.unpack(
            "Q", _metadata_issue("array length truncated")
        )
        if element_count > MAX_ARRAY_ELEMENTS:
            raise GGUFIssue(
                "corrupt",
                "GGUF_METADATA_INVALID",
                "GGUF array length exceeds the safety bound",
            )
        if element_type == 8:
            for index in range(element_count):
                raw, text = _read_string(
                    reader, _metadata_issue("string array truncated")
                )
                if index < 3 and len(raw) <= 64:
                    samples.append(text)
        else:
            _, _, element_width = GGUF_VALUE_TYPES[element_type]
            assert element_width is not None
            total = _checked_multiply(
                element_count,
                element_width,
                _metadata_issue("array byte count overflow"),
            )
            if element_type == 7:
                remaining = total
                while remaining:
                    amount = min(1024 * 1024, remaining)
                    values = reader.read(
                        amount, _metadata_issue("boolean array truncated")
                    )
                    if any(value not in (0, 1) for value in values):
                        raise GGUFIssue(
                            "corrupt",
                            "GGUF_METADATA_INVALID",
                            "GGUF boolean array contains a non-boolean value",
                        )
                    remaining -= amount
            else:
                reader.skip(total, _metadata_issue("array truncated"))
        type_name = f"array<{GGUF_VALUE_TYPES[element_type][0]}>"
    else:
        assert code is not None and width is not None
        raw = reader.read(width, _metadata_issue("scalar truncated"))
        scalar = struct.unpack("<" + code, raw)[0]
        if type_code == 7:
            if scalar not in (0, 1):
                raise GGUFIssue(
                    "corrupt",
                    "GGUF_METADATA_INVALID",
                    "GGUF boolean is not physically 0 or 1",
                )
            scalar = bool(scalar)
    end = reader.offset
    return {
        "value_type": type_name,
        "element_count": element_count,
        "encoded_byte_count": end - start,
        "value_sha256": reader.hash_range(start, end),
        "scalar": scalar,
        "samples": samples,
    }


def _stat_identity(details: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _revalidate(path: Path, descriptor: int, opened: os.stat_result) -> None:
    current_descriptor = os.fstat(descriptor)
    try:
        current_path = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "ARTIFACT_CHANGED_DURING_INSPECTION",
            "GGUF path disappeared during inspection",
        ) from error
    if (
        stat.S_ISLNK(current_path.st_mode)
        or _stat_identity(current_descriptor) != _stat_identity(opened)
        or _stat_identity(current_path) != _stat_identity(opened)
    ):
        raise InspectorError(
            "ARTIFACT_CHANGED_DURING_INSPECTION",
            "GGUF source identity changed during inspection",
        )


def _parse(reader: _Reader) -> GGUFResult | None:
    if reader.size < 4:
        return None
    magic = reader.read(
        4,
        GGUFIssue("unknown", "FORMAT_EVIDENCE_UNKNOWN", "magic unavailable"),
    )
    if magic != GGUF_MAGIC:
        return None
    header_issue = GGUFIssue(
        "incomplete", "GGUF_HEADER_TRUNCATED", "GGUF header is truncated"
    )
    version = reader.unpack("I", header_issue)
    if version not in SUPPORTED_VERSIONS:
        raise GGUFIssue(
            "unknown",
            "GGUF_VERSION_UNSUPPORTED",
            "GGUF structural version is unsupported",
        )
    tensor_count = reader.unpack("Q", header_issue)
    metadata_count = reader.unpack("Q", header_issue)
    if tensor_count > MAX_TENSOR_COUNT or metadata_count > MAX_METADATA_COUNT:
        raise GGUFIssue(
            "corrupt",
            "GGUF_METADATA_INVALID",
            "GGUF header count exceeds a safety bound",
        )

    metadata: dict[str, dict[str, Any]] = {}
    retained: list[dict[str, Any]] = []
    tokenizer_present = False
    token_count: int | None = None
    token_identity: str | None = None
    chat_template_present = False
    chat_template_identity: str | None = None
    key_pattern = re.compile(r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
    for _ in range(metadata_count):
        key_raw, key = _read_string(
            reader,
            _metadata_issue("metadata key truncated"),
            maximum=4096,
        )
        if (
            not key_raw
            or not key_raw.isascii()
            or key_pattern.fullmatch(key) is None
            or key in metadata
        ):
            raise GGUFIssue(
                "corrupt",
                "GGUF_METADATA_INVALID",
                "GGUF metadata key is invalid or duplicated",
            )
        type_code = reader.unpack(
            "I", _metadata_issue("metadata type truncated")
        )
        summary = _parse_value(reader, type_code, depth=0)
        metadata[key] = summary
        if len(retained) < MAX_RETAINED_METADATA and (
            key.startswith("general.")
            or key.startswith("tokenizer.")
            or key.endswith(".model")
        ):
            retained.append(
                {
                    "key": key,
                    "value_type": summary["value_type"],
                    "element_count": summary["element_count"],
                    "encoded_byte_count": summary["encoded_byte_count"],
                    "value_sha256": summary["value_sha256"],
                    "scalar": summary["scalar"],
                    "samples": summary["samples"],
                }
            )
        if key.startswith("tokenizer."):
            tokenizer_present = True
        if key in {"tokenizer.ggml.tokens", "tokenizer.tokens"}:
            token_count = summary["element_count"]
            token_identity = summary["value_sha256"]
        if key in {"tokenizer.chat_template", "tokenizer.ggml.chat_template"}:
            chat_template_present = True
            chat_template_identity = summary["value_sha256"]

    architecture_value = metadata.get("general.architecture", {}).get("scalar")
    if not isinstance(architecture_value, str) or not architecture_value:
        raise GGUFIssue(
            "corrupt",
            "GGUF_REQUIRED_ARCHITECTURE_MISSING",
            "GGUF general.architecture is missing or invalid",
        )
    alignment_value = metadata.get("general.alignment")
    alignment = DEFAULT_ALIGNMENT
    if alignment_value is not None:
        scalar = alignment_value.get("scalar")
        if (
            alignment_value.get("value_type") != "uint32"
            or not isinstance(scalar, int)
            or scalar <= 0
            or scalar & (scalar - 1)
        ):
            raise GGUFIssue(
                "corrupt",
                "GGUF_METADATA_INVALID",
                "GGUF general.alignment is invalid",
            )
        alignment = scalar
    file_type = metadata.get("general.file_type", {}).get("scalar")
    if not isinstance(file_type, int):
        file_type = None
    quantization_version = metadata.get(
        "general.quantization_version", {}
    ).get("scalar")
    if not isinstance(quantization_version, int):
        quantization_version = None
    explicit_model_type = metadata.get("general.type", {}).get("scalar")
    model_type = (
        explicit_model_type
        if isinstance(explicit_model_type, str) and explicit_model_type
        else architecture_value
    )

    tensor_issue = GGUFIssue(
        "incomplete",
        "GGUF_TENSOR_TABLE_TRUNCATED",
        "GGUF tensor table is truncated",
    )
    tensor_names: set[str] = set()
    tensor_name_hasher = hashlib.sha256()
    histogram: dict[str, int] = {}
    ranges: list[tuple[int, int, str]] = []
    for _ in range(tensor_count):
        _, name = _read_string(
            reader, tensor_issue, maximum=MAX_TENSOR_NAME_BYTES
        )
        if not name or name in tensor_names:
            raise GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_DESCRIPTOR_INVALID",
                "GGUF tensor name is empty or duplicated",
            )
        tensor_names.add(name)
        encoded_name = name.encode("utf-8")
        tensor_name_hasher.update(struct.pack("<Q", len(encoded_name)))
        tensor_name_hasher.update(encoded_name)
        dimensions_count = reader.unpack("I", tensor_issue)
        if not 1 <= dimensions_count <= MAX_TENSOR_DIMENSIONS:
            raise GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_DESCRIPTOR_INVALID",
                "GGUF tensor dimension count is invalid",
            )
        dimensions = [reader.unpack("Q", tensor_issue) for _ in range(dimensions_count)]
        if any(value <= 0 for value in dimensions):
            raise GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_DESCRIPTOR_INVALID",
                "GGUF tensor dimension is not positive",
            )
        type_code = reader.unpack("I", tensor_issue)
        geometry = GGML_TYPE_GEOMETRY.get(type_code)
        if geometry is None or not geometry[3]:
            raise GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_TYPE_UNKNOWN",
                "GGUF tensor type is unknown or not file-permitted",
            )
        type_name, block_elements, block_bytes, _ = geometry
        if dimensions[0] % block_elements:
            raise GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_DESCRIPTOR_INVALID",
                "GGUF tensor row violates block divisibility",
            )
        elements = 1
        overflow_issue = GGUFIssue(
            "corrupt",
            "GGUF_TENSOR_DESCRIPTOR_INVALID",
            "GGUF tensor element count overflows",
        )
        for dimension in dimensions:
            elements = _checked_multiply(elements, dimension, overflow_issue)
        byte_count = _checked_multiply(
            elements // block_elements, block_bytes, overflow_issue
        )
        relative_offset = reader.unpack("Q", tensor_issue)
        if relative_offset % alignment:
            raise GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_RANGE_INVALID",
                "GGUF tensor offset is not aligned",
            )
        ranges.append((relative_offset, byte_count, name))
        histogram[type_name] = histogram.get(type_name, 0) + 1

    data_offset = (
        (reader.offset + alignment - 1) // alignment * alignment
        if tensor_count
        else reader.offset
    )
    if data_offset > reader.size:
        raise GGUFIssue(
            "incomplete",
            "GGUF_TENSOR_DATA_TRUNCATED",
            "GGUF tensor data offset is outside the file",
        )
    previous_end = data_offset
    maximum_end = data_offset
    for relative_offset, byte_count, name in sorted(ranges):
        absolute_start = _checked_add(
            data_offset,
            relative_offset,
            GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_RANGE_INVALID",
                "GGUF tensor offset overflows",
            ),
        )
        absolute_end = _checked_add(
            absolute_start,
            byte_count,
            GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_RANGE_INVALID",
                "GGUF tensor extent overflows",
            ),
        )
        if absolute_start < previous_end:
            raise GGUFIssue(
                "corrupt",
                "GGUF_TENSOR_RANGE_INVALID",
                f"GGUF tensor range overlaps at {name}",
            )
        if absolute_end > reader.size:
            raise GGUFIssue(
                "incomplete",
                "GGUF_TENSOR_DATA_TRUNCATED",
                "GGUF tensor data ends beyond the file",
            )
        previous_end = absolute_end
        maximum_end = max(maximum_end, absolute_end)

    return GGUFResult(
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        alignment=alignment,
        tensor_data_offset=data_offset,
        tensor_data_byte_count=maximum_end - data_offset,
        architecture=architecture_value,
        model_type=model_type,
        general_file_type=file_type,
        quantization_version=quantization_version,
        tensor_type_histogram=dict(sorted(histogram.items())),
        tokenizer_metadata_present=tokenizer_present,
        tokenizer_token_count=token_count,
        tokenizer_token_identity=token_identity,
        chat_template_present=chat_template_present,
        chat_template_identity=chat_template_identity,
        retained_metadata=tuple(retained),
        tensor_name_count=len(tensor_names),
        tensor_name_identity="sha256:" + tensor_name_hasher.hexdigest(),
        format_definition_identity=FORMAT_DEFINITION_IDENTITY,
        tensor_payload_bytes_read=0,
    )


def inspect_gguf(path: Path) -> GGUFResult | None:
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "GGUF candidate does not exist"
        ) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise InspectorError(
            "ARTIFACT_READ_FAILED",
            "GGUF candidate must be a regular non-symlink file",
        )
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as error:
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "GGUF candidate open failed"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(opened):
            raise InspectorError(
                "ARTIFACT_CHANGED_DURING_INSPECTION",
                "GGUF path and descriptor identity differ",
            )
        result = _parse(_Reader(descriptor, opened.st_size))
        _revalidate(path, descriptor, opened)
        return result
    except GGUFIssue:
        _revalidate(path, descriptor, opened)
        raise
    finally:
        os.close(descriptor)
