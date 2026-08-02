"""Physical-format classification and conservative normalized metadata."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import InspectorError
from .gguf import GGUFIssue, GGUFResult, inspect_gguf
from .native import NativeIssue, NativeResult, inspect_native


TERMINAL_CLASSES = (
    "GGUF",
    "NATIVE",
    "UNKNOWN",
    "CONTRADICTORY",
    "CORRUPT",
    "INCOMPLETE",
)

NORMALIZED_FIELDS = (
    "physical_format",
    "format_version",
    "model_type",
    "architectures",
    "task",
    "modalities",
    "quantization",
    "configuration_source",
    "tokenizer_source",
    "weight_format",
    "shard_count",
    "artifact_size",
    "artifact_identity",
    "inspection_confidence",
    "reason_codes",
)

REASON_ORDER = (
    "INSPECTION_COMPLETE",
    "ARTIFACT_CHANGED_DURING_INSPECTION",
    "ARTIFACT_READ_FAILED",
    "ARTIFACT_IDENTITY_FAILED",
    "INSPECTION_RECORD_COLLISION",
    "INSPECTION_INTERNAL_ERROR",
    "GGUF_MAGIC_CONFIRMED",
    "GGUF_STRUCTURE_VALID",
    "GGUF_VERSION_UNSUPPORTED",
    "GGUF_HEADER_TRUNCATED",
    "GGUF_METADATA_TRUNCATED",
    "GGUF_METADATA_INVALID",
    "GGUF_TENSOR_TABLE_TRUNCATED",
    "GGUF_TENSOR_DESCRIPTOR_INVALID",
    "GGUF_TENSOR_TYPE_UNKNOWN",
    "GGUF_TENSOR_RANGE_INVALID",
    "GGUF_TENSOR_DATA_TRUNCATED",
    "GGUF_REQUIRED_ARCHITECTURE_MISSING",
    "NATIVE_CONFIG_VALID",
    "NATIVE_CONFIG_MISSING",
    "NATIVE_CONFIG_INVALID",
    "NATIVE_WEIGHT_LAYOUT_VALID",
    "NATIVE_WEIGHT_MISSING",
    "NATIVE_WEIGHT_FORMAT_UNKNOWN",
    "NATIVE_SHARD_INDEX_INVALID",
    "NATIVE_SHARD_INDEX_MISSING",
    "NATIVE_SHARD_MISSING",
    "NATIVE_SHARD_DUPLICATE_TENSOR",
    "NATIVE_SAFETENSORS_HEADER_INVALID",
    "NATIVE_SAFETENSORS_RANGE_INVALID",
    "NATIVE_PYTORCH_STRUCTURAL_ONLY",
    "NATIVE_MULTIPLE_WEIGHT_REPRESENTATIONS",
    "NATIVE_TOKENIZER_NOT_FOUND",
    "NATIVE_REMOTE_CODE_DECLARED",
    "FORMAT_EVIDENCE_CONTRADICTORY",
    "FORMAT_EVIDENCE_UNKNOWN",
    "MISLEADING_EXTENSION_IGNORED",
    "PYTORCH_PAYLOAD_NOT_DESERIALIZED",
)
_REASON_RANK = {reason: index for index, reason in enumerate(REASON_ORDER)}


@dataclass(frozen=True)
class ClassificationResult:
    terminal_class: str
    normalized: dict[str, Any]
    parser_evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "terminal_class": self.terminal_class,
            "normalized": dict(self.normalized),
            "parser_evidence": dict(self.parser_evidence),
        }


def ordered_reasons(reasons: Iterable[str]) -> list[str]:
    unique = set(reasons)
    return sorted(
        unique,
        key=lambda reason: (_REASON_RANK.get(reason, len(REASON_ORDER)), reason),
    )


def _task_and_modalities(
    architectures: tuple[str, ...],
    *,
    tokenizer_evidence: bool,
) -> tuple[str, list[str]]:
    task = "unknown"
    for architecture in architectures:
        if architecture.endswith(("LlavaForConditionalGeneration", "MllamaForConditionalGeneration")):
            return "multimodal_generation", ["text", "image"]
        suffixes = (
            ("ForCausalLM", "causal_language_modeling"),
            ("ForSeq2SeqLM", "sequence_to_sequence_generation"),
            ("ForMaskedLM", "masked_language_modeling"),
            ("ForSequenceClassification", "sequence_classification"),
            ("ForTokenClassification", "token_classification"),
            ("ForQuestionAnswering", "question_answering"),
            ("ForImageClassification", "image_classification"),
            ("ForSpeechRecognition", "speech_recognition"),
        )
        for suffix, candidate in suffixes:
            if architecture.endswith(suffix):
                task = candidate
                break
        if task != "unknown":
            break
    if task == "image_classification":
        return task, ["image"]
    if task == "speech_recognition":
        return task, ["audio"]
    if task != "unknown":
        return task, ["text"]
    if tokenizer_evidence:
        return task, ["text"]
    return task, ["unknown"]


def _base_normalized(
    *,
    artifact_identity: str,
    artifact_size: int,
    reasons: Iterable[str],
    physical_format: str = "unknown",
    confidence: str = "low",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "physical_format": physical_format,
        "format_version": None,
        "model_type": None,
        "architectures": [],
        "task": "unknown",
        "modalities": ["unknown"],
        "quantization": None,
        "configuration_source": None,
        "tokenizer_source": [],
        "weight_format": None,
        "shard_count": None,
        "artifact_size": artifact_size,
        "artifact_identity": artifact_identity,
        "inspection_confidence": confidence,
        "reason_codes": ordered_reasons(reasons),
    }
    if tuple(value) != NORMALIZED_FIELDS:
        raise RuntimeError("normalized field order or membership drifted")
    return value


def _gguf_normalized(
    result: GGUFResult,
    *,
    artifact_identity: str,
    artifact_size: int,
    extra_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    architectures = (result.architecture,)
    task, modalities = _task_and_modalities(
        architectures, tokenizer_evidence=result.tokenizer_metadata_present
    )
    value = _base_normalized(
        artifact_identity=artifact_identity,
        artifact_size=artifact_size,
        physical_format="GGUF",
        confidence="high",
        reasons=(
            "INSPECTION_COMPLETE",
            "GGUF_MAGIC_CONFIRMED",
            "GGUF_STRUCTURE_VALID",
            *extra_reasons,
        ),
    )
    value.update(
        {
            "format_version": result.version,
            "model_type": result.model_type,
            "architectures": list(architectures),
            "task": task,
            "modalities": modalities,
            "quantization": {
                "kind": "gguf",
                "general_file_type": result.general_file_type,
                "quantization_version": result.quantization_version,
                "tensor_type_histogram": dict(result.tensor_type_histogram),
                "mixed_tensor_types": len(result.tensor_type_histogram) > 1,
            },
            "configuration_source": "embedded_gguf_metadata",
            "tokenizer_source": (
                ["gguf_metadata"] if result.tokenizer_metadata_present else []
            ),
            "weight_format": "gguf_tensor_table",
            "shard_count": 1,
        }
    )
    return value


def _native_normalized(
    result: NativeResult,
    *,
    artifact_identity: str,
    artifact_size: int,
) -> dict[str, Any]:
    task, modalities = _task_and_modalities(
        result.architectures,
        tokenizer_evidence=bool(result.tokenizer_source),
    )
    value = _base_normalized(
        artifact_identity=artifact_identity,
        artifact_size=artifact_size,
        physical_format="NATIVE",
        confidence="high",
        reasons=("INSPECTION_COMPLETE", *result.reason_codes),
    )
    value.update(
        {
            "format_version": (
                "safetensors.v1"
                if result.weight_format == "safetensors"
                else "multiple"
                if result.weight_format == "multiple_alternative_representations"
                else "pytorch.zip.structural"
            ),
            "model_type": result.model_type,
            "architectures": list(result.architectures),
            "task": task,
            "modalities": modalities,
            "quantization": {
                "kind": "native",
                "quantization_config": result.quantization,
                "declared_dtype": result.declared_dtype,
                "dtype_histogram": dict(result.dtype_histogram),
                "mixed_dtypes": len(result.dtype_histogram) > 1,
            },
            "configuration_source": result.configuration_source,
            "tokenizer_source": list(result.tokenizer_source),
            "weight_format": result.weight_format,
            "shard_count": result.shard_count,
        }
    )
    return value


def _tree_size(path: Path) -> int:
    details = path.lstat()
    if stat.S_ISREG(details.st_mode):
        return details.st_size
    total = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                child = Path(entry.path)
                child_details = child.lstat()
                if stat.S_ISREG(child_details.st_mode):
                    total += child_details.st_size
                elif stat.S_ISDIR(child_details.st_mode):
                    pending.append(child)
    return total


def _directory_gguf_evidence(
    root: Path,
) -> tuple[list[tuple[Path, GGUFResult]], list[tuple[Path, GGUFIssue]]]:
    valid: list[tuple[Path, GGUFResult]] = []
    invalid: list[tuple[Path, GGUFIssue]] = []
    with os.scandir(root) as entries:
        candidates = sorted(
            (Path(entry.path) for entry in entries),
            key=lambda item: item.name,
        )
    for candidate in candidates:
        if not stat.S_ISREG(candidate.lstat().st_mode):
            continue
        try:
            result = inspect_gguf(candidate)
        except GGUFIssue as issue:
            invalid.append((candidate, issue))
        if result is not None:
            valid.append((candidate, result))
    return valid, invalid


def classify_artifact(
    path: Path,
    *,
    artifact_identity: str,
    artifact_size: int | None = None,
) -> ClassificationResult:
    """Return exactly one terminal class based on physical content evidence."""

    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise InspectorError(
            "ARTIFACT_READ_FAILED", "artifact disappeared before classification"
        ) from error
    size = _tree_size(path) if artifact_size is None else artifact_size
    if stat.S_ISREG(details.st_mode):
        try:
            gguf = inspect_gguf(path)
        except GGUFIssue as issue:
            terminal = issue.category.upper()
            reasons = ["GGUF_MAGIC_CONFIRMED", issue.reason_code]
            normalized = _base_normalized(
                artifact_identity=artifact_identity,
                artifact_size=size,
                physical_format="GGUF",
                confidence="high",
                reasons=reasons,
            )
            return ClassificationResult(
                terminal, normalized, {"gguf_issue": issue.reason_code}
            )
        if gguf is not None:
            normalized = _gguf_normalized(
                gguf,
                artifact_identity=artifact_identity,
                artifact_size=size,
            )
            return ClassificationResult("GGUF", normalized, gguf.as_dict())
        reasons = ["FORMAT_EVIDENCE_UNKNOWN"]
        if path.suffix.lower() == ".gguf":
            reasons.append("MISLEADING_EXTENSION_IGNORED")
        normalized = _base_normalized(
            artifact_identity=artifact_identity,
            artifact_size=size,
            reasons=reasons,
        )
        return ClassificationResult("UNKNOWN", normalized, {})

    if not stat.S_ISDIR(details.st_mode):
        normalized = _base_normalized(
            artifact_identity=artifact_identity,
            artifact_size=size,
            reasons=("FORMAT_EVIDENCE_UNKNOWN",),
        )
        return ClassificationResult("UNKNOWN", normalized, {})

    valid_gguf, invalid_gguf = _directory_gguf_evidence(path)
    native_result: NativeResult | None = None
    native_issue: NativeIssue | None = None
    try:
        native_result = inspect_native(path)
    except NativeIssue as issue:
        native_issue = issue

    if native_result is not None and (valid_gguf or invalid_gguf):
        reasons = [
            "FORMAT_EVIDENCE_CONTRADICTORY",
            *native_result.reason_codes,
        ]
        if valid_gguf:
            reasons.extend(("GGUF_MAGIC_CONFIRMED", "GGUF_STRUCTURE_VALID"))
        reasons.extend(issue.reason_code for _candidate, issue in invalid_gguf)
        normalized = _base_normalized(
            artifact_identity=artifact_identity,
            artifact_size=size,
            physical_format="contradictory",
            confidence="high",
            reasons=reasons,
        )
        return ClassificationResult(
            "CONTRADICTORY",
            normalized,
            {
                "native": native_result.as_dict(),
                "gguf_valid_count": len(valid_gguf),
                "gguf_invalid_count": len(invalid_gguf),
            },
        )
    if native_result is not None:
        normalized = _native_normalized(
            native_result,
            artifact_identity=artifact_identity,
            artifact_size=size,
        )
        return ClassificationResult(
            "NATIVE", normalized, native_result.as_dict()
        )
    if len(valid_gguf) == 1 and not invalid_gguf:
        candidate, gguf = valid_gguf[0]
        normalized = _gguf_normalized(
            gguf,
            artifact_identity=artifact_identity,
            artifact_size=size,
        )
        normalized["shard_count"] = 1
        return ClassificationResult(
            "GGUF",
            normalized,
            {"container_member": candidate.name, **gguf.as_dict()},
        )
    if len(valid_gguf) > 1 or (valid_gguf and invalid_gguf):
        normalized = _base_normalized(
            artifact_identity=artifact_identity,
            artifact_size=size,
            physical_format="contradictory",
            confidence="high",
            reasons=("FORMAT_EVIDENCE_CONTRADICTORY",),
        )
        return ClassificationResult(
            "CONTRADICTORY",
            normalized,
            {
                "gguf_valid_count": len(valid_gguf),
                "gguf_invalid_count": len(invalid_gguf),
            },
        )
    if invalid_gguf:
        issue = invalid_gguf[0][1]
        normalized = _base_normalized(
            artifact_identity=artifact_identity,
            artifact_size=size,
            physical_format="GGUF",
            confidence="high",
            reasons=("GGUF_MAGIC_CONFIRMED", issue.reason_code),
        )
        return ClassificationResult(
            issue.category.upper(),
            normalized,
            {"gguf_issue": issue.reason_code},
        )
    assert native_issue is not None
    normalized = _base_normalized(
        artifact_identity=artifact_identity,
        artifact_size=size,
        physical_format="NATIVE",
        confidence="high",
        reasons=(native_issue.reason_code,),
    )
    return ClassificationResult(
        native_issue.category.upper(),
        normalized,
        {"native_issue": native_issue.reason_code},
    )
