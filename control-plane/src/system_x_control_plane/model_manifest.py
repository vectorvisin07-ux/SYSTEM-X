"""Strict engine-neutral model manifests."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import StrEnum
import hashlib, json
from pathlib import Path

class ArtifactFamily(StrEnum):
    GGUF = "GGUF"
    NATIVE_HF = "NATIVE_HF"

class RequiredEngine(StrEnum):
    LLAMA_CPP = "llama_cpp"
    VLLM_NATIVE = "vllm_native"

@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    immutable_model_id: str
    artifact_family: ArtifactFamily
    required_engine: RequiredEngine
    architecture: str
    model_type: str
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    tokenizer_identity: str
    chat_template_identity: str
    context_limit: int
    dtype: str
    quantization: str | None
    tensor_files: tuple[str, ...]
    source_identity: str
    source_revision: str
    file_hash_map: dict[str, str]
    total_bytes: int
    capabilities: tuple[str, ...]
    resource_estimate: dict[str, int | str]
    admission_status: str
    registration_generation: int
    created_utc: str

    def validate(self) -> None:
        if self.artifact_family == ArtifactFamily.NATIVE_HF and self.required_engine != RequiredEngine.VLLM_NATIVE:
            raise ValueError("native artifacts require vllm_native")
        if self.artifact_family == ArtifactFamily.GGUF and self.required_engine != RequiredEngine.LLAMA_CPP:
            raise ValueError("GGUF artifacts require llama_cpp")
        if self.context_limit <= 0 or self.total_bytes < 0 or not self.file_hash_map:
            raise ValueError("invalid manifest bounds")
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in self.tensor_files):
            raise ValueError("unsafe tensor path")

    def canonical_bytes(self) -> bytes:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str).encode()

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

def inspect_native(root: Path, *, model_id: str, source_revision: str, created_utc: str) -> ModelManifest:
    root = root.resolve(strict=True)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not (root / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("incomplete native file graph")
    if any(p.is_symlink() or p.stat().st_mode & 0o002 for p in files):
        raise ValueError("unsafe native file graph")
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = ModelManifest("v7", model_id, ArtifactFamily.NATIVE_HF, RequiredEngine.VLLM_NATIVE,
        "unknown", "causal-language", ("text",), ("text-generation",), "sha256:" + hashes.get("tokenizer.json", "missing"),
        "unknown", 2048, "auto", None, tuple(str(p.relative_to(root)) for p in files if p.suffix == ".safetensors"),
        "local", source_revision, hashes, sum(p.stat().st_size for p in files), ("chat", "streaming"),
        {"artifact_bytes": sum(p.stat().st_size for p in files)}, "REGISTERED", 0, created_utc)
    manifest.validate()
    return manifest
