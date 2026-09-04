from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .model_manifest import ArtifactFamily, ModelManifest, RequiredEngine

class EngineAdapter(Protocol):
    engine_id: str
    def validate_model(self, manifest: ModelManifest) -> None: ...
    def create_launch_plan(self, manifest: ModelManifest) -> dict[str, str]: ...

@dataclass(frozen=True)
class LlamaCppEngineAdapter:
    engine_id: str = "llama-cpp"
    def validate_model(self, manifest: ModelManifest) -> None:
        if manifest.artifact_family != ArtifactFamily.GGUF or manifest.required_engine != RequiredEngine.LLAMA_CPP: raise ValueError("not a GGUF manifest")
    def create_launch_plan(self, manifest: ModelManifest) -> dict[str, str]: self.validate_model(manifest); return {"engine":self.engine_id,"model":manifest.immutable_model_id}

@dataclass(frozen=True)
class VllmNativeEngineAdapter:
    engine_id: str = "vllm-native"
    def validate_model(self, manifest: ModelManifest) -> None:
        if manifest.artifact_family != ArtifactFamily.NATIVE_HF or manifest.required_engine != RequiredEngine.VLLM_NATIVE: raise ValueError("not a native manifest")
    def create_launch_plan(self, manifest: ModelManifest) -> dict[str, str]: self.validate_model(manifest); return {"engine":self.engine_id,"model":manifest.immutable_model_id,"VLLM_PLUGINS":"","trust_remote_code":"false"}
