"""Product-owned fresh-install capability authority construction."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system_x_inspector.capabilities import (  # type: ignore  # noqa: E402
    _git_commit, _identity, _manifest_basis, build_binding,
    build_capability_record, initialize_capability_store, load_binding,
    load_capability_record, publish_binding, publish_capability_record,
    verify_installed_tuple,
)
from system_x_inspector.paths import InspectorPaths  # type: ignore  # noqa: E402

DONOR_IDENTITY = "sha256:d98cff5f907e2186ac8243ac5676a47dc24e1822cd6ed674c9c92c7503915a1b"
ACCEPTED_TOKENIZER = "sha256:8602a5aeb57ffeab205af479d8d776f6d26a87ea71908599b5beef21a9e10001"
ACCEPTED_CHAT_TEMPLATE = "sha256:b0dc26e8c89e083bad45b60433cb3fff3b883ca14384cb6b678f202eefc54a7e"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _file(path: Path) -> dict[str, Any]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise RuntimeError(f"unsafe capability component: {path}")
    digest = hashlib.sha256(); count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk); count += len(chunk)
    return {"byte_count": count, "sha256": "sha256:" + digest.hexdigest()}


def _component(name: str, branch_root: Path, relative: str) -> dict[str, Any]:
    return {"name": name, "root": "branch", "path": relative, **_file(branch_root / relative)}


def _api_manifest(branch_root: Path) -> dict[str, Any]:
    files = []
    api_root = branch_root / "api_service" / "src" / "system_x_gguf_api"
    for path in sorted(api_root.glob("*.py")):
        relative = path.relative_to(branch_root).as_posix()
        files.append({"name": "api_source_" + path.stem.replace("__", "_").replace("-", "_"), "root": "branch", "path": relative, **_file(path)})
    if not files: raise RuntimeError("API source graph is empty")
    return {"name": "api_source_graph", "identity": _identity(_manifest_basis(files)), "file_count": len(files), "byte_count": sum(item["byte_count"] for item in files), "files": files}


def _installed_tuple(branch_root: Path) -> dict[str, Any]:
    source_commit = _git_commit(branch_root)
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError("vendored llama source identity is unavailable")
    components = [
        _component("api_service_controller", branch_root, "api_service_controller/controller.py"),
        _component("branch_controller", branch_root, "branch_controller/controller.py"),
        _component("llama_server_binary", branch_root, "llama.cpp/build/bin/llama-server"),
        _component("llama_source_identity", branch_root, "LLAMA_CPP_SOURCE_IDENTITY.json"),
        _component("service_control_supervisor", branch_root, "service_control/supervisor.py"),
    ]
    return {"source_commit": source_commit, "accepted_tag": "b10092", "clean_worktree_required": True, "components": components, "manifests": [_api_manifest(branch_root)], "platform_registration": {"schema_version": "system-x.platform-service-adapter-manifest.v1", "adapter_identity": "system-x.linux-systemd-user-service.v1", "registered": True, "enabled": True}}


def _record(branch_root: Path, created: str) -> dict[str, Any]:
    installed = _installed_tuple(branch_root)
    supported = {"supported_exact_artifact_identities": [DONOR_IDENTITY], "accepted_format_versions": [3], "accepted_architectures": ["qwen35"], "accepted_primary_model_types": ["model"], "accepted_modalities": ["text"], "accepted_tensor_type_evidence": ["BF16", "F32"], "accepted_tokenizer_evidence": ACCEPTED_TOKENIZER, "accepted_chat_template_evidence": ACCEPTED_CHAT_TEMPLATE, "accepted_runtime_capabilities": ["HEYCHAT-compatible", "Messages-compatible", "OpenAI-compatible", "Responses", "accepted reasoning control mode", "generate/chat", "reasoning output", "streaming", "structured output", "token count", "tool calling"], "public_model_id": "cold-install-unproven", "accepted_capability_manifest_identity": _identity(installed["manifests"])}
    return build_capability_record(created_utc=created, branch_identity="model-api-gguf", supported_physical_format="GGUF", availability="AVAILABLE", runtime_engine="llama-server", installed_tuple=installed, accepted_evidence=[], supported_evidence=supported, unsupported_primary_artifact_roles=["adapter"], unproven_valid_policy="RUNTIME_SMOKE_REQUIRED", reason_code=None)


def ensure_current_capability_authority(inspector_root: Path, branch_root: Path, user_config_root: Path) -> dict[str, Any]:
    paths = InspectorPaths.discover(inspector_root, explicit_user_config_root=user_config_root)
    initialize_capability_store(paths)
    binding_path = paths.capability_bindings / "model-api-gguf.json"
    if binding_path.exists() or binding_path.is_symlink():
        binding = load_binding(paths, "model-api-gguf")
        current = load_capability_record(paths, binding["capability_record_id"])
        verification = verify_installed_tuple(paths, current, branch_root=branch_root, user_config_root=user_config_root, binding=binding)
        if verification.get("verified") is True:
            return {"ok": True, "constructed": False, "reused": True, "capability_record_id": current["capability_record_id"], "capability_record_identity": current["capability_record_identity"], "binding_identity": binding["binding_identity"], "binding_generation": binding["binding_generation"], "installed_tuple_verified": True}
        next_generation = binding["binding_generation"] + 1
    else:
        next_generation = 1
    record = _record(branch_root, _now())
    publish_capability_record(paths, record)
    new_binding = build_binding(record, binding_generation=next_generation, updated_utc=_now())
    publish_binding(paths, new_binding)
    verification = verify_installed_tuple(paths, record, branch_root=branch_root, user_config_root=user_config_root, binding=new_binding)
    if verification.get("verified") is not True: raise RuntimeError("new capability authority failed installed-tuple verification")
    return {"ok": True, "constructed": True, "reused": False, "capability_record_id": record["capability_record_id"], "capability_record_identity": record["capability_record_identity"], "binding_identity": new_binding["binding_identity"], "binding_generation": new_binding["binding_generation"], "installed_tuple_verified": True, "verification": verification}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--inspector-root", type=Path, required=True); parser.add_argument("--branch-root", type=Path, required=True); parser.add_argument("--user-config-root", type=Path, required=True); parser.add_argument("--platform-registered", choices=("true", "false"), required=True); parser.add_argument("--platform-enabled", choices=("true", "false"), required=True); args = parser.parse_args()
    if args.platform_registered != "true" or args.platform_enabled != "true": raise SystemExit("platform service is not enabled")
    try: result = ensure_current_capability_authority(args.inspector_root.resolve(strict=True), args.branch_root.resolve(strict=True), args.user_config_root.resolve(strict=True))
    except Exception as error:
        print(json.dumps({"ok": False, "reason_code": type(error).__name__, "message": str(error)}, sort_keys=True)); return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"))); return 0


if __name__ == "__main__": raise SystemExit(main())
