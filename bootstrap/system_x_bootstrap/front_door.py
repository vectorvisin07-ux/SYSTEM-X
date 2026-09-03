"""Small, secret-safe public command surface for System X."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = "system-x.front-door-result.v1"
PUBLIC_OPERATIONS = ("install", "status", "connection", "doctor", "help", "chat", "open", "verify-code")
_PUBLIC_OPERATION_SET = frozenset(PUBLIC_OPERATIONS)


@dataclass(frozen=True)
class ChildRun:
    argv: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    payload: Mapping[str, Any] | None
    output_valid: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _repository_root() -> Path:
    module = Path(__file__).resolve(strict=True)
    root = module.parents[2]
    markers = (
        root / "SYSTEM_X_REPOSITORY_MANIFEST.json",
        root / "bootstrap" / "run_bootstrap.py",
        root / "INSPECTOR" / "run_inspector.py",
    )
    if not all(path.is_file() and not path.is_symlink() for path in markers):
        raise RuntimeError("System X repository root is invalid")
    return root


def _clean_environment() -> dict[str, str]:
    """Build the narrow environment needed by product child commands."""

    identity = pwd.getpwuid(os.getuid())
    values = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "USER": identity.pw_name,
        "LOGNAME": identity.pw_name,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        value = os.environ.get(name)
        if value:
            values[name] = value
    return values


def _decode_child_output(raw: bytes) -> tuple[Mapping[str, Any] | None, bool]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, False
    return (value, True) if isinstance(value, dict) else (None, False)


def _run_child(argv: Sequence[str], *, cwd: Path, timeout: float) -> ChildRun:
    command = tuple(str(item) for item in argv)
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=_clean_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ChildRun(command, None, True, None, False)
    except (OSError, ValueError):
        return ChildRun(command, None, False, None, False)
    payload, valid = _decode_child_output(completed.stdout)
    return ChildRun(command, completed.returncode, False, payload, valid)


def _safe_identity(run: ChildRun, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = payload if payload is not None else run.payload
    result: dict[str, Any] = {
        "exit_status": run.returncode,
        "output_valid": bool(value is not None and run.output_valid),
        "timeout": run.timed_out,
    }
    if isinstance(value, Mapping):
        for key in (
            "operation", "status", "state", "ok", "reason_code", "receipt_id",
            "transaction_id", "result_class", "schema", "schema_version",
        ):
            item = value.get(key)
            if isinstance(item, (str, int, bool)) or item is None:
                result[key] = item
    return result


def _child_identities(runs: Sequence[ChildRun]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for run in runs:
        identities.append(_safe_identity(run))
        payload = run.payload
        if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
            for item in payload["results"]:
                if isinstance(item, Mapping):
                    identities.append(_safe_identity(run, item))
    return identities[:32]


def _result(
    operation: str,
    *,
    ok: bool,
    reason_code: str,
    message: str,
    installation_state: str,
    service_state: str,
    readiness_state: str,
    model_state: str,
    connection_state: str,
    recommended_model: str | None,
    child_result_identities: Sequence[Mapping[str, Any]],
    connection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "ok": bool(ok),
        "reason_code": str(reason_code),
        "message": str(message)[:512],
        "installation_state": installation_state,
        "service_state": service_state,
        "readiness_state": readiness_state,
        "model_state": model_state,
        "connection_state": connection_state,
        "recommended_model": recommended_model,
        "child_result_identities": [dict(item) for item in child_result_identities],
        "timestamp_utc": _utc_now(),
    }
    if connection is not None:
        result["connection"] = dict(connection)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            return {}
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _collect_observations(value: object, result: dict[str, Any]) -> None:
    keys = {
        "active", "enabled", "registered", "unit_present", "adapter_manifest_present",
        "operating_profile_present", "desired_state_present", "ready", "inference_ready",
        "warm", "warm_model_present", "service_operational", "supervisor_state",
        "public_origin",
        "service_readiness_state", "model_service_state", "desired_state", "lifecycle_state",
        "state", "result_class", "model_state", "default_alias", "recommended_reference",
        "resolved_immutable_model_id", "resolved_model_id", "health_state", "ready_model_count",
        "current_model_count", "model_count", "registered_model_count", "http_status",
        "service_available", "receipt_id", "reason_code", "schema_version",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and (isinstance(item, (str, int, float, bool)) or item is None):
                result.setdefault(str(key), item)
            if isinstance(item, (Mapping, list)):
                _collect_observations(item, result)
    elif isinstance(value, list):
        for item in value[:64]:
            _collect_observations(item, result)


def _runtime_observations(root: Path) -> dict[str, Any]:
    relative_paths = (
        "model-api-gguf/RUNTIME/service_control/status/supervisor.json",
        "model-api-gguf/RUNTIME/api/status/service.json",
        "INSPECTOR/RUNTIME/status/current.json",
        "INSPECTOR/RUNTIME/status/api-connection.json",
    )
    observations: dict[str, Any] = {}
    for relative in relative_paths:
        _collect_observations(_load_json(root / relative), observations)
    return observations


def _contained(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or "\\" in relative or any(part in ("", ".", "..") for part in candidate.parts):
        return None
    return root.joinpath(*candidate.parts)


def _service_coordinates(root: Path) -> tuple[Path, Path, str, Path] | None:
    config = _load_json(root / "bootstrap" / "configuration" / "service-registration.json")
    adapter = config.get("adapter")
    if not isinstance(adapter, Mapping):
        return None
    source = _contained(root, adapter.get("source_entrypoint"))
    runtime = _contained(root, adapter.get("runtime_root"))
    service_name = adapter.get("service_name")
    unit_relative = config.get("future_generated_unit_relative_to_home")
    if source is None or runtime is None or not isinstance(service_name, str) or not service_name:
        return None
    if not isinstance(unit_relative, str) or unit_relative.startswith("/"):
        return None
    unit_path = Path.home().joinpath(*Path(unit_relative).parts)
    return source, runtime, service_name, unit_path


def _bootstrap_argv(root: Path, operation: str) -> tuple[str, ...]:
    return (sys.executable, "-I", "-S", "-B", str(root / "bootstrap" / "run_bootstrap.py"), operation)


def _inspector_argv(root: Path, operation: str) -> tuple[str, ...]:
    return (sys.executable, "-I", "-S", "-B", str(root / "INSPECTOR" / "run_inspector.py"), operation)


def _adapter_status_argv(root: Path) -> tuple[str, ...] | None:
    coordinates = _service_coordinates(root)
    if coordinates is None:
        return None
    source, runtime, service_name, unit_path = coordinates
    return (
        sys.executable, "-I", "-S", "-B", str(source),
        "--adapter-runtime-root", str(runtime),
        "--service-name", service_name,
        "--unit-path", str(unit_path), "status",
    )


def _child_success(run: ChildRun) -> bool:
    if run.returncode != 0 or run.timed_out or not isinstance(run.payload, Mapping):
        return False
    results = run.payload.get("results")
    if isinstance(results, list):
        return all(isinstance(item, Mapping) and item.get("status") == "ok" for item in results)
    if "ok" in run.payload:
        return run.payload.get("ok") is True
    return run.payload.get("status", "ok") == "ok"


def _state_from_payload(payload: Mapping[str, Any] | None) -> str | None:
    observations: dict[str, Any] = {}
    if payload is not None:
        _collect_observations(payload, observations)
    for key in ("state", "supervisor_state", "service_readiness_state"):
        value = observations.get(key)
        if isinstance(value, str):
            return value
    return None


def _default_alias(root: Path, observations: Mapping[str, Any] | None = None) -> str | None:
    if observations is not None:
        for key in ("default_alias", "recommended_reference"):
            value = observations.get(key)
            if isinstance(value, str) and value:
                return value
    config = _load_json(root / "bootstrap" / "configuration" / "service-registration.json")
    profile = config.get("operating_profile")
    template = profile.get("template") if isinstance(profile, Mapping) else None
    value = template.get("default_model_alias") if isinstance(template, Mapping) else None
    return value if isinstance(value, str) and value else None


def _activation_completed(durable: Mapping[str, Any]) -> bool:
    completed = durable.get("completed_operations")
    return isinstance(completed, list) and all(isinstance(item, str) and item for item in completed) and "activate-platform-service" in completed


def _installation_state(
    root: Path,
    observations: Mapping[str, Any] | None = None,
    *,
    durable: Mapping[str, Any] | None = None,
) -> str:
    document = durable if durable is not None else _load_json(root / ".system-x-bootstrap-state" / "status.json")
    state = _state_from_payload(document)
    if state in (None, "CLONED") and not observations:
        return "SOURCE_ONLY"
    if state in {"FAILED_CLEAN", "FAIL_CLOSED"}:
        return "REPAIR_REQUIRED"
    if state == "SERVICE_REGISTERED" and _activation_completed(document):
        return "INSTALLED"
    if state in {"HOST_INSPECTED", "HOST_PLAN_READY", "HOST_READY", "SUBMODULES_READY", "PYTHON_ENVIRONMENTS_READY", "LLAMA_SERVER_BUILT", "RUNTIME_INITIALIZED", "CREDENTIAL_READY", "SERVICE_REGISTERED"}:
        return "INSTALLING"
    return "INSTALLED"


def _connection_view(data: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {"result_class": "CONNECTION_RECORD_INVALID", "reason_code": "CONNECTION_RECORD_INVALID"}
    result_class = data.get("result_class")
    reason_code = data.get("reason_code")
    view: dict[str, Any] = {
        "result_class": result_class if isinstance(result_class, str) else "CONNECTION_RECORD_INVALID",
        "reason_code": reason_code if isinstance(reason_code, str) else "CONNECTION_RECORD_INVALID",
    }
    receipt = data.get("receipt")
    if not isinstance(receipt, Mapping):
        return view
    for key in ("receipt_id", "receipt_identity", "generated_utc", "receipt_source", "deployment_id", "deployment_result_identity"):
        value = receipt.get(key)
        if isinstance(value, str) or value is None:
            view[key] = value
    field_groups = {
        "service": ("public_origin", "service_available", "inference_ready", "service_readiness", "model_service_state", "desired_state", "always_on", "authentication_required"),
        "model": ("recommended_reference", "default_alias", "resolved_immutable_model_id", "source_label", "physical_architecture", "physical_model_type", "official_checkpoint_attested", "artifact_sha256", "artifact_version_id", "capability_manifest_identity", "model_state", "warm", "context_window_tokens", "maximum_output_tokens"),
        "authentication": ("required", "accepted_schemes", "non_secret_key_id", "raw_api_key_returned"),
        "capabilities": ("protocol_families", "streaming", "token_counting", "reasoning_output", "reasoning_control", "tool_calling", "structured_output", "context_window_tokens"),
        "proof": ("health_http_status", "model_list_http_status", "model_detail_http_status", "proof_request_id", "proof_request_http_status", "response_model_matches", "artifact_version_matches", "final_content_nonempty", "operation_record_correlated", "openai_model_list_http_status", "openai_model_list_contains_recommended_model", "openai_model_list_contains_resolved_model", "messages_model_list_http_status", "messages_model_list_contains_recommended_model", "messages_model_list_contains_resolved_model", "messages_token_count_http_status", "messages_token_count_result_valid"),
        "lifecycle": ("promotion_result", "rollback_result", "retirement_result", "service_left_running", "service_left_ready", "current_receipt_updated"),
    }
    for group, fields in field_groups.items():
        value = receipt.get(group)
        if not isinstance(value, Mapping):
            continue
        safe: dict[str, Any] = {}
        for key in fields:
            item = value.get(key)
            if isinstance(item, (str, int, float, bool)) or item is None:
                safe[key] = item
            elif key in {"accepted_schemes", "protocol_families"} and isinstance(item, list) and all(isinstance(child, str) for child in item):
                safe[key] = list(item)
        view[group] = safe
    connections = receipt.get("connections")
    if isinstance(connections, Mapping):
        safe_connections: dict[str, Any] = {}
        for family in ("system_x_native", "openai_compatible", "messages_compatible"):
            value = connections.get(family)
            if not isinstance(value, Mapping):
                continue
            safe: dict[str, Any] = {}
            for key in ("protocol_family", "base_url", "endpoint_semantics", "model_reference", "compatibility_version"):
                item = value.get(key)
                if isinstance(item, str) or item is None:
                    safe[key] = item
            item = value.get("authentication")
            if isinstance(item, list) and all(isinstance(child, str) for child in item):
                safe["authentication"] = list(item)
            for key in ("endpoints", "required_headers"):
                item = value.get(key)
                if isinstance(item, Mapping) and all(isinstance(child, str) for child in item.values()):
                    safe[key] = {str(name): child for name, child in item.items()}
            safe_connections[family] = safe
        view["connections"] = safe_connections
    return view


def _install(root: Path) -> dict[str, Any]:
    run = _run_child(_bootstrap_argv(root, "reconstruct") + ("--authorize",), cwd=root, timeout=3600)
    identities = _child_identities((run,))
    state = _state_from_payload(run.payload)
    if _child_success(run):
        if state == "READY":
            return _result("install", ok=True, reason_code="READY", message="System X installation is READY", installation_state="INSTALLED", service_state="RUNNING", readiness_state="READY", model_state="READY", connection_state="READY", recommended_model=None, child_result_identities=identities)
        return _result("install", ok=True, reason_code="WAITING_FOR_MODEL", message="System X installation is complete; add one stable GGUF and wait for READY", installation_state="INSTALLED", service_state="RUNNING", readiness_state="WAITING_FOR_MODEL", model_state="ABSENT", connection_state="NOT_READY", recommended_model=None, child_result_identities=identities)
    if run.timed_out:
        return _result("install", ok=False, reason_code="INSTALL_TIMEOUT", message="System X installation is still in progress or requires inspection", installation_state="INSTALLING", service_state="UNKNOWN", readiness_state="DEGRADED", model_state="UNKNOWN", connection_state="NOT_READY", recommended_model=None, child_result_identities=identities)
    return _result("install", ok=False, reason_code="INSTALL_FAILED", message="System X installation requires repair", installation_state="REPAIR_REQUIRED", service_state="UNKNOWN", readiness_state="DEGRADED", model_state="UNKNOWN", connection_state="NOT_READY", recommended_model=None, child_result_identities=identities)


def _status(root: Path) -> dict[str, Any]:
    runs: list[ChildRun] = []
    bootstrap = _run_child(_bootstrap_argv(root, "status"), cwd=root, timeout=120)
    runs.append(bootstrap)
    adapter_argv = _adapter_status_argv(root)
    if adapter_argv is not None:
        runs.append(_run_child(adapter_argv, cwd=root, timeout=60))
    inspector = _run_child(_inspector_argv(root, "status"), cwd=root / "INSPECTOR", timeout=60)
    runs.append(inspector)
    connection_run: ChildRun | None = None
    connection_path = root / "INSPECTOR" / "RUNTIME" / "status" / "api-connection.json"
    if connection_path.is_file() and not connection_path.is_symlink():
        connection_run = _run_child(_inspector_argv(root, "show-connection"), cwd=root / "INSPECTOR", timeout=60)
        runs.append(connection_run)
    observations = _runtime_observations(root)
    durable = _load_json(root / ".system-x-bootstrap-state" / "status.json")
    durable_state = _state_from_payload(durable) or _state_from_payload(bootstrap.payload)
    if durable_state in {"FAILED_CLEAN", "FAIL_CLOSED"} or (bootstrap.returncode not in (0, None) and not bootstrap.timed_out):
        installation = "REPAIR_REQUIRED"
    elif durable_state in (None, "CLONED") and not observations and connection_run is None:
        installation = "SOURCE_ONLY"
    else:
        installation_document: Mapping[str, Any] = durable
        if _state_from_payload(durable) is None and isinstance(bootstrap.payload, Mapping):
            installation_document = bootstrap.payload
        installation = _installation_state(root, observations, durable=installation_document)
    active = observations.get("active") is True or observations.get("supervisor_state") == "RUNNING"
    service = "RUNNING" if active else "STOPPED"
    if installation == "SOURCE_ONLY":
        service = "STOPPED"
    if installation == "REPAIR_REQUIRED":
        service = "DEGRADED"
    connection_class = None
    connection_view: dict[str, Any] | None = None
    if connection_run is not None and isinstance(connection_run.payload, Mapping):
        data = connection_run.payload.get("data")
        if isinstance(data, Mapping):
            connection_class = data.get("result_class")
            connection_view = _connection_view(data)
    if connection_class == "CONNECTION_READY":
        readiness, connection_state = "READY", "READY"
    elif connection_class == "CONNECTION_STALE":
        readiness, connection_state = "DEGRADED", "STALE"
    elif service == "STOPPED" and installation == "INSTALLED":
        readiness, connection_state = "STOPPED", "NOT_READY"
    else:
        service_readiness = observations.get("service_readiness_state") or observations.get("model_service_state")
        if service_readiness in {"READY", "WAITING_FOR_MODEL", "DEGRADED", "STOPPED"}:
            readiness = str(service_readiness)
        elif installation == "SOURCE_ONLY":
            readiness = "WAITING_FOR_MODEL"
        elif installation == "INSTALLING":
            readiness = "INSTALLING"
        else:
            readiness = "DEGRADED"
        connection_state = "NOT_READY"
    if readiness == "READY" or observations.get("health_state") == "ready" or observations.get("warm_model_present") is True:
        model = "READY"
    elif readiness == "WAITING_FOR_MODEL" or observations.get("ready_model_count") == 0:
        model = "ABSENT"
    elif observations.get("model_service_state") in {"LOADING", "PROBING"}:
        model = "LOADING"
    else:
        model = "UNKNOWN"
    if installation == "SOURCE_ONLY":
        model = "ABSENT"
    recommended = _default_alias(root, observations) if installation != "SOURCE_ONLY" else None
    reason = {"SOURCE_ONLY": "SOURCE_ONLY", "INSTALLING": "INSTALLING", "REPAIR_REQUIRED": "REPAIR_REQUIRED"}.get(installation)
    if reason is None:
        reason = {"READY": "READY", "WAITING_FOR_MODEL": "WAITING_FOR_MODEL", "STOPPED": "STOPPED", "DEGRADED": "DEGRADED"}.get(readiness, "STATUS_OBSERVED")
    ok = installation != "REPAIR_REQUIRED" and readiness != "DEGRADED"
    return _result("status", ok=ok, reason_code=reason, message="System X status observed", installation_state=installation, service_state=service, readiness_state=readiness, model_state=model, connection_state=connection_state, recommended_model=recommended, child_result_identities=_child_identities(runs), connection=connection_view)


def _connection(root: Path) -> dict[str, Any]:
    run = _run_child(_inspector_argv(root, "show-connection"), cwd=root / "INSPECTOR", timeout=120)
    identities = _child_identities((run,))
    data = run.payload.get("data") if isinstance(run.payload, Mapping) else None
    view = _connection_view(data if isinstance(data, Mapping) else None)
    ready = run.returncode == 0 and view.get("result_class") == "CONNECTION_READY"
    installation = _installation_state(root)
    if ready and installation == "INSTALLED":
        model = view.get("model")
        recommended = model.get("recommended_reference") if isinstance(model, Mapping) else None
        return _result("connection", ok=True, reason_code="CONNECTION_READY", message="System X API connection is READY; use the default model", installation_state="INSTALLED", service_state="RUNNING", readiness_state="READY", model_state="READY", connection_state="READY", recommended_model=recommended if isinstance(recommended, str) else None, child_result_identities=identities, connection=view)
    if ready:
        view = dict(view)
        view["result_class"] = "CONNECTION_NOT_READY"
        view["reason_code"] = "CONNECTION_NOT_READY"
        if installation == "SOURCE_ONLY":
            service, readiness, model = "STOPPED", "WAITING_FOR_MODEL", "ABSENT"
        elif installation == "INSTALLING":
            service, readiness, model = "RUNNING", "INSTALLING", "UNKNOWN"
        else:
            service, readiness, model = "DEGRADED", "DEGRADED", "UNKNOWN"
        return _result("connection", ok=False, reason_code="CONNECTION_NOT_READY", message="System X API connection is not ready", installation_state=installation, service_state=service, readiness_state=readiness, model_state=model, connection_state="NOT_READY", recommended_model=None, child_result_identities=identities, connection=view)
    result_class = view.get("result_class")
    if result_class == "CONNECTION_STALE":
        reason, readiness, state, model = "CONNECTION_STALE", "DEGRADED", "STALE", "UNKNOWN"
    elif run.timed_out:
        reason, readiness, state, model = "CONNECTION_PROBE_TIMEOUT", "DEGRADED", "UNKNOWN", "UNKNOWN"
    else:
        reason, readiness, state, model = str(view.get("reason_code") or "CONNECTION_NOT_READY"), "WAITING_FOR_MODEL", "NOT_READY", "ABSENT"
    return _result("connection", ok=False, reason_code=reason, message="System X API connection is not ready", installation_state=installation, service_state="STOPPED" if installation == "SOURCE_ONLY" else "RUNNING", readiness_state=readiness, model_state=model, connection_state=state, recommended_model=None, child_result_identities=identities, connection=view)


def _doctor(root: Path) -> dict[str, Any]:
    runs = [
        _run_child(_bootstrap_argv(root, "identify"), cwd=root, timeout=60),
        _run_child(_bootstrap_argv(root, "inspect-host"), cwd=root, timeout=120),
        _run_child(_bootstrap_argv(root, "plan"), cwd=root, timeout=120),
    ]
    ok = all(_child_success(run) for run in runs)
    state = _state_from_payload(runs[0].payload)
    installed = state not in (None, "CLONED")
    return _result("doctor", ok=ok, reason_code="DOCTOR_OK" if ok else "DOCTOR_REQUIRES_ATTENTION", message="Read-only System X diagnostics completed" if ok else "Read-only System X diagnostics found an issue", installation_state="INSTALLED" if installed else "SOURCE_ONLY", service_state="STOPPED", readiness_state="WAITING_FOR_MODEL", model_state="ABSENT", connection_state="NOT_READY", recommended_model=None, child_result_identities=_child_identities(runs))


def _help() -> dict[str, Any]:
    return _result("help", ok=True, reason_code="HELP", message="Clone once, run system-x install, wait for READY, then use system-x chat or system-x verify-code for local source verification. Use system-x connection for API details. OpenClaw is not required.", installation_state="SOURCE_ONLY", service_state="STOPPED", readiness_state="WAITING_FOR_MODEL", model_state="ABSENT", connection_state="NOT_READY", recommended_model=None, child_result_identities=())


def _open(root: Path) -> dict[str, Any]:
    """Return the canonical same-origin Studio entry without exposing secrets."""
    observed = _runtime_observations(root)
    origin = observed.get("public_origin")
    if not isinstance(origin, str) or not origin.startswith(("http://", "https://")):
        origin = "http://127.0.0.1:8080"
    return _result("open", ok=True, reason_code="STUDIO_ENTRY", message="System X Studio entry is available at the canonical local origin", installation_state="INSTALLED", service_state="RUNNING", readiness_state="READY", model_state="READY", connection_state="READY", recommended_model=_default_alias(root, observed), child_result_identities=(), connection={"origin": origin, "path": "/ui/chat/", "same_origin": True})


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1 or arguments[0] not in _PUBLIC_OPERATION_SET:
        result = _result("unknown", ok=False, reason_code="UNKNOWN_OPERATION", message="Unknown public operation; run system-x help", installation_state="SOURCE_ONLY", service_state="STOPPED", readiness_state="WAITING_FOR_MODEL", model_state="ABSENT", connection_state="NOT_READY", recommended_model=None, child_result_identities=())
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        return 2
    operation = arguments[0]
    if operation == "chat":
        from .chat_client import run_chat
        return run_chat()
    if operation == "verify-code":
        from .code_verify import main as verify_main
        return verify_main(_repository_root(), machine=False)
    try:
        root = _repository_root()
        if operation == "help":
            result = _help()
        elif operation == "install":
            result = _install(root)
        elif operation == "status":
            result = _status(root)
        elif operation == "connection":
            result = _connection(root)
        elif operation == "open":
            result = _open(root)
        elif operation == "verify-code":
            result = _result("verify-code", ok=True, reason_code="VERIFY_CODE", message="Source gate completed", installation_state="SOURCE_ONLY", service_state="STOPPED", readiness_state="SOURCE_ONLY", model_state="UNKNOWN", connection_state="NOT_READY", recommended_model=None, child_result_identities=())
        else:
            result = _doctor(root)
    except Exception:
        result = _result(operation, ok=False, reason_code="INTERNAL_ERROR", message="System X public operation failed safely", installation_state="REPAIR_REQUIRED", service_state="UNKNOWN", readiness_state="DEGRADED", model_state="UNKNOWN", connection_state="UNKNOWN", recommended_model=None, child_result_identities=())
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if result["ok"] else 2
