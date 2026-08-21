#!/usr/bin/env python3
"""Branch-local API-service lifecycle controller.

Exit classes are stable: 0 success, 2 input/dependency rejection,
3 state/ownership conflict, 4 runtime start/stop failure, and 5 internal error.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import datetime as dt
import errno
import hashlib
import ipaddress
import json
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "system-x.gguf-api-service-controller.v1"
AUTHENTICATION_CONTRACT = "system-x.private-authentication.v1"
OPERATIONS = (
    "plan",
    "start",
    "status",
    "stop",
    "reconcile",
    "alias-transaction",
)
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")
DEPENDENCY_FILES = {
    "requirements.in": (
        113,
        "d4865cbaa25917c8d3f3e9b65fbdb574e6474afeb4b73b633bb09ee818339677",
    ),
    "requirements.lock": (
        420,
        "c6293b88a7edf3354a83f1efb772c8a275fac4ee2880a1359256d88a85f95d5e",
    ),
    "configuration.schema.json": (
        10105,
        "5519472b704db16b4bd4dd9469ec4312a4dd501b1ab10bbb8c502765704fb3ed",
    ),
    "configuration.example.json": (
        2076,
        "fcc1b8fd0eef4bed7bfdb0374c456be0ab27ceeef33a2a3d982e840796cbeb03",
    ),
    "src/system_x_gguf_api/__init__.py": (
        205,
        "6de72ee10a917feec2bf041b76b904819dd4c748d3663080bf90cffa475440c6",
    ),
    "src/system_x_gguf_api/application.py": (
        19781,
        "5a60ecd87b6ba1807ae9f4c8b2e57283b5f5525a2bd242d81e0b16d202c053b3",
    ),
    "src/system_x_gguf_api/anthropic_adapter.py": (
        10973,
        "1b21a9ee8102709ce56b63c3ae1930de89087e679cba02f6bd5592126d536c64",
    ),
    "src/system_x_gguf_api/anthropic_contract.py": (
        1764,
        "fe4bd3c3de68bd519b657305aac99ec0cd5f17a48ca98b795aa11095b0b35915",
    ),
    "src/system_x_gguf_api/anthropic_errors.py": (
        5878,
        "5ee605b490d45a80bbefdf017abb7cd99cccc0d43ad51a48f86f4494d774881d",
    ),
    "src/system_x_gguf_api/anthropic_routes.py": (
        3014,
        "3356ed4db9171b9dc29397ad4ac265ee07d0e1b044cead8e06d9193058e62ec8",
    ),
    "src/system_x_gguf_api/anthropic_schemas.py": (
        13367,
        "6a4ba3064244aa9e7d1a281fc1dc09e2d7888ec3417dedce8e675bdcc157511b",
    ),
    "src/system_x_gguf_api/anthropic_stream.py": (
        15162,
        "c9268bb8d859e9d63a0c43f495534f3e8a6588eb8874dcc4500a3aa452c94278",
    ),
    "src/system_x_gguf_api/artifact_inspector.py": (
        21875,
        "d3fa631d1a9c24e3e8045f77d33398c78e3a9bb4141fbac37fbdd47f8a6a11cd",
    ),
    "src/system_x_gguf_api/backend.py": (
        51383,
        "c079342b236147502b876dff0a53248562213a5e27c8995711594a3d696689f2",
    ),
    "src/system_x_gguf_api/capability_inspector.py": (
        6822,
        "3a65078a19b86a4ebc1be0ec3c9cb939bde30cd9449745546723cea2732adc51",
    ),
    "src/system_x_gguf_api/authentication.py": (
        10292,
        "a9a86df5745a8bf24f9890a7940f5b2e9f3d803e75e681f91fbf9a0d18b7c070",
    ),
    "src/system_x_gguf_api/authentication_openapi.py": (
        7755,
        "3753365b416a1f5479792f07d8b16bd21948f58b18a4b9eedf24df365bb24612",
    ),
    "src/system_x_gguf_api/request_governance.py": (
        10084,
        "94ca9a320dfbcf9c0628782a61363113bf6c142c77bab3cb7c036b6291eb4c6d",
    ),
    "src/system_x_gguf_api/credential_admin.py": (
        2289,
        "3d0e0955b81596f962818ce847ac913cd23225ccffeeabb0ffad486b90e79bcf",
    ),
    "src/system_x_gguf_api/credential_store.py": (
        24855,
        "00de00fc555f2e5dcd1e6c02f5b6d636a2215de168479fa0fa58a5f380784dbd",
    ),
    "src/system_x_gguf_api/credential_types.py": (
        3867,
        "462370b2089908c304e27f0e62bf1c5dd3d41e854455f4e794f17131b1089166",
    ),
    "src/system_x_gguf_api/controller_client.py": (
        5867,
        "ca8c9e6d55613314d04af5354c92ed19d694349d82207ccb62f23b54cd83b45a",
    ),
    "src/system_x_gguf_api/compatibility_models.py": (
        2006,
        "15af07e0016e099d29f1b0e963b3a03e1aa2bd6fd806bbd978e972f3f1a39281",
    ),
    "src/system_x_gguf_api/errors.py": (
        27258,
        "2cb0966209ce3783077a37cd791fe4c31c7b750c9c8b46ddfe30b74c1877d58f",
    ),
    "src/system_x_gguf_api/external_static.py": (
        7801,
        "b4396d2e4f7d12f483a8baa3ac25f6b123f111bb8ea5d464dc8f2168ad1ed6ff",
    ),
    "src/system_x_gguf_api/finalization_policy.py": (
        2371,
        "1e94665f7dcd50553cfb903b949c9bf71c84aab7d1e64a387e601f4ebeec9687",
    ),
    "src/system_x_gguf_api/inference_service.py": (
        31794,
        "b125008b916e060f86d58c294bfa5b5970050417a67c8a8ff59288ede9aabce0",
    ),
    "src/system_x_gguf_api/model_catalogue.py": (
        21837,
        "1debf12288213845e77e7e14ae872e1f60c17245c698ac4ca80cc3448e16e9b1",
    ),
    "src/system_x_gguf_api/model_lifecycle.py": (
        5398,
        "479524e0ea33cd5f4fb638e24615b2f73baab2129d7f92b9b9556af99cfe692a",
    ),
    "src/system_x_gguf_api/model_monitor.py": (
        5224,
        "40b0d4ed2d4f8b54bf064884cec4ea4c727e297a1ad1a874bc9f930344769452",
    ),
    "src/system_x_gguf_api/model_registry.py": (
        25183,
        "5d3a8cb19f1a74f4dc012bd4a82a7754376e774b8e46be3ce634d387f9a33d2a",
    ),
    "src/system_x_gguf_api/openai_adapter.py": (
        24174,
        "9681eff81700d4cdab2b3bad2af286d736eba3b395a0db4b18b288a811a05edd",
    ),
    "src/system_x_gguf_api/openai_contract.py": (
        2208,
        "b6a2cd713e882804372b6f3f98d0780aa069c8af8a546c0fe93ab2ebbcbe1136",
    ),
    "src/system_x_gguf_api/openai_errors.py": (
        8882,
        "3b756f5d1a0b0081ab9a368e5fbe583381b41b4a050ddc876a48380a6459ce6d",
    ),
    "src/system_x_gguf_api/openai_responses_stream.py": (
        15895,
        "bb9ad7035608a81d1648725ee60a85b8da183c3ee42632e7f20ffda84c4ef3e7",
    ),
    "src/system_x_gguf_api/openai_routes.py": (
        4344,
        "cebe183252163c0d0f625e58f7a4e304ef5376b2efc3977f25840b2f75067667",
    ),
    "src/system_x_gguf_api/openai_schemas.py": (
        24341,
        "2a628461d1d40fc2fb91af6ef3be0dc13ba8d97b8098e61da6a286293c5c8063",
    ),
    "src/system_x_gguf_api/openai_stream.py": (
        10755,
        "c98cb66f4b84b3c3f9b2bd6d58e537c7e8dca275f0ddd44a3421f4ebbd78732d",
    ),
    "src/system_x_gguf_api/operation_records.py": (
        30407,
        "affa00eec23f496054706cc5265936d3befd90309f362d708c81539b63c18571",
    ),
    "src/system_x_gguf_api/privacy_diagnostics.py": (
        3148,
        "d044ee703c35b6ec53073a01e48ba8c80c04f419f641b086025e63dcaa1a325c",
    ),
    "src/system_x_gguf_api/operation_metrics.py": (
        8866,
        "ed5c3c7a5511627488306c41c8a04779b93130ca4a3cc3fcce09836d17463daa",
    ),
    "src/system_x_gguf_api/registry_store.py": (
        108127,
        "37feea1c68916e6d6944c3f014446b602c8e009178ee03a5da9986ec8e2214d9",
    ),
    "src/system_x_gguf_api/registry_types.py": (
        4737,
        "2c5fab07724dda9759119ef217f32267c5b312cb192104625447d95ad9377840",
    ),
    "src/system_x_gguf_api/request_context.py": (
        1309,
        "db29a56f885702cab5e465bfb90c5b56abf22e2587a0565b1a08976e5f746ac0",
    ),
    "src/system_x_gguf_api/response_normalizer.py": (
        18054,
        "89da46ea2754a41f13539a71d6ac2caf6e3dce8679cc90af51a99a09f28b9d12",
    ),
    "src/system_x_gguf_api/router_client.py": (
        37844,
        "7dc0578d5d0b551ab3c6eafdef923444b329a44e5044da6866adbe38c2b8edaa",
    ),
    "src/system_x_gguf_api/schemas.py": (
        19215,
        "676b9e0ced58c3f009630b9c6a9e2115902375e13256a56e882a0211af856afd",
    ),
    "src/system_x_gguf_api/secret_redaction.py": (
        1264,
        "a9f077def7a3920052878f7d39b30909b4d23ec5312b705df35db6a091cf3038",
    ),
    "src/system_x_gguf_api/settings.py": (
        12143,
        "2cbaf065de14c0f60f1a52c9e59a87054ab552d974de8eb2bcc10bf97ecc86da",
    ),
    "src/system_x_gguf_api/warm_model.py": (
        22630,
        "950eb9a2d772650771d0571d15945b392e3e928238c32c83798b03f10717d95c",
    ),
    "src/system_x_gguf_api/runtime_recovery.py": (
        29383,
        "efc1c1a3eca03aeb4106ccde3c750db3476a407e0d62711dca20e9a93cd1a8fc",
    ),
    "src/system_x_gguf_api/sse.py": (
        8306,
        "ab3a6f79389ddebf42e058b1e5bbee6f3aef883e44e482a5ae4a159c1ae02c19",
    ),
    "src/system_x_gguf_api/stream_control.py": (
        11098,
        "dee52cd9e1c239c2ba6369d304bf6977ca13c744c01f7fa817547851413ab455",
    ),
    "src/system_x_gguf_api/stream_types.py": (
        14974,
        "5b7b13e5359240d655d65386a7bc5c2ae27c11bc5cab16b1f0c641bc0eb5f69e",
    ),
    "src/system_x_gguf_api/streaming_inference.py": (
        50167,
        "cce1428a9a53de2352ae4838b2d216064e43157c6f3bc8eee5f5281beb5884b0",
    ),
    "src/system_x_gguf_api/system_routes.py": (
        3915,
        "ab230a89b5aaff0710603db645e483bce94d2501a100c99b65ab0cd09e00aadb",
    ),
    "src/system_x_gguf_api/system_stream.py": (
        1409,
        "86ed1452be7208d631d489c300280fada782f483f8c294ca47a9a41d187ce049",
    ),
    "src/system_x_gguf_api/tool_contract.py": (
        22544,
        "2720fd24bab51b5121032298c38f5b53539e504cfa01e6e259758d8b0e57f346",
    ),
    "src/system_x_gguf_api/tool_schema.py": (
        12510,
        "fbd131de024a201f003be09d1b4a6e4bdbdbc96db903411cf4932b80cdb46a49",
    ),
}
REASON_EXIT = {
    "OK": 0,
    "INVALID_INPUT": 2,
    "DEPENDENCY_MISSING": 2,
    "DEPENDENCY_INVALID": 2,
    "RUNTIME_LAYOUT_INVALID": 2,
    "ALIAS_TRANSACTION_INVALID": 2,
    "ALIAS_TRANSACTION_CONFLICT": 3,
    "ENDPOINT_IN_USE": 3,
    "SERVICE_LOCK_ACTIVE": 3,
    "SERVICE_NOT_ACTIVE": 3,
    "SERVICE_STATE_INCONSISTENT": 3,
    "PROCESS_IDENTITY_MISMATCH": 3,
    "LISTENER_OWNERSHIP_MISMATCH": 3,
    "OWNERSHIP_UNCERTAIN": 3,
    "ENDPOINT_CONFLICT": 3,
    "PROCESS_START_FAILED": 4,
    "PROCESS_EXITED_EARLY": 4,
    "GRACEFUL_STOP_TIMEOUT": 4,
    "FORCED_STOP_REQUIRED": 4,
    "OWNED_PROCESS_GROUP_NOT_GONE": 4,
    "INTERNAL_ERROR": 5,
}


class ControllerError(Exception):
    def __init__(self, reason_code: str, message: str, **sections: Any) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.sections = sections


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ControllerError("INVALID_INPUT", message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def derive_paths(controller_file: str | Path | None = None) -> dict[str, Path]:
    source = Path(controller_file or __file__).resolve(strict=True)
    controller_dir = source.parent
    branch_root = controller_dir.parent
    api_service_root = branch_root / "api_service"
    runtime_api_root = branch_root / "RUNTIME" / "api"
    return {
        "controller_file": source,
        "controller_dir": controller_dir,
        "branch_root": branch_root,
        "api_service_root": api_service_root,
        "venv_python": api_service_root / ".venv" / "bin" / "python",
        "application_source_root": api_service_root / "src",
        "runtime_api_root": runtime_api_root,
        "log_root": runtime_api_root / "logs",
        "status_root": runtime_api_root / "status",
        "pid_root": runtime_api_root / "pids",
        "lock_root": runtime_api_root / "locks",
        "transaction_root": runtime_api_root / "transactions",
        "database_root": runtime_api_root / "database",
        "registry_database": runtime_api_root
        / "database"
        / "model_registry.sqlite3",
        "auth_root": runtime_api_root / "auth",
        "credential_database": runtime_api_root / "auth" / "credentials.sqlite3",
        "credential_pepper": runtime_api_root / "auth" / "pepper.bin",
        "credential_handoff_root": runtime_api_root / "auth" / "handoff",
        "active_lock": runtime_api_root / "locks" / "active.lock",
        "active_pid": runtime_api_root / "pids" / "active.json",
        "service_status": runtime_api_root / "status" / "service.json",
    }


def contained(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def validate_dependency(paths: dict[str, Path]) -> dict[str, Any]:
    root = paths["api_service_root"]
    if not root.exists():
        raise ControllerError("DEPENDENCY_MISSING", "API-service dependency root is missing")
    if root.is_symlink() or not root.is_dir():
        raise ControllerError("DEPENDENCY_INVALID", "API-service dependency root is not a physical directory")
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in DEPENDENCY_FILES.items():
        path = root / relative
        if not path.exists():
            raise ControllerError("DEPENDENCY_MISSING", f"required dependency path is missing: {relative}")
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ControllerError("DEPENDENCY_INVALID", f"required dependency path is not a regular file: {relative}")
        if not contained(path, root):
            raise ControllerError("DEPENDENCY_INVALID", f"required dependency path escapes its root: {relative}")
        data = path.read_bytes()
        actual = (len(data), hashlib.sha256(data).hexdigest())
        if actual != expected:
            raise ControllerError(
                "DEPENDENCY_INVALID",
                f"required dependency identity does not match: {relative}",
                dependency={"path": str(path), "bytes": actual[0], "sha256": actual[1]},
            )
        if relative.endswith(".py"):
            try:
                tree = ast.parse(data.decode("utf-8"), filename=str(path))
                compile(tree, str(path), "exec")
            except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
                raise ControllerError("DEPENDENCY_INVALID", f"dependency source is invalid: {relative}: {exc}") from exc
        verified[relative] = {"bytes": actual[0], "sha256": actual[1]}
    python = paths["venv_python"]
    if not python.exists():
        raise ControllerError("DEPENDENCY_MISSING", "branch-local Python executable is missing")
    if python.is_symlink():
        try:
            target = python.resolve(strict=True)
        except OSError as exc:
            raise ControllerError("DEPENDENCY_INVALID", "branch-local Python symlink target is unavailable") from exc
        if target != Path("/usr/bin/python3.14") or not os.access(target, os.X_OK):
            raise ControllerError("DEPENDENCY_INVALID", "branch-local Python symlink target is not the pinned CPython 3.14 executable")
    else:
        metadata = python.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ControllerError("DEPENDENCY_INVALID", "branch-local Python is not a regular file")
        if not contained(python, root) or not os.access(python, os.X_OK):
            raise ControllerError("DEPENDENCY_INVALID", "branch-local Python is uncontained or non-executable")
    source_root = paths["application_source_root"]
    if source_root.is_symlink() or not source_root.is_dir() or not contained(source_root, root):
        raise ControllerError("DEPENDENCY_INVALID", "application source root is invalid")
    return {
        "verified_files": len(verified),
        "venv_python": str(python),
        "application_source_root": str(source_root),
    }


def validate_runtime_layout(paths: dict[str, Path]) -> None:
    branch_root = paths["branch_root"].resolve(strict=True)
    for key in (
        "runtime_api_root",
        "log_root",
        "status_root",
        "pid_root",
        "lock_root",
        "transaction_root",
        "database_root",
    ):
        path = paths[key]
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ControllerError("RUNTIME_LAYOUT_INVALID", f"runtime path is missing: {key}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ControllerError("RUNTIME_LAYOUT_INVALID", f"runtime path is not a physical directory: {key}")
        if branch_root not in resolved.parents:
            raise ControllerError("RUNTIME_LAYOUT_INVALID", f"runtime path escapes branch root: {key}")
        if key == "database_root" and stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID", "database_root mode must be 0700"
            )
    database = paths["registry_database"]
    if os.path.lexists(database):
        metadata = database.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                "registry database is not a direct regular file",
            )
        if not contained(database, paths["database_root"]):
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID", "registry database escapes database_root"
            )
    validate_auth_layout(paths)


def validate_auth_layout(paths: dict[str, Path]) -> dict[str, Any]:
    """Validate dynamic authentication state without reading secret bytes."""

    branch_root = paths["branch_root"].resolve(strict=True)
    for key in ("auth_root", "credential_handoff_root"):
        path = paths[key]
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication directory is missing: {key}",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication path is not a physical directory: {key}",
            )
        if branch_root not in resolved.parents:
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication path escapes branch root: {key}",
            )
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication directory mode must be 0700: {key}",
            )
    file_sizes: dict[str, int] = {}
    for key in ("credential_database", "credential_pepper"):
        path = paths[key]
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication file is missing: {key}",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication path is not a physical regular file: {key}",
            )
        if paths["auth_root"].resolve(strict=True) not in resolved.parents:
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication file escapes auth root: {key}",
            )
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ControllerError(
                "RUNTIME_LAYOUT_INVALID",
                f"authentication file mode must be 0600: {key}",
            )
        file_sizes[key] = metadata.st_size
    if file_sizes["credential_pepper"] != 32:
        raise ControllerError(
            "RUNTIME_LAYOUT_INVALID",
            "credential pepper must contain exactly 32 bytes",
        )
    return {
        "authentication_contract": AUTHENTICATION_CONTRACT,
        "authentication_enabled": True,
        "auth_root": str(paths["auth_root"]),
        "credential_database": str(paths["credential_database"]),
        "credential_database_bytes": file_sizes["credential_database"],
        "credential_pepper_bytes": file_sizes["credential_pepper"],
    }


def parse_loopback(value: str, field: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ControllerError("INVALID_INPUT", f"{field} must be an IPv4 loopback address") from exc
    if address.version != 4 or not address.is_loopback:
        raise ControllerError("INVALID_INPUT", f"{field} must be an IPv4 loopback address")
    return address.compressed


def parse_port(value: str | int | None, field: str, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    try:
        port = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ControllerError("INVALID_INPUT", f"{field} must be an integer") from exc
    if isinstance(value, str) and str(port) != value.strip():
        raise ControllerError("INVALID_INPUT", f"{field} must be a canonical integer")
    if not 1 <= port <= 65535:
        raise ControllerError("INVALID_INPUT", f"{field} must be in range 1..65535")
    return port


def parse_boolean(value: str | bool, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ControllerError("INVALID_INPUT", f"{field} must be a boolean")


def parse_bounded_float(
    value: str | float, field: str, minimum: float, maximum: float
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ControllerError("INVALID_INPUT", f"{field} must be numeric") from exc
    if not minimum <= number <= maximum:
        raise ControllerError(
            "INVALID_INPUT", f"{field} must be in range {minimum}..{maximum}"
        )
    return number


def parse_bounded_integer(
    value: str | int, field: str, minimum: int, maximum: int
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ControllerError("INVALID_INPUT", f"{field} must be an integer") from exc
    if isinstance(value, str) and str(number) != value.strip():
        raise ControllerError(
            "INVALID_INPUT", f"{field} must be a canonical integer"
        )
    if not minimum <= number <= maximum:
        raise ControllerError(
            "INVALID_INPUT", f"{field} must be in range {minimum}..{maximum}"
        )
    return number


def parse_registry_alias(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", normalized) is None:
        raise ControllerError(
            "INVALID_INPUT",
            "registry_default_alias must match [a-z0-9][a-z0-9._-]{0,63}",
        )
    return normalized


def parse_startup_model_policy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"always_warm", "router_control", "registry_control", "api_only"}:
        raise ControllerError(
            "INVALID_INPUT",
            "startup_model_policy must equal always_warm, router_control, registry_control or api_only",
        )
    return normalized


def parse_profile_identity(value: str) -> str:
    normalized = str(value).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise ControllerError(
            "INVALID_INPUT",
            "service_control_profile_identity is invalid",
        )
    return normalized


def parse_desired_state_path(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ControllerError(
            "INVALID_INPUT",
            "service_control_desired_state_path is invalid",
        )
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ControllerError(
            "INVALID_INPUT",
            "service_control_desired_state_path must be absolute",
        )
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ControllerError(
            "INVALID_INPUT",
            "service_control_desired_state_path is unavailable",
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ControllerError(
            "INVALID_INPUT",
            "service_control_desired_state_path must be a direct regular file",
        )
    return str(resolved)


def parse_external_static_mount_path(value: str) -> str:
    normalized = value.strip()
    if (
        re.fullmatch(
            r"/[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*",
            normalized,
        )
        is None
    ):
        raise ControllerError(
            "INVALID_INPUT",
            "external_static_mount_path must be a canonical lower-case URL path",
        )
    return normalized


def parse_external_static_distribution_root(
    value: str | None,
    *,
    enabled: bool,
) -> str | None:
    if value is None or value == "":
        if enabled:
            raise ControllerError(
                "INVALID_INPUT",
                "external_static_distribution_root is required when enabled",
            )
        return None
    if "\x00" in value or value.strip() != value:
        raise ControllerError(
            "INVALID_INPUT",
            "external_static_distribution_root is not canonical",
        )
    root = Path(value)
    if not root.is_absolute():
        raise ControllerError(
            "INVALID_INPUT",
            "external_static_distribution_root must be absolute",
        )
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
        index_metadata = (root / "index.html").lstat()
    except OSError as exc:
        raise ControllerError(
            "INVALID_INPUT",
            "external_static_distribution_root or index.html is missing",
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != root
        or stat.S_ISLNK(index_metadata.st_mode)
        or not stat.S_ISREG(index_metadata.st_mode)
    ):
        raise ControllerError(
            "INVALID_INPUT",
            "external static distribution must be a physical directory with "
            "a direct regular index.html",
        )
    return str(root)


def validated_input(namespace: argparse.Namespace) -> dict[str, Any]:
    log_level = namespace.log_level.lower()
    if log_level not in LOG_LEVELS:
        raise ControllerError("INVALID_INPUT", f"log_level must be one of: {', '.join(LOG_LEVELS)}")
    external_static_enabled = parse_boolean(
        namespace.external_static_enabled,
        "external_static_enabled",
    )
    values = {
        "host": parse_loopback(namespace.host, "host"),
        "port": parse_port(namespace.port, "port"),
        "authentication_enabled": parse_boolean(
            namespace.authentication_enabled, "authentication_enabled"
        ),
        "request_max_body_bytes": parse_bounded_integer(
            namespace.request_max_body_bytes,
            "request_max_body_bytes",
            1024,
            16_777_216,
        ),
        "request_max_total_tokens": parse_bounded_integer(
            namespace.request_max_total_tokens,
            "request_max_total_tokens",
            1,
            1_048_576,
        ),
        "request_timeout_seconds": parse_bounded_float(
            namespace.request_timeout_seconds,
            "request_timeout_seconds",
            0.1,
            3600.0,
        ),
        "request_concurrency_limit_per_key": parse_bounded_integer(
            namespace.request_concurrency_limit_per_key,
            "request_concurrency_limit_per_key",
            1,
            64,
        ),
        "request_rate_limit_requests_per_key": parse_bounded_integer(
            namespace.request_rate_limit_requests_per_key,
            "request_rate_limit_requests_per_key",
            1,
            100_000,
        ),
        "request_rate_limit_window_seconds": parse_bounded_float(
            namespace.request_rate_limit_window_seconds,
            "request_rate_limit_window_seconds",
            0.1,
            86_400.0,
        ),
        "private_backend_host": parse_loopback(namespace.private_backend_host, "private_backend_host"),
        "private_backend_port": parse_port(
            namespace.private_backend_port,
            "private_backend_port",
            nullable=True,
        ),
        "private_backend_enabled": parse_boolean(
            namespace.private_backend_enabled, "private_backend_enabled"
        ),
        "private_backend_models_max": parse_port(
            namespace.private_backend_models_max, "private_backend_models_max"
        ),
        "private_backend_start_timeout_seconds": parse_bounded_float(
            namespace.private_backend_start_timeout_seconds,
            "private_backend_start_timeout_seconds",
            0.1,
            120.0,
        ),
        "private_backend_model_timeout_seconds": parse_bounded_float(
            namespace.private_backend_model_timeout_seconds,
            "private_backend_model_timeout_seconds",
            0.1,
            300.0,
        ),
        "private_backend_inference_timeout_seconds": parse_bounded_float(
            namespace.private_backend_inference_timeout_seconds,
            "private_backend_inference_timeout_seconds",
            0.1,
            3600.0,
        ),
        "private_backend_poll_interval_seconds": parse_bounded_float(
            namespace.private_backend_poll_interval_seconds,
            "private_backend_poll_interval_seconds",
            0.05,
            5.0,
        ),
        "registry_enabled": parse_boolean(
            namespace.registry_enabled, "registry_enabled"
        ),
        "registry_reconcile_interval_seconds": parse_bounded_float(
            namespace.registry_reconcile_interval_seconds,
            "registry_reconcile_interval_seconds",
            5.0,
            3600.0,
        ),
        "registry_watch_debounce_milliseconds": parse_bounded_integer(
            namespace.registry_watch_debounce_milliseconds,
            "registry_watch_debounce_milliseconds",
            100,
            10_000,
        ),
        "registry_stability_samples": parse_bounded_integer(
            namespace.registry_stability_samples,
            "registry_stability_samples",
            2,
            10,
        ),
        "registry_stability_interval_seconds": parse_bounded_float(
            namespace.registry_stability_interval_seconds,
            "registry_stability_interval_seconds",
            0.1,
            10.0,
        ),
        "registry_database_busy_timeout_milliseconds": parse_bounded_integer(
            namespace.registry_database_busy_timeout_milliseconds,
            "registry_database_busy_timeout_milliseconds",
            100,
            60_000,
        ),
        "registry_default_alias": parse_registry_alias(
            namespace.registry_default_alias
        ),
        "startup_model_policy": parse_startup_model_policy(
            namespace.startup_model_policy
        ),
        "automatic_recovery_enabled": parse_boolean(
            namespace.automatic_recovery_enabled,
            "automatic_recovery_enabled",
        ),
        "recovery_delay_initial_seconds": parse_bounded_float(
            namespace.recovery_delay_initial_seconds,
            "recovery_delay_initial_seconds",
            0.0,
            3600.0,
        ),
        "recovery_delay_maximum_seconds": parse_bounded_float(
            namespace.recovery_delay_maximum_seconds,
            "recovery_delay_maximum_seconds",
            0.01,
            3600.0,
        ),
        "recovery_delay_multiplier": parse_bounded_float(
            namespace.recovery_delay_multiplier,
            "recovery_delay_multiplier",
            1.0,
            16.0,
        ),
        "recovery_maximum_attempts_in_window": parse_bounded_integer(
            namespace.recovery_maximum_attempts_in_window,
            "recovery_maximum_attempts_in_window",
            1,
            16,
        ),
        "recovery_attempt_window_seconds": parse_bounded_float(
            namespace.recovery_attempt_window_seconds,
            "recovery_attempt_window_seconds",
            1.0,
            3600.0,
        ),
        "recovery_stable_reset_seconds": parse_bounded_float(
            namespace.recovery_stable_reset_seconds,
            "recovery_stable_reset_seconds",
            1.0,
            3600.0,
        ),
        "service_control_profile_identity": parse_profile_identity(
            namespace.service_control_profile_identity
        ),
        "service_control_desired_state_path": parse_desired_state_path(
            namespace.service_control_desired_state_path
        ),
        "external_static_enabled": external_static_enabled,
        "external_static_distribution_root": (
            parse_external_static_distribution_root(
                namespace.external_static_distribution_root,
                enabled=external_static_enabled,
            )
        ),
        "external_static_mount_path": parse_external_static_mount_path(
            namespace.external_static_mount_path
        ),
        "log_level": log_level,
    }
    if values["request_timeout_seconds"] > values["private_backend_inference_timeout_seconds"]:
        raise ControllerError(
            "INVALID_INPUT",
            "request_timeout_seconds must not exceed private backend inference timeout",
        )
    if values["private_backend_models_max"] != 1:
        raise ControllerError(
            "INVALID_INPUT", "private_backend_models_max must equal 1"
        )
    if values["private_backend_enabled"]:
        if values["private_backend_port"] is None:
            raise ControllerError(
                "INVALID_INPUT",
                "private_backend_port is required when private backend is enabled",
            )
        if values["private_backend_port"] == values["port"]:
            raise ControllerError(
                "INVALID_INPUT", "public and private backend ports must differ"
            )
    if values["registry_enabled"] and not values["private_backend_enabled"]:
        raise ControllerError(
            "INVALID_INPUT",
            "private_backend_enabled must be true when registry is enabled",
        )
    if values["startup_model_policy"] == "registry_control":
        if not values["private_backend_enabled"]:
            raise ControllerError(
                "INVALID_INPUT",
                "private_backend_enabled must be true for registry_control",
            )
        if not values["registry_enabled"]:
            raise ControllerError(
                "INVALID_INPUT",
                "registry_enabled must be true for registry_control",
            )
        if values["automatic_recovery_enabled"]:
            raise ControllerError(
                "INVALID_INPUT",
                "automatic_recovery_enabled must be false for registry_control",
            )
    if values["startup_model_policy"] == "router_control":
        if not values["private_backend_enabled"]:
            raise ControllerError(
                "INVALID_INPUT",
                "private_backend_enabled must be true for router_control",
            )
        if values["registry_enabled"]:
            raise ControllerError(
                "INVALID_INPUT",
                "registry_enabled must be false for router_control",
            )
        if values["automatic_recovery_enabled"]:
            raise ControllerError(
                "INVALID_INPUT",
                "automatic_recovery_enabled must be false for router_control",
            )
    if (
        values["recovery_delay_initial_seconds"]
        > values["recovery_delay_maximum_seconds"]
    ):
        raise ControllerError(
            "INVALID_INPUT",
            "recovery initial delay must not exceed maximum delay",
        )
    values["service_start_timeout_seconds"] = max(
        10.0,
        values["private_backend_start_timeout_seconds"]
        + (
            values["private_backend_model_timeout_seconds"]
            + (
                values["registry_stability_samples"]
                * values["registry_stability_interval_seconds"]
            )
            if values["registry_enabled"]
            else 0.0
        ),
    )
    return values


def build_plan(paths: dict[str, Path], values: dict[str, Any]) -> dict[str, Any]:
    argv = [
        str(paths["venv_python"]),
        "-B",
        "-m",
        "uvicorn",
        "system_x_gguf_api.application:app",
        "--app-dir",
        str(paths["application_source_root"]),
        "--host",
        values["host"],
        "--port",
        str(values["port"]),
        "--workers",
        "1",
        "--no-access-log",
        "--log-level",
        values["log_level"],
    ]
    environment_overrides: dict[str, str | None] = {
        "PYTHONPATH": str(paths["application_source_root"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SYSTEM_X_GGUF_API_PUBLIC_HOST": values["host"],
        "SYSTEM_X_GGUF_API_PUBLIC_PORT": str(values["port"]),
        "SYSTEM_X_GGUF_API_AUTHENTICATION_ENABLED": (
            "true" if values["authentication_enabled"] else "false"
        ),
        "SYSTEM_X_GGUF_API_REQUEST_MAX_BODY_BYTES": str(
            values["request_max_body_bytes"]
        ),
        "SYSTEM_X_GGUF_API_REQUEST_MAX_TOTAL_TOKENS": str(
            values["request_max_total_tokens"]
        ),
        "SYSTEM_X_GGUF_API_REQUEST_TIMEOUT_SECONDS": str(
            values["request_timeout_seconds"]
        ),
        "SYSTEM_X_GGUF_API_REQUEST_CONCURRENCY_LIMIT_PER_KEY": str(
            values["request_concurrency_limit_per_key"]
        ),
        "SYSTEM_X_GGUF_API_REQUEST_RATE_LIMIT_REQUESTS_PER_KEY": str(
            values["request_rate_limit_requests_per_key"]
        ),
        "SYSTEM_X_GGUF_API_REQUEST_RATE_LIMIT_WINDOW_SECONDS": str(
            values["request_rate_limit_window_seconds"]
        ),
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_HOST": values["private_backend_host"],
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_PORT": (
            str(values["private_backend_port"]) if values["private_backend_port"] is not None else None
        ),
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_ENABLED": (
            "true" if values["private_backend_enabled"] else "false"
        ),
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_MODELS_MAX": str(
            values["private_backend_models_max"]
        ),
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_START_TIMEOUT_SECONDS": str(
            values["private_backend_start_timeout_seconds"]
        ),
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_MODEL_TIMEOUT_SECONDS": str(
            values["private_backend_model_timeout_seconds"]
        ),
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_INFERENCE_TIMEOUT_SECONDS": str(
            values["private_backend_inference_timeout_seconds"]
        ),
        "SYSTEM_X_GGUF_API_PRIVATE_BACKEND_POLL_INTERVAL_SECONDS": str(
            values["private_backend_poll_interval_seconds"]
        ),
        "SYSTEM_X_GGUF_API_REGISTRY_ENABLED": (
            "true" if values["registry_enabled"] else "false"
        ),
        "SYSTEM_X_GGUF_API_REGISTRY_RECONCILE_INTERVAL_SECONDS": str(
            values["registry_reconcile_interval_seconds"]
        ),
        "SYSTEM_X_GGUF_API_REGISTRY_WATCH_DEBOUNCE_MILLISECONDS": str(
            values["registry_watch_debounce_milliseconds"]
        ),
        "SYSTEM_X_GGUF_API_REGISTRY_STABILITY_SAMPLES": str(
            values["registry_stability_samples"]
        ),
        "SYSTEM_X_GGUF_API_REGISTRY_STABILITY_INTERVAL_SECONDS": str(
            values["registry_stability_interval_seconds"]
        ),
        "SYSTEM_X_GGUF_API_REGISTRY_DATABASE_BUSY_TIMEOUT_MILLISECONDS": str(
            values["registry_database_busy_timeout_milliseconds"]
        ),
        "SYSTEM_X_GGUF_API_REGISTRY_DEFAULT_ALIAS": values[
            "registry_default_alias"
        ],
        "SYSTEM_X_GGUF_API_STARTUP_MODEL_POLICY": values[
            "startup_model_policy"
        ],
        "SYSTEM_X_GGUF_API_AUTOMATIC_RECOVERY_ENABLED": (
            "true" if values["automatic_recovery_enabled"] else "false"
        ),
        "SYSTEM_X_GGUF_API_RECOVERY_DELAY_INITIAL_SECONDS": str(
            values["recovery_delay_initial_seconds"]
        ),
        "SYSTEM_X_GGUF_API_RECOVERY_DELAY_MAXIMUM_SECONDS": str(
            values["recovery_delay_maximum_seconds"]
        ),
        "SYSTEM_X_GGUF_API_RECOVERY_DELAY_MULTIPLIER": str(
            values["recovery_delay_multiplier"]
        ),
        "SYSTEM_X_GGUF_API_RECOVERY_MAXIMUM_ATTEMPTS_IN_WINDOW": str(
            values["recovery_maximum_attempts_in_window"]
        ),
        "SYSTEM_X_GGUF_API_RECOVERY_ATTEMPT_WINDOW_SECONDS": str(
            values["recovery_attempt_window_seconds"]
        ),
        "SYSTEM_X_GGUF_API_RECOVERY_STABLE_RESET_SECONDS": str(
            values["recovery_stable_reset_seconds"]
        ),
        "SYSTEM_X_GGUF_API_SERVICE_CONTROL_PROFILE_IDENTITY": values[
            "service_control_profile_identity"
        ],
        "SYSTEM_X_GGUF_API_SERVICE_CONTROL_DESIRED_STATE_PATH": values[
            "service_control_desired_state_path"
        ],
        "SYSTEM_X_GGUF_API_EXTERNAL_STATIC_ENABLED": (
            "true" if values["external_static_enabled"] else "false"
        ),
        "SYSTEM_X_GGUF_API_EXTERNAL_STATIC_DISTRIBUTION_ROOT": values[
            "external_static_distribution_root"
        ],
        "SYSTEM_X_GGUF_API_EXTERNAL_STATIC_MOUNT_PATH": values[
            "external_static_mount_path"
        ],
    }
    return {
        "executable": str(paths["venv_python"]),
        "argv": argv,
        "cwd": str(paths["api_service_root"]),
        "shell": False,
        "start_new_session": True,
        "workers": 1,
        "reload": False,
        "access_log": False,
        "listener_start_timeout_seconds": values[
            "service_start_timeout_seconds"
        ],
        "environment_overrides": environment_overrides,
    }


def child_environment(plan: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    for key in tuple(environment):
        if key.startswith("SYSTEM_X_GGUF_API_") or key == (
            "SYSTEM_X_API_SERVICE_TRANSACTION_ID"
        ):
            environment.pop(key)
    for key, value in plan["environment_overrides"].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    return environment


def endpoint_available(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        return True
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            return False
        raise
    finally:
        probe.close()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def encode_record(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def atomic_write_json(path: Path, record: dict[str, Any], mode: int = 0o640) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        os.write(descriptor, encode_record(record))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_json_lock(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
    try:
        os.write(descriptor, encode_record(record))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ControllerError("SERVICE_LOCK_ACTIVE", "active service lock already exists") from exc
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_directory(path.parent)


def read_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("not a physical regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControllerError("SERVICE_STATE_INCONSISTENT", f"invalid state record: {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError("SERVICE_STATE_INCONSISTENT", f"state record is not an object: {path.name}")
    return value


def unlink_transaction_owned(path: Path, transaction_id: str) -> bool:
    if not path.exists():
        return False
    record = read_json(path)
    if record.get("transaction_id") != transaction_id:
        raise ControllerError("SERVICE_STATE_INCONSISTENT", f"refusing to remove foreign state record: {path.name}")
    path.unlink()
    fsync_directory(path.parent)
    return True


def create_log(path: Path) -> int:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
    os.fsync(descriptor)
    fsync_directory(path.parent)
    return descriptor


def process_stat(pid: int) -> dict[str, Any]:
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise ProcessLookupError(pid)
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise ProcessLookupError(pid)
    return {
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_identity": fields[19],
    }


def read_argv(pid: int) -> list[str]:
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]


def process_identity(pid: int) -> dict[str, Any]:
    proc_root = Path("/proc") / str(pid)
    values = process_stat(pid)
    if values["state"] == "Z":
        raise ProcessLookupError(pid)
    executable = os.readlink(proc_root / "exe")
    executable_metadata = os.stat(proc_root / "exe")
    return {
        "pid": pid,
        "process_start_identity": values["start_identity"],
        "pgid": values["pgid"],
        "sid": values["sid"],
        "executable": executable,
        "executable_device": executable_metadata.st_dev,
        "executable_inode": executable_metadata.st_ino,
        "argv": read_argv(pid),
    }


def wait_for_expected_identity(pid: int, plan: dict[str, Any], timeout: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    planned_executable = Path(plan["executable"])
    try:
        expected_executable = str(planned_executable.resolve(strict=True)) if planned_executable.is_symlink() else str(planned_executable)
    except OSError:
        expected_executable = str(planned_executable)
    while time.monotonic() < deadline:
        try:
            last = process_identity(pid)
        except (OSError, ProcessLookupError):
            time.sleep(0.02)
            continue
        if last["executable"] == expected_executable and last["argv"] == plan["argv"]:
            return last
        time.sleep(0.02)
    if last is None:
        raise ControllerError("PROCESS_EXITED_EARLY", "service process exited before identity capture")
    raise ControllerError(
        "PROCESS_IDENTITY_MISMATCH",
        "spawned process identity does not match the structured plan",
        ownership={"observed": last, "expected_argv": plan["argv"]},
    )


def identity_matches(record: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    try:
        observed = process_identity(int(record["pid"]))
    except (KeyError, TypeError, ValueError, OSError, ProcessLookupError):
        return False, None
    keys = (
        "pid",
        "process_start_identity",
        "pgid",
        "sid",
        "executable",
        "executable_device",
        "executable_inode",
        "argv",
    )
    return all(observed.get(key) == record.get(key) for key in keys), observed


def group_members(pgid: int) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            values = process_stat(int(entry.name))
        except (OSError, ProcessLookupError, ValueError):
            continue
        if values["pgid"] == pgid and values["state"] != "Z":
            members.append(
                {
                    "pid": int(entry.name),
                    "state": values["state"],
                    "process_start_identity": values["start_identity"],
                    "sid": values["sid"],
                }
            )
    return sorted(members, key=lambda item: int(item["pid"]))


def tcp_listeners() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    table = Path("/proc/net/tcp")
    for row in table.read_text(encoding="ascii").splitlines()[1:]:
        fields = row.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        address_hex, port_hex = fields[1].split(":")
        address = socket.inet_ntoa(bytes.fromhex(address_hex)[::-1])
        result.append(
            {
                "host": address,
                "port": int(port_hex, 16),
                "socket_inode": fields[9],
            }
        )
    return result


def socket_fds(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in (Path("/proc") / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            result[target[8:-1]] = int(entry.name)
    return result


def find_listener(pgid: int, host: str, port: int) -> dict[str, Any] | None:
    candidates = {
        str(item["socket_inode"])
        for item in tcp_listeners()
        if item["host"] == host and item["port"] == port
    }
    if not candidates:
        return None
    for member in group_members(pgid):
        pid = int(member["pid"])
        try:
            descriptors = socket_fds(pid)
        except OSError:
            continue
        for inode in sorted(candidates):
            if inode in descriptors:
                return {
                    "host": host,
                    "port": port,
                    "socket_inode": inode,
                    "fd": descriptors[inode],
                    "owning_pid": pid,
                    "owning_pgid": pgid,
                }
    return None


def endpoint_listener_owners(host: str, port: int) -> list[dict[str, Any]]:
    """Return coherent live owner evidence for one IPv4 listening endpoint."""

    candidates = {
        str(item["socket_inode"])
        for item in tcp_listeners()
        if item["host"] == host and item["port"] == port
    }
    if not candidates:
        return []
    owners: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            values = process_stat(pid)
            descriptors = socket_fds(pid)
        except (OSError, ProcessLookupError, ValueError):
            continue
        if values["state"] == "Z":
            continue
        for inode in sorted(candidates):
            if inode in descriptors:
                owners.append(
                    {
                        "pid": pid,
                        "process_start_identity": values["start_identity"],
                        "pgid": values["pgid"],
                        "sid": values["sid"],
                        "socket_inode": inode,
                        "fd": descriptors[inode],
                    }
                )
    return sorted(owners, key=lambda value: (value["pid"], value["fd"]))


def matching_recorded_processes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Find complete executable/argv matches without treating them as owned."""

    executable = record.get("executable")
    argv = record.get("argv")
    if (
        not isinstance(executable, str)
        or not executable
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) for value in argv)
    ):
        return []
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            observed = process_identity(int(entry.name))
        except (OSError, ProcessLookupError, ValueError):
            continue
        if observed.get("executable") == executable and observed.get("argv") == argv:
            matches.append(
                {
                    "pid": observed["pid"],
                    "process_start_identity": observed[
                        "process_start_identity"
                    ],
                    "pgid": observed["pgid"],
                    "sid": observed["sid"],
                }
            )
    return sorted(matches, key=lambda value: value["pid"])


def listener_matches(listener: dict[str, Any], pgid: int) -> bool:
    observed = find_listener(pgid, str(listener["host"]), int(listener["port"]))
    if observed is None:
        return False
    return (
        str(observed["socket_inode"]) == str(listener.get("socket_inode"))
        and int(observed["fd"]) == int(listener.get("fd"))
        and int(observed["owning_pid"]) == int(listener.get("owning_pid"))
    )


def wait_for_listener(
    process: subprocess.Popen[bytes],
    identity: dict[str, Any],
    host: str,
    port: int,
    timeout: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise ControllerError(
                "PROCESS_EXITED_EARLY",
                f"service process exited before listener establishment with status {return_code}",
                runtime={"exit_status": return_code},
            )
        listener = find_listener(int(identity["pgid"]), host, port)
        if listener is not None:
            return listener
        time.sleep(0.05)
    raise ControllerError("PROCESS_START_FAILED", "service listener was not established before timeout")


def make_record(
    record_type: str,
    transaction_id: str,
    lifecycle_state: str,
    values: dict[str, Any],
    plan: dict[str, Any],
    paths: dict[str, Path],
    created_timestamp: str,
    identity: dict[str, Any] | None = None,
    listener: dict[str, Any] | None = None,
    cleanup: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ownership = identity or {}
    transaction_path = paths["transaction_root"] / f"{transaction_id}.json"
    log_path = paths["log_root"] / f"{transaction_id}.log"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "transaction_id": transaction_id,
        "lifecycle_state": lifecycle_state,
        "pid": ownership.get("pid"),
        "process_start_identity": ownership.get("process_start_identity"),
        "pgid": ownership.get("pgid"),
        "sid": ownership.get("sid"),
        "executable": ownership.get("executable", plan["executable"]),
        "executable_device": ownership.get("executable_device"),
        "executable_inode": ownership.get("executable_inode"),
        "argv": ownership.get("argv", plan["argv"]),
        "host": values["host"],
        "port": values["port"],
        "authentication_enabled": values["authentication_enabled"],
        "request_max_body_bytes": values["request_max_body_bytes"],
        "request_max_total_tokens": values["request_max_total_tokens"],
        "request_timeout_seconds": values["request_timeout_seconds"],
        "request_concurrency_limit_per_key": values[
            "request_concurrency_limit_per_key"
        ],
        "request_rate_limit_requests_per_key": values[
            "request_rate_limit_requests_per_key"
        ],
        "request_rate_limit_window_seconds": values[
            "request_rate_limit_window_seconds"
        ],
        "private_backend_host": values["private_backend_host"],
        "private_backend_port": values["private_backend_port"],
        "private_backend_enabled": values["private_backend_enabled"],
        "private_backend_models_max": values["private_backend_models_max"],
        "private_backend_start_timeout_seconds": values[
            "private_backend_start_timeout_seconds"
        ],
        "private_backend_model_timeout_seconds": values[
            "private_backend_model_timeout_seconds"
        ],
        "private_backend_inference_timeout_seconds": values[
            "private_backend_inference_timeout_seconds"
        ],
        "private_backend_poll_interval_seconds": values[
            "private_backend_poll_interval_seconds"
        ],
        "registry_enabled": values["registry_enabled"],
        "registry_reconcile_interval_seconds": values[
            "registry_reconcile_interval_seconds"
        ],
        "registry_watch_debounce_milliseconds": values[
            "registry_watch_debounce_milliseconds"
        ],
        "registry_stability_samples": values["registry_stability_samples"],
        "registry_stability_interval_seconds": values[
            "registry_stability_interval_seconds"
        ],
        "registry_database_busy_timeout_milliseconds": values[
            "registry_database_busy_timeout_milliseconds"
        ],
        "registry_default_alias": values["registry_default_alias"],
        "startup_model_policy": values["startup_model_policy"],
        "automatic_recovery_enabled": values[
            "automatic_recovery_enabled"
        ],
        "recovery_delay_initial_seconds": values[
            "recovery_delay_initial_seconds"
        ],
        "recovery_delay_maximum_seconds": values[
            "recovery_delay_maximum_seconds"
        ],
        "recovery_delay_multiplier": values[
            "recovery_delay_multiplier"
        ],
        "recovery_maximum_attempts_in_window": values[
            "recovery_maximum_attempts_in_window"
        ],
        "recovery_attempt_window_seconds": values[
            "recovery_attempt_window_seconds"
        ],
        "recovery_stable_reset_seconds": values[
            "recovery_stable_reset_seconds"
        ],
        "service_control_profile_identity": values[
            "service_control_profile_identity"
        ],
        "service_control_desired_state_path": values[
            "service_control_desired_state_path"
        ],
        "external_static_enabled": values["external_static_enabled"],
        "external_static_distribution_root": values[
            "external_static_distribution_root"
        ],
        "external_static_mount_path": values["external_static_mount_path"],
        "registry_database": str(paths["registry_database"]),
        "authentication_contract": AUTHENTICATION_CONTRACT,
        "authentication_enabled": values["authentication_enabled"],
        "auth_root": str(paths["auth_root"]),
        "credential_database": str(paths["credential_database"]),
        "service_start_timeout_seconds": values[
            "service_start_timeout_seconds"
        ],
        "log_level": values["log_level"],
        "environment_overrides": plan["environment_overrides"],
        "listener": listener,
        "log_path": str(log_path),
        "status_path": str(paths["service_status"]),
        "pid_path": str(paths["active_pid"]),
        "lock_path": str(paths["active_lock"]),
        "transaction_path": str(transaction_path),
        "created_timestamp_utc": created_timestamp,
        "updated_timestamp_utc": utc_now(),
        "cleanup": cleanup,
        "failure": failure,
    }


def write_lifecycle_records(
    paths: dict[str, Path],
    transaction_id: str,
    lifecycle_state: str,
    values: dict[str, Any],
    plan: dict[str, Any],
    created_timestamp: str,
    identity: dict[str, Any] | None,
    listener: dict[str, Any] | None,
    include_active_pid: bool,
    include_active_lock: bool,
    cleanup: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> None:
    transaction_path = paths["transaction_root"] / f"{transaction_id}.json"
    if include_active_lock:
        atomic_write_json(
            paths["active_lock"],
            make_record(
                "active_lock",
                transaction_id,
                lifecycle_state,
                values,
                plan,
                paths,
                created_timestamp,
                identity,
                listener,
                cleanup,
                failure,
            ),
        )
    if include_active_pid:
        atomic_write_json(
            paths["active_pid"],
            make_record(
                "active_pid",
                transaction_id,
                lifecycle_state,
                values,
                plan,
                paths,
                created_timestamp,
                identity,
                listener,
                cleanup,
                failure,
            ),
        )
    atomic_write_json(
        paths["service_status"],
        make_record(
            "service_status",
            transaction_id,
            lifecycle_state,
            values,
            plan,
            paths,
            created_timestamp,
            identity,
            listener,
            cleanup,
            failure,
        ),
    )
    atomic_write_json(
        transaction_path,
        make_record(
            "transaction",
            transaction_id,
            lifecycle_state,
            values,
            plan,
            paths,
            created_timestamp,
            identity,
            listener,
            cleanup,
            failure,
        ),
    )


def result(
    path_map: dict[str, Path],
    operation: str,
    ok: bool,
    reason_code: str,
    message: str,
    **sections: Any,
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "ok": ok,
        "reason_code": reason_code,
        "message": message,
        "timestamp_utc": utc_now(),
        "branch_root": str(path_map["branch_root"]),
        "api_service_root": str(path_map["api_service_root"]),
        "runtime_api_root": str(path_map["runtime_api_root"]),
    }
    value.update(sections)
    return value


def path_section(paths: dict[str, Path], transaction_id: str | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "active_lock": str(paths["active_lock"]),
        "active_pid": str(paths["active_pid"]),
        "service_status": str(paths["service_status"]),
        "database_root": str(paths["database_root"]),
        "registry_database": str(paths["registry_database"]),
        "auth_root": str(paths["auth_root"]),
        "credential_database": str(paths["credential_database"]),
        "credential_handoff_root": str(paths["credential_handoff_root"]),
    }
    if transaction_id:
        values["log"] = str(paths["log_root"] / f"{transaction_id}.log")
        values["transaction"] = str(paths["transaction_root"] / f"{transaction_id}.json")
    return values


ALIAS_TRANSACTION_SCHEMA = "system-x.gguf-alias-transaction.v1"
ALIAS_TRANSACTION_INPUT_LIMIT = 65_536
ALIAS_TRANSACTION_KEYS = frozenset(
    {
        "schema_version",
        "action",
        "promotion_transaction_id",
        "alias",
        "expected_current_target",
        "new_target",
        "expected_registry_generation",
        "target_artifact_version_id",
        "target_capability_manifest_identity",
        "target_relative_root",
        "promotion_alias_event_identity",
    }
)


def decode_alias_transaction_request(raw: bytes) -> dict[str, Any]:
    """Decode one exact bounded administrative request from standard input."""

    if not raw or len(raw) > ALIAS_TRANSACTION_INPUT_LIMIT:
        raise ControllerError(
            "ALIAS_TRANSACTION_INVALID",
            "alias transaction input is absent or exceeds its bound",
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError(
            "ALIAS_TRANSACTION_INVALID",
            "alias transaction input is not one UTF-8 JSON value",
        ) from exc
    if not isinstance(value, dict) or frozenset(value) != ALIAS_TRANSACTION_KEYS:
        raise ControllerError(
            "ALIAS_TRANSACTION_INVALID",
            "alias transaction input fields are not exact",
        )
    if value.get("schema_version") != ALIAS_TRANSACTION_SCHEMA:
        raise ControllerError(
            "ALIAS_TRANSACTION_INVALID",
            "alias transaction schema is not accepted",
        )
    return value


def operation_alias_transaction(
    paths: dict[str, Path],
    raw: bytes,
) -> dict[str, Any]:
    """Run the branch-owned registry CAS without exposing an HTTP route."""

    dependency = validate_dependency(paths)
    validate_runtime_layout(paths)
    request = decode_alias_transaction_request(raw)
    source_root = str(paths["application_source_root"])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        from system_x_gguf_api.registry_store import (
            AliasTransactionConflict,
            RegistryStore,
            RegistryStoreError,
        )

        async def execute() -> dict[str, Any]:
            store = RegistryStore(paths["registry_database"], 5_000)
            await store.initialize()
            return await store.compare_and_swap_default_alias(
                action=request["action"],
                promotion_transaction_id=request[
                    "promotion_transaction_id"
                ],
                alias=request["alias"],
                expected_current_target=request[
                    "expected_current_target"
                ],
                new_target=request["new_target"],
                expected_registry_generation=request[
                    "expected_registry_generation"
                ],
                target_artifact_version_id=request[
                    "target_artifact_version_id"
                ],
                target_capability_manifest_identity=request[
                    "target_capability_manifest_identity"
                ],
                target_relative_root=request["target_relative_root"],
                promotion_alias_event_identity=request[
                    "promotion_alias_event_identity"
                ],
            )

        alias_result = asyncio.run(execute())
    except AliasTransactionConflict as exc:
        raise ControllerError(
            "ALIAS_TRANSACTION_CONFLICT",
            str(exc),
        ) from exc
    except RegistryStoreError as exc:
        raise ControllerError(
            "ALIAS_TRANSACTION_INVALID",
            str(exc),
        ) from exc
    return result(
        paths,
        "alias-transaction",
        True,
        "OK",
        "default alias transaction committed",
        alias_transaction=alias_result,
        dependency=dependency,
        paths={
            "registry_database": str(paths["registry_database"]),
        },
    )


def operation_plan(paths: dict[str, Path], namespace: argparse.Namespace) -> dict[str, Any]:
    dependency = validate_dependency(paths)
    validate_runtime_layout(paths)
    values = validated_input(namespace)
    plan = build_plan(paths, values)
    reconciliation = None
    if paths["active_lock"].exists() or paths["active_pid"].exists():
        reconciliation = operation_reconcile(paths)
    if paths["active_lock"].exists():
        raise ControllerError(
            "SERVICE_LOCK_ACTIVE",
            "active service lock already exists",
            paths=path_section(paths),
        )
    if paths["active_pid"].exists():
        raise ControllerError(
            "SERVICE_STATE_INCONSISTENT",
            "active PID record exists without an active lock",
            paths=path_section(paths),
        )
    if not endpoint_available(values["host"], values["port"]):
        raise ControllerError(
            "ENDPOINT_IN_USE",
            "requested endpoint is not available",
            input=values,
            paths=path_section(paths),
        )
    if values["private_backend_enabled"] and not endpoint_available(
        values["private_backend_host"], values["private_backend_port"]
    ):
        raise ControllerError(
            "ENDPOINT_IN_USE",
            "requested private backend endpoint is not available",
            input=values,
            paths=path_section(paths),
        )
    return result(
        paths,
        "plan",
        True,
        "OK",
        "launch plan constructed without runtime mutation",
        input=values,
        dependency=dependency,
        plan=plan,
        paths=path_section(paths),
    )


def cleanup_failed_start(
    paths: dict[str, Path],
    transaction_id: str,
    values: dict[str, Any],
    plan: dict[str, Any],
    created_timestamp: str,
    identity: dict[str, Any] | None,
    original: ControllerError,
) -> dict[str, Any]:
    cleanup: dict[str, Any] = {
        "signal": None,
        "force_used": False,
        "owned_group_gone": True,
        "active_pid_removed": False,
        "active_lock_removed": False,
    }
    if identity is not None:
        matched, _ = identity_matches(identity)
        members = group_members(int(identity["pgid"]))
        if matched and members:
            os.killpg(int(identity["pgid"]), signal.SIGTERM)
            cleanup["signal"] = "SIGTERM"
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and group_members(int(identity["pgid"])):
                time.sleep(0.05)
            if group_members(int(identity["pgid"])):
                matched_again, _ = identity_matches(identity)
                if matched_again:
                    os.killpg(int(identity["pgid"]), signal.SIGKILL)
                    cleanup["signal"] = "SIGTERM,SIGKILL"
                    cleanup["force_used"] = True
                    deadline = time.monotonic() + 3.0
                    while time.monotonic() < deadline and group_members(int(identity["pgid"])):
                        time.sleep(0.05)
            cleanup["owned_group_gone"] = not group_members(int(identity["pgid"]))
    failure = {"reason_code": original.reason_code, "message": original.message}
    write_lifecycle_records(
        paths,
        transaction_id,
        "FAILED",
        values,
        plan,
        created_timestamp,
        identity,
        None,
        paths["active_pid"].exists(),
        True,
        cleanup,
        failure,
    )
    cleanup["active_pid_removed"] = unlink_transaction_owned(paths["active_pid"], transaction_id)
    cleanup["active_lock_removed"] = unlink_transaction_owned(paths["active_lock"], transaction_id)
    return cleanup


def operation_start(paths: dict[str, Path], namespace: argparse.Namespace) -> dict[str, Any]:
    dependency = validate_dependency(paths)
    validate_runtime_layout(paths)
    values = validated_input(namespace)
    plan = build_plan(paths, values)
    if paths["active_lock"].exists():
        raise ControllerError("SERVICE_LOCK_ACTIVE", "active service lock already exists", paths=path_section(paths))
    if paths["active_pid"].exists():
        raise ControllerError(
            "SERVICE_STATE_INCONSISTENT",
            "active PID record exists without an active lock",
            paths=path_section(paths),
        )
    transaction_id = f"tx-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(6)}"
    plan["environment_overrides"][
        "SYSTEM_X_API_SERVICE_TRANSACTION_ID"
    ] = transaction_id
    created_timestamp = utc_now()
    initial = make_record(
        "active_lock",
        transaction_id,
        "PREPARING",
        values,
        plan,
        paths,
        created_timestamp,
    )
    acquire_json_lock(paths["active_lock"], initial)
    identity: dict[str, Any] | None = None
    log_descriptor: int | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        if paths["active_pid"].exists():
            raise ControllerError("SERVICE_STATE_INCONSISTENT", "active PID record appeared while lock was held")
        if not endpoint_available(values["host"], int(values["port"])):
            unlink_transaction_owned(paths["active_lock"], transaction_id)
            raise ControllerError(
                "ENDPOINT_IN_USE",
                "requested endpoint is not available",
                input=values,
                paths=path_section(paths),
            )
        log_path = paths["log_root"] / f"{transaction_id}.log"
        log_descriptor = create_log(log_path)
        write_lifecycle_records(
            paths,
            transaction_id,
            "PREPARING",
            values,
            plan,
            created_timestamp,
            None,
            None,
            False,
            True,
        )
        process = subprocess.Popen(
            plan["argv"],
            executable=plan["executable"],
            cwd=plan["cwd"],
            env=child_environment(plan),
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=subprocess.STDOUT,
            close_fds=True,
            shell=False,
            start_new_session=True,
        )
        identity = wait_for_expected_identity(process.pid, plan)
        if identity["pid"] != identity["pgid"] or identity["pid"] != identity["sid"]:
            raise ControllerError(
                "PROCESS_IDENTITY_MISMATCH",
                "service process is not its process-group and session leader",
                ownership=identity,
            )
        write_lifecycle_records(
            paths,
            transaction_id,
            "STARTING",
            values,
            plan,
            created_timestamp,
            identity,
            None,
            True,
            True,
        )
        listener = wait_for_listener(
            process,
            identity,
            values["host"],
            int(values["port"]),
            timeout=values["service_start_timeout_seconds"],
        )
        write_lifecycle_records(
            paths,
            transaction_id,
            "STARTED",
            values,
            plan,
            created_timestamp,
            identity,
            listener,
            True,
            True,
        )
        return result(
            paths,
            "start",
            True,
            "OK",
            "API service started with verified process and listener ownership",
            input=values,
            dependency=dependency,
            plan=plan,
            runtime={
                "active": True,
                "consistent": True,
                "lifecycle_state": "STARTED",
                "transaction_id": transaction_id,
                **identity,
            },
            listener=listener,
            paths=path_section(paths, transaction_id),
        )
    except ControllerError as original:
        if paths["active_lock"].exists():
            cleanup = cleanup_failed_start(
                paths,
                transaction_id,
                values,
                plan,
                created_timestamp,
                identity,
                original,
            )
            original.sections["cleanup"] = cleanup
            original.sections["paths"] = path_section(paths, transaction_id)
        raise
    finally:
        if log_descriptor is not None:
            os.close(log_descriptor)


def active_records(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_exists = paths["active_lock"].exists()
    pid_exists = paths["active_pid"].exists()
    if not lock_exists and not pid_exists:
        raise ControllerError("SERVICE_NOT_ACTIVE", "service has no active lock or PID record")
    if lock_exists != pid_exists:
        raise ControllerError("SERVICE_STATE_INCONSISTENT", "active lock and PID record presence differs")
    lock_record = read_json(paths["active_lock"])
    pid_record = read_json(paths["active_pid"])
    if lock_record.get("transaction_id") != pid_record.get("transaction_id"):
        raise ControllerError("SERVICE_STATE_INCONSISTENT", "active lock and PID transaction IDs differ")
    return lock_record, pid_record


def verify_active_ownership(
    paths: dict[str, Path],
    lock_record: dict[str, Any],
    pid_record: dict[str, Any],
    *,
    require_listener: bool = True,
    allow_process_absent: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    transaction_id = str(pid_record["transaction_id"])
    status = read_json(paths["service_status"])
    transaction_path = paths["transaction_root"] / f"{transaction_id}.json"
    transaction = read_json(transaction_path)
    if status.get("transaction_id") != transaction_id or transaction.get("transaction_id") != transaction_id:
        raise ControllerError("SERVICE_STATE_INCONSISTENT", "status or transaction correspondence failed")
    identity_keys = (
        "pid",
        "process_start_identity",
        "pgid",
        "sid",
        "executable",
        "executable_device",
        "executable_inode",
        "argv",
        "host",
        "port",
    )
    if any(lock_record.get(key) != pid_record.get(key) for key in identity_keys):
        raise ControllerError("SERVICE_STATE_INCONSISTENT", "active lock and PID ownership fields differ")
    matched, observed = identity_matches(pid_record)
    if not matched:
        if allow_process_absent and observed is None:
            return {}, {}
        raise ControllerError(
            "PROCESS_IDENTITY_MISMATCH",
            "active process identity does not match its PID record",
            ownership={"recorded": pid_record, "observed": observed},
        )
    listener = pid_record.get("listener")
    listener_matches_now = bool(
        isinstance(listener, dict)
        and listener_matches(listener, int(pid_record["pgid"]))
    )
    if require_listener and not listener_matches_now:
        raise ControllerError(
            "LISTENER_OWNERSHIP_MISMATCH",
            "recorded listener is absent or no longer owned by the recorded process group",
            listener={"recorded": listener},
        )
    return observed or {}, listener if isinstance(listener, dict) else {}


def operation_status(paths: dict[str, Path]) -> dict[str, Any]:
    validate_dependency(paths)
    validate_runtime_layout(paths)
    lock_exists = paths["active_lock"].exists()
    pid_exists = paths["active_pid"].exists()
    if not lock_exists and not pid_exists:
        status: dict[str, Any] | None = None
        if paths["service_status"].exists():
            status = read_json(paths["service_status"])
        return result(
            paths,
            "status",
            True,
            "OK",
            "service is inactive and active state is consistent",
            runtime={
                "active": False,
                "consistent": True,
                "lifecycle_state": status.get("lifecycle_state") if status else None,
                "transaction_id": status.get("transaction_id") if status else None,
                "pid": None,
                "pgid": None,
                "sid": None,
            },
            listener=None,
            paths=path_section(paths, str(status["transaction_id"])) if status else path_section(paths),
        )
    lock_record, pid_record = active_records(paths)
    observed, listener = verify_active_ownership(paths, lock_record, pid_record)
    return result(
        paths,
        "status",
        True,
        "OK",
        "service is active with consistent process and listener ownership",
        runtime={
            "active": True,
            "consistent": True,
            "lifecycle_state": pid_record.get("lifecycle_state"),
            "transaction_id": pid_record.get("transaction_id"),
            **observed,
        },
        listener=listener,
        paths=path_section(paths, str(pid_record["transaction_id"])),
    )


def operation_reconcile(paths: dict[str, Path]) -> dict[str, Any]:
    """Remove only provably stale controller-owned API active records."""

    validate_dependency(paths)
    validate_runtime_layout(paths)
    lock_present = (
        paths["active_lock"].exists() or paths["active_lock"].is_symlink()
    )
    pid_present = (
        paths["active_pid"].exists() or paths["active_pid"].is_symlink()
    )
    if not lock_present and not pid_present:
        return result(
            paths,
            "reconcile",
            True,
            "OK",
            "API active state is already inactive and consistent",
            runtime={
                "active": False,
                "consistent": True,
                "reconciled": False,
            },
            paths=path_section(paths),
        )

    lock_record = read_json(paths["active_lock"]) if lock_present else None
    pid_record = read_json(paths["active_pid"]) if pid_present else None
    for record, expected_type in (
        (lock_record, "active_lock"),
        (pid_record, "active_pid"),
    ):
        if record is None:
            continue
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("record_type") != expected_type
        ):
            raise ControllerError(
                "OWNERSHIP_UNCERTAIN",
                "active API record controller identity is invalid",
            )
    transaction_ids = {
        str(record.get("transaction_id"))
        for record in (lock_record, pid_record)
        if record is not None
    }
    if len(transaction_ids) != 1 or "None" in transaction_ids:
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "active API transaction identities are ambiguous",
        )
    transaction_id = next(iter(transaction_ids))
    transaction_path = paths["transaction_root"] / f"{transaction_id}.json"
    transaction = (
        read_json(transaction_path) if transaction_path.exists() else None
    )
    if transaction is not None and (
        transaction.get("schema_version") != SCHEMA_VERSION
        or transaction.get("record_type") != "transaction"
        or transaction.get("transaction_id") != transaction_id
    ):
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "API transaction history does not validate for active records",
        )
    record = pid_record or lock_record
    assert record is not None
    host = record.get("host")
    port = record.get("port")
    if not isinstance(host, str) or not isinstance(port, int):
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "active API record lacks an exact endpoint",
        )

    matched, observed = identity_matches(record)
    endpoint_owners = endpoint_listener_owners(host, port)
    if matched and observed is not None:
        listener = find_listener(int(observed["pgid"]), host, port)
        if listener is not None:
            return result(
                paths,
                "reconcile",
                True,
                "OK",
                "API active process and listener remain consistently owned",
                runtime={
                    "active": True,
                    "consistent": True,
                    "reconciled": False,
                    **observed,
                },
                listener=listener,
                paths=path_section(paths, transaction_id),
            )
        foreign = [
            owner
            for owner in endpoint_owners
            if owner.get("pgid") != observed.get("pgid")
        ]
        if foreign:
            raise ControllerError(
                "ENDPOINT_CONFLICT",
                "public endpoint is owned by a foreign process",
                listener={
                    "foreign_endpoint_owners": foreign,
                    "unrelated_process_signaled": False,
                },
            )
        return result(
            paths,
            "reconcile",
            True,
            "PUBLIC_LISTENER_LOST",
            "owned API process is alive but its public listener is absent",
            runtime={
                "active": True,
                "consistent": False,
                "reconciled": False,
                **observed,
            },
            listener={
                "recorded": record.get("listener"),
                "present": False,
                "unrelated_process_signaled": False,
            },
            recovery={"selected_action": "CONTROLLER_STOP_RESTART"},
            paths=path_section(paths, transaction_id),
        )

    if observed is not None:
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "recorded API PID is alive but exact start identity does not match",
            ownership={"recorded": record, "observed": observed},
        )
    matches = matching_recorded_processes(record)
    if matches:
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "complete API executable/argv match remains alive under another identity",
            ownership={"matching_processes": matches},
        )
    if endpoint_owners:
        raise ControllerError(
            "ENDPOINT_CONFLICT",
            "public endpoint has a foreign live owner",
            listener={
                "foreign_endpoint_owners": endpoint_owners,
                "unrelated_process_signaled": False,
            },
        )

    removed: list[str] = []
    if pid_present and unlink_transaction_owned(
        paths["active_pid"], transaction_id
    ):
        removed.append("active_pid")
    if lock_present and unlink_transaction_owned(
        paths["active_lock"], transaction_id
    ):
        removed.append("active_lock")
    reconciliation = {
        "reason_code": "API_STATE_STALE",
        "reconciled_utc": utc_now(),
        "removed_records": removed,
        "process_absent": True,
        "listener_absent": True,
        "unrelated_process_signaled": False,
    }
    for record_path in (transaction_path, paths["service_status"]):
        if not record_path.exists():
            continue
        historical = read_json(record_path)
        if historical.get("transaction_id") != transaction_id:
            raise ControllerError(
                "OWNERSHIP_UNCERTAIN",
                "historical API state changed ownership during reconciliation",
            )
        historical["lifecycle_state"] = "RECONCILED"
        historical["updated_timestamp_utc"] = utc_now()
        historical["reconciliation"] = reconciliation
        atomic_write_json(record_path, historical)
    return result(
        paths,
        "reconcile",
        True,
        "API_STATE_STALE",
        "stale controller-owned API active records reconciled",
        runtime={
            "active": False,
            "consistent": True,
            "reconciled": True,
            "transaction_id": transaction_id,
        },
        reconciliation=reconciliation,
        paths=path_section(paths, transaction_id),
    )

def _reconciled_absent_stop(
    paths: dict[str, Path], transaction_id: str
) -> dict[str, Any]:
    reconciliation_result = operation_reconcile(paths)
    runtime = reconciliation_result.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("active") is not False:
        raise ControllerError(
            "OWNERSHIP_UNCERTAIN",
            "API active state changed before absent-process reconciliation completed",
            reconciliation=reconciliation_result,
        )
    reconciliation = reconciliation_result.get("reconciliation", {})
    if not isinstance(reconciliation, dict):
        reconciliation = {}
    removed_records = reconciliation.get("removed_records", [])
    if not isinstance(removed_records, list):
        removed_records = []
    cleanup = {
        "signal": None,
        "force_used": False,
        "owned_process_group_gone": True,
        "listener_absent": True,
        "active_pid_removed": "active_pid" in removed_records,
        "active_lock_removed": "active_lock" in removed_records,
        "unrelated_process_signaled": False,
    }
    return result(
        paths,
        "stop",
        True,
        str(reconciliation_result.get("reason_code", "API_STATE_STALE")),
        "owned API-service process was already absent; stale controller records reconciled",
        runtime={
            "active": False,
            "consistent": True,
            "lifecycle_state": "RECONCILED",
            "transaction_id": transaction_id,
            "pid": None,
            "pgid": None,
            "sid": None,
            "process_start_identity": None,
        },
        listener={
            "recorded": None,
            "absent": True,
            "unrelated_process_signaled": False,
        },
        cleanup=cleanup,
        reconciliation=reconciliation,
        paths=path_section(paths, transaction_id),
    )


def operation_stop(paths: dict[str, Path]) -> dict[str, Any]:
    validate_dependency(paths)
    validate_runtime_layout(paths)
    lock_record, pid_record = active_records(paths)
    observed, listener = verify_active_ownership(
        paths,
        lock_record,
        pid_record,
        require_listener=False,
        allow_process_absent=True,
    )
    transaction_id = str(pid_record["transaction_id"])
    if not observed:
        return _reconciled_absent_stop(paths, transaction_id)
    endpoint_owners = endpoint_listener_owners(
        str(pid_record["host"]), int(pid_record["port"])
    )
    foreign_owners = [
        owner
        for owner in endpoint_owners
        if owner.get("pgid") != observed.get("pgid")
    ]
    if foreign_owners:
        raise ControllerError(
            "ENDPOINT_CONFLICT",
            "public endpoint is owned by a foreign process",
            listener={
                "foreign_endpoint_owners": foreign_owners,
                "unrelated_process_signaled": False,
            },
        )
    values = {
        "host": pid_record["host"],
        "authentication_enabled": pid_record.get("authentication_enabled", True),
        "request_max_body_bytes": pid_record.get("request_max_body_bytes", 2_097_152),
        "request_max_total_tokens": pid_record.get("request_max_total_tokens", 32_768),
        "request_timeout_seconds": pid_record.get("request_timeout_seconds", 120.0),
        "request_concurrency_limit_per_key": pid_record.get(
            "request_concurrency_limit_per_key", 2
        ),
        "request_rate_limit_requests_per_key": pid_record.get(
            "request_rate_limit_requests_per_key", 60
        ),
        "request_rate_limit_window_seconds": pid_record.get(
            "request_rate_limit_window_seconds", 60.0
        ),
        "port": pid_record["port"],
        "private_backend_host": pid_record["private_backend_host"],
        "private_backend_port": pid_record["private_backend_port"],
        "private_backend_enabled": pid_record["private_backend_enabled"],
        "private_backend_models_max": pid_record["private_backend_models_max"],
        "private_backend_start_timeout_seconds": pid_record[
            "private_backend_start_timeout_seconds"
        ],
        "private_backend_model_timeout_seconds": pid_record[
            "private_backend_model_timeout_seconds"
        ],
        "private_backend_inference_timeout_seconds": pid_record[
            "private_backend_inference_timeout_seconds"
        ],
        "private_backend_poll_interval_seconds": pid_record[
            "private_backend_poll_interval_seconds"
        ],
        "registry_enabled": pid_record["registry_enabled"],
        "registry_reconcile_interval_seconds": pid_record[
            "registry_reconcile_interval_seconds"
        ],
        "registry_watch_debounce_milliseconds": pid_record[
            "registry_watch_debounce_milliseconds"
        ],
        "registry_stability_samples": pid_record["registry_stability_samples"],
        "registry_stability_interval_seconds": pid_record[
            "registry_stability_interval_seconds"
        ],
        "registry_database_busy_timeout_milliseconds": pid_record[
            "registry_database_busy_timeout_milliseconds"
        ],
        "registry_default_alias": pid_record["registry_default_alias"],
        "startup_model_policy": pid_record["startup_model_policy"],
        "automatic_recovery_enabled": pid_record[
            "automatic_recovery_enabled"
        ],
        "recovery_delay_initial_seconds": pid_record[
            "recovery_delay_initial_seconds"
        ],
        "recovery_delay_maximum_seconds": pid_record[
            "recovery_delay_maximum_seconds"
        ],
        "recovery_delay_multiplier": pid_record[
            "recovery_delay_multiplier"
        ],
        "recovery_maximum_attempts_in_window": pid_record[
            "recovery_maximum_attempts_in_window"
        ],
        "recovery_attempt_window_seconds": pid_record[
            "recovery_attempt_window_seconds"
        ],
        "recovery_stable_reset_seconds": pid_record[
            "recovery_stable_reset_seconds"
        ],
        "service_control_profile_identity": pid_record[
            "service_control_profile_identity"
        ],
        "service_control_desired_state_path": pid_record[
            "service_control_desired_state_path"
        ],
        "external_static_enabled": pid_record["external_static_enabled"],
        "external_static_distribution_root": pid_record[
            "external_static_distribution_root"
        ],
        "external_static_mount_path": pid_record[
            "external_static_mount_path"
        ],
        "service_start_timeout_seconds": pid_record[
            "service_start_timeout_seconds"
        ],
        "log_level": pid_record["log_level"],
    }
    plan = {
        "executable": pid_record["executable"],
        "argv": pid_record["argv"],
        "cwd": str(paths["api_service_root"]),
        "shell": False,
        "start_new_session": True,
        "workers": 1,
        "reload": False,
        "access_log": False,
        "listener_start_timeout_seconds": pid_record[
            "service_start_timeout_seconds"
        ],
        "environment_overrides": pid_record["environment_overrides"],
    }
    created_timestamp = str(pid_record["created_timestamp_utc"])
    stopping_cleanup = {"signal": None, "force_used": False}
    write_lifecycle_records(
        paths,
        transaction_id,
        "STOPPING",
        values,
        plan,
        created_timestamp,
        observed,
        listener,
        True,
        True,
        stopping_cleanup,
    )
    matched, reobserved = identity_matches(pid_record)
    if not matched:
        if reobserved is None:
            return _reconciled_absent_stop(paths, transaction_id)
        raise ControllerError("PROCESS_IDENTITY_MISMATCH", "ownership changed before graceful signal")
    pgid = int(reobserved["pgid"])
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and group_members(pgid):
        time.sleep(0.05)
    remaining = group_members(pgid)
    if remaining:
        raise ControllerError(
            "GRACEFUL_STOP_TIMEOUT",
            "owned process group remains after graceful-stop timeout",
            ownership={"remaining_group_members": remaining},
            cleanup={"signal": "SIGTERM", "force_used": False},
        )
    if find_listener(pgid, str(values["host"]), int(values["port"])) is not None:
        raise ControllerError(
            "OWNED_PROCESS_GROUP_NOT_GONE",
            "owned listener remains after process-group exit",
            cleanup={"signal": "SIGTERM", "force_used": False},
        )
    cleanup = {
        "signal": "SIGTERM",
        "target_pgid": pgid,
        "force_used": False,
        "owned_process_group_gone": True,
        "listener_absent": True,
        "active_pid_removed": False,
        "active_lock_removed": False,
    }
    cleanup["active_pid_removed"] = unlink_transaction_owned(paths["active_pid"], transaction_id)
    cleanup["active_lock_removed"] = unlink_transaction_owned(paths["active_lock"], transaction_id)
    write_lifecycle_records(
        paths,
        transaction_id,
        "STOPPED",
        values,
        plan,
        created_timestamp,
        observed,
        listener,
        False,
        False,
        cleanup,
    )
    return result(
        paths,
        "stop",
        True,
        "OK",
        "owned API-service process group stopped gracefully",
        runtime={
            "active": False,
            "consistent": True,
            "lifecycle_state": "STOPPED",
            "transaction_id": transaction_id,
            "pid": observed["pid"],
            "pgid": observed["pgid"],
            "sid": observed["sid"],
            "process_start_identity": observed["process_start_identity"],
        },
        listener={"recorded": listener, "absent": True},
        cleanup=cleanup,
        paths=path_section(paths, transaction_id),
    )


def parser() -> JsonArgumentParser:
    root = JsonArgumentParser(prog="controller.py", add_help=False)
    subparsers = root.add_subparsers(dest="operation", required=True)
    for operation in ("plan", "start"):
        command = subparsers.add_parser(operation, add_help=False)
        command.add_argument("--host", required=True)
        command.add_argument("--port", required=True)
        command.add_argument("--authentication-enabled", default="true")
        command.add_argument(
            "--request-max-body-bytes", default="2097152"
        )
        command.add_argument(
            "--request-max-total-tokens", default="32768"
        )
        command.add_argument(
            "--request-timeout-seconds", default="120.0"
        )
        command.add_argument(
            "--request-concurrency-limit-per-key", default="2"
        )
        command.add_argument(
            "--request-rate-limit-requests-per-key", default="60"
        )
        command.add_argument(
            "--request-rate-limit-window-seconds", default="60.0"
        )
        command.add_argument("--private-backend-host", default="127.0.0.1")
        command.add_argument("--private-backend-port", default=None)
        command.add_argument("--private-backend-enabled", default="false")
        command.add_argument("--private-backend-models-max", default="1")
        command.add_argument(
            "--private-backend-start-timeout-seconds", default="30.0"
        )
        command.add_argument(
            "--private-backend-model-timeout-seconds", default="120.0"
        )
        command.add_argument(
            "--private-backend-inference-timeout-seconds", default="900.0"
        )
        command.add_argument(
            "--private-backend-poll-interval-seconds", default="0.25"
        )
        command.add_argument("--registry-enabled", default="false")
        command.add_argument(
            "--registry-reconcile-interval-seconds", default="30.0"
        )
        command.add_argument(
            "--registry-watch-debounce-milliseconds", default="1600"
        )
        command.add_argument("--registry-stability-samples", default="3")
        command.add_argument(
            "--registry-stability-interval-seconds", default="1.0"
        )
        command.add_argument(
            "--registry-database-busy-timeout-milliseconds", default="5000"
        )
        command.add_argument("--registry-default-alias", default="default")
        command.add_argument(
            "--startup-model-policy", default="always_warm"
        )
        command.add_argument(
            "--automatic-recovery-enabled", default="false"
        )
        command.add_argument(
            "--recovery-delay-initial-seconds", default="0.25"
        )
        command.add_argument(
            "--recovery-delay-maximum-seconds", default="30.0"
        )
        command.add_argument(
            "--recovery-delay-multiplier", default="2.0"
        )
        command.add_argument(
            "--recovery-maximum-attempts-in-window", default="3"
        )
        command.add_argument(
            "--recovery-attempt-window-seconds", default="60.0"
        )
        command.add_argument(
            "--recovery-stable-reset-seconds", default="30.0"
        )
        command.add_argument(
            "--service-control-profile-identity", required=True
        )
        command.add_argument(
            "--service-control-desired-state-path", required=True
        )
        command.add_argument("--external-static-enabled", default="false")
        command.add_argument("--external-static-distribution-root", default=None)
        command.add_argument(
            "--external-static-mount-path",
            default="/ui/chat",
        )
        command.add_argument("--log-level", default="info")
    subparsers.add_parser("status", add_help=False)
    subparsers.add_parser("stop", add_help=False)
    subparsers.add_parser("reconcile", add_help=False)
    subparsers.add_parser("alias-transaction", add_help=False)
    return root


def emit(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    paths = derive_paths()
    arguments = list(sys.argv[1:] if argv is None else argv)
    operation = arguments[0] if arguments and arguments[0] in OPERATIONS else "unknown"
    try:
        namespace = parser().parse_args(arguments)
        operation = namespace.operation
        if operation == "plan":
            output = operation_plan(paths, namespace)
        elif operation == "start":
            output = operation_start(paths, namespace)
        elif operation == "status":
            output = operation_status(paths)
        elif operation == "stop":
            output = operation_stop(paths)
        elif operation == "reconcile":
            output = operation_reconcile(paths)
        elif operation == "alias-transaction":
            output = operation_alias_transaction(
                paths,
                sys.stdin.buffer.read(ALIAS_TRANSACTION_INPUT_LIMIT + 1),
            )
        else:
            raise ControllerError("INVALID_INPUT", "unknown operation")
        emit(output)
        return 0
    except ControllerError as exc:
        emit(
            result(
                paths,
                operation,
                False,
                exc.reason_code,
                exc.message,
                **exc.sections,
            )
        )
        return REASON_EXIT.get(exc.reason_code, 5)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        emit(result(paths, operation, False, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}"))
        return REASON_EXIT["INTERNAL_ERROR"]


if __name__ == "__main__":
    sys.exit(main())
