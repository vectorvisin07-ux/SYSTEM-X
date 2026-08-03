"""Operation contract, ordered state transitions, and receipts."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Callable, Mapping

from . import BOOTSTRAP_VERSION
from .command import Runner, SubprocessRunner
from .config import LoadedConfiguration, canonical_json_bytes, load_registry
from .credentials import credential_status, initialize_credentials
from .environments import build_environments, environment_status
from .errors import BootstrapError, ErrorCode
from .host import HostInspector
from .llama import build_llama_server, initialize_submodules, inspect_vendored_source
from .packages import apply_host, build_host_plan
from .paths import RepositoryPaths
from .result import MachineResult
from .runtime import expand_runtime_layout, initialize_runtime, runtime_status
from .service import register_platform_service, service_status
from .state import StateDocument, read_state, write_failure_state, write_receipt, write_success_state
from .transaction import BootstrapTransaction, incomplete_transactions
from .verify import verify_level


CONFIGURATION_NAMES = (
    "ubuntu-26.04-wsl2-host.json",
    "ubuntu-package.lock.json",
    "cuda-wsl.lock.json",
    "python-environments.lock.json",
    "llama-build.lock.json",
    "runtime-layout.json",
    "credential-initialization.json",
    "service-registration.json",
)


class BootstrapOrchestrator:
    def __init__(
        self,
        paths: RepositoryPaths | None = None,
        *,
        runner: Runner | None = None,
        home: Path | None = None,
    ) -> None:
        self.paths = paths or RepositoryPaths.discover()
        self.runner = runner or SubprocessRunner()
        self.home = home
        self.loaded: dict[str, LoadedConfiguration] = load_registry(self.paths, CONFIGURATION_NAMES)
        self.configs = {name: item.data for name, item in self.loaded.items()}

    def _inspection(self) -> dict[str, Any]:
        packages = self.configs["ubuntu-package.lock.json"]
        cuda = self.configs["cuda-wsl.lock.json"]
        return HostInspector(self.paths.root, runner=self.runner).inspect(
            [item["name"] for item in packages["packages"]], cuda["forbidden_package_patterns"]
        )

    def identify(self) -> MachineResult:
        commit: str | None = None
        if (self.paths.root / ".git").exists():
            result = self.runner(("git", "-C", str(self.paths.root), "rev-parse", "HEAD"), timeout=30)
            if result.returncode == 0:
                commit = result.stdout.strip()
        state = read_state(self.paths)
        return MachineResult(
            "identify", "ok", state.state,
            details={
                "bootstrap_version": BOOTSTRAP_VERSION,
                "repository_root": str(self.paths.root),
                "source_commit": commit,
                "host_profile_identity": self.loaded["ubuntu-26.04-wsl2-host.json"].identity,
                "configuration_identities": {
                    name: {"identity": config.identity, "version": config.version, "sha256": config.sha256}
                    for name, config in sorted(self.loaded.items())
                },
            },
        )

    def inspect_host(self) -> MachineResult:
        inspection = self._inspection()
        return MachineResult("inspect-host", "ok", "HOST_INSPECTED", details=inspection)

    def plan(self) -> MachineResult:
        inspection = self._inspection()
        plan = build_host_plan(
            inspection,
            self.configs["ubuntu-26.04-wsl2-host.json"],
            self.configs["ubuntu-package.lock.json"],
            self.configs["cuda-wsl.lock.json"],
        )
        environment_lock = self.configs["python-environments.lock.json"]
        environment_states = {
            item["environment_identity"]: environment_status(self.paths, environment_lock, item)
            for item in environment_lock["environments"]
        }
        llama_source = inspect_vendored_source(self.paths, self.configs["llama-build.lock.json"])
        layout_entries = expand_runtime_layout(self.paths, self.configs["runtime-layout.json"])
        runtime_physical = any((self.paths.root / item["path"]).exists() for item in layout_entries)
        credential = credential_status(self.paths, self.configs["credential-initialization.json"])
        service = service_status(self.paths, self.configs["service-registration.json"], home=self.home)

        plan["would_create"].extend(
            ["runtime directory layout and empty registry database"] if not runtime_physical else []
        )
        plan["would_build"].extend(
            [f"private environment {name}" for name, state in environment_states.items() if state == "absent"]
        )
        if not llama_source["exact"]:
            plan["blockers"].append("vendored llama.cpp source identity mismatch")
        binary = self.paths.root / self.configs["llama-build.lock.json"]["binary"]
        if not binary.is_file():
            plan["would_build"].append("locked CUDA llama-server")
        if credential["state"] == "absent":
            plan["would_generate"].append("new local primary API credential")
        if not service["unit_present"]:
            plan["would_register"].append("Linux systemd user service through accepted adapter")
        external_states = {name: value for name, value in environment_states.items() if value not in ("absent", "ready")}
        if runtime_physical and runtime_status(self.paths, self.configs["runtime-layout.json"]) != "ready":
            plan["would_leave_external"].append("pre-existing unbound runtime state")
        if credential["state"] == "collision":
            plan["would_leave_external"].append("pre-existing unbound credential state")
        if external_states:
            plan["would_leave_external"].append("pre-existing unbound private environments")
        plan["component_observations"] = {
            "llama_source": llama_source,
            "environments": environment_states,
            "runtime": runtime_status(self.paths, self.configs["runtime-layout.json"]),
            "credential": credential["state"],
            "service": service,
        }
        plan["plan_identity"] = hashlib.sha256(canonical_json_bytes({k: v for k, v in plan.items() if k != "plan_identity"})).hexdigest()
        status = "blocked" if plan["blockers"] else "ok"
        return MachineResult("plan", status, "HOST_PLAN_READY", details=plan)

    def status(self) -> MachineResult:
        state = read_state(self.paths)
        environment_lock = self.configs["python-environments.lock.json"]
        details = {
            "persistent_state": state.as_dict(),
            "incomplete_transactions": [path.name for path in incomplete_transactions(self.paths.transaction_directory)],
            "llama_source": inspect_vendored_source(self.paths, self.configs["llama-build.lock.json"]),
            "environments": {
                item["environment_identity"]: environment_status(self.paths, environment_lock, item)
                for item in environment_lock["environments"]
            },
            "runtime": runtime_status(self.paths, self.configs["runtime-layout.json"]),
            "credential": credential_status(self.paths, self.configs["credential-initialization.json"]),
            "service": service_status(self.paths, self.configs["service-registration.json"], home=self.home),
            "model_loaded": False,
            "api_called": False,
        }
        return MachineResult("status", "ok", state.state, details=details)

    def verify(self, level: str) -> MachineResult:
        details = verify_level(self.paths, self.configs, level, runner=self.runner, home=self.home)
        state = {
            "source-only": "CLONED",
            "host-ready": "HOST_READY",
            "build-ready": "LLAMA_SERVER_BUILT",
            "service-process-ready": "SERVICE_REGISTERED",
            "waiting-for-model": "WAITING_FOR_MODEL",
            "full-ready": "READY",
        }[level]
        return MachineResult("verify", "ok", state, details=details)

    def _mutate(
        self,
        operation: str,
        target: str,
        allowed: tuple[str, ...],
        authorized: bool,
        provider: Callable[[BootstrapTransaction], Mapping[str, Any]],
    ) -> MachineResult:
        if not authorized:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{operation} requires --authorize")
        previous = read_state(self.paths)
        if previous.state in ("FAILED_CLEAN", "FAIL_CLOSED"):
            raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "bootstrap failure state requires explicit recovery")
        pending = incomplete_transactions(self.paths.transaction_directory)
        if pending:
            raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "incomplete bootstrap transaction requires recovery")
        if previous.stable_state not in allowed and previous.stable_state != target:
            raise BootstrapError(
                ErrorCode.PRECONDITION_FAILED,
                "bootstrap operation is out of order",
                context={"operation": operation, "state": previous.stable_state, "allowed": list(allowed)},
            )
        plan_identity = hashlib.sha256(
            canonical_json_bytes(
                {
                    "operation": operation,
                    "target": target,
                    "prestate": previous.as_dict(),
                    "configuration_sha256": {name: value.sha256 for name, value in sorted(self.loaded.items())},
                }
            )
        ).hexdigest()
        transaction = BootstrapTransaction(
            self.paths, operation, plan_identity, previous.as_dict(), True
        )
        receipt_id = secrets.token_hex(16)
        try:
            with transaction:
                try:
                    details = dict(provider(transaction))
                    changed = bool(details.get("changed"))
                    write_receipt(
                        self.paths,
                        receipt_id=receipt_id,
                        operation=operation,
                        prestate=previous,
                        poststate=target,
                        plan_identity=plan_identity,
                        changed=changed,
                        details=details,
                    )
                    write_success_state(self.paths, previous, target, operation, receipt_id)
                    transaction.complete({"receipt_id": receipt_id, "poststate": target})
                except BootstrapError as exc:
                    failure = "FAIL_CLOSED" if exc.code in {
                        ErrorCode.CREDENTIAL_COLLISION, ErrorCode.RUNTIME_COLLISION, ErrorCode.UNKNOWN_STATE,
                        ErrorCode.INTEGRITY_FAILURE, ErrorCode.SECRET_POLICY_VIOLATION,
                    } else "FAILED_CLEAN"
                    write_failure_state(self.paths, previous, failure, operation)
                    raise
        except BootstrapError:
            raise
        return MachineResult(operation, "ok", target, changed=changed, details=details, receipt_id=receipt_id)

    def apply_host(self, *, authorized: bool, allow_patch_difference: bool = False) -> MachineResult:
        inspection = self._inspection()
        plan = build_host_plan(
            inspection, self.configs["ubuntu-26.04-wsl2-host.json"],
            self.configs["ubuntu-package.lock.json"], self.configs["cuda-wsl.lock.json"],
        )
        if plan["blockers"]:
            raise BootstrapError(ErrorCode.HOST_UNSUPPORTED, "host has non-installable blockers", context={"blockers": plan["blockers"]})
        return self._mutate(
            "apply-host", "HOST_READY", ("CLONED", "HOST_INSPECTED", "HOST_PLAN_READY"), authorized,
            lambda transaction: apply_host(
                inspection, self.configs["ubuntu-package.lock.json"], self.configs["cuda-wsl.lock.json"],
                transaction=transaction, authorized=True, elevated=os.geteuid() == 0,
                allow_patch_difference=allow_patch_difference, runner=self.runner,
            ),
        )

    def initialize_submodules(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "initialize-submodules", "SUBMODULES_READY", ("HOST_READY",), authorized,
            lambda transaction: initialize_submodules(
                self.paths, self.configs["llama-build.lock.json"], transaction=transaction, authorized=True, runner=self.runner
            ),
        )

    def build_environments(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "build-environments", "PYTHON_ENVIRONMENTS_READY", ("SUBMODULES_READY",), authorized,
            lambda transaction: build_environments(
                self.paths, self.configs["python-environments.lock.json"], transaction=transaction, authorized=True, runner=self.runner
            ),
        )

    def build_llama_server(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "build-llama-server", "LLAMA_SERVER_BUILT", ("PYTHON_ENVIRONMENTS_READY",), authorized,
            lambda transaction: build_llama_server(
                self.paths, self.configs["llama-build.lock.json"], transaction=transaction, authorized=True, runner=self.runner
            ),
        )

    def initialize_runtime(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "initialize-runtime", "RUNTIME_INITIALIZED", ("LLAMA_SERVER_BUILT",), authorized,
            lambda transaction: initialize_runtime(
                self.paths, self.configs["runtime-layout.json"], transaction=transaction, authorized=True, runner=self.runner
            ),
        )

    def initialize_credentials(self, *, authorized: bool, mode: str = "generate-new") -> MachineResult:
        return self._mutate(
            "initialize-credentials", "CREDENTIAL_READY", ("RUNTIME_INITIALIZED",), authorized,
            lambda transaction: initialize_credentials(
                self.paths, self.configs["credential-initialization.json"], transaction=transaction,
                authorized=True, mode=mode, runner=self.runner,
            ),
        )

    def register_platform_service(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "register-platform-service", "SERVICE_REGISTERED", ("CREDENTIAL_READY",), authorized,
            lambda transaction: register_platform_service(
                self.paths, self.configs["service-registration.json"], transaction=transaction,
                authorized=True, runner=self.runner, home=self.home,
            ),
        )

    def reconstruct(self, *, authorized: bool, allow_patch_difference: bool = False) -> list[MachineResult]:
        if not authorized:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "reconstruct requires --authorize")
        operations: list[Callable[[], MachineResult]] = [
            lambda: self.apply_host(authorized=True, allow_patch_difference=allow_patch_difference),
            lambda: self.initialize_submodules(authorized=True),
            lambda: self.build_environments(authorized=True),
            lambda: self.build_llama_server(authorized=True),
            lambda: self.initialize_runtime(authorized=True),
            lambda: self.initialize_credentials(authorized=True),
            lambda: self.register_platform_service(authorized=True),
            lambda: self.verify("waiting-for-model"),
        ]
        return [operation() for operation in operations]
