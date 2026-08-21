"""Operation contract, ordered state transitions, and receipts."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Callable, Mapping

from . import BOOTSTRAP_VERSION
from .command import Runner, SubprocessRunner, ensure_user_manager, exec_elevated_reconstruct, installation_user_context, resolve_installation_user, user_manager_ready
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
from .service import activate_platform_service, register_platform_service, service_status
from .state import StateDocument, read_state, write_failure_state, write_receipt, write_success_state
from .transaction import BootstrapTransaction, incomplete_transactions, recover_failed_clean_transactions
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
        installation_user: str | None = None,
    ) -> None:
        self.paths = paths or RepositoryPaths.discover()
        self.runner = runner or SubprocessRunner()
        self.installation_user = resolve_installation_user(installation_user)
        self.installation_user.validate_repository(self.paths.root)
        self.home = home or self.installation_user.home
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
        with installation_user_context(self.installation_user):
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
        user_owned: bool = False,
        allow_failure_recovery: bool = False,
    ) -> MachineResult:
        if not authorized:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, f"{operation} requires --authorize")
        previous = read_state(self.paths)
        recovered_transactions: list[str] = []
        if previous.state in ("FAILED_CLEAN", "FAIL_CLOSED"):
            if not allow_failure_recovery or previous.state != "FAILED_CLEAN":
                raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "bootstrap failure state requires explicit recovery")
            recovered_transactions = recover_failed_clean_transactions(
                self.paths.transaction_directory, authorized=authorized
            )
            if not recovered_transactions:
                raise BootstrapError(
                    ErrorCode.TRANSACTION_RECOVERY_REQUIRED,
                    "failed-clean state has no bounded transaction recovery record",
                )
            previous = read_state(self.paths)
        pending = incomplete_transactions(self.paths.transaction_directory)
        resume_record: Path | None = None
        if pending:
            if operation != "apply-host" or len(pending) != 1:
                raise BootstrapError(ErrorCode.TRANSACTION_RECOVERY_REQUIRED, "incomplete bootstrap transaction requires recovery")
            resume_record = pending[0]
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
            self.paths, operation, plan_identity, previous.as_dict(), True,
            resume_record=resume_record,
        )
        receipt_id = secrets.token_hex(16)
        try:
            with transaction:
                try:
                    if user_owned:
                        with installation_user_context(self.installation_user):
                            details = dict(provider(transaction))
                    else:
                        details = dict(provider(transaction))
                    changed = bool(details.get("changed"))
                    if recovered_transactions:
                        details["recovered_transactions"] = list(recovered_transactions)
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
            user_owned=True,
        )

    def build_environments(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "build-environments", "PYTHON_ENVIRONMENTS_READY", ("SUBMODULES_READY",), authorized,
            lambda transaction: build_environments(
                self.paths, self.configs["python-environments.lock.json"], transaction=transaction, authorized=True, runner=self.runner
            ),
            user_owned=True,
        )

    def build_llama_server(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "build-llama-server", "LLAMA_SERVER_BUILT", ("PYTHON_ENVIRONMENTS_READY",), authorized,
            lambda transaction: build_llama_server(
                self.paths, self.configs["llama-build.lock.json"], transaction=transaction, authorized=True, runner=self.runner
            ),
            user_owned=True,
        )

    def initialize_runtime(self, *, authorized: bool) -> MachineResult:
        return self._mutate(
            "initialize-runtime", "RUNTIME_INITIALIZED", ("LLAMA_SERVER_BUILT",), authorized,
            lambda transaction: initialize_runtime(
                self.paths, self.configs["runtime-layout.json"], transaction=transaction, authorized=True, runner=self.runner
            ),
            user_owned=True,
        )

    def initialize_credentials(self, *, authorized: bool, mode: str = "generate-new") -> MachineResult:
        return self._mutate(
            "initialize-credentials", "CREDENTIAL_READY", ("RUNTIME_INITIALIZED",), authorized,
            lambda transaction: initialize_credentials(
                self.paths, self.configs["credential-initialization.json"], transaction=transaction,
                authorized=True, mode=mode, runner=self.runner,
            ),
            user_owned=True,
        )

    def register_platform_service(self, *, authorized: bool, allow_failure_recovery: bool = False) -> MachineResult:
        return self._mutate(
            "register-platform-service", "SERVICE_REGISTERED", ("CREDENTIAL_READY",), authorized,
            lambda transaction: register_platform_service(
                self.paths, self.configs["service-registration.json"], transaction=transaction,
                authorized=True, runner=self.runner, home=self.home,
            ),
            user_owned=True,
            allow_failure_recovery=allow_failure_recovery,
        )

    def activate_platform_service(self, *, authorized: bool, allow_failure_recovery: bool = False) -> MachineResult:
        return self._mutate(
            "activate-platform-service", "SERVICE_REGISTERED", ("SERVICE_REGISTERED",), authorized,
            lambda transaction: activate_platform_service(
                self.paths, self.configs["service-registration.json"], transaction=transaction,
                authorized=True, runner=self.runner, home=self.home,
            ),
            user_owned=True,
            allow_failure_recovery=allow_failure_recovery,
        )

    def _prepare_reconstruct_host(
        self,
        *,
        allow_patch_difference: bool,
        current_state: str | None = None,
    ) -> None:
        """Plan host changes before any mutation transaction or privilege switch."""
        inspection = self._inspection()
        plan = build_host_plan(
            inspection,
            self.configs["ubuntu-26.04-wsl2-host.json"],
            self.configs["ubuntu-package.lock.json"],
            self.configs["cuda-wsl.lock.json"],
        )
        if plan["blockers"]:
            raise BootstrapError(
                ErrorCode.HOST_UNSUPPORTED,
                "host has non-installable blockers",
                context={"blockers": plan["blockers"]},
            )
        manager_required = current_state in {"SERVICE_REGISTERED", "WAITING_FOR_MODEL", "READY"}
        runner = getattr(self, "runner", None) or SubprocessRunner()
        manager_ready = True
        if manager_required:
            manager_ready = user_manager_ready(self.installation_user, runner)
        if plan["would_install"] or (manager_required and not manager_ready):
            if os.geteuid() != 0:
                exec_elevated_reconstruct(
                    self.paths.root,
                    self.installation_user,
                    allow_patch_difference=allow_patch_difference,
                )
            if manager_required and not manager_ready:
                ensure_user_manager(self.installation_user, runner)

    def reconstruct(self, *, authorized: bool, allow_patch_difference: bool = False) -> list[MachineResult]:
        if not authorized:
            raise BootstrapError(ErrorCode.AUTHORIZATION_REQUIRED, "reconstruct requires --authorize")
        state = read_state(self.paths)
        state_order = {
            "CLONED": 0,
            "HOST_INSPECTED": 1,
            "HOST_PLAN_READY": 2,
            "HOST_READY": 3,
            "SUBMODULES_READY": 4,
            "PYTHON_ENVIRONMENTS_READY": 5,
            "LLAMA_SERVER_BUILT": 6,
            "RUNTIME_INITIALIZED": 7,
            "CREDENTIAL_READY": 8,
            "SERVICE_REGISTERED": 9,
            "WAITING_FOR_MODEL": 10,
            "READY": 11,
        }
        phases: list[tuple[str, str, str, Callable[[], MachineResult]]] = [
            ("apply-host", "HOST_READY", "host-ready", lambda: self.apply_host(authorized=True, allow_patch_difference=allow_patch_difference)),
            ("initialize-submodules", "SUBMODULES_READY", "source-only", lambda: self.initialize_submodules(authorized=True)),
            ("build-environments", "PYTHON_ENVIRONMENTS_READY", "build-ready", lambda: self.build_environments(authorized=True)),
            ("build-llama-server", "LLAMA_SERVER_BUILT", "build-ready", lambda: self.build_llama_server(authorized=True)),
            ("initialize-runtime", "RUNTIME_INITIALIZED", "waiting-for-model", lambda: self.initialize_runtime(authorized=True)),
            ("initialize-credentials", "CREDENTIAL_READY", "waiting-for-model", lambda: self.initialize_credentials(authorized=True)),
            ("register-platform-service", "SERVICE_REGISTERED", "service-process-ready", lambda: self.register_platform_service(authorized=True, allow_failure_recovery=allow_patch_difference)),
            ("activate-platform-service", "SERVICE_REGISTERED", "waiting-for-model", lambda: self.activate_platform_service(authorized=True, allow_failure_recovery=allow_patch_difference)),
        ]
        current_index = state_order.get(state.stable_state, -1)
        if current_index < state_order["HOST_READY"] or (state.stable_state == "SERVICE_REGISTERED" and hasattr(self, "configs")):
            self._prepare_reconstruct_host(allow_patch_difference=allow_patch_difference, current_state=state.stable_state)
        results: list[MachineResult] = []
        for operation, target, level, action in phases:
            if operation == "activate-platform-service" and state.stable_state == "SERVICE_REGISTERED" and hasattr(self, "configs"):
                results.append(action())
                continue
            if operation == "register-platform-service" and state.stable_state == "SERVICE_REGISTERED" and hasattr(self, "configs"):
                results.append(action())
                continue
            if current_index >= state_order[target]:
                verification = self.verify(level)
                results.append(MachineResult(
                    operation,
                    "ok",
                    target,
                    changed=False,
                    details={
                        "reused": True,
                        "reuse_reason": "completed phase verified from current stable state",
                        "current_stable_state": state.stable_state,
                        "verification_level": level,
                        "verification": verification.details,
                    },
                ))
            else:
                results.append(action())
                current_index = state_order[target]
        results.append(self.verify("waiting-for-model"))
        return results
