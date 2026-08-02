"""One-result JSON command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .errors import BootstrapError
from .orchestrator import BootstrapOrchestrator
from .result import MachineResult
from .verify import VERIFICATION_LEVELS


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="system-x-bootstrap")
    subparsers = value.add_subparsers(dest="operation", required=True)
    for name in ("identify", "inspect-host", "plan", "status"):
        subparsers.add_parser(name)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--level", choices=VERIFICATION_LEVELS, required=True)
    for name in (
        "apply-host", "initialize-submodules", "build-environments", "build-llama-server",
        "initialize-runtime", "initialize-credentials", "register-platform-service", "reconstruct",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--authorize", action="store_true")
        if name in ("apply-host", "reconstruct"):
            command.add_argument("--allow-patch-difference", action="store_true")
        if name == "initialize-credentials":
            command.add_argument("--mode", choices=("generate-new", "import-encrypted"), default="generate-new")
    return value


def execute(arguments: argparse.Namespace, orchestrator: BootstrapOrchestrator) -> MachineResult | list[MachineResult]:
    operation = arguments.operation
    if operation == "identify": return orchestrator.identify()
    if operation == "inspect-host": return orchestrator.inspect_host()
    if operation == "plan": return orchestrator.plan()
    if operation == "status": return orchestrator.status()
    if operation == "verify": return orchestrator.verify(arguments.level)
    if operation == "apply-host": return orchestrator.apply_host(authorized=arguments.authorize, allow_patch_difference=arguments.allow_patch_difference)
    if operation == "initialize-submodules": return orchestrator.initialize_submodules(authorized=arguments.authorize)
    if operation == "build-environments": return orchestrator.build_environments(authorized=arguments.authorize)
    if operation == "build-llama-server": return orchestrator.build_llama_server(authorized=arguments.authorize)
    if operation == "initialize-runtime": return orchestrator.initialize_runtime(authorized=arguments.authorize)
    if operation == "initialize-credentials": return orchestrator.initialize_credentials(authorized=arguments.authorize, mode=arguments.mode)
    if operation == "register-platform-service": return orchestrator.register_platform_service(authorized=arguments.authorize)
    if operation == "reconstruct": return orchestrator.reconstruct(authorized=arguments.authorize, allow_patch_difference=arguments.allow_patch_difference)
    raise AssertionError(operation)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = execute(arguments, BootstrapOrchestrator())
    except BootstrapError as error:
        result = MachineResult.from_error(arguments.operation, "FAIL_CLOSED", error)
    if isinstance(result, list):
        payload = {
            "schema": "system-x.bootstrap.ordered-results.v1",
            "version": 1,
            "results": [item.as_dict() for item in result],
        }
        status = 0
    else:
        payload = result.as_dict()
        status = 0 if result.status == "ok" else 2
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return status
