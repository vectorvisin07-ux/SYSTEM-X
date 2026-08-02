"""Bounded JSON machine interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from .constants import (
    COMPONENT_IDENTITY,
    DEPLOYMENT_VERSION,
    FOUNDATION_VERSION,
    HANDOFF_VERSION,
    PUBLICATION_VERSION,
    PROMOTION_VERSION,
    RETIREMENT_VERSION,
    QUALIFICATION_VERSION,
    QUALIFICATION_PROFILES,
    INSPECTION_VERSION,
    OPERATIONS,
    SAFETY_MAXIMA,
    SCHEMA_IDENTITIES,
)
from .config import load_configuration, validate_configuration_values
from .capabilities import capability_inventory
from .errors import InspectorError
from .intake import list_intake
from .handoff import handoff_transaction
from .service_publication import publish_service_transaction
from .qualification import qualify_transaction
from .promotion import promote_transaction
from .retirement import retire_transaction
from .connection_receipt import render_connection, show_connection
from .deployment import (
    CurrentSourceDeploymentAdapter,
    deploy_transaction,
)
from .paths import InspectorPaths
from .results import machine_result
from .runtime import (
    layout_report,
    status_report,
    inspect_transaction,
    decide_transaction,
    validate_intake_transaction,
)


def identify(paths: InspectorPaths) -> dict[str, object]:
    environment_identity = None
    if paths.environment_lock.is_file() and not paths.environment_lock.is_symlink():
        environment_identity = (
            "sha256:" + hashlib.sha256(paths.environment_lock.read_bytes()).hexdigest()
        )
    return machine_result(
        operation="identify",
        ok=True,
        reason_code="OK",
        message="Inspector foundation identified",
        inspector_root=paths.inspector_root,
        data={
            "component_identity": COMPONENT_IDENTITY,
            "foundation_version": FOUNDATION_VERSION,
            "inspection_version": INSPECTION_VERSION,
            "handoff_version": HANDOFF_VERSION,
            "publication_version": PUBLICATION_VERSION,
            "qualification_version": QUALIFICATION_VERSION,
            "promotion_version": PROMOTION_VERSION,
            "retirement_version": RETIREMENT_VERSION,
            "deployment_version": DEPLOYMENT_VERSION,
            "source_root": str(paths.source_root),
            "runtime_root": str(paths.runtime_root),
            "intake_root": str(paths.intake_root),
            "schema_identities": SCHEMA_IDENTITIES,
            "environment_lock_identity": environment_identity,
            "implemented_operations": list(OPERATIONS),
        },
        paths={
            key: str(value)
            for key, value in paths.as_mapping().items()
        },
    )


class MachineArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise InspectorError("CONFIG_INVALID", message)


def build_parser() -> argparse.ArgumentParser:
    parser = MachineArgumentParser(prog="system-x-inspector")
    parser.add_argument("--inspector-root", type=Path)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("identify")
    config_parser = subparsers.add_parser("validate-config")
    config_parser.add_argument("--config", type=Path, required=True)
    subparsers.add_parser("layout")
    subparsers.add_parser("status")
    subparsers.add_parser("list-intake")
    intake_parser = subparsers.add_parser("validate-intake")
    intake_parser.add_argument("--config", type=Path, required=True)
    intake_parser.add_argument("--target")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", type=Path)
    inspect_parser.add_argument("--target")
    subparsers.add_parser("capabilities")
    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--inspection-id", required=True)
    qualification_parser = subparsers.add_parser("qualify-gguf")
    qualification_parser.add_argument("--inspection-id", required=True)
    qualification_parser.add_argument(
        "--candidate-artifact-identity", required=True
    )
    qualification_parser.add_argument(
        "--required-capability-profile",
        choices=QUALIFICATION_PROFILES,
        required=True,
    )
    promotion_parser = subparsers.add_parser("promote-gguf")
    promotion_parser.add_argument("--qualification-id", required=True)
    promotion_parser.add_argument("--candidate-name", required=True)
    retirement_parser = subparsers.add_parser("retire-gguf")
    retirement_parser.add_argument("--public-model-id", required=True)
    retirement_parser.add_argument("--artifact-identity", required=True)
    retirement_parser.add_argument(
        "--managed-location-identity", required=True
    )
    retirement_parser.add_argument(
        "--expected-registry-generation", type=int, required=True
    )
    retirement_parser.add_argument("--retirement-reason", required=True)
    retirement_parser.add_argument(
        "--last-model-policy",
        choices=("REJECT", "ENTER_WAITING_FOR_MODEL"),
        default="REJECT",
    )
    handoff_parser = subparsers.add_parser("handoff")
    handoff_parser.add_argument("--decision-id", required=True)
    handoff_parser.add_argument("--source-candidate", required=True)
    handoff_parser.add_argument("--managed-name", required=True)
    handoff_parser.add_argument("--qualification-id")
    publication_parser = subparsers.add_parser("publish-service")
    publication_parser.add_argument("--handoff-id", required=True)
    deploy_parser = subparsers.add_parser("deploy-gguf")
    deploy_parser.add_argument("--candidate-name", required=True)
    deploy_parser.add_argument(
        "--deployment-mode",
        choices=("install-first", "add", "replace-default"),
        required=True,
    )
    deploy_parser.add_argument(
        "--required-capability-profile",
        choices=QUALIFICATION_PROFILES,
        required=True,
    )
    deploy_parser.add_argument(
        "--retirement-policy",
        choices=(
            "retain-incumbent",
            "retire-incumbent-after-acceptance",
        ),
        required=True,
    )
    subparsers.add_parser("show-connection")
    return parser


def _inspection_configuration(
    paths: InspectorPaths, config_path: Path | None
):
    if config_path is not None:
        return load_configuration(config_path, paths)
    return validate_configuration_values(
        {
            "schema_version": SCHEMA_IDENTITIES["configuration"],
            "intake_root": str(paths.intake_root),
            "runtime_root": str(paths.runtime_root),
            "intake_bounds": dict(SAFETY_MAXIMA),
            "record_policy": {
                "status_file_mode": "0600",
                "transaction_file_mode": "0600",
                "log_file_mode": "0600",
            },
            "result_roots": {
                "inspection": str(paths.inspection_results),
                "decision": str(paths.decision_results),
                "handoff": str(paths.handoff_results),
                "publication": str(paths.publication_results),
            },
        },
        paths,
    )


def execute(arguments: argparse.Namespace) -> tuple[int, dict[str, object]]:
    paths = InspectorPaths.discover(arguments.inspector_root)
    if arguments.operation == "identify":
        return 0, identify(paths)
    if arguments.operation == "validate-config":
        configuration = load_configuration(arguments.config, paths)
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="OK",
            message="Inspector configuration is valid",
            inspector_root=paths.inspector_root,
            data={
                "configuration": configuration.values,
                "configuration_identity": configuration.identity,
            },
        )
    if arguments.operation == "layout":
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="OK",
            message="Inspector layout is valid",
            inspector_root=paths.inspector_root,
            data=layout_report(paths),
        )
    if arguments.operation == "status":
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="OK",
            message="Inspector foundation status inspected",
            inspector_root=paths.inspector_root,
            data=status_report(paths),
        )
    if arguments.operation == "list-intake":
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="OK",
            message="Inspector intake candidates listed",
            inspector_root=paths.inspector_root,
            data=list_intake(paths),
        )
    if arguments.operation == "validate-intake":
        configuration = load_configuration(arguments.config, paths)
        transaction_id, candidate = validate_intake_transaction(
            paths, configuration, arguments.target
        )
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="OK",
            message="Inspector intake candidate validated",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={"candidate": candidate},
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
            },
        )
    if arguments.operation == "inspect":
        configuration = _inspection_configuration(paths, arguments.config)
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = inspect_transaction(paths, configuration, arguments.target)
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="INSPECTION_COMPLETE",
            message="Inspector physical-format inspection completed",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={
                "inspection_id": record["inspection_id"],
                "result_path": str(result_path),
                "result_identity": result_identity,
                "terminal_class": record["classification"][
                    "terminal_class"
                ],
                "detected_family": record["classification"][
                    "detected_family"
                ],
                "artifact_identity": record["artifact"]["identity"],
                "artifact_size": record["artifact"]["byte_count"],
                "normalized": record["normalized"],
            },
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
                "result": str(result_path),
            },
        )
    if arguments.operation == "capabilities":
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="OK",
            message="Installed Inspector capabilities inspected",
            inspector_root=paths.inspector_root,
            data=capability_inventory(paths),
        )
    if arguments.operation == "decide":
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = decide_transaction(paths, arguments.inspection_id)
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code=record["reason_code"],
            message="Inspector branch decision completed",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={
                "decision_id": record["decision_id"],
                "decision_basis_identity": record[
                    "decision_basis_identity"
                ],
                "result_identity": result_identity,
                "capability_result": record["capability"][
                    "capability_result"
                ],
                "selected_branch": record["selected_branch"],
                "handoff_allowed": record["handoff_allowed"],
                "spawn_allowed": record["spawn_allowed"],
                "reason_codes": record["reason_codes"],
            },
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
                "result": str(result_path),
            },
        )
    if arguments.operation == "qualify-gguf":
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = qualify_transaction(
            paths,
            arguments.inspection_id,
            arguments.candidate_artifact_identity,
            arguments.required_capability_profile,
        )
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code=record["reason_codes"][0],
            message="Inspector GGUF qualification completed",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={
                "qualification_id": record["qualification_id"],
                "result_identity": result_identity,
                "result_class": record["result_class"],
                "requested_profile": record["requested_profile"],
                "supported_profiles": record["supported_profiles"],
                "reason_codes": record["reason_codes"],
            },
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
                "result": str(result_path),
            },
        )
    if arguments.operation == "promote-gguf":
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = promote_transaction(
            paths,
            arguments.qualification_id,
            arguments.candidate_name,
        )
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code=record["reason_codes"][0],
            message="Inspector GGUF promotion transaction completed",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={
                "promotion_id": record["promotion_id"],
                "result_identity": result_identity,
                "result_class": record["result_class"],
                "qualification_id": record["qualification"][
                    "qualification_id"
                ],
                "candidate_public_model_id": record["candidate"].get(
                    "public_model_id"
                ),
                "reason_codes": record["reason_codes"],
            },
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
                "result": str(result_path),
            },
        )
    if arguments.operation == "retire-gguf":
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = retire_transaction(
            paths,
            public_model_id=arguments.public_model_id,
            artifact_identity=arguments.artifact_identity,
            managed_location_identity=(
                arguments.managed_location_identity
            ),
            expected_registry_generation=(
                arguments.expected_registry_generation
            ),
            retirement_reason=arguments.retirement_reason,
            last_model_policy=arguments.last_model_policy,
        )
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code=record["reason_codes"][0],
            message="Inspector GGUF retirement transaction completed",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={
                "retirement_id": record["retirement_id"],
                "result_identity": result_identity,
                "result_class": record["result_class"],
                "public_model_id": record["input"]["public_model_id"],
                "last_model_policy": record["input"][
                    "last_model_policy"
                ],
                "reason_codes": record["reason_codes"],
            },
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
                "result": str(result_path),
            },
        )
    if arguments.operation == "handoff":
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = handoff_transaction(
            paths,
            arguments.decision_id,
            arguments.source_candidate,
            arguments.managed_name,
            qualification_id=arguments.qualification_id,
        )
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="HANDOFF_COMPLETE",
            message="Inspector GGUF branch handoff completed",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={
                "handoff_id": record["handoff_id"],
                "result_identity": result_identity,
                "decision_id": record["decision"]["decision_id"],
                "inspection_id": record["inspection"]["inspection_id"],
                "artifact_identity": record["inspection"][
                    "artifact_identity"
                ],
                "managed_relative_path": record["publication"][
                    "managed_relative_path"
                ],
                "registry_observation": record["registry_observation"],
            },
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
                "result": str(result_path),
            },
        )
    if arguments.operation == "publish-service":
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = publish_service_transaction(paths, arguments.handoff_id)
        return 0, machine_result(
            operation=arguments.operation,
            ok=True,
            reason_code="SERVICE_PUBLICATION_COMPLETE",
            message="Inspector service publication proof completed",
            inspector_root=paths.inspector_root,
            transaction_id=transaction_id,
            data={
                "publication_id": record["publication_id"],
                "result_identity": result_identity,
                "selected_branch": record["handoff"]["selected_branch"],
                "public_base_url": record["public_service"]["base_url"],
                "public_model_id": record["public_service"][
                    "public_model_id"
                ],
                "aliases": record["public_service"]["aliases"],
                "artifact_version_id": record["correlation"][
                    "artifact_version_id"
                ],
                "service_readiness": record["public_service"][
                    "readiness_state"
                ],
                "request_id": record["request"]["request_id"],
                "request_http_status": record["request"]["http_status"],
                "default_warm_ready": (
                    record["restoration"]["final_warm_health"] == "ready"
                    and record["restoration"]["final_public_health"]
                    == "READY"
                ),
            },
            paths={
                "status": str(paths.status / "current.json"),
                "transaction": str(
                    paths.transactions / f"{transaction_id}.json"
                ),
                "result": str(result_path),
            },
        )
    if arguments.operation == "show-connection":
        connection = show_connection(paths)
        ready = connection["result_class"] == "CONNECTION_READY"
        receipt = connection["receipt"]
        data = dict(connection)
        if ready:
            data["rendered"] = render_connection(receipt)
        return (
            0 if ready else 2,
            machine_result(
                operation=arguments.operation,
                ok=ready,
                reason_code=str(connection["reason_code"]),
                message=(
                    "Current API connection receipt verified"
                    if ready
                    else "Current API connection receipt is not ready"
                ),
                inspector_root=paths.inspector_root,
                data=data,
                paths={
                    "current_connection": str(
                        paths.current_connection_status
                    )
                },
            ),
        )
    if arguments.operation == "deploy-gguf":
        (
            transaction_id,
            record,
            result_path,
            result_identity,
        ) = deploy_transaction(
            paths,
            {
                "candidate_name": arguments.candidate_name,
                "deployment_mode": arguments.deployment_mode,
                "required_capability_profile": (
                    arguments.required_capability_profile
                ),
                "retirement_policy": arguments.retirement_policy,
            },
            adapter=CurrentSourceDeploymentAdapter(),
        )
        complete = record["result_class"] == "DEPLOYMENT_COMPLETE"
        return (
            0 if complete else 2,
            machine_result(
                operation=arguments.operation,
                ok=complete,
                reason_code=record["reason_code"],
                message="Inspector GGUF deployment transaction completed",
                inspector_root=paths.inspector_root,
                transaction_id=transaction_id,
                data={
                    "deployment_id": record["deployment_id"],
                    "result_identity": result_identity,
                    "result_class": record["result_class"],
                    "deployment_mode": record["deployment_mode"],
                    "connection_receipt_identity": (
                        record["connection_receipt"][
                            "receipt_identity"
                        ]
                        if record["connection_receipt"] is not None
                        else None
                    ),
                    "child_results": record["child_results"],
                },
                paths={"result": str(result_path)},
            ),
        )
    raise InspectorError("CONFIG_INVALID", "unknown Inspector operation")


def main(argv: Sequence[str] | None = None) -> int:
    operation = "unknown"
    root = Path(__file__).resolve().parent.parent
    try:
        arguments = build_parser().parse_args(argv)
        operation = arguments.operation
        root = (
            arguments.inspector_root.resolve(strict=False)
            if arguments.inspector_root is not None
            else root
        )
        exit_status, result = execute(arguments)
    except InspectorError as error:
        exit_status = error.exit_status
        transaction_id = error.data.get("transaction_id")
        result = machine_result(
            operation=operation,
            ok=False,
            reason_code=error.reason_code,
            message=error.message,
            inspector_root=root,
            transaction_id=(
                transaction_id if isinstance(transaction_id, str) else None
            ),
            data=error.data,
        )
    except Exception:
        exit_status = 70
        result = machine_result(
            operation=operation,
            ok=False,
            reason_code="INTERNAL_ERROR",
            message="Unexpected Inspector internal failure",
            inspector_root=root,
        )
    sys.stdout.write(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return exit_status
