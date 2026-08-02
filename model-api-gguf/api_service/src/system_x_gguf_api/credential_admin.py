"""Local-only administration CLI for System X private API credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .credential_store import CredentialStore, CredentialStoreError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="system-x-credential-admin",
        description="Administer the fixed branch-local System X credential store.",
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("initialize")
    issue = subparsers.add_parser("issue")
    issue.add_argument("--label", required=True)
    issue.add_argument("--output-file", required=True, type=Path)
    subparsers.add_parser("list")
    revoke = subparsers.add_parser("revoke")
    revoke.add_argument("--key-id", required=True)
    subparsers.add_parser("inspect")
    return parser


def execute(namespace: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    store = CredentialStore()
    if namespace.operation == "initialize":
        return store.initialize()
    if namespace.operation == "issue":
        return store.issue(namespace.label, namespace.output_file)
    if namespace.operation == "list":
        return store.list_keys()
    if namespace.operation == "revoke":
        return store.revoke(namespace.key_id)
    if namespace.operation == "inspect":
        return store.inspect()
    raise CredentialStoreError("unsupported credential administration operation")


def main(argv: list[str] | None = None) -> int:
    try:
        result = execute(_parser().parse_args(argv))
    except (CredentialStoreError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"ok": True, "result": result},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
