"""Explicit identity and bounded surface for local Messages compatibility."""

from __future__ import annotations

from typing import Final


COMPATIBILITY_VERSION: Final = "system-x.anthropic-compatible.v1"
STREAMING_VERSION: Final = "system-x.anthropic-streaming.v1"
COMPATIBILITY_HEADER: Final = "X-System-X-Anthropic-Compatibility"
ANTHROPIC_VERSION_HEADER: Final = "anthropic-version"
ANTHROPIC_BETA_HEADER: Final = "anthropic-beta"
ACCEPTED_ANTHROPIC_VERSION: Final = "2023-06-01"
ANTHROPIC_REQUEST_ID_HEADER: Final = "request-id"
ANTHROPIC_PATH_PREFIX: Final = "/v1/messages"
TEXT_BLOCK_SEPARATOR: Final = "\n\n"

COMPATIBILITY_CONTRACT: Final = {
    "identity": COMPATIBILITY_VERSION,
    "anthropic_version": ACCEPTED_ANTHROPIC_VERSION,
    "routes": [
        "POST /v1/messages",
        "POST /v1/messages/count_tokens",
    ],
    "streaming": True,
    "tools": True,
    "structured_output": True,
    "parallel_tool_calling": False,
    "client_tool_execution": False,
    "extensions": [
        "system-x.anthropic-tools.v1",
        "system-x.structured-output.v1",
        STREAMING_VERSION,
    ],
    "thinking": False,
    "multimodal": False,
    "authentication": True,
}


def request_suffix(request_id: str) -> str:
    prefix = "sx_req_"
    suffix = request_id.removeprefix(prefix)
    if (
        not request_id.startswith(prefix)
        or len(suffix) != 32
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise RuntimeError("System X request identity is invalid")
    return suffix


def anthropic_request_id(request_id: str) -> str:
    return f"req_sx_{request_suffix(request_id)}"


def anthropic_message_id(request_id: str) -> str:
    return f"msg_sx_{request_suffix(request_id)}"
