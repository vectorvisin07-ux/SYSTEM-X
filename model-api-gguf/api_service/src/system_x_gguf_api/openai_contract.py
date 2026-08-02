"""Explicit identity and bounded surface for local OpenAI compatibility."""

from __future__ import annotations

from typing import Final


COMPATIBILITY_VERSION: Final = "system-x.openai-compatible.v1"
STREAMING_VERSION: Final = "system-x.openai-streaming.v1"
COMPATIBILITY_HEADER: Final = "X-System-X-Compatibility-Version"
OPENAI_REQUEST_ID_HEADER: Final = "x-request-id"
OPENAI_PATH_PREFIX: Final = "/v1"

COMPATIBILITY_CONTRACT: Final = {
    "identity": COMPATIBILITY_VERSION,
    "routes": {
        "GET /v1/models": [],
        "POST /v1/completions": [
            "model",
            "prompt",
            "max_tokens",
            "temperature",
            "stop",
            "stream",
            "stream_options",
            "n",
            "echo",
            "logprobs",
            "suffix",
            "best_of",
        ],
        "POST /v1/chat/completions": [
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "stop",
            "stream",
            "stream_options",
            "n",
            "logprobs",
            "top_logprobs",
            "parallel_tool_calls",
            "tools",
            "tool_choice",
            "response_format",
        ],
        "POST /v1/responses": [
            "model",
            "input",
            "instructions",
            "max_output_tokens",
            "temperature",
            "stream",
            "tools",
            "tool_choice",
            "text",
            "parallel_tool_calls",
            "background",
            "store",
        ],
    },
    "finish_reason_mapping": {
        "completed": "stop",
        "stop_sequence": "stop",
        "output_limit": "length",
        "context_limit": "length",
        "unknown": "error",
    },
    "streaming": True,
    "tool_calling": True,
    "structured_output": True,
    "parallel_tool_calling": False,
    "client_tool_execution": False,
    "extensions": [
        "system-x.openai-tools.v1",
        "system-x.structured-output.v1",
        STREAMING_VERSION,
    ],
    "authentication_enforced": True,
    "text_only": True,
    "maximum_choices": 1,
}
