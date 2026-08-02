"""OpenAPI security contract for private System X functional operations."""

from __future__ import annotations

from typing import Any

from .credential_types import AUTHENTICATION_CONTRACT


SECURITY_REQUIREMENT = [
    {"SystemXBearer": []},
    {"SystemXApiKey": []},
]
PROTECTED_OPENAPI_OPERATIONS = (
    ("/system/v1/version", "get", "system"),
    ("/system/v1/models", "get", "system"),
    ("/system/v1/models/{model_id}", "get", "system"),
    ("/system/v1/generate", "post", "system"),
    ("/system/v1/chat", "post", "system"),
    ("/system/v1/responses", "post", "system"),
    ("/system/v1/tokens/count", "post", "system"),
    ("/v1/models", "get", "openai"),
    ("/v1/completions", "post", "openai"),
    ("/v1/chat/completions", "post", "openai"),
    ("/v1/responses", "post", "openai"),
    ("/v1/messages", "post", "anthropic"),
    ("/v1/messages/count_tokens", "post", "anthropic"),
)


def _system_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["request_id", "status", "error"],
        "properties": {
            "request_id": {
                "type": "string",
                "pattern": "^sx_req_[0-9a-f]{32}$",
            },
            "status": {"type": "string", "const": "error"},
            "error": {
                "type": "object",
                "additionalProperties": False,
                "required": ["code", "message", "retryable", "details"],
                "properties": {
                    "code": {
                        "type": "string",
                        "const": "system_x_authentication_error",
                    },
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean", "const": False},
                    "details": {
                        "type": "object",
                        "additionalProperties": False,
                    },
                },
            },
        },
    }


def _openai_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["error"],
        "properties": {
            "error": {
                "type": "object",
                "additionalProperties": False,
                "required": ["message", "type", "param", "code"],
                "properties": {
                    "message": {"type": "string"},
                    "type": {
                        "type": "string",
                        "const": "authentication_error",
                    },
                    "param": {"type": "null"},
                    "code": {"type": "string", "const": "invalid_api_key"},
                },
            }
        },
    }


def _anthropic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "error", "request_id"],
        "properties": {
            "type": {"type": "string", "const": "error"},
            "error": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "message"],
                "properties": {
                    "type": {
                        "type": "string",
                        "const": "authentication_error",
                    },
                    "message": {"type": "string"},
                },
            },
            "request_id": {
                "type": "string",
                "pattern": "^req_sx_[0-9a-f]{32}$",
            },
        },
    }


def _headers(family: str) -> dict[str, Any]:
    headers: dict[str, Any] = {
        "WWW-Authenticate": {
            "description": "Required Bearer authentication challenge",
            "schema": {
                "type": "string",
                "const": 'Bearer realm="system-x"',
            },
        },
        "X-System-X-Request-ID": {
            "description": "System X request identity",
            "schema": {"type": "string"},
        },
    }
    if family == "openai":
        headers["x-request-id"] = {
            "description": "System X request identity",
            "schema": {"type": "string"},
        }
    elif family == "anthropic":
        headers["request-id"] = {
            "description": "Messages compatibility request identity",
            "schema": {"type": "string"},
        }
    return headers


def apply_authentication_openapi(
    schema: dict[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Apply two alternative schemes and exact per-operation 401 contracts."""

    components = schema.setdefault("components", {})
    schemes = components.setdefault("securitySchemes", {})
    schemes["SystemXBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "System-X-API-Key",
    }
    schemes["SystemXApiKey"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
    }
    family_schemas = {
        "system": _system_schema(),
        "openai": _openai_schema(),
        "anthropic": _anthropic_schema(),
    }
    for path, method, family in PROTECTED_OPENAPI_OPERATIONS:
        operation = schema["paths"][path][method]
        operation["security"] = [
            {"SystemXBearer": []},
            {"SystemXApiKey": []},
        ]
        operation.setdefault("responses", {})["401"] = {
            "description": "Authentication credentials are missing or invalid.",
            "headers": _headers(family),
            "content": {
                "application/json": {
                    "schema": family_schemas[family],
                }
            },
        }
    health = schema["paths"]["/system/v1/health"]["get"]
    health["security"] = []
    schema["x-system-x-authentication-contract"] = AUTHENTICATION_CONTRACT
    schema["x-system-x-authentication-enabled"] = enabled
    return schema
