"""Normalize private router metadata without claiming untested runtime support."""

from __future__ import annotations

import hashlib
from typing import Any

from .registry_types import (
    CAPABILITY_SCHEMA_IDENTITY,
    CapabilityEvidence,
    RouterModelEvidence,
    canonical_json,
    utc_now,
)


CAPABILITY_FIELDS = (
    "supports_tools",
    "supports_tool_calls",
    "supports_system_role",
    "supports_parallel_tool_calls",
    "supports_preserve_reasoning",
    "supports_string_content",
    "supports_typed_content",
    "supports_object_arguments",
)


class CapabilityInspectionError(RuntimeError):
    """Private router metadata failed bounded structural validation."""


def normalize_router_model(model: Any) -> RouterModelEvidence:
    if model.source != "models_dir":
        raise CapabilityInspectionError("router model source is not models_dir")
    if not model.model_id or not isinstance(model.model_id, str):
        raise CapabilityInspectionError("router model ID is invalid")
    if not isinstance(model.status, str) or not isinstance(model.raw, dict):
        raise CapabilityInspectionError("router model status payload is invalid")
    payload = {
        "router_model_id": model.model_id,
        "router_source": model.source,
        "router_status": model.status,
        "input_modalities": list(model.input_modalities),
        "output_modalities": list(model.output_modalities),
    }
    metadata_json = canonical_json(payload)
    return RouterModelEvidence(
        router_model_id=model.model_id,
        router_source=model.source,
        router_status=model.status,
        display_name=model.model_id,
        physical_path=model.physical_path,
        connected_paths=tuple(model.connected_paths),
        input_modalities=tuple(model.input_modalities),
        output_modalities=tuple(model.output_modalities),
        observed_utc=utc_now(),
        metadata_json=metadata_json,
        metadata_sha256=hashlib.sha256(metadata_json.encode("utf-8")).hexdigest(),
    )


def _exact_boolean(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _derived_boolean(value: bool | None) -> bool | str:
    return value if value is not None else "unknown"


def build_capability_evidence(
    model_version_id: str,
    bundle_id: str,
    router: RouterModelEvidence,
    props_payload: Any,
) -> CapabilityEvidence:
    if not isinstance(props_payload, dict):
        raise CapabilityInspectionError("props payload is not an object")
    raw_props_json = canonical_json(props_payload)
    props_sha256 = hashlib.sha256(raw_props_json.encode("utf-8")).hexdigest()
    template = props_payload.get("chat_template")
    template_present = isinstance(template, str) and bool(template.strip())
    caps_payload = props_payload.get("chat_template_caps")
    caps_object = caps_payload if isinstance(caps_payload, dict) else None
    exact_caps = {
        field: _exact_boolean(caps_object.get(field)) if caps_object else None
        for field in CAPABILITY_FIELDS
    }
    if not template_present:
        tool_calling: bool | str = False
    elif (
        exact_caps["supports_tools"] is None
        or exact_caps["supports_tool_calls"] is None
    ):
        tool_calling = "unknown"
    else:
        tool_calling = bool(
            exact_caps["supports_tools"] is True
            and exact_caps["supports_tool_calls"] is True
        )
    if tool_calling is True:
        parallel_cap = exact_caps["supports_parallel_tool_calls"]
        parallel_tool_calling: bool | str = (
            parallel_cap if parallel_cap is not None else "unknown"
        )
    elif tool_calling is False:
        parallel_tool_calling = False
    else:
        parallel_tool_calling = "unknown"

    inputs = set(router.input_modalities)
    outputs = set(router.output_modalities)
    props_modalities = props_payload.get("modalities")
    if isinstance(props_modalities, dict):
        for key, enabled in props_modalities.items():
            if (
                isinstance(key, str)
                and len(key) <= 64
                and type(enabled) is bool
                and enabled
            ):
                inputs.add(key)
    context_value = None
    generation = props_payload.get("default_generation_settings")
    if isinstance(generation, dict):
        candidate = generation.get("n_ctx")
        if type(candidate) is int and candidate >= 0:
            context_value = candidate
    observed = utc_now()
    manifest = {
        "schema": CAPABILITY_SCHEMA_IDENTITY,
        "model_version_id": model_version_id,
        "bundle_id": bundle_id,
        "router_model_id": router.router_model_id,
        "observed_utc": observed,
        "evidence_layers": {
            "static_artifact": {
                "bundle_id": bundle_id,
                "runtime_compatibility_claimed": False,
            },
            "router_properties": {
                "source": "props",
                "props_payload_sha256": props_sha256,
            },
            "runtime_generation": "NOT_TESTED",
        },
        "modalities": {
            "input": sorted(inputs),
            "output": sorted(outputs),
        },
        "chat_template": {
            "present": template_present,
            "source": "props",
        },
        "chat_template_caps": exact_caps,
        "derived_template_capabilities": {
            "tool_calling": tool_calling,
            "parallel_tool_calling": parallel_tool_calling,
            "system_role": _derived_boolean(
                exact_caps["supports_system_role"]
            ),
            "reasoning_preservation": _derived_boolean(
                exact_caps["supports_preserve_reasoning"]
            ),
            "string_content": _derived_boolean(
                exact_caps["supports_string_content"]
            ),
            "typed_content": _derived_boolean(
                exact_caps["supports_typed_content"]
            ),
            "object_arguments": _derived_boolean(
                exact_caps["supports_object_arguments"]
            ),
        },
        "context": {"default_n_ctx": context_value},
        "is_sleeping": _exact_boolean(props_payload.get("is_sleeping")),
        "runtime_generation_tests": {
            "completion": "NOT_TESTED",
            "chat": "NOT_TESTED",
            "responses": "NOT_TESTED",
            "streaming": "NOT_TESTED",
            "embeddings": "NOT_TESTED",
        },
    }
    manifest_json = canonical_json(manifest)
    return CapabilityEvidence(
        model_version_id=model_version_id,
        manifest_json=manifest_json,
        manifest_sha256=hashlib.sha256(
            manifest_json.encode("utf-8")
        ).hexdigest(),
        props_payload_sha256=props_sha256,
        observed_utc=observed,
    )
