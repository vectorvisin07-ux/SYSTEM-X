"""Bounded local Draft 2020-12 policy for tools and structured output."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


MAX_SCHEMA_BYTES = 65_536
MAX_AGGREGATE_SCHEMA_BYTES = 262_144
MAX_SCHEMA_DEPTH = 16
MAX_SCHEMA_PROPERTIES = 256
MAX_ENUM_MEMBERS = 256
MAX_PATTERN_LENGTH = 256
MAX_INSTANCE_BYTES = 1_048_576

_ALLOWED_KEYWORDS = {
    "$schema",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "anyOf",
    "description",
    "title",
    "default",
}
_PROHIBITED_REFERENCE_KEYWORDS = {
    "$id",
    "$ref",
    "$dynamicRef",
    "$recursiveRef",
    "$anchor",
    "$dynamicAnchor",
}
_PROHIBITED_COMPLEX_KEYWORDS = {
    "allOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contentEncoding",
    "contentMediaType",
}
_JSON_TYPES = {"null", "boolean", "object", "array", "number", "integer", "string"}


@dataclass(frozen=True, slots=True)
class SchemaPolicyError(ValueError):
    kind: str
    path: tuple[str | int, ...]
    detail: str

    def __str__(self) -> str:
        location = ".".join(str(item) for item in self.path) or "$"
        return f"{self.kind} at {location}: {self.detail}"


def _raise(kind: str, path: tuple[str | int, ...], detail: str) -> None:
    raise SchemaPolicyError(kind, path, detail[:240])


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaPolicyError("json_value_invalid", (), "value is not finite JSON") from exc


def canonical_json_size(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def _finite_number(value: Any, path: tuple[str | int, ...]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _raise("json_number_invalid", path, "non-finite numbers are prohibited")


def _validate_type(value: Any, path: tuple[str | int, ...]) -> None:
    if isinstance(value, str):
        if value not in _JSON_TYPES:
            _raise("schema_type_invalid", path, "unknown JSON type")
        return
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, str) and item in _JSON_TYPES for item in value)
        and len(set(value)) == 2
        and "null" in value
    ):
        return
    _raise(
        "schema_type_invalid",
        path,
        "type must be one JSON type or one nullable two-type union",
    )


def _validate_any_of(
    value: Any,
    path: tuple[str | int, ...],
    depth: int,
    counters: dict[str, int],
) -> None:
    if not isinstance(value, list) or len(value) != 2:
        _raise("schema_union_unsupported", path, "anyOf must contain exactly two schemas")
    null_members = [
        member
        for member in value
        if isinstance(member, dict) and member.get("type") == "null"
    ]
    if len(null_members) != 1:
        _raise("schema_union_unsupported", path, "anyOf must be a nullable union")
    for index, member in enumerate(value):
        _scan_schema(member, path + (index,), depth + 1, counters)


def _scan_schema(
    schema: Any,
    path: tuple[str | int, ...] = (),
    depth: int = 0,
    counters: dict[str, int] | None = None,
) -> None:
    if counters is None:
        counters = {"properties": 0}
    if depth > MAX_SCHEMA_DEPTH:
        _raise("schema_depth_exceeded", path, "schema nesting exceeds the configured bound")
    if not isinstance(schema, dict):
        _raise("schema_shape_invalid", path, "schema node must be an object")
    for keyword in schema:
        if keyword in _PROHIBITED_REFERENCE_KEYWORDS:
            _raise("schema_reference_unsupported", path + (keyword,), "references are prohibited")
        if keyword in _PROHIBITED_COMPLEX_KEYWORDS:
            _raise("schema_keyword_unsupported", path + (keyword,), "keyword is unsupported")
        if keyword not in _ALLOWED_KEYWORDS:
            _raise("schema_keyword_unsupported", path + (keyword,), "keyword is unsupported")
    if "$schema" in schema and schema["$schema"] not in {
        "https://json-schema.org/draft/2020-12/schema",
        "https://json-schema.org/draft/2020-12/schema#",
    }:
        _raise("schema_dialect_unsupported", path + ("$schema",), "only Draft 2020-12 is accepted")
    if "type" in schema:
        _validate_type(schema["type"], path + ("type",))
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict) or any(
            not isinstance(name, str) or not name or len(name) > 128
            for name in properties
        ):
            _raise("schema_properties_invalid", path + ("properties",), "properties must be a bounded object")
        counters["properties"] += len(properties)
        if counters["properties"] > MAX_SCHEMA_PROPERTIES:
            _raise(
                "schema_properties_exceeded",
                path + ("properties",),
                "aggregate property count exceeds the configured bound",
            )
        for name, child in properties.items():
            _scan_schema(child, path + ("properties", name), depth + 1, counters)
    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list)
        or any(not isinstance(item, str) for item in required)
        or len(required) != len(set(required))
    ):
        _raise("schema_required_invalid", path + ("required",), "required must contain unique strings")
    additional = schema.get("additionalProperties")
    if additional is not None and type(additional) is not bool:
        _raise(
            "schema_additional_properties_unsupported",
            path + ("additionalProperties",),
            "additionalProperties must be boolean",
        )
    if "items" in schema:
        _scan_schema(schema["items"], path + ("items",), depth + 1, counters)
    prefix = schema.get("prefixItems")
    if prefix is not None:
        if not isinstance(prefix, list) or len(prefix) > MAX_SCHEMA_PROPERTIES:
            _raise("schema_prefix_items_invalid", path + ("prefixItems",), "prefixItems is invalid")
        for index, child in enumerate(prefix):
            _scan_schema(child, path + ("prefixItems", index), depth + 1, counters)
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not 1 <= len(enum) <= MAX_ENUM_MEMBERS:
            _raise("schema_enum_invalid", path + ("enum",), "enum member count is invalid")
        canonical_members = [canonical_json(item) for item in enum]
        if len(set(canonical_members)) != len(canonical_members):
            _raise("schema_enum_invalid", path + ("enum",), "enum members must be unique")
    pattern = schema.get("pattern")
    if pattern is not None and (
        not isinstance(pattern, str) or len(pattern.encode("utf-8")) > MAX_PATTERN_LENGTH
    ):
        _raise("schema_pattern_invalid", path + ("pattern",), "pattern exceeds the configured bound")
    if "anyOf" in schema:
        _validate_any_of(schema["anyOf"], path + ("anyOf",), depth, counters)
    for keyword in (
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "default",
        "const",
    ):
        if keyword in schema:
            _finite_number(schema[keyword], path + (keyword,))


def _strict_objects(schema: dict[str, Any], path: tuple[str | int, ...] = ()) -> None:
    schema_type = schema.get("type")
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if is_object:
        properties = schema.get("properties", {})
        required = schema.get("required")
        if schema.get("additionalProperties") is not False:
            _raise(
                "strict_additional_properties_required",
                path + ("additionalProperties",),
                "strict object schemas require additionalProperties false",
            )
        if not isinstance(properties, dict) or required != list(properties):
            if not isinstance(required, list) or set(required) != set(properties):
                _raise(
                    "strict_required_incomplete",
                    path + ("required",),
                    "strict object schemas require every property",
                )
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            _strict_objects(child, path + ("properties", name))
    items = schema.get("items")
    if isinstance(items, dict):
        _strict_objects(items, path + ("items",))
    prefix = schema.get("prefixItems")
    if isinstance(prefix, list):
        for index, child in enumerate(prefix):
            _strict_objects(child, path + ("prefixItems", index))
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        for index, child in enumerate(any_of):
            _strict_objects(child, path + ("anyOf", index))


def validate_schema(
    schema: Any,
    *,
    strict: bool,
    require_object: bool,
    maximum_bytes: int = MAX_SCHEMA_BYTES,
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        _raise("schema_shape_invalid", (), "schema must be an object")
    size = canonical_json_size(schema)
    if size > maximum_bytes:
        _raise("schema_size_exceeded", (), "serialized schema exceeds the configured bound")
    _scan_schema(schema)
    if require_object and schema.get("type") != "object":
        _raise("schema_top_level_invalid", ("type",), "top-level schema must be object")
    if strict:
        _strict_objects(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        path = tuple(exc.absolute_schema_path)
        raise SchemaPolicyError("schema_invalid", path, exc.message) from exc
    return schema


def validate_instance(
    schema: dict[str, Any],
    value: Any,
    *,
    maximum_bytes: int = MAX_INSTANCE_BYTES,
) -> Any:
    if canonical_json_size(value) > maximum_bytes:
        _raise("instance_size_exceeded", (), "serialized value exceeds the configured bound")
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        raise SchemaPolicyError(
            "instance_schema_mismatch",
            tuple(exc.absolute_path),
            exc.message,
        ) from exc
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaPolicyError("duplicate_json_key", (key,), "duplicate object key")
        result[key] = value
    return result


def parse_json_value(value: str, *, maximum_bytes: int = MAX_INSTANCE_BYTES) -> Any:
    if not isinstance(value, str):
        _raise("json_text_invalid", (), "JSON input must be text")
    if len(value.encode("utf-8")) > maximum_bytes:
        _raise("instance_size_exceeded", (), "serialized value exceeds the configured bound")
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: _raise(
                "json_number_invalid", (), f"non-finite number {token} is prohibited"
            ),
        )
    except SchemaPolicyError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SchemaPolicyError("json_text_invalid", (), "value is not exactly one JSON value") from exc


def parse_json_object(value: str | dict[str, Any]) -> dict[str, Any]:
    parsed = parse_json_value(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        _raise("tool_arguments_not_object", (), "tool arguments must be one JSON object")
    canonical_json(parsed)
    return parsed
