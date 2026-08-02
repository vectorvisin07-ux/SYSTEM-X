"""Canonical client-function and structured-output contract shared by all APIs."""

from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from .tool_schema import (
    MAX_AGGREGATE_SCHEMA_BYTES,
    SchemaPolicyError,
    canonical_json,
    canonical_json_size,
    parse_json_object,
    parse_json_value,
    validate_instance,
    validate_schema,
)


AGENT_CLIENT_CONTRACT = "system-x.agent-client-tools.v1"
STRUCTURED_OUTPUT_CONTRACT = "system-x.structured-output.v1"
OPENAI_TOOL_EXTENSION = "system-x.openai-tools.v1"
ANTHROPIC_TOOL_EXTENSION = "system-x.anthropic-tools.v1"

MAX_TOOLS = 20
MAX_TOOL_CALLS = 8
MAX_TOOL_RESULT_BYTES = 262_144
MAX_AGGREGATE_TOOL_RESULT_BYTES = 1_048_576
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SYSTEM_CALL_ID_PATTERN = re.compile(r"^sx_call_([0-9a-f]{32})$")
OPENAI_CALL_ID_PATTERN = re.compile(r"^call_sx_([0-9a-f]{32})$")
ANTHROPIC_CALL_ID_PATTERN = re.compile(r"^toolu_sx_([0-9a-f]{32})$")
PRIVATE_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


class ToolContractError(ValueError):
    """A bounded canonical tool or structured-output contract failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _custom(kind: str, message: str, param: str) -> PydanticCustomError:
    return PydanticCustomError(kind, message, {"param": param})


class ToolStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class FunctionTool(ToolStrictModel):
    type: Literal["function"] = "function"
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1, max_length=4096)
    parameters: dict[str, Any]
    strict: Literal[True] = True

    @field_validator("description")
    @classmethod
    def description_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool description must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def strict_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise _custom(
                "system_x_tool_schema_invalid",
                str(exc),
                "tools.parameters",
            ) from exc


class ToolChoiceNone(ToolStrictModel):
    type: Literal["none"]


class ToolChoiceAuto(ToolStrictModel):
    type: Literal["auto"]


class ToolChoiceRequired(ToolStrictModel):
    type: Literal["required"]


class ToolChoiceFunction(ToolStrictModel):
    type: Literal["function"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


ToolChoice = Annotated[
    ToolChoiceNone | ToolChoiceAuto | ToolChoiceRequired | ToolChoiceFunction,
    Field(discriminator="type"),
]


class ToolCall(ToolStrictModel):
    id: str = Field(pattern=r"^sx_call_[0-9a-f]{32}$")
    type: Literal["function"] = "function"
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    arguments: dict[str, Any]

    @field_validator("arguments")
    @classmethod
    def arguments_are_finite_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        canonical_json(value)
        return value


class CanonicalMessage(ToolStrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None
    tool_calls: list[ToolCall] | None = Field(default=None, max_length=MAX_TOOL_CALLS)
    reasoning: list[str] = Field(default_factory=list, max_length=8)
    tool_call_id: str | None = Field(
        default=None,
        pattern=r"^sx_call_[0-9a-f]{32}$",
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    is_error: bool | None = None

    @model_validator(mode="after")
    def validate_role_shape(self) -> "CanonicalMessage":
        if any(
            not isinstance(item, str)
            or not item.strip()
            or len(item) > 1_048_576
            for item in self.reasoning
        ):
            raise ValueError("reasoning history is invalid")
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool message requires tool_call_id")
            if self.tool_calls is not None:
                raise ValueError("tool message may not contain tool_calls")
            if self.reasoning:
                raise ValueError("tool message may not contain reasoning")
            if self.is_error is None:
                object.__setattr__(self, "is_error", False)
            canonical_json(self.content)
            return self
        if self.tool_call_id is not None or self.name is not None or self.is_error is not None:
            raise ValueError("non-tool message contains tool-result fields")
        if self.role != "assistant" and self.reasoning:
            raise ValueError("reasoning requires assistant role")
        if self.role == "assistant" and self.tool_calls:
            if self.content is not None and (
                not isinstance(self.content, str)
                or len(self.content) > 1_048_576
            ):
                raise ValueError("assistant tool-call preamble is invalid")
            return self
        if self.tool_calls is not None:
            raise ValueError("tool_calls require assistant role")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("text message content must be non-blank text")
        if len(self.content) > 1_048_576:
            raise ValueError("text message content exceeds the bound")
        return self


class StructuredOutputFormat(ToolStrictModel):
    type: Literal["json_schema"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = Field(default=None, min_length=1, max_length=4096)
    schema_value: dict[str, Any] = Field(alias="schema")
    strict: Literal[True] = True

    @field_validator("schema_value")
    @classmethod
    def strict_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise _custom(
                "system_x_structured_output_schema_invalid",
                str(exc),
                "output_format.schema",
            ) from exc


@dataclass(frozen=True, slots=True)
class ToolHistory:
    unresolved: tuple[ToolCall, ...]
    resolved_ids: tuple[str, ...]


def new_tool_call_id() -> str:
    value = f"sx_call_{secrets.token_hex(16)}"
    if SYSTEM_CALL_ID_PATTERN.fullmatch(value) is None:
        raise RuntimeError("generated tool-call identity is invalid")
    return value


def _suffix(value: str, pattern: re.Pattern[str]) -> str:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError("public tool-call identity is invalid")
    return match.group(1)


def openai_call_id(system_call_id: str) -> str:
    return f"call_sx_{_suffix(system_call_id, SYSTEM_CALL_ID_PATTERN)}"


def system_call_id_from_openai(value: str) -> str:
    return f"sx_call_{_suffix(value, OPENAI_CALL_ID_PATTERN)}"


def anthropic_call_id(system_call_id: str) -> str:
    return f"toolu_sx_{_suffix(system_call_id, SYSTEM_CALL_ID_PATTERN)}"


def system_call_id_from_anthropic(value: str) -> str:
    return f"sx_call_{_suffix(value, ANTHROPIC_CALL_ID_PATTERN)}"


def validate_tools(tools: list[FunctionTool]) -> dict[str, FunctionTool]:
    if len(tools) > MAX_TOOLS:
        raise ToolContractError(
            "system_x_tool_schema_invalid",
            "tool definition count exceeds the configured bound",
        )
    result: dict[str, FunctionTool] = {}
    aggregate = 0
    for tool in tools:
        if tool.name in result:
            raise ToolContractError(
                "system_x_tool_schema_invalid",
                "tool names must be unique",
            )
        aggregate += canonical_json_size(tool.parameters)
        if aggregate > MAX_AGGREGATE_SCHEMA_BYTES:
            raise ToolContractError(
                "system_x_tool_schema_invalid",
                "aggregate tool schema bytes exceed the configured bound",
            )
        result[tool.name] = tool
    return result


def validate_tool_choice(
    choice: ToolChoice | None,
    tools: list[FunctionTool],
) -> ToolChoice:
    selected: ToolChoice = choice or (
        ToolChoiceAuto(type="auto") if tools else ToolChoiceNone(type="none")
    )
    by_name = validate_tools(tools)
    if isinstance(selected, ToolChoiceFunction) and selected.name not in by_name:
        raise ToolContractError(
            "system_x_tool_choice_invalid",
            "forced tool choice names an undeclared function",
        )
    if not tools and not isinstance(selected, ToolChoiceNone):
        raise ToolContractError(
            "system_x_tool_choice_invalid",
            "tool choice requires at least one declared function",
        )
    return selected


def validate_history(
    messages: list[CanonicalMessage],
    tools: list[FunctionTool],
    *,
    maximum_result_bytes: int = MAX_TOOL_RESULT_BYTES,
    maximum_aggregate_result_bytes: int = MAX_AGGREGATE_TOOL_RESULT_BYTES,
) -> ToolHistory:
    definitions = validate_tools(tools)
    unresolved: dict[str, ToolCall] = {}
    seen_calls: set[str] = set()
    resolved: list[str] = []
    aggregate_results = 0
    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_calls:
            if unresolved:
                raise ToolContractError(
                    "system_x_tool_result_missing",
                    "assistant tool calls require prior calls to be resolved",
                )
            for call in message.tool_calls:
                if call.id in seen_calls:
                    raise ToolContractError(
                        "system_x_tool_call_invalid",
                        "tool-call IDs must be unique in history",
                    )
                tool = definitions.get(call.name)
                if tool is None:
                    raise ToolContractError(
                        "system_x_tool_call_invalid",
                        "assistant history calls an undeclared function",
                    )
                try:
                    validate_instance(tool.parameters, call.arguments)
                except SchemaPolicyError as exc:
                    raise ToolContractError(
                        "system_x_tool_arguments_invalid",
                        "assistant history contains invalid tool arguments",
                    ) from exc
                seen_calls.add(call.id)
                unresolved[call.id] = call
            continue
        if message.role == "tool":
            call_id = message.tool_call_id
            if call_id is None or call_id not in unresolved:
                if call_id in resolved:
                    raise ToolContractError(
                        "system_x_tool_result_duplicate",
                        "tool-result ID is duplicated",
                    )
                raise ToolContractError(
                    "system_x_tool_result_mismatch",
                    "tool-result ID does not match an unresolved call",
                )
            call = unresolved[call_id]
            if message.name is not None and message.name != call.name:
                raise ToolContractError(
                    "system_x_tool_result_mismatch",
                    "tool-result name does not match its call",
                )
            result_size = canonical_json_size(message.content)
            if result_size > maximum_result_bytes:
                raise ToolContractError(
                    "system_x_tool_result_mismatch",
                    "tool result exceeds the per-result byte bound",
                )
            aggregate_results += result_size
            if aggregate_results > maximum_aggregate_result_bytes:
                raise ToolContractError(
                    "system_x_tool_result_mismatch",
                    "aggregate tool-result bytes exceed the configured bound",
                )
            del unresolved[call_id]
            resolved.append(call_id)
            continue
        if unresolved:
            raise ToolContractError(
                "system_x_tool_result_missing",
                f"message {index} appears before all tool results",
            )
    if unresolved:
        raise ToolContractError(
            "system_x_tool_result_missing",
            "one or more tool results are missing",
        )
    return ToolHistory((), tuple(resolved))


def validate_returned_tool_calls(
    raw_calls: Any,
    tools: list[FunctionTool],
    choice: ToolChoice,
) -> list[ToolCall]:
    definitions = validate_tools(tools)
    if not isinstance(raw_calls, list) or not 1 <= len(raw_calls) <= MAX_TOOL_CALLS:
        raise ToolContractError(
            "system_x_tool_call_invalid",
            "backend tool-call array is invalid",
        )
    if len(raw_calls) != 1:
        raise ToolContractError(
            "system_x_tool_call_invalid",
            "parallel tool calls are not exposed",
        )
    calls: list[ToolCall] = []
    names: list[str] = []
    private_ids: set[str] = set()
    for raw in raw_calls:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"id", "type", "function"}
            or raw.get("type") != "function"
            or not isinstance(raw.get("id"), str)
            or PRIVATE_CALL_ID_PATTERN.fullmatch(raw["id"]) is None
            or raw["id"] in private_ids
        ):
            raise ToolContractError(
                "system_x_tool_call_invalid",
                "backend returned an invalid function-call identity or shape",
            )
        private_ids.add(raw["id"])
        function = raw.get("function")
        if not isinstance(function, dict) or set(function) != {
            "name",
            "arguments",
        }:
            raise ToolContractError(
                "system_x_tool_call_invalid",
                "backend returned an invalid function-call payload",
            )
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or name not in definitions:
            raise ToolContractError(
                "system_x_tool_call_invalid",
                "backend returned an undeclared function",
            )
        try:
            parsed = parse_json_object(arguments)
            validate_instance(definitions[name].parameters, parsed)
        except SchemaPolicyError as exc:
            raise ToolContractError(
                "system_x_tool_arguments_invalid",
                "backend returned invalid function arguments",
            ) from exc
        calls.append(ToolCall(id=new_tool_call_id(), name=name, arguments=parsed))
        names.append(name)
    if isinstance(choice, ToolChoiceNone):
        raise ToolContractError(
            "system_x_tool_call_invalid",
            "backend returned a tool call when tool choice was none",
        )
    if isinstance(choice, ToolChoiceFunction) and choice.name not in names:
        raise ToolContractError(
            "system_x_tool_call_invalid",
            "backend did not satisfy the forced function choice",
        )
    return calls


def _private_selected_tools(
    tools: list[FunctionTool],
    choice: ToolChoice | None,
) -> list[FunctionTool]:
    definitions = validate_tools(tools)
    if isinstance(choice, ToolChoiceNone):
        return []
    if isinstance(choice, ToolChoiceFunction):
        return [definitions[choice.name]]
    return list(tools)


def private_chat_tools(
    tools: list[FunctionTool],
    choice: ToolChoice | None = None,
) -> list[dict[str, Any]] | None:
    selected = _private_selected_tools(tools, choice)
    if not selected:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": True,
            },
        }
        for tool in selected
    ]


def private_responses_tools(
    tools: list[FunctionTool],
    choice: ToolChoice | None = None,
) -> list[dict[str, Any]] | None:
    selected = _private_selected_tools(tools, choice)
    if not selected:
        return None
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": True,
        }
        for tool in selected
    ]


def private_tool_choice(choice: ToolChoice) -> str:
    if isinstance(choice, ToolChoiceNone):
        return "none"
    if isinstance(choice, ToolChoiceAuto):
        return "auto"
    if isinstance(choice, ToolChoiceRequired):
        return "required"
    return "required"


def _private_tool_result_content(message: CanonicalMessage) -> str:
    if message.role != "tool":
        raise ValueError("private tool result requires a tool message")
    return canonical_json(
        {
            "is_error": bool(message.is_error),
            "result": message.content,
        }
    )


def private_messages(messages: list[CanonicalMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            value = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": canonical_json(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ],
            }
            if message.reasoning:
                value["reasoning_content"] = "\n".join(message.reasoning)
            result.append(value)
        elif message.role == "tool":
            value = {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": _private_tool_result_content(message),
            }
            if message.name is not None:
                value["name"] = message.name
            result.append(value)
        else:
            value = {"role": message.role, "content": message.content}
            if message.role == "assistant" and message.reasoning:
                value["reasoning_content"] = "\n".join(message.reasoning)
            result.append(value)
    return result


def private_responses_input(messages: list[CanonicalMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "assistant":
            for reasoning in message.reasoning:
                result.append(
                    {
                        "type": "reasoning",
                        "summary": [],
                        "content": [
                            {
                                "type": "reasoning_text",
                                "text": reasoning,
                            }
                        ],
                    }
                )
        if message.role == "assistant" and message.tool_calls:
            if message.content:
                result.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": message.content}
                        ],
                    }
                )
            for call in message.tool_calls:
                suffix = _suffix(call.id, SYSTEM_CALL_ID_PATTERN)
                result.append(
                    {
                        "type": "function_call",
                        "id": f"fc_sx_{suffix}",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": canonical_json(call.arguments),
                        "status": "completed",
                    }
                )
        elif message.role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _private_tool_result_content(message),
                }
            )
        elif message.role == "assistant":
            result.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": message.content}
                    ],
                }
            )
        else:
            result.append({"role": message.role, "content": message.content})
    return result


def private_response_format(
    output_format: StructuredOutputFormat,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": output_format.name,
            "schema": output_format.schema_value,
            "strict": True,
        },
    }
    if output_format.description is not None:
        value["json_schema"]["description"] = output_format.description
    return value


def parse_structured_output(
    output_format: StructuredOutputFormat,
    final_content: str,
) -> tuple[str, Any]:
    try:
        parsed = parse_json_value(final_content)
        validate_instance(output_format.schema_value, parsed)
    except SchemaPolicyError as exc:
        raise ToolContractError(
            "system_x_structured_output_invalid",
            "backend structured output is invalid",
        ) from exc
    return canonical_json(parsed), parsed
