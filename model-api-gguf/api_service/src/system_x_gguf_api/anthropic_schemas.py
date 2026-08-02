"""Strict schemas for the bounded local Messages-compatible subset."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .tool_schema import (
    SchemaPolicyError,
    canonical_json,
    validate_schema,
)


ModelReference = Annotated[StrictStr, Field(min_length=1, max_length=1024)]
TextValue = Annotated[StrictStr, Field(min_length=1, max_length=1_048_576)]
SystemText = Annotated[StrictStr, Field(min_length=1, max_length=65_536)]
StopText = Annotated[StrictStr, Field(min_length=1, max_length=256)]
PositiveTokenLimit = Annotated[int, Field(strict=True, ge=1, le=1_048_576)]
Temperature = Annotated[
    float, Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False)
]


def _unsupported(param: str) -> PydanticCustomError:
    return PydanticCustomError(
        "unsupported_parameter",
        f"Parameter '{param}' is unsupported",
        {"param": param},
    )


class AnthropicStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AnthropicModelInfo(AnthropicStrictModel):
    type: Literal["model"] = "model"
    id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=128)
    created_at: str = Field(min_length=20, max_length=40)


class AnthropicModelPage(AnthropicStrictModel):
    data: list[AnthropicModelInfo]
    has_more: Literal[False] = False
    first_id: str | None = Field(default=None, min_length=1, max_length=128)
    last_id: str | None = Field(default=None, min_length=1, max_length=128)


class AnthropicTextBlock(AnthropicStrictModel):
    type: Literal["text"]
    text: TextValue

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AnthropicToolUseBlock(AnthropicStrictModel):
    type: Literal["tool_use"]
    id: str = Field(pattern=r"^toolu_sx_[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    input: dict[str, Any]

    @field_validator("input")
    @classmethod
    def finite_input(cls, value: dict[str, Any]) -> dict[str, Any]:
        canonical_json(value)
        return value


class AnthropicToolResultBlock(AnthropicStrictModel):
    type: Literal["tool_result"]
    tool_use_id: str = Field(pattern=r"^toolu_sx_[0-9a-f]{32}$")
    content: TextValue
    is_error: StrictBool = False


AnthropicInputContentBlock = Annotated[
    AnthropicTextBlock | AnthropicToolUseBlock | AnthropicToolResultBlock,
    Field(discriminator="type"),
]
AnthropicInputContentBlocks = Annotated[
    list[AnthropicInputContentBlock],
    Field(min_length=1, max_length=32),
]


class AnthropicInputMessage(AnthropicStrictModel):
    role: Literal["user", "assistant"]
    content: TextValue | AnthropicInputContentBlocks

    @field_validator("content")
    @classmethod
    def reject_empty_content(
        cls, value: str | list[AnthropicInputContentBlock]
    ) -> str | list[AnthropicInputContentBlock]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("message content must not be blank")
        elif not value:
            raise ValueError("message content blocks must not be empty")
        return value

    @model_validator(mode="after")
    def validate_role_blocks(self) -> "AnthropicInputMessage":
        if isinstance(self.content, str):
            return self
        if self.role == "assistant":
            if any(
                not isinstance(
                    block, (AnthropicTextBlock, AnthropicToolUseBlock)
                )
                for block in self.content
            ):
                raise ValueError(
                    "assistant messages accept only text and tool_use blocks"
                )
            saw_tool = False
            for block in self.content:
                if isinstance(block, AnthropicToolUseBlock):
                    saw_tool = True
                elif saw_tool:
                    raise ValueError(
                        "assistant text must precede tool_use blocks"
                    )
            return self
        if any(
            not isinstance(
                block, (AnthropicTextBlock, AnthropicToolResultBlock)
            )
            for block in self.content
        ):
            raise ValueError(
                "user messages accept only text and tool_result blocks"
            )
        saw_text = False
        for block in self.content:
            if isinstance(block, AnthropicTextBlock):
                saw_text = True
            elif saw_text:
                raise PydanticCustomError(
                    "system_x_tool_result_mismatch",
                    "tool_result blocks must precede text",
                    {"param": "messages"},
                )
        return self


class AnthropicTool(AnthropicStrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1, max_length=4096)
    input_schema: dict[str, Any]
    strict: Literal[True]

    @field_validator("description")
    @classmethod
    def description_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool description must not be blank")
        return value

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise PydanticCustomError(
                "system_x_tool_schema_invalid",
                str(exc),
                {"param": "tools[0].input_schema"},
            ) from exc


class AnthropicToolChoiceNone(AnthropicStrictModel):
    type: Literal["none"]


class AnthropicToolChoiceAuto(AnthropicStrictModel):
    type: Literal["auto"]


class AnthropicToolChoiceAny(AnthropicStrictModel):
    type: Literal["any"]


class AnthropicToolChoiceTool(AnthropicStrictModel):
    type: Literal["tool"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


AnthropicToolChoice = Annotated[
    AnthropicToolChoiceNone
    | AnthropicToolChoiceAuto
    | AnthropicToolChoiceAny
    | AnthropicToolChoiceTool,
    Field(discriminator="type"),
]


class AnthropicJSONOutputFormat(AnthropicStrictModel):
    type: Literal["json_schema"]
    schema_value: dict[str, Any] = Field(alias="schema")

    @field_validator("schema_value")
    @classmethod
    def validate_output_schema(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise PydanticCustomError(
                "system_x_structured_output_schema_invalid",
                str(exc),
                {"param": "output_config.format.schema"},
            ) from exc


class AnthropicOutputConfig(AnthropicStrictModel):
    format: AnthropicJSONOutputFormat


class AnthropicThinkingEnabled(AnthropicStrictModel):
    type: Literal["enabled"]
    budget_tokens: PositiveTokenLimit


class AnthropicRequestBase(AnthropicStrictModel):
    model: ModelReference
    messages: list[AnthropicInputMessage] = Field(min_length=1, max_length=256)
    system: SystemText | list[AnthropicTextBlock] | None = None

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("system")
    @classmethod
    def reject_empty_system(
        cls, value: str | list[AnthropicTextBlock] | None
    ) -> str | list[AnthropicTextBlock] | None:
        if isinstance(value, str) and not value.strip():
            raise ValueError("system must not be blank")
        if isinstance(value, list) and not value:
            raise ValueError("system blocks must not be empty")
        return value

    @model_validator(mode="after")
    def require_final_user(self) -> "AnthropicRequestBase":
        if self.messages[-1].role != "user":
            raise ValueError("messages must end with a user turn")
        return self


class AnthropicMessageRequest(AnthropicRequestBase):
    max_tokens: PositiveTokenLimit
    temperature: Temperature | None = None
    stop_sequences: list[StopText] | None = Field(
        default=None, min_length=1, max_length=16
    )
    stream: StrictBool = False
    tools: list[AnthropicTool] = Field(default_factory=list, max_length=20)
    tool_choice: AnthropicToolChoice | None = None
    output_config: AnthropicOutputConfig | None = None
    thinking: AnthropicThinkingEnabled | None = None

    @field_validator("stop_sequences")
    @classmethod
    def reject_blank_stops(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not item.strip() for item in value):
            raise ValueError("stop sequences must not be blank")
        return value

    @model_validator(mode="after")
    def validate_tool_contract(self) -> "AnthropicMessageRequest":
        if self.thinking is not None:
            raise _unsupported("thinking")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise PydanticCustomError(
                "system_x_tool_schema_invalid",
                "tool names must be unique",
                {"param": "tools"},
            )
        if isinstance(self.tool_choice, AnthropicToolChoiceTool):
            if self.tool_choice.name not in names:
                raise PydanticCustomError(
                    "system_x_tool_choice_invalid",
                    "forced tool choice names an undeclared function",
                    {"param": "tool_choice"},
                )
        elif isinstance(
            self.tool_choice,
            (AnthropicToolChoiceAuto, AnthropicToolChoiceAny),
        ) and not self.tools:
            raise PydanticCustomError(
                "system_x_tool_choice_invalid",
                "tool choice requires at least one tool",
                {"param": "tool_choice"},
            )
        if self.tools and self.output_config is not None:
            raise PydanticCustomError(
                "system_x_tool_and_output_format_conflict",
                "tools and structured output cannot be combined",
                {"param": "output_config"},
            )
        return self


class AnthropicCountTokensRequest(AnthropicRequestBase):
    tools: list[AnthropicTool] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_tool_definitions(self) -> "AnthropicCountTokensRequest":
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise PydanticCustomError(
                "system_x_tool_schema_invalid",
                "tool names must be unique",
                {"param": "tools"},
            )
        return self


class AnthropicOutputTextBlock(AnthropicStrictModel):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=4_194_304)


class AnthropicOutputThinkingBlock(AnthropicStrictModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = Field(min_length=1, max_length=4_194_304)


class AnthropicOutputToolUseBlock(AnthropicStrictModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(pattern=r"^toolu_sx_[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    input: dict[str, Any]


AnthropicOutputContentBlock = Annotated[
    AnthropicOutputThinkingBlock
    | AnthropicOutputTextBlock
    | AnthropicOutputToolUseBlock,
    Field(discriminator="type"),
]


class AnthropicUsage(AnthropicStrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class AnthropicMessage(AnthropicStrictModel):
    id: str = Field(pattern=r"^msg_sx_[0-9a-f]{32}$")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicOutputContentBlock] = Field(max_length=10)
    model: str = Field(min_length=1, max_length=128)
    stop_reason: Literal[
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "model_context_window_exceeded",
        "tool_use",
    ]
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicMessageTokensCount(AnthropicStrictModel):
    input_tokens: int = Field(ge=0)


AnthropicErrorType = Literal[
    "invalid_request_error",
    "not_found_error",
    "conflict_error",
    "request_too_large",
    "overloaded_error",
    "timeout_error",
    "api_error",
]


class AnthropicErrorDetail(AnthropicStrictModel):
    type: AnthropicErrorType
    message: str = Field(min_length=1, max_length=240)


class AnthropicErrorResponse(AnthropicStrictModel):
    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
    request_id: str = Field(pattern=r"^req_sx_[0-9a-f]{32}$")
