"""Strict schemas for the explicitly bounded OpenAI-compatible v1 subset."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .tool_schema import SchemaPolicyError, validate_schema


PositiveTokenLimit = Annotated[
    int,
    Field(strict=True, ge=1, le=1_048_576),
]
Temperature = Annotated[
    float,
    Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False),
]
ModelReference = Annotated[StrictStr, Field(min_length=1, max_length=1024)]
PromptText = Annotated[
    StrictStr,
    Field(min_length=1, max_length=1_048_576),
]
InstructionText = Annotated[
    StrictStr,
    Field(min_length=1, max_length=65_536),
]
StopText = Annotated[StrictStr, Field(min_length=1, max_length=256)]
PublicModelId = Annotated[StrictStr, Field(min_length=1, max_length=128)]
ObjectId = Annotated[StrictStr, Field(min_length=1, max_length=128)]


def _unsupported(param: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(
        "unsupported_parameter",
        message,
        {"param": param},
    )


class OpenAIStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OpenAIStreamOptions(OpenAIStrictModel):
    include_usage: StrictBool = False


class OpenAICompletionRequest(OpenAIStrictModel):
    model: ModelReference
    prompt: PromptText
    max_tokens: PositiveTokenLimit
    temperature: Temperature | None = None
    stop: StopText | list[StopText] | None = Field(
        default=None,
        min_length=1,
        max_length=16,
    )
    stream: StrictBool = False
    stream_options: OpenAIStreamOptions | None = None
    n: StrictInt = 1
    echo: StrictBool = False
    logprobs: Any = None
    suffix: Any = None
    best_of: StrictInt = 1

    @field_validator("model", "prompt")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("stop")
    @classmethod
    def reject_blank_stop(
        cls, value: str | list[str] | None
    ) -> str | list[str] | None:
        values = [value] if isinstance(value, str) else value
        if values is not None and any(not item.strip() for item in values):
            raise ValueError("stop values must not be blank")
        return value

    @field_validator("n")
    @classmethod
    def require_one_choice(cls, value: int) -> int:
        if value != 1:
            raise _unsupported("n", "n must equal 1")
        return value

    @field_validator("echo")
    @classmethod
    def reject_echo(cls, value: bool) -> bool:
        if value:
            raise _unsupported("echo", "echo is unsupported")
        return value

    @field_validator("logprobs")
    @classmethod
    def reject_logprobs(cls, value: Any) -> None:
        if value is not None:
            raise _unsupported("logprobs", "logprobs is unsupported")
        return None

    @field_validator("suffix")
    @classmethod
    def reject_suffix(cls, value: Any) -> None:
        if value is not None:
            raise _unsupported("suffix", "suffix is unsupported")
        return None

    @field_validator("best_of")
    @classmethod
    def require_one_best_of(cls, value: int) -> int:
        if value != 1:
            raise _unsupported("best_of", "best_of must equal 1")
        return value

    @model_validator(mode="after")
    def validate_stream_options(self) -> "OpenAICompletionRequest":
        if self.stream_options is not None and not self.stream:
            raise _unsupported(
                "stream_options",
                "stream_options requires stream=true",
            )
        return self


def _schema_invalid(param: str, exc: SchemaPolicyError) -> PydanticCustomError:
    return PydanticCustomError(
        "system_x_tool_schema_invalid",
        str(exc),
        {"param": param},
    )


def _structured_schema_invalid(
    param: str, exc: SchemaPolicyError
) -> PydanticCustomError:
    return PydanticCustomError(
        "system_x_structured_output_schema_invalid",
        str(exc),
        {"param": param},
    )


class OpenAIFunctionDefinition(OpenAIStrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1, max_length=4096)
    parameters: dict[str, Any]
    strict: Literal[True]

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("function description must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise _schema_invalid(
                "tools[0].function.parameters", exc
            ) from exc


class OpenAIChatTool(OpenAIStrictModel):
    type: Literal["function"]
    function: OpenAIFunctionDefinition


class OpenAIResponsesTool(OpenAIStrictModel):
    type: Literal["function"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1, max_length=4096)
    parameters: dict[str, Any]
    strict: Literal[True]

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("function description must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise _schema_invalid("tools[0].parameters", exc) from exc


class OpenAIChatFunctionChoice(OpenAIStrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class OpenAIChatNamedToolChoice(OpenAIStrictModel):
    type: Literal["function"]
    function: OpenAIChatFunctionChoice


OpenAIChatToolChoice = (
    Literal["none", "auto", "required"] | OpenAIChatNamedToolChoice
)


class OpenAIResponsesNamedToolChoice(OpenAIStrictModel):
    type: Literal["function"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


OpenAIResponsesToolChoice = (
    Literal["none", "auto", "required"] | OpenAIResponsesNamedToolChoice
)


class OpenAIChatToolCallFunction(OpenAIStrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    arguments: str = Field(min_length=2, max_length=1_048_576)


class OpenAIChatToolCall(OpenAIStrictModel):
    id: str = Field(pattern=r"^call_sx_[0-9a-f]{32}$")
    type: Literal["function"]
    function: OpenAIChatToolCallFunction


class OpenAIChatTextMessage(OpenAIStrictModel):
    role: Literal["developer", "system", "user"]
    content: PromptText

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class OpenAIChatAssistantInput(OpenAIStrictModel):
    role: Literal["assistant"]
    content: str | None = Field(default=None, max_length=1_048_576)
    tool_calls: list[OpenAIChatToolCall] | None = Field(
        default=None, min_length=1, max_length=8
    )
    refusal: None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self) -> "OpenAIChatAssistantInput":
        if isinstance(self.content, str) and not self.content.strip():
            raise ValueError("assistant content must not be blank")
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant message requires content or tool calls")
        return self


class OpenAIChatToolMessage(OpenAIStrictModel):
    role: Literal["tool"]
    content: PromptText
    tool_call_id: str = Field(pattern=r"^call_sx_[0-9a-f]{32}$")


OpenAIChatMessage = Annotated[
    OpenAIChatTextMessage | OpenAIChatAssistantInput | OpenAIChatToolMessage,
    Field(discriminator="role"),
]


class OpenAIJSONSchemaDefinition(OpenAIStrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = Field(default=None, min_length=1, max_length=4096)
    schema_value: dict[str, Any] = Field(alias="schema")
    strict: Literal[True]

    @field_validator("schema_value")
    @classmethod
    def validate_output_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise _structured_schema_invalid(
                "response_format.json_schema.schema", exc
            ) from exc


class OpenAIChatResponseFormatText(OpenAIStrictModel):
    type: Literal["text"]


class OpenAIChatResponseFormatJSONSchema(OpenAIStrictModel):
    type: Literal["json_schema"]
    json_schema: OpenAIJSONSchemaDefinition


OpenAIChatResponseFormat = Annotated[
    OpenAIChatResponseFormatText | OpenAIChatResponseFormatJSONSchema,
    Field(discriminator="type"),
]


class OpenAIChatCompletionRequest(OpenAIStrictModel):
    model: ModelReference
    messages: list[OpenAIChatMessage] = Field(min_length=1, max_length=256)
    max_tokens: PositiveTokenLimit | None = None
    max_completion_tokens: PositiveTokenLimit | None = None
    temperature: Temperature | None = None
    stop: StopText | list[StopText] | None = Field(
        default=None,
        min_length=1,
        max_length=16,
    )
    stream: StrictBool = False
    stream_options: OpenAIStreamOptions | None = None
    n: StrictInt = 1
    logprobs: StrictBool = False
    top_logprobs: StrictInt = 0
    parallel_tool_calls: StrictBool = False
    tools: list[OpenAIChatTool] = Field(default_factory=list, max_length=20)
    tool_choice: OpenAIChatToolChoice | None = None
    response_format: OpenAIChatResponseFormat | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("stop")
    @classmethod
    def reject_blank_stop(
        cls, value: str | list[str] | None
    ) -> str | list[str] | None:
        values = [value] if isinstance(value, str) else value
        if values is not None and any(not item.strip() for item in values):
            raise ValueError("stop values must not be blank")
        return value

    @field_validator("n")
    @classmethod
    def require_one_choice(cls, value: int) -> int:
        if value != 1:
            raise _unsupported("n", "n must equal 1")
        return value

    @field_validator("logprobs")
    @classmethod
    def reject_logprobs(cls, value: bool) -> bool:
        if value:
            raise _unsupported("logprobs", "logprobs is unsupported")
        return value

    @field_validator("top_logprobs")
    @classmethod
    def reject_top_logprobs(cls, value: int) -> int:
        if value != 0:
            raise _unsupported("top_logprobs", "top_logprobs is unsupported")
        return value

    @field_validator("parallel_tool_calls")
    @classmethod
    def reject_parallel_tools(cls, value: bool) -> bool:
        if value:
            raise _unsupported(
                "parallel_tool_calls",
                "parallel tool calls are unsupported",
            )
        return value

    @model_validator(mode="after")
    def require_one_token_limit(self) -> "OpenAIChatCompletionRequest":
        supplied = (
            self.max_tokens is not None,
            self.max_completion_tokens is not None,
        )
        if supplied == (False, False):
            raise PydanticCustomError(
                "invalid_request",
                "exactly one token limit is required",
                {"param": "max_tokens"},
            )
        if (
            supplied == (True, True)
            and self.max_tokens != self.max_completion_tokens
        ):
            raise _unsupported(
                "max_completion_tokens",
                "max_tokens and max_completion_tokens conflict",
            )
        if self.reasoning_effort is not None:
            raise _unsupported(
                "reasoning_effort",
                "reasoning effort control is unavailable",
            )
        names = [tool.function.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise PydanticCustomError(
                "system_x_tool_schema_invalid",
                "tool names must be unique",
                {"param": "tools"},
            )
        if isinstance(self.tool_choice, OpenAIChatNamedToolChoice):
            if self.tool_choice.function.name not in names:
                raise PydanticCustomError(
                    "system_x_tool_choice_invalid",
                    "forced tool choice names an undeclared function",
                    {"param": "tool_choice"},
                )
        elif self.tool_choice in {"auto", "required"} and not self.tools:
            raise PydanticCustomError(
                "system_x_tool_choice_invalid",
                "tool choice requires at least one function",
                {"param": "tool_choice"},
            )
        if (
            self.tools
            and isinstance(
                self.response_format,
                OpenAIChatResponseFormatJSONSchema,
            )
        ):
            raise PydanticCustomError(
                "system_x_tool_and_output_format_conflict",
                "tools and structured output cannot be combined",
                {"param": "response_format"},
            )
        if self.stream_options is not None and not self.stream:
            raise _unsupported(
                "stream_options",
                "stream_options requires stream=true",
            )
        return self

    @property
    def output_limit(self) -> int:
        value = (
            self.max_tokens
            if self.max_tokens is not None
            else self.max_completion_tokens
        )
        if value is None:
            raise RuntimeError("validated chat token limit is unavailable")
        return value


class OpenAIResponsesEasyInputMessage(OpenAIStrictModel):
    type: Literal["message"] | None = None
    role: Literal["developer", "system", "user", "assistant"]
    content: PromptText


class OpenAIResponseReasoningText(OpenAIStrictModel):
    type: Literal["reasoning_text"]
    text: PromptText


class OpenAIResponseReasoningItem(OpenAIStrictModel):
    id: ObjectId | None = None
    type: Literal["reasoning"]
    summary: list[Any] = Field(default_factory=list, max_length=0)
    content: list[OpenAIResponseReasoningText] = Field(
        min_length=1, max_length=8
    )
    encrypted_content: str | None = Field(default=None, max_length=1_048_576)
    status: Literal["completed", "incomplete"] | None = None


class OpenAIResponseOutputText(OpenAIStrictModel):
    type: Literal["output_text"] = "output_text"
    text: str = Field(min_length=1, max_length=4_194_304)
    annotations: list[Any] = Field(default_factory=list, max_length=0)


class OpenAIResponseOutputMessage(OpenAIStrictModel):
    id: ObjectId
    type: Literal["message"] = "message"
    status: Literal["completed", "incomplete"]
    role: Literal["assistant"] = "assistant"
    content: list[OpenAIResponseOutputText] = Field(
        min_length=1,
        max_length=1,
    )


class OpenAIResponseFunctionCallItem(OpenAIStrictModel):
    id: str = Field(pattern=r"^fc_sx_[0-9a-f]{32}$")
    type: Literal["function_call"]
    call_id: str = Field(pattern=r"^call_sx_[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    arguments: str = Field(min_length=2, max_length=1_048_576)
    status: Literal["completed"]


class OpenAIResponseFunctionCallOutput(OpenAIStrictModel):
    type: Literal["function_call_output"]
    call_id: str = Field(pattern=r"^call_sx_[0-9a-f]{32}$")
    output: PromptText


OpenAIResponsesInputItem = (
    OpenAIResponseOutputMessage
    | OpenAIResponsesEasyInputMessage
    | OpenAIResponseReasoningItem
    | OpenAIResponseFunctionCallItem
    | OpenAIResponseFunctionCallOutput
)


class OpenAIResponsesJSONSchemaFormat(OpenAIStrictModel):
    type: Literal["json_schema"]
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = Field(default=None, min_length=1, max_length=4096)
    schema_value: dict[str, Any] = Field(alias="schema")
    strict: Literal[True]

    @field_validator("schema_value")
    @classmethod
    def validate_output_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_schema(value, strict=True, require_object=True)
        except SchemaPolicyError as exc:
            raise _structured_schema_invalid(
                "text.format.schema", exc
            ) from exc


class OpenAIResponsesTextFormat(OpenAIStrictModel):
    format: OpenAIResponsesJSONSchemaFormat


class OpenAIResponsesReasoning(OpenAIStrictModel):
    effort: Literal["low", "medium", "high"]


class OpenAIResponsesRequest(OpenAIStrictModel):
    model: ModelReference
    input: PromptText | list[OpenAIResponsesInputItem] = Field(
        min_length=1, max_length=512
    )
    instructions: InstructionText | None = None
    max_output_tokens: PositiveTokenLimit
    temperature: Temperature | None = None
    stream: StrictBool = False
    tools: list[OpenAIResponsesTool] = Field(default_factory=list, max_length=20)
    tool_choice: OpenAIResponsesToolChoice | None = None
    text: OpenAIResponsesTextFormat | None = None
    parallel_tool_calls: StrictBool = False
    background: StrictBool = False
    store: StrictBool = False
    reasoning: OpenAIResponsesReasoning | None = None

    @field_validator("model", "instructions")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("input")
    @classmethod
    def reject_blank_input(
        cls, value: str | list[OpenAIResponsesInputItem]
    ) -> str | list[OpenAIResponsesInputItem]:
        if isinstance(value, str) and not value.strip():
            raise ValueError("input must not be blank")
        return value

    @field_validator("parallel_tool_calls")
    @classmethod
    def reject_parallel_tools(cls, value: bool) -> bool:
        if value:
            raise _unsupported(
                "parallel_tool_calls",
                "parallel tool calls are unsupported",
            )
        return value

    @field_validator("background")
    @classmethod
    def reject_background(cls, value: bool) -> bool:
        if value:
            raise _unsupported("background", "background mode is unsupported")
        return value

    @field_validator("store")
    @classmethod
    def reject_store(cls, value: bool) -> bool:
        if value:
            raise _unsupported("store", "stored responses are unsupported")
        return value

    @model_validator(mode="after")
    def validate_tool_contract(self) -> "OpenAIResponsesRequest":
        if self.reasoning is not None:
            raise _unsupported(
                "reasoning",
                "reasoning effort control is unavailable",
            )
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise PydanticCustomError(
                "system_x_tool_schema_invalid",
                "tool names must be unique",
                {"param": "tools"},
            )
        if isinstance(self.tool_choice, OpenAIResponsesNamedToolChoice):
            if self.tool_choice.name not in names:
                raise PydanticCustomError(
                    "system_x_tool_choice_invalid",
                    "forced tool choice names an undeclared function",
                    {"param": "tool_choice"},
                )
        elif self.tool_choice in {"auto", "required"} and not self.tools:
            raise PydanticCustomError(
                "system_x_tool_choice_invalid",
                "tool choice requires at least one function",
                {"param": "tool_choice"},
            )
        if self.tools and self.text is not None:
            raise PydanticCustomError(
                "system_x_tool_and_output_format_conflict",
                "tools and structured output cannot be combined",
                {"param": "text"},
            )
        return self


class OpenAIModel(OpenAIStrictModel):
    id: PublicModelId
    object: Literal["model"] = "model"
    created: int = Field(ge=0)
    owned_by: Literal["system-x"] = "system-x"


class OpenAIModelList(OpenAIStrictModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class OpenAIUsage(OpenAIStrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def require_consistent_total(self) -> "OpenAIUsage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens is inconsistent")
        return self


class OpenAICompletionChoice(OpenAIStrictModel):
    text: str
    index: Literal[0] = 0
    logprobs: None = None
    finish_reason: Literal["stop", "length"]


class OpenAICompletion(OpenAIStrictModel):
    id: ObjectId
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(ge=0)
    model: PublicModelId
    choices: list[OpenAICompletionChoice] = Field(min_length=1, max_length=1)
    usage: OpenAIUsage


class OpenAIChatAssistantMessage(OpenAIStrictModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    refusal: None = None
    reasoning_content: str | None = None
    tool_calls: list[OpenAIChatToolCall] | None = Field(
        default=None, min_length=1, max_length=8
    )


class OpenAIChatChoice(OpenAIStrictModel):
    index: Literal[0] = 0
    message: OpenAIChatAssistantMessage
    logprobs: None = None
    finish_reason: Literal["stop", "length", "tool_calls"]


class OpenAIChatCompletion(OpenAIStrictModel):
    id: ObjectId
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(ge=0)
    model: PublicModelId
    choices: list[OpenAIChatChoice] = Field(min_length=1, max_length=1)
    usage: OpenAIUsage


class OpenAIResponseIncompleteDetails(OpenAIStrictModel):
    reason: Literal["max_output_tokens"]


OpenAIResponseOutputItem = Annotated[
    OpenAIResponseOutputMessage
    | OpenAIResponseReasoningItem
    | OpenAIResponseFunctionCallItem,
    Field(discriminator="type"),
]


class OpenAIResponse(OpenAIStrictModel):
    id: ObjectId
    object: Literal["response"] = "response"
    created_at: float = Field(ge=0.0)
    status: Literal["completed", "incomplete"]
    error: None = None
    incomplete_details: OpenAIResponseIncompleteDetails | None = None
    instructions: str | None
    model: PublicModelId
    output: list[OpenAIResponseOutputItem] = Field(max_length=17)
    parallel_tool_calls: Literal[False] = False
    tool_choice: OpenAIResponsesToolChoice = "none"
    tools: list[OpenAIResponsesTool] = Field(default_factory=list, max_length=20)
    max_output_tokens: PositiveTokenLimit


class OpenAIErrorDetail(OpenAIStrictModel):
    message: str = Field(min_length=1, max_length=240)
    type: Literal[
        "invalid_request_error",
        "server_error",
        "conflict_error",
    ]
    param: str | None
    code: str = Field(min_length=1, max_length=64)


class OpenAIErrorResponse(OpenAIStrictModel):
    error: OpenAIErrorDetail
