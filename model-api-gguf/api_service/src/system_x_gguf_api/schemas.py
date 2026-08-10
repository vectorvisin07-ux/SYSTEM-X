"""Strict public schemas for the System X native GGUF API contract."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .operation_records import OperationRecordSchema
from .tool_contract import (
    CanonicalMessage,
    FunctionTool,
    StructuredOutputFormat,
    ToolCall,
    ToolChoice,
    validate_tool_choice,
    validate_tools,
)


RequestId = Annotated[
    str,
    StringConstraints(pattern=r"^sx_req_[0-9a-f]{32}$"),
]
ModelReference = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
PublicModelId = Annotated[str, StringConstraints(min_length=1, max_length=128)]
PublicAlias = Annotated[str, StringConstraints(min_length=1, max_length=64)]
InputText = Annotated[str, StringConstraints(min_length=1, max_length=1_048_576)]
InstructionText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=65_536),
]
ReasoningText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1_048_576),
]
StopText = Annotated[str, StringConstraints(min_length=1, max_length=256)]
CapabilityState = Literal["available", "unavailable", "not_tested", "not_exposed"]
RuntimeState = Literal["unloaded", "loading", "loaded", "unavailable"]
ServiceReadinessState = Literal[
    "SUPERVISOR_STARTING",
    "API_STARTING",
    "BACKEND_STARTING",
    "MODEL_LOADING",
    "WAITING_FOR_MODEL",
    "MODEL_CANDIDATE_LOADING",
    "READY",
    "DEGRADED",
    "STOPPED",
    "FAIL_CLOSED",
]
ModelServiceState = Literal[
    "WAITING_FOR_MODEL",
    "MODEL_CANDIDATE_LOADING",
    "READY",
    "DEGRADED",
    "STOPPED",
    "FAIL_CLOSED",
]
InferenceStatus = Literal["completed", "incomplete", "requires_action", "error"]
FinishReason = Literal[
    "completed",
    "output_limit",
    "stop_sequence",
    "context_limit",
    "tool_call",
    "unknown",
]
SystemXErrorCode = Literal[
    "system_x_authentication_error",
    "system_x_validation_error",
    "system_x_route_not_found",
    "system_x_method_not_allowed",
    "system_x_model_not_found",
    "system_x_no_ready_model",
    "system_x_model_unavailable",
    "system_x_model_conflict",
    "system_x_capability_unavailable",
    "system_x_backend_unavailable",
    "system_x_backend_timeout",
    "system_x_backend_response_invalid",
    "system_x_output_invalid",
    "system_x_tool_schema_invalid",
    "system_x_tool_choice_invalid",
    "system_x_tool_capability_unavailable",
    "system_x_tool_call_invalid",
    "system_x_tool_arguments_invalid",
    "system_x_tool_result_mismatch",
    "system_x_tool_result_duplicate",
    "system_x_tool_result_missing",
    "system_x_structured_output_schema_invalid",
    "system_x_structured_output_invalid",
    "system_x_streaming_structured_output_unsupported",
    "system_x_tool_and_output_format_conflict",
    "system_x_internal_error",
    "system_x_request_too_large",
    "system_x_token_budget_exceeded",
    "system_x_concurrency_limit_exceeded",
    "system_x_rate_limit_exceeded",
    "system_x_request_deadline_exceeded",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthDetail(StrictModel):
    public_listener_configured: bool


class WarmIdentityResponse(StrictModel):
    requested_alias: PublicAlias
    resolved_public_model_id: PublicModelId
    artifact_version_id: str = Field(min_length=1, max_length=128)
    registry_generation: int = Field(ge=0)
    capability_manifest_identity: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    router_transaction_id: str = Field(min_length=1, max_length=128)
    model_child_pid: int = Field(ge=1)
    model_child_start_identity: str = Field(min_length=1, max_length=128)
    model_child_parent: int = Field(ge=1)
    model_child_process_group: int = Field(ge=1)
    model_child_session: int = Field(ge=1)
    warm_since_utc: str
    last_verified_utc: str
    health_state: Literal["ready", "stopped"]


class HealthResponse(StrictModel):
    request_id: RequestId
    service_name: str
    service_readiness_state: ServiceReadinessState
    model_service_state: ModelServiceState
    service_available: bool
    inference_ready: bool
    ready: bool
    service_status: Literal["starting", "ready", "degraded", "stopped"]
    contract_version: str
    backend_integration: Literal["llama-server-router"] = "llama-server-router"
    backend_status: Literal["disabled", "router_ready", "unavailable"]
    backend_process_running: bool
    backend_control_plane_ready: bool
    loaded_model_count: int = Field(ge=0, le=1)
    maximum_loaded_models: Literal[1] = 1
    model_ready: bool
    environment_name: str
    registry_status: Literal["disabled", "starting", "ready", "degraded"]
    registered_model_count: int = Field(ge=0)
    ready_model_count: int = Field(ge=0)
    candidate_model_count: int = Field(ge=0)
    rejected_artifact_count: int = Field(ge=0)
    registry_generation: int = Field(ge=0)
    last_reconcile_utc: str | None = None
    default_alias: PublicAlias
    configured_default_alias: PublicAlias
    resolved_default_alias: PublicAlias | None = None
    resolved_public_model_id: PublicModelId | None = None
    artifact_version_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    reason_code: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    warm_identity: WarmIdentityResponse | None = None
    recovery_state: Literal[
        "IDLE",
        "DETECTED",
        "DELAYING",
        "RESTARTING_ROUTER",
        "RELOADING_MODEL",
        "VERIFYING",
        "RECOVERED",
        "FAIL_CLOSED",
        "STOPPED",
    ] = "IDLE"
    recovery_reason_code: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    recovery_attempt: int = Field(default=0, ge=0, le=16)
    detail: HealthDetail | None = None


class VersionResponse(StrictModel):
    request_id: RequestId
    service_name: str
    service_version: str
    contract_version: str
    request_governance_contract: Literal["system-x.request-governance.v1"]
    authentication_contract: Literal["system-x.private-authentication.v1"]
    authentication_enabled: bool
    operation_record_contract: OperationRecordSchema
    model_adaptation_contract: Literal["system-x.gguf-model-adaptation.v1"]
    compatibility_version: Literal["system-x.openai-compatible.v1"]
    anthropic_compatibility_version: Literal["system-x.anthropic-compatible.v1"]
    agent_client_contract: Literal["system-x.agent-client-tools.v1"]
    structured_output_contract: Literal["system-x.structured-output.v1"]
    streaming_contract: Literal["system-x.streaming.v1"]
    openai_streaming_contract: Literal["system-x.openai-streaming.v1"]
    anthropic_streaming_contract: Literal["system-x.anthropic-streaming.v1"]
    openai_tool_extension: Literal["system-x.openai-tools.v1"]
    anthropic_tool_extension: Literal["system-x.anthropic-tools.v1"]
    python_version: str
    fastapi_version: str
    uvicorn_version: str
    pydantic_version: str
    private_backend_adapter: Literal["llama-server-router"] = "llama-server-router"
    registry_schema_identity: Literal["system-x.gguf-model-registry.v1"]
    registry_schema_version: Literal[2] = 2
    watchfiles_version: str
    sqlite_library_version: str


class ErrorField(StrictModel):
    location: list[str | int] = Field(min_length=1, max_length=16)
    issue: str = Field(min_length=1, max_length=64)


class SystemXErrorDetail(StrictModel):
    code: SystemXErrorCode
    message: str = Field(min_length=1, max_length=240)
    retryable: bool
    fields: list[ErrorField] | None = Field(default=None, max_length=16)


class SystemXErrorResponse(StrictModel):
    request_id: RequestId
    error: SystemXErrorDetail


class ModelCapabilities(StrictModel):
    generate: CapabilityState
    chat: CapabilityState
    responses: CapabilityState
    token_count: CapabilityState
    tool_calling: CapabilityState
    structured_output: CapabilityState
    parallel_tool_calling: CapabilityState
    streaming: CapabilityState


class PlannerModelCapabilities(StrictModel):
    chat: CapabilityState
    completion: CapabilityState
    responses: CapabilityState
    streaming: CapabilityState
    tool_calling: CapabilityState
    structured_output: CapabilityState
    reasoning_output: CapabilityState
    reasoning_control: CapabilityState


class PlannerDefaultGeneration(StrictModel):
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, allow_inf_nan=False
    )


class TokenCountOperations(StrictModel):
    system_x_native: CapabilityState
    messages_compatible: CapabilityState
    openai_compatible: CapabilityState


class PublicModelRecord(StrictModel):
    id: PublicModelId
    aliases: list[PublicAlias]
    branch: Literal["gguf"] = "gguf"
    registration_state: Literal["ready"] = "ready"
    runtime_state: RuntimeState
    capabilities: ModelCapabilities


class PublicModelDetail(StrictModel):
    public_model_id: PublicModelId
    requested_alias: PublicAlias | None = None
    resolved_model_id: PublicModelId
    artifact_version_id: str = Field(min_length=1, max_length=128)
    state: Literal["ready"] = "ready"
    runtime_state: RuntimeState
    capabilities: PlannerModelCapabilities
    reasoning_formats: list[str] = Field(max_length=8)
    reasoning_control_modes: list[str] = Field(max_length=8)
    context_window_tokens: int | None = Field(default=None, ge=1)
    model_declared_context_window_tokens: int | None = Field(
        default=None, ge=1
    )
    active_context_window_tokens: int | None = Field(default=None, ge=1)
    maximum_output_tokens: int | None = Field(default=None, ge=1)
    default_generation: PlannerDefaultGeneration
    token_count_operations: TokenCountOperations


class ModelListResponse(StrictModel):
    request_id: RequestId
    object: Literal["system_x.model.list"] = "system_x.model.list"
    registry_generation: int = Field(ge=0)
    models: list[PublicModelRecord]


class ModelDetailResponse(StrictModel):
    request_id: RequestId
    object: Literal["system_x.model"] = "system_x.model"
    registry_generation: int = Field(ge=0)
    model: PublicModelDetail


class ReasoningRequest(StrictModel):
    mode: Literal["standard", "pro_extended", "custom"]
    effort: Literal["low", "medium", "high"] | None = None
    budget_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    final_answer_reserve_tokens: int | None = Field(
        default=None, ge=0, le=1_048_576
    )


class InferenceRequestBase(StrictModel):
    model: ModelReference
    max_output_tokens: int = Field(ge=1, le=1_048_576)
    stream: bool = False
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        allow_inf_nan=False,
    )

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value


class StopRequestBase(InferenceRequestBase):
    stop: list[StopText] | None = Field(default=None, min_length=1, max_length=16)

    @field_validator("stop")
    @classmethod
    def reject_blank_stops(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not item.strip() for item in value):
            raise ValueError("stop values must not be blank")
        return value


class GenerateRequest(StopRequestBase):
    input: InputText

    @field_validator("input")
    @classmethod
    def reject_blank_input(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input must not be blank")
        return value


class ChatMessage(CanonicalMessage):
    """Canonical text, assistant tool-call, or client tool-result turn."""


class ChatRequest(StopRequestBase):
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    reasoning: ReasoningRequest | None = None
    tools: list[FunctionTool] = Field(default_factory=list, max_length=20)
    tool_choice: ToolChoice | None = None
    output_format: StructuredOutputFormat | None = None

    @model_validator(mode="after")
    def validate_tool_request(self) -> "ChatRequest":
        validate_tools(self.tools)
        validate_tool_choice(self.tool_choice, self.tools)
        if self.tools and self.output_format is not None:
            raise ValueError("tools and output_format cannot be combined")
        return self


class ResponsesRequest(InferenceRequestBase):
    input: InputText | list[ChatMessage]
    instructions: InstructionText | None = None
    tools: list[FunctionTool] = Field(default_factory=list, max_length=20)
    tool_choice: ToolChoice | None = None
    output_format: StructuredOutputFormat | None = None
    reasoning: ReasoningRequest | None = None

    @field_validator("instructions")
    @classmethod
    def reject_blank_responses_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("input")
    @classmethod
    def reject_blank_response_input(
        cls, value: str | list[ChatMessage]
    ) -> str | list[ChatMessage]:
        if isinstance(value, str) and not value.strip():
            raise ValueError("text must not be blank")
        if isinstance(value, list) and not value:
            raise ValueError("response input messages must not be empty")
        return value

    @model_validator(mode="after")
    def validate_tool_request(self) -> "ResponsesRequest":
        validate_tools(self.tools)
        validate_tool_choice(self.tool_choice, self.tools)
        if self.tools and self.output_format is not None:
            raise ValueError("tools and output_format cannot be combined")
        return self


class GenerateTokenCountRequest(StrictModel):
    model: ModelReference
    operation: Literal["generate"]
    input: InputText

    @field_validator("model", "input")
    @classmethod
    def reject_blank_generate_count(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class ChatTokenCountRequest(StrictModel):
    model: ModelReference
    operation: Literal["chat"]
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    tools: list[FunctionTool] = Field(default_factory=list, max_length=20)

    @field_validator("model")
    @classmethod
    def reject_blank_chat_count_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value


class ResponsesTokenCountRequest(StrictModel):
    model: ModelReference
    operation: Literal["responses"]
    input: InputText
    instructions: InstructionText | None = None

    @field_validator("model", "input", "instructions")
    @classmethod
    def reject_blank_responses_count(
        cls, value: str | None
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value


TokenCountRequest = Annotated[
    GenerateTokenCountRequest | ChatTokenCountRequest | ResponsesTokenCountRequest,
    Field(discriminator="operation"),
]


class TokenUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_consistent_total(self) -> "TokenUsage":
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens is inconsistent")
        return self


class TextOutput(StrictModel):
    text: str = Field(min_length=1, max_length=4_194_304)


class AssistantOutput(StrictModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = Field(default=None, max_length=4_194_304)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=8)
    reasoning: list[ReasoningText] = Field(default_factory=list, max_length=8)
    structured: Any | None = None


class GenerationResponse(StrictModel):
    request_id: RequestId
    object: Literal["system_x.generation"] = "system_x.generation"
    status: InferenceStatus
    model: PublicModelId
    output: TextOutput
    finish_reason: FinishReason
    usage: TokenUsage


class ChatResponse(StrictModel):
    request_id: RequestId
    object: Literal["system_x.chat"] = "system_x.chat"
    status: InferenceStatus
    model: PublicModelId
    output: AssistantOutput
    finish_reason: FinishReason
    usage: TokenUsage

    @model_validator(mode="after")
    def require_consistent_turn_state(self) -> "ChatResponse":
        if self.status == "requires_action":
            if self.finish_reason != "tool_call" or not self.output.tool_calls:
                raise ValueError("requires_action must contain tool calls")
        elif self.output.tool_calls or self.finish_reason == "tool_call":
            raise ValueError("tool calls require requires_action state")
        elif self.status == "completed":
            if self.output.content is None and self.output.structured is None:
                raise ValueError("completed turn requires final output")
        elif self.status == "incomplete":
            if self.finish_reason not in {"output_limit", "context_limit", "unknown"}:
                raise ValueError("incomplete turn has an invalid finish reason")
        return self


class ResponsesResponse(StrictModel):
    request_id: RequestId
    object: Literal["system_x.response"] = "system_x.response"
    status: InferenceStatus
    model: PublicModelId
    output: AssistantOutput
    finish_reason: FinishReason
    usage: TokenUsage

    @model_validator(mode="after")
    def require_consistent_turn_state(self) -> "ResponsesResponse":
        if self.status == "requires_action":
            if self.finish_reason != "tool_call" or not self.output.tool_calls:
                raise ValueError("requires_action must contain tool calls")
        elif self.output.tool_calls or self.finish_reason == "tool_call":
            raise ValueError("tool calls require requires_action state")
        elif self.status == "completed":
            if self.output.content is None and self.output.structured is None:
                raise ValueError("completed turn requires final output")
        return self


class TokenCountResponse(StrictModel):
    request_id: RequestId
    object: Literal["system_x.token_count"] = "system_x.token_count"
    model: PublicModelId
    operation: Literal["generate", "chat", "responses"]
    input_tokens: int = Field(ge=1)
