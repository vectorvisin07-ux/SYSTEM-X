"""Authoritative System X inference orchestration over the private router."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from typing import Any, NoReturn

from .backend import (
    BackendError,
    BackendModelConflict,
    BackendModelUnavailable,
    BackendSnapshotConflict,
    InferenceBackendLease,
)
from .errors import SystemXError
from .finalization_policy import (
    classify_turn_intent,
    private_chat_template_kwargs,
    retain_declared_tools,
)
from .model_catalogue import ModelCatalogue, ModelSnapshot
from .operation_records import OperationRecorder
from .response_normalizer import (
    ResponseNormalizationError,
    normalize_chat_turn,
    normalize_completion,
    normalize_responses_turn,
    normalize_token_count,
)
from .router_client import RouterObservation
from .schemas import (
    AssistantOutput,
    ChatRequest,
    ChatResponse,
    ChatTokenCountRequest,
    GenerateRequest,
    GenerateTokenCountRequest,
    GenerationResponse,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesTokenCountRequest,
    TextOutput,
    TokenCountRequest,
    TokenCountResponse,
    ReasoningRequest,
)
from .tool_contract import (
    MAX_AGGREGATE_TOOL_RESULT_BYTES,
    MAX_TOOL_CALLS,
    MAX_TOOL_RESULT_BYTES,
    MAX_TOOLS,
    ToolContractError,
    ToolChoice,
    ToolHistory,
    private_chat_tools,
    private_messages,
    private_response_format,
    private_responses_input,
    private_responses_tools,
    private_tool_choice,
    validate_history,
    validate_tool_choice,
    validate_tools,
)
from .tool_schema import (
    MAX_AGGREGATE_SCHEMA_BYTES,
    MAX_ENUM_MEMBERS,
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_PROPERTIES,
)


LOGGER = logging.getLogger("uvicorn.error")
SERVICE_TRANSACTION_ENV = "SYSTEM_X_API_SERVICE_TRANSACTION_ID"


@dataclass(frozen=True, slots=True)
class PreparedGenerate:
    snapshot: ModelSnapshot


@dataclass(frozen=True, slots=True)
class PreparedChat:
    snapshot: ModelSnapshot
    selected: ToolChoice
    messages: list[dict[str, Any]]
    private_tools: list[dict[str, Any]] | None
    private_choice: str | dict[str, Any] | None
    private_format: dict[str, Any] | None
    private_template_kwargs: dict[str, bool] | None


@dataclass(frozen=True, slots=True)
class PreparedResponses:
    snapshot: ModelSnapshot
    selected: ToolChoice
    private_input: str | list[dict[str, Any]]
    private_tools: list[dict[str, Any]] | None
    private_choice: str | dict[str, Any] | None
    private_format: dict[str, Any] | None
    private_template_kwargs: dict[str, bool] | None


class InferenceService:
    """Keep route functions free of private process and endpoint concerns."""

    def __init__(
        self,
        catalogue: ModelCatalogue,
        backend: object,
        operations: OperationRecorder,
    ) -> None:
        self.catalogue = catalogue
        self.backend = backend
        self.operations = operations
        settings = getattr(backend, "settings", None)
        expected_limits = {
            "tool_max_definitions": MAX_TOOLS,
            "tool_max_calls_per_turn": MAX_TOOL_CALLS,
            "tool_schema_max_bytes": MAX_SCHEMA_BYTES,
            "tool_schema_max_aggregate_bytes": MAX_AGGREGATE_SCHEMA_BYTES,
            "tool_schema_max_depth": MAX_SCHEMA_DEPTH,
            "tool_schema_max_properties": MAX_SCHEMA_PROPERTIES,
            "tool_schema_max_enum_members": MAX_ENUM_MEMBERS,
            "tool_result_max_bytes": MAX_TOOL_RESULT_BYTES,
            "tool_result_max_aggregate_bytes": (
                MAX_AGGREGATE_TOOL_RESULT_BYTES
            ),
        }
        if settings is not None:
            for name, expected in expected_limits.items():
                if getattr(settings, name, None) != expected:
                    raise RuntimeError(
                        f"tool policy setting is inconsistent: {name}"
                    )
        self.maximum_tool_result_bytes = expected_limits[
            "tool_result_max_bytes"
        ]
        self.maximum_aggregate_tool_result_bytes = expected_limits[
            "tool_result_max_aggregate_bytes"
        ]

    @staticmethod
    def _validate_output_bound(
        snapshot: ModelSnapshot, max_output_tokens: int
    ) -> None:
        if (
            snapshot.context_bound is not None
            and max_output_tokens > snapshot.context_bound
        ):
            raise SystemXError(
                422,
                "system_x_validation_error",
                "max_output_tokens exceeds the registered context bound",
            )

    @staticmethod
    def _record_private_result(
        request_id: str,
        endpoint: str,
        lease: InferenceBackendLease,
        observation: RouterObservation,
    ) -> None:
        body_bytes = observation.body.encode("utf-8")
        LOGGER.info(
            "private inference result "
            "service_transaction=%s request_id=%s "
            "router_transaction=%s router_pid=%s router_pgid=%s router_sid=%s "
            "router_start=%s router_model=%s endpoint=%s "
            "status=%s transport_error=%s body_bytes=%s body_sha256=%s",
            os.environ.get(SERVICE_TRANSACTION_ENV, "unavailable"),
            request_id,
            lease.router_identity.transaction_id,
            lease.router_identity.pid,
            lease.router_identity.pgid,
            lease.router_identity.sid,
            lease.router_identity.process_start_identity,
            lease.router_model_id,
            endpoint,
            observation.status_code,
            observation.error,
            len(body_bytes),
            hashlib.sha256(body_bytes).hexdigest(),
        )

    @staticmethod
    def _service_transaction_id() -> str:
        value = os.environ.get(SERVICE_TRANSACTION_ENV)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 128
        ):
            raise SystemXError(
                503,
                "system_x_backend_unavailable",
                "API-service transaction identity is unavailable",
                retryable=True,
            )
        return value

    @staticmethod
    def _raise_backend(exc: BackendError) -> NoReturn:
        if isinstance(exc, (BackendModelConflict, BackendSnapshotConflict)):
            raise SystemXError(
                409,
                "system_x_model_conflict",
                "Requested model mapping conflicts with current backend state",
            ) from exc
        if isinstance(exc, BackendModelUnavailable):
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Requested model is unavailable",
                retryable=True,
            ) from exc
        raise SystemXError(
            503,
            "system_x_backend_unavailable",
            "Private inference backend is unavailable",
            retryable=True,
        ) from exc

    @staticmethod
    def _raise_normalization(exc: ResponseNormalizationError) -> NoReturn:
        if exc.kind == "timeout":
            raise SystemXError(
                504,
                "system_x_backend_timeout",
                "Private inference backend timed out",
                retryable=True,
            ) from exc
        if exc.kind == "connection_failure":
            raise SystemXError(
                503,
                "system_x_backend_unavailable",
                "Private inference backend is unavailable",
                retryable=True,
            ) from exc
        if exc.kind.startswith("empty_"):
            raise SystemXError(
                502,
                "system_x_output_invalid",
                "Private inference produced no usable public output",
            ) from exc
        if exc.kind in {"tool_call_invalid", "required_tool_call_missing"}:
            raise SystemXError(
                502,
                "system_x_tool_call_invalid",
                "Private inference returned an invalid tool call",
            ) from exc
        if exc.kind == "tool_arguments_invalid":
            raise SystemXError(
                502,
                "system_x_tool_arguments_invalid",
                "Private inference returned invalid tool arguments",
            ) from exc
        if exc.kind == "structured_output_invalid":
            raise SystemXError(
                502,
                "system_x_structured_output_invalid",
                "Private inference returned invalid structured output",
            ) from exc
        if exc.kind == "tool_and_output_format_conflict":
            raise SystemXError(
                502,
                "system_x_tool_and_output_format_conflict",
                "Private inference mixed tool and structured output",
            ) from exc
        raise SystemXError(
            502,
            "system_x_backend_response_invalid",
            "Private inference backend returned an invalid response",
        ) from exc

    @staticmethod
    def _raise_contract(exc: ToolContractError) -> NoReturn:
        raise SystemXError(
            422,
            exc.code,
            str(exc),
        ) from exc

    async def _reasoning_template_kwargs(
        self,
        snapshot: ModelSnapshot,
        intent: object,
        reasoning: ReasoningRequest | None,
    ) -> dict[str, bool] | None:
        if reasoning is not None and (
            reasoning.effort is not None
            or reasoning.budget_tokens is not None
        ):
            raise SystemXError(
                422,
                "system_x_capability_unavailable",
                "Requested reasoning effort or token-budget control is unavailable",
            )
        available = await self.catalogue.reasoning_template_control_available(
            snapshot
        )
        enable: bool | None = None
        if reasoning is not None and available:
            enable = reasoning.mode == "pro_extended"
        return private_chat_template_kwargs(intent, enable)

    async def prepare_generate(
        self, request_id: str, request: GenerateRequest
    ) -> PreparedGenerate:
        snapshot = await self.catalogue.resolve(request.model)
        self.operations.note_model(
            request_id,
            snapshot.public_model_id,
            snapshot.bundle_id,
        )
        self._validate_output_bound(snapshot, request.max_output_tokens)
        return PreparedGenerate(snapshot)

    async def prepare_chat(
        self, request_id: str, request: ChatRequest
    ) -> PreparedChat:
        try:
            definitions = validate_tools(request.tools)
            selected = validate_tool_choice(
                request.tool_choice, request.tools
            )
            history = validate_history(
                request.messages,
                request.tools,
                maximum_result_bytes=self.maximum_tool_result_bytes,
                maximum_aggregate_result_bytes=(
                    self.maximum_aggregate_tool_result_bytes
                ),
            )
        except ToolContractError as exc:
            self._raise_contract(exc)
        if request.tools and request.output_format is not None:
            raise SystemXError(
                422,
                "system_x_tool_and_output_format_conflict",
                "Tools and structured output cannot be combined",
            )
        intent = classify_turn_intent(
            history=history,
            messages=request.messages,
            has_tools=bool(definitions),
            has_output_format=request.output_format is not None,
        )
        snapshot = await self.catalogue.resolve(request.model)
        self.operations.note_model(
            request_id,
            snapshot.public_model_id,
            snapshot.bundle_id,
        )
        self._validate_output_bound(snapshot, request.max_output_tokens)
        if not snapshot.chat_template_present:
            raise SystemXError(
                503,
                "system_x_capability_unavailable",
                "Requested model has no registered chat template",
            )
        if definitions:
            self.catalogue.require_capability(snapshot, "tool_calling")
        if request.output_format is not None:
            self.catalogue.require_capability(snapshot, "structured_output")
        messages = private_messages(request.messages)
        private_tools = (
            private_chat_tools(
                request.tools,
                None if retain_declared_tools(intent) else selected,
            )
            if definitions
            else None
        )
        private_choice = (
            private_tool_choice(selected) if definitions else None
        )
        private_format = (
            private_response_format(request.output_format)
            if request.output_format is not None
            else None
        )
        return PreparedChat(
            snapshot=snapshot,
            selected=selected,
            messages=messages,
            private_tools=private_tools,
            private_choice=private_choice,
            private_format=private_format,
            private_template_kwargs=await self._reasoning_template_kwargs(
                snapshot, intent, request.reasoning
            ),
        )

    async def prepare_responses(
        self, request_id: str, request: ResponsesRequest
    ) -> PreparedResponses:
        history = ToolHistory((), ())
        try:
            definitions = validate_tools(request.tools)
            selected = validate_tool_choice(
                request.tool_choice, request.tools
            )
            if isinstance(request.input, list):
                history = validate_history(
                    request.input,
                    request.tools,
                    maximum_result_bytes=self.maximum_tool_result_bytes,
                    maximum_aggregate_result_bytes=(
                        self.maximum_aggregate_tool_result_bytes
                    ),
                )
        except ToolContractError as exc:
            self._raise_contract(exc)
        if request.tools and request.output_format is not None:
            raise SystemXError(
                422,
                "system_x_tool_and_output_format_conflict",
                "Tools and structured output cannot be combined",
            )
        canonical_messages = (
            request.input if isinstance(request.input, list) else ()
        )
        intent = classify_turn_intent(
            history=history,
            messages=canonical_messages,
            has_tools=bool(definitions),
            has_output_format=request.output_format is not None,
        )
        snapshot = await self.catalogue.resolve(request.model)
        self.operations.note_model(
            request_id,
            snapshot.public_model_id,
            snapshot.bundle_id,
        )
        self._validate_output_bound(snapshot, request.max_output_tokens)
        if definitions:
            self.catalogue.require_capability(snapshot, "tool_calling")
        if request.output_format is not None:
            self.catalogue.require_capability(snapshot, "structured_output")
        private_input = (
            private_responses_input(request.input)
            if isinstance(request.input, list)
            else request.input
        )
        private_tools = (
            private_responses_tools(
                request.tools,
                None if retain_declared_tools(intent) else selected,
            )
            if definitions
            else None
        )
        private_choice = (
            private_tool_choice(selected) if definitions else None
        )
        private_format = (
            private_response_format(request.output_format)
            if request.output_format is not None
            else None
        )
        return PreparedResponses(
            snapshot=snapshot,
            selected=selected,
            private_input=private_input,
            private_tools=private_tools,
            private_choice=private_choice,
            private_format=private_format,
            private_template_kwargs=await self._reasoning_template_kwargs(
                snapshot, intent, request.reasoning
            ),
        )

    async def generate(
        self, request_id: str, request: GenerateRequest
    ) -> GenerationResponse:
        prepared = await self.prepare_generate(request_id, request)
        snapshot = prepared.snapshot
        try:
            async with self.backend.inference_session(
                snapshot.router_model_id,
                lambda: self.catalogue.verify(snapshot),
            ) as lease:
                self.operations.note_router(
                    request_id,
                    lease.router_identity.transaction_id,
                )
                observation = await lease.router.completion(
                    lease.router_model_id,
                    request.input,
                    request.max_output_tokens,
                    request.temperature,
                    request.stop,
                )
                self._record_private_result(
                    request_id, "/v1/completions", lease, observation
                )
                normalized = normalize_completion(observation)
        except ResponseNormalizationError as exc:
            self._raise_normalization(exc)
        except BackendError as exc:
            self._raise_backend(exc)
        self.catalogue.mark_operation_proven(
            snapshot.public_model_id, "generate"
        )
        response = GenerationResponse(
            request_id=request_id,
            status=normalized.status,
            model=snapshot.public_model_id,
            output=TextOutput(text=normalized.text),
            finish_reason=normalized.finish_reason,
            usage=normalized.usage,
        )
        self.operations.note_terminal(
            request_id,
            state=normalized.status,
            finish_reason=normalized.finish_reason,
            input_tokens=normalized.usage.input_tokens,
            output_tokens=normalized.usage.output_tokens,
        )
        return response

    async def chat(
        self, request_id: str, request: ChatRequest
    ) -> ChatResponse:
        prepared = await self.prepare_chat(request_id, request)
        snapshot = prepared.snapshot
        selected = prepared.selected
        try:
            async with self.backend.inference_session(
                snapshot.router_model_id,
                lambda: self.catalogue.verify(snapshot),
            ) as lease:
                self.operations.note_router(
                    request_id,
                    lease.router_identity.transaction_id,
                )
                observation = await lease.router.chat_completion(
                    lease.router_model_id,
                    prepared.messages,
                    request.max_output_tokens,
                    request.temperature,
                    request.stop,
                    prepared.private_tools,
                    prepared.private_choice,
                    prepared.private_format,
                    prepared.private_template_kwargs,
                )
                self._record_private_result(
                    request_id,
                    "/v1/chat/completions",
                    lease,
                    observation,
                )
                normalized = normalize_chat_turn(
                    observation,
                    request.tools,
                    selected,
                    request.output_format,
                )
        except ResponseNormalizationError as exc:
            self._raise_normalization(exc)
        except BackendError as exc:
            self._raise_backend(exc)
        if normalized.tool_calls:
            await self.catalogue.mark_capability_proven(
                snapshot.public_model_id,
                "tool_calling",
                request_id,
                self._service_transaction_id(),
                lease.router_identity.transaction_id,
            )
        if normalized.structured is not None:
            await self.catalogue.mark_capability_proven(
                snapshot.public_model_id,
                "structured_output",
                request_id,
                self._service_transaction_id(),
                lease.router_identity.transaction_id,
            )
        self.catalogue.mark_operation_proven(snapshot.public_model_id, "chat")
        response = ChatResponse(
            request_id=request_id,
            status=normalized.status,
            model=snapshot.public_model_id,
            output=AssistantOutput(
                content=normalized.content,
                tool_calls=list(normalized.tool_calls),
                reasoning=list(normalized.reasoning),
                structured=normalized.structured,
            ),
            finish_reason=normalized.finish_reason,
            usage=normalized.usage,
        )
        self.operations.note_terminal(
            request_id,
            state=normalized.status,
            finish_reason=normalized.finish_reason,
            input_tokens=normalized.usage.input_tokens,
            output_tokens=normalized.usage.output_tokens,
        )
        return response

    async def responses(
        self, request_id: str, request: ResponsesRequest
    ) -> ResponsesResponse:
        prepared = await self.prepare_responses(request_id, request)
        snapshot = prepared.snapshot
        selected = prepared.selected
        try:
            async with self.backend.inference_session(
                snapshot.router_model_id,
                lambda: self.catalogue.verify(snapshot),
            ) as lease:
                self.operations.note_router(
                    request_id,
                    lease.router_identity.transaction_id,
                )
                observation = await lease.router.responses(
                    lease.router_model_id,
                    prepared.private_input,
                    request.max_output_tokens,
                    request.instructions,
                    request.temperature,
                    prepared.private_tools,
                    prepared.private_choice,
                    prepared.private_format,
                    prepared.private_template_kwargs,
                )
                self._record_private_result(
                    request_id, "/v1/responses", lease, observation
                )
                normalized = normalize_responses_turn(
                    observation,
                    request.tools,
                    selected,
                    request.output_format,
                    request.max_output_tokens,
                )
        except ResponseNormalizationError as exc:
            self._raise_normalization(exc)
        except BackendError as exc:
            self._raise_backend(exc)
        if normalized.tool_calls:
            await self.catalogue.mark_capability_proven(
                snapshot.public_model_id,
                "tool_calling",
                request_id,
                self._service_transaction_id(),
                lease.router_identity.transaction_id,
            )
        if normalized.structured is not None:
            await self.catalogue.mark_capability_proven(
                snapshot.public_model_id,
                "structured_output",
                request_id,
                self._service_transaction_id(),
                lease.router_identity.transaction_id,
            )
        self.catalogue.mark_operation_proven(
            snapshot.public_model_id, "responses"
        )
        response = ResponsesResponse(
            request_id=request_id,
            status=normalized.status,
            model=snapshot.public_model_id,
            output=AssistantOutput(
                content=normalized.content,
                tool_calls=list(normalized.tool_calls),
                reasoning=list(normalized.reasoning),
                structured=normalized.structured,
            ),
            finish_reason=normalized.finish_reason,
            usage=normalized.usage,
        )
        self.operations.note_terminal(
            request_id,
            state=normalized.status,
            finish_reason=normalized.finish_reason,
            input_tokens=normalized.usage.input_tokens,
            output_tokens=normalized.usage.output_tokens,
        )
        return response

    async def count_tokens(
        self, request_id: str, request: TokenCountRequest
    ) -> TokenCountResponse:
        definitions: dict[str, object] = {}
        if isinstance(request, ChatTokenCountRequest):
            try:
                definitions = validate_tools(request.tools)
                validate_history(
                    request.messages,
                    request.tools,
                    maximum_result_bytes=self.maximum_tool_result_bytes,
                    maximum_aggregate_result_bytes=(
                        self.maximum_aggregate_tool_result_bytes
                    ),
                )
            except ToolContractError as exc:
                self._raise_contract(exc)
        snapshot = await self.catalogue.resolve(request.model)
        self.operations.note_model(
            request_id,
            snapshot.public_model_id,
            snapshot.bundle_id,
        )
        if isinstance(request, ChatTokenCountRequest):
            if not snapshot.chat_template_present:
                raise SystemXError(
                    503,
                    "system_x_capability_unavailable",
                    "Requested model has no registered chat template",
                )
            if definitions:
                self.catalogue.require_capability(snapshot, "tool_calling")
        try:
            async with self.backend.inference_session(
                snapshot.router_model_id,
                lambda: self.catalogue.verify(snapshot),
            ) as lease:
                self.operations.note_router(
                    request_id,
                    lease.router_identity.transaction_id,
                )
                if isinstance(request, GenerateTokenCountRequest):
                    endpoint = "/tokenize"
                    observation = await lease.router.tokenize(
                        lease.router_model_id, request.input
                    )
                    self._record_private_result(
                        request_id, endpoint, lease, observation
                    )
                    count = normalize_token_count(
                        observation, tokenize=True
                    )
                elif isinstance(request, ChatTokenCountRequest):
                    endpoint = "/v1/chat/completions/input_tokens"
                    observation = await lease.router.chat_input_tokens(
                        lease.router_model_id,
                        private_messages(request.messages),
                        (
                            private_chat_tools(request.tools)
                            if definitions
                            else None
                        ),
                    )
                    self._record_private_result(
                        request_id, endpoint, lease, observation
                    )
                    count = normalize_token_count(observation)
                elif isinstance(request, ResponsesTokenCountRequest):
                    endpoint = "/v1/responses/input_tokens"
                    observation = await lease.router.responses_input_tokens(
                        lease.router_model_id,
                        request.input,
                        request.instructions,
                    )
                    self._record_private_result(
                        request_id, endpoint, lease, observation
                    )
                    count = normalize_token_count(observation)
                else:
                    raise ResponseNormalizationError(
                        "unknown_token_count_operation"
                    )
        except ResponseNormalizationError as exc:
            self._raise_normalization(exc)
        except BackendError as exc:
            self._raise_backend(exc)
        self.catalogue.mark_operation_proven(
            snapshot.public_model_id, "token_count"
        )
        response = TokenCountResponse(
            request_id=request_id,
            model=snapshot.public_model_id,
            operation=request.operation,
            input_tokens=count,
        )
        self.operations.note_terminal(
            request_id,
            state="completed",
            finish_reason=None,
            input_tokens=count,
            output_tokens=None,
        )
        return response
