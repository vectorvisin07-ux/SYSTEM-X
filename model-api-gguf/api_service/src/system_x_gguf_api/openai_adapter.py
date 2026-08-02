"""In-process translation over the canonical System X catalogue and service."""

from __future__ import annotations

import datetime as dt
import time

from .errors import SystemXError
from .inference_service import InferenceService
from .model_catalogue import ModelCatalogue, ModelSnapshot
from .openai_schemas import (
    OpenAIChatAssistantInput,
    OpenAIChatAssistantMessage,
    OpenAIChatChoice,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequest,
    OpenAIChatNamedToolChoice,
    OpenAIChatResponseFormatJSONSchema,
    OpenAIChatTextMessage,
    OpenAIChatToolCall,
    OpenAIChatToolCallFunction,
    OpenAIChatToolMessage,
    OpenAICompletion,
    OpenAICompletionChoice,
    OpenAICompletionRequest,
    OpenAIModel,
    OpenAIModelList,
    OpenAIResponse,
    OpenAIResponseFunctionCallItem,
    OpenAIResponseFunctionCallOutput,
    OpenAIResponseIncompleteDetails,
    OpenAIResponseReasoningItem,
    OpenAIResponseReasoningText,
    OpenAIResponseOutputMessage,
    OpenAIResponseOutputText,
    OpenAIResponsesEasyInputMessage,
    OpenAIResponsesNamedToolChoice,
    OpenAIResponsesRequest,
    OpenAIUsage,
)
from .schemas import (
    ChatMessage,
    ChatRequest,
    FinishReason,
    GenerateRequest,
    ResponsesRequest,
    TokenUsage,
)
from .tool_contract import (
    FunctionTool,
    StructuredOutputFormat,
    ToolCall,
    ToolChoiceAuto,
    ToolChoiceFunction,
    ToolChoiceNone,
    ToolChoiceRequired,
    openai_call_id,
    system_call_id_from_openai,
)
from .tool_schema import (
    SchemaPolicyError,
    canonical_json,
    parse_json_object,
)


class OpenAICompatibilityAdapter:
    """Translate compatibility objects without owning backend state."""

    def __init__(
        self,
        catalogue: ModelCatalogue,
        inference: InferenceService,
    ) -> None:
        self.catalogue = catalogue
        self.inference = inference

    @staticmethod
    def _created_seconds(snapshot: ModelSnapshot) -> int:
        try:
            value = dt.datetime.fromisoformat(
                snapshot.created_utc.replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model creation time is unavailable",
                retryable=True,
            ) from exc
        if value.tzinfo is None:
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model creation time is unavailable",
                retryable=True,
            )
        seconds = int(value.timestamp())
        if seconds < 0:
            raise SystemXError(
                503,
                "system_x_model_unavailable",
                "Registered model creation time is unavailable",
                retryable=True,
            )
        return seconds

    async def models(self, request_id: str) -> OpenAIModelList:
        from .compatibility_models import compatibility_model_references

        snapshots = await self.catalogue.compatibility_snapshots()
        references = compatibility_model_references(snapshots)
        created_by_id = {
            snapshot.public_model_id: self._created_seconds(snapshot)
            for snapshot in snapshots
        }
        default_created = next(
            (
                self._created_seconds(snapshot)
                for snapshot in snapshots
                if "default" in snapshot.aliases
            ),
            0,
        )
        response = OpenAIModelList(
            data=[
                OpenAIModel(
                    id=model_id,
                    created=(
                        default_created
                        if model_id == "default"
                        else created_by_id[model_id]
                    ),
                )
                for model_id, _created_at in references
            ]
        )
        self.inference.operations.note_terminal(
            request_id,
            state="completed",
            finish_reason=None,
        )
        return response

    @staticmethod
    def _request_suffix(request_id: str) -> str:
        prefix = "sx_req_"
        suffix = request_id.removeprefix(prefix)
        if (
            not request_id.startswith(prefix)
            or len(suffix) != 32
            or any(character not in "0123456789abcdef" for character in suffix)
        ):
            raise RuntimeError("System X request identity is invalid")
        return suffix

    @staticmethod
    def _finish_reason(reason: FinishReason) -> str:
        if reason in {"completed", "stop_sequence"}:
            return "stop"
        if reason in {"output_limit", "context_limit"}:
            return "length"
        raise SystemXError(
            502,
            "system_x_backend_response_invalid",
            "Private inference backend returned an unknown finish state",
        )

    @staticmethod
    def _chat_tools(
        request: OpenAIChatCompletionRequest,
    ) -> list[FunctionTool]:
        return [
            FunctionTool(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters,
                strict=True,
            )
            for tool in request.tools
        ]

    @staticmethod
    def _responses_tools(
        request: OpenAIResponsesRequest,
    ) -> list[FunctionTool]:
        return [
            FunctionTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=True,
            )
            for tool in request.tools
        ]

    @staticmethod
    def _chat_tool_choice(request: OpenAIChatCompletionRequest):
        value = request.tool_choice
        if value is None:
            return ToolChoiceAuto(type="auto") if request.tools else None
        if value == "none":
            return ToolChoiceNone(type="none")
        if value == "auto":
            return ToolChoiceAuto(type="auto")
        if value == "required":
            return ToolChoiceRequired(type="required")
        if isinstance(value, OpenAIChatNamedToolChoice):
            return ToolChoiceFunction(
                type="function",
                name=value.function.name,
            )
        raise RuntimeError("validated Chat tool choice is unavailable")

    @staticmethod
    def _responses_tool_choice(request: OpenAIResponsesRequest):
        value = request.tool_choice
        if value is None:
            return ToolChoiceAuto(type="auto") if request.tools else None
        if value == "none":
            return ToolChoiceNone(type="none")
        if value == "auto":
            return ToolChoiceAuto(type="auto")
        if value == "required":
            return ToolChoiceRequired(type="required")
        if isinstance(value, OpenAIResponsesNamedToolChoice):
            return ToolChoiceFunction(type="function", name=value.name)
        raise RuntimeError("validated Responses tool choice is unavailable")

    @staticmethod
    def _chat_output_format(
        request: OpenAIChatCompletionRequest,
    ) -> StructuredOutputFormat | None:
        value = request.response_format
        if not isinstance(value, OpenAIChatResponseFormatJSONSchema):
            return None
        definition = value.json_schema
        return StructuredOutputFormat(
            type="json_schema",
            name=definition.name,
            description=definition.description,
            schema=definition.schema_value,
            strict=True,
        )

    @staticmethod
    def _responses_output_format(
        request: OpenAIResponsesRequest,
    ) -> StructuredOutputFormat | None:
        if request.text is None:
            return None
        value = request.text.format
        return StructuredOutputFormat(
            type="json_schema",
            name=value.name,
            description=value.description,
            schema=value.schema_value,
            strict=True,
        )

    @staticmethod
    def _arguments(value: str) -> dict:
        try:
            return parse_json_object(value)
        except SchemaPolicyError as exc:
            raise SystemXError(
                422,
                "system_x_tool_arguments_invalid",
                "Tool-call arguments are not a valid JSON object",
            ) from exc

    @classmethod
    def _chat_messages(
        cls, request: OpenAIChatCompletionRequest
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for message in request.messages:
            if isinstance(message, OpenAIChatTextMessage):
                messages.append(
                    ChatMessage(
                        role=(
                            "system"
                            if message.role == "developer"
                            else message.role
                        ),
                        content=message.content,
                    )
                )
            elif isinstance(message, OpenAIChatAssistantInput):
                try:
                    calls = [
                        ToolCall(
                            id=system_call_id_from_openai(call.id),
                            name=call.function.name,
                            arguments=cls._arguments(
                                call.function.arguments
                            ),
                        )
                        for call in message.tool_calls or []
                    ]
                except ValueError as exc:
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Assistant tool-call history is invalid",
                    ) from exc
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=message.content,
                        tool_calls=calls or None,
                    )
                )
            elif isinstance(message, OpenAIChatToolMessage):
                try:
                    call_id = system_call_id_from_openai(
                        message.tool_call_id
                    )
                except ValueError as exc:
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Tool-result call identity is invalid",
                    ) from exc
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=message.content,
                        tool_call_id=call_id,
                    )
                )
            else:
                raise RuntimeError("validated Chat message is unavailable")
        return messages

    @classmethod
    def _responses_input(
        cls, request: OpenAIResponsesRequest
    ) -> str | list[ChatMessage]:
        if isinstance(request.input, str):
            return request.input
        messages: list[ChatMessage] = []
        pending_reasoning: list[str] = []
        pending_content: str | None = None
        for item in request.input:
            if isinstance(item, OpenAIResponseReasoningItem):
                if pending_content is not None:
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Reasoning history is out of output-item order",
                    )
                pending_reasoning.extend(
                    part.text for part in item.content
                )
                continue
            if isinstance(item, OpenAIResponseOutputMessage):
                if pending_content is not None:
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Responses output-message history is duplicated",
                    )
                pending_content = "".join(
                    part.text for part in item.content
                )
                continue
            if isinstance(item, OpenAIResponseFunctionCallItem):
                try:
                    call = ToolCall(
                        id=system_call_id_from_openai(item.call_id),
                        name=item.name,
                        arguments=cls._arguments(item.arguments),
                    )
                except ValueError as exc:
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Responses function-call history is invalid",
                    ) from exc
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=pending_content,
                        tool_calls=[call],
                        reasoning=pending_reasoning,
                    )
                )
                pending_reasoning = []
                pending_content = None
                continue
            if pending_content is not None:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=pending_content,
                        reasoning=pending_reasoning,
                    )
                )
                pending_reasoning = []
                pending_content = None
            if isinstance(item, OpenAIResponseFunctionCallOutput):
                if pending_reasoning:
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Reasoning history is missing its function call",
                    )
                try:
                    call_id = system_call_id_from_openai(item.call_id)
                except ValueError as exc:
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Responses tool-result identity is invalid",
                    ) from exc
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=item.output,
                        tool_call_id=call_id,
                    )
                )
                continue
            if isinstance(item, OpenAIResponsesEasyInputMessage):
                if pending_reasoning and item.role != "assistant":
                    raise SystemXError(
                        422,
                        "system_x_tool_call_invalid",
                        "Reasoning history is not attached to an assistant item",
                    )
                messages.append(
                    ChatMessage(
                        role=(
                            "system"
                            if item.role == "developer"
                            else item.role
                        ),
                        content=item.content,
                        reasoning=(
                            pending_reasoning
                            if item.role == "assistant"
                            else []
                        ),
                    )
                )
                pending_reasoning = []
                continue
            raise RuntimeError("validated Responses input item is unavailable")
        if pending_content is not None:
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=pending_content,
                    reasoning=pending_reasoning,
                )
            )
            pending_reasoning = []
        if pending_reasoning:
            raise SystemXError(
                422,
                "system_x_tool_call_invalid",
                "Reasoning history is missing its following output item",
            )
        return messages

    @staticmethod
    def _completion_usage(usage: TokenUsage) -> OpenAIUsage:
        if (
            usage.input_tokens is None
            or usage.output_tokens is None
            or usage.total_tokens is None
        ):
            raise SystemXError(
                502,
                "system_x_backend_response_invalid",
                "Private inference backend omitted required usage",
            )
        return OpenAIUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    @staticmethod
    def _stop_list(value: str | list[str] | None) -> list[str] | None:
        return [value] if isinstance(value, str) else value

    def canonical_completion_request(
        self,
        request: OpenAICompletionRequest,
    ) -> GenerateRequest:
        return GenerateRequest(
            model=request.model,
            input=request.prompt,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=self._stop_list(request.stop),
            stream=request.stream,
        )

    def canonical_chat_request(
        self,
        request: OpenAIChatCompletionRequest,
    ) -> ChatRequest:
        return ChatRequest(
            model=request.model,
            messages=self._chat_messages(request),
            max_output_tokens=request.output_limit,
            temperature=request.temperature,
            stop=self._stop_list(request.stop),
            tools=self._chat_tools(request),
            tool_choice=self._chat_tool_choice(request),
            output_format=self._chat_output_format(request),
            stream=request.stream,
        )

    def canonical_responses_request(
        self,
        request: OpenAIResponsesRequest,
    ) -> ResponsesRequest:
        return ResponsesRequest(
            model=request.model,
            input=self._responses_input(request),
            instructions=request.instructions,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            tools=self._responses_tools(request),
            tool_choice=self._responses_tool_choice(request),
            output_format=self._responses_output_format(request),
            stream=request.stream,
        )

    async def completion(
        self,
        request_id: str,
        request: OpenAICompletionRequest,
        *,
        created: int | None = None,
    ) -> OpenAICompletion:
        canonical = self.canonical_completion_request(request)
        result = await self.inference.generate(request_id, canonical)
        suffix = self._request_suffix(request_id)
        return OpenAICompletion(
            id=f"cmpl-sx-{suffix}",
            created=int(time.time()) if created is None else created,
            model=result.model,
            choices=[
                OpenAICompletionChoice(
                    text=result.output.text,
                    finish_reason=self._finish_reason(
                        result.finish_reason
                    ),
                )
            ],
            usage=self._completion_usage(result.usage),
        )

    async def chat(
        self,
        request_id: str,
        request: OpenAIChatCompletionRequest,
        *,
        created: int | None = None,
    ) -> OpenAIChatCompletion:
        canonical = self.canonical_chat_request(request)
        result = await self.inference.chat(request_id, canonical)
        suffix = self._request_suffix(request_id)
        tool_calls = [
            OpenAIChatToolCall(
                id=openai_call_id(call.id),
                type="function",
                function=OpenAIChatToolCallFunction(
                    name=call.name,
                    arguments=canonical_json(call.arguments),
                ),
            )
            for call in result.output.tool_calls
        ]
        finish_reason = (
            "tool_calls"
            if tool_calls
            else self._finish_reason(result.finish_reason)
        )
        return OpenAIChatCompletion(
            id=f"chatcmpl-sx-{suffix}",
            created=int(time.time()) if created is None else created,
            model=result.model,
            choices=[
                OpenAIChatChoice(
                    message=OpenAIChatAssistantMessage(
                        content=result.output.content,
                        reasoning_content=(
                            "\n".join(result.output.reasoning)
                            if result.output.reasoning
                            else None
                        ),
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=self._completion_usage(result.usage),
        )

    async def response(
        self,
        request_id: str,
        request: OpenAIResponsesRequest,
        *,
        created_at: float | None = None,
    ) -> OpenAIResponse:
        canonical = self.canonical_responses_request(request)
        result = await self.inference.responses(request_id, canonical)
        if result.finish_reason == "tool_call":
            status = "completed"
            incomplete_details = None
        elif result.finish_reason in {"completed", "stop_sequence"}:
            status = "completed"
            incomplete_details = None
        elif result.finish_reason in {"output_limit", "context_limit"}:
            status = "incomplete"
            incomplete_details = OpenAIResponseIncompleteDetails(
                reason="max_output_tokens"
            )
        else:
            raise SystemXError(
                502,
                "system_x_backend_response_invalid",
                "Private inference backend returned an unknown finish state",
            )
        suffix = self._request_suffix(request_id)
        output = [
            OpenAIResponseReasoningItem(
                id=f"rs_sx_{suffix}_{index}",
                type="reasoning",
                summary=[],
                content=[
                    OpenAIResponseReasoningText(
                        type="reasoning_text",
                        text=text,
                    )
                ],
                encrypted_content="",
                status=status,
            )
            for index, text in enumerate(result.output.reasoning)
        ]
        if result.output.content is not None:
            output.append(
                OpenAIResponseOutputMessage(
                    id=f"msg_sx_{suffix}",
                    status=status,
                    content=[
                        OpenAIResponseOutputText(
                            text=result.output.content,
                        )
                    ],
                )
            )
        for call in result.output.tool_calls:
            call_id = openai_call_id(call.id)
            output.append(
                OpenAIResponseFunctionCallItem(
                    id=f"fc_sx_{call.id.removeprefix('sx_call_')}",
                    type="function_call",
                    call_id=call_id,
                    name=call.name,
                    arguments=canonical_json(call.arguments),
                    status="completed",
                )
            )
        public_tool_choice = request.tool_choice
        if public_tool_choice is None:
            public_tool_choice = "auto" if request.tools else "none"
        return OpenAIResponse(
            id=f"resp_sx_{suffix}",
            created_at=(
                float(time.time())
                if created_at is None
                else created_at
            ),
            status=status,
            incomplete_details=incomplete_details,
            instructions=request.instructions,
            model=result.model,
            output=output,
            tool_choice=public_tool_choice,
            tools=request.tools,
            max_output_tokens=request.max_output_tokens,
        )
