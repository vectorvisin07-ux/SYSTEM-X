"""In-process Messages translation over canonical System X services."""

from __future__ import annotations

from .anthropic_contract import (
    TEXT_BLOCK_SEPARATOR,
    anthropic_message_id,
    anthropic_request_id,
)
from .anthropic_schemas import (
    AnthropicCountTokensRequest,
    AnthropicJSONOutputFormat,
    AnthropicMessage,
    AnthropicMessageRequest,
    AnthropicMessageTokensCount,
    AnthropicModelInfo,
    AnthropicModelPage,
    AnthropicOutputToolUseBlock,
    AnthropicOutputTextBlock,
    AnthropicOutputThinkingBlock,
    AnthropicTextBlock,
    AnthropicToolChoiceAny,
    AnthropicToolChoiceAuto,
    AnthropicToolChoiceNone,
    AnthropicToolChoiceTool,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
)
from .errors import SystemXError
from .inference_service import InferenceService
from .model_catalogue import ModelCatalogue
from .schemas import ChatMessage, ChatRequest, ChatTokenCountRequest, TokenUsage
from .tool_contract import (
    FunctionTool,
    StructuredOutputFormat,
    ToolCall,
    ToolChoiceAuto,
    ToolChoiceFunction,
    ToolChoiceNone,
    ToolChoiceRequired,
    anthropic_call_id,
    system_call_id_from_anthropic,
)


class AnthropicCompatibilityAdapter:
    """Translate Messages objects without owning backend state."""

    def __init__(
        self, catalogue: ModelCatalogue, inference: InferenceService
    ) -> None:
        self.catalogue = catalogue
        self.inference = inference

    async def models(self, request_id: str) -> AnthropicModelPage:
        from .compatibility_models import compatibility_model_references

        snapshots = await self.catalogue.compatibility_snapshots()
        references = compatibility_model_references(snapshots)
        response = AnthropicModelPage(
            data=[
                AnthropicModelInfo(
                    id=model_id,
                    display_name=model_id,
                    created_at=created_at,
                )
                for model_id, created_at in references
            ],
            first_id=references[0][0] if references else None,
            last_id=references[-1][0] if references else None,
        )
        self.inference.operations.note_terminal(
            request_id,
            state="completed",
            finish_reason=None,
        )
        return response

    @staticmethod
    def _text(value: str | list[AnthropicTextBlock]) -> str:
        if isinstance(value, str):
            return value
        return TEXT_BLOCK_SEPARATOR.join(block.text for block in value)

    @classmethod
    def canonical_messages(
        cls,
        body: AnthropicMessageRequest | AnthropicCountTokensRequest,
    ) -> list[ChatMessage]:
        result: list[ChatMessage] = []
        if body.system is not None:
            result.append(
                ChatMessage(role="system", content=cls._text(body.system))
            )
        for item in body.messages:
            if isinstance(item.content, str):
                result.append(
                    ChatMessage(role=item.role, content=item.content)
                )
                continue
            if item.role == "assistant":
                text = TEXT_BLOCK_SEPARATOR.join(
                    block.text
                    for block in item.content
                    if isinstance(block, AnthropicTextBlock)
                )
                calls: list[ToolCall] = []
                for block in item.content:
                    if not isinstance(block, AnthropicToolUseBlock):
                        continue
                    try:
                        call_id = system_call_id_from_anthropic(block.id)
                    except ValueError as exc:
                        raise SystemXError(
                            422,
                            "system_x_tool_call_invalid",
                            "Messages tool-use identity is invalid",
                        ) from exc
                    calls.append(
                        ToolCall(
                            id=call_id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )
                result.append(
                    ChatMessage(
                        role="assistant",
                        content=text or None,
                        tool_calls=calls or None,
                    )
                )
                continue
            text_blocks: list[str] = []
            for block in item.content:
                if isinstance(block, AnthropicToolResultBlock):
                    try:
                        call_id = system_call_id_from_anthropic(
                            block.tool_use_id
                        )
                    except ValueError as exc:
                        raise SystemXError(
                            422,
                            "system_x_tool_call_invalid",
                            "Messages tool-result identity is invalid",
                        ) from exc
                    result.append(
                        ChatMessage(
                            role="tool",
                            content=block.content,
                            tool_call_id=call_id,
                            is_error=block.is_error,
                        )
                    )
                elif isinstance(block, AnthropicTextBlock):
                    text_blocks.append(block.text)
                else:
                    raise RuntimeError(
                        "validated Messages content block is unavailable"
                    )
            if text_blocks:
                result.append(
                    ChatMessage(
                        role="user",
                        content=TEXT_BLOCK_SEPARATOR.join(text_blocks),
                    )
                )
        return result

    @staticmethod
    def _tools(
        body: AnthropicMessageRequest | AnthropicCountTokensRequest,
    ) -> list[FunctionTool]:
        return [
            FunctionTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
                strict=True,
            )
            for tool in body.tools
        ]

    @staticmethod
    def _tool_choice(body: AnthropicMessageRequest):
        value = body.tool_choice
        if value is None:
            return ToolChoiceAuto(type="auto") if body.tools else None
        if isinstance(value, AnthropicToolChoiceNone):
            return ToolChoiceNone(type="none")
        if isinstance(value, AnthropicToolChoiceAuto):
            return ToolChoiceAuto(type="auto")
        if isinstance(value, AnthropicToolChoiceAny):
            return ToolChoiceRequired(type="required")
        if isinstance(value, AnthropicToolChoiceTool):
            return ToolChoiceFunction(type="function", name=value.name)
        raise RuntimeError("validated Messages tool choice is unavailable")

    @staticmethod
    def _output_format(
        body: AnthropicMessageRequest,
    ) -> StructuredOutputFormat | None:
        if body.output_config is None:
            return None
        value: AnthropicJSONOutputFormat = body.output_config.format
        return StructuredOutputFormat(
            type="json_schema",
            name="anthropic_output",
            schema=value.schema_value,
            strict=True,
        )

    @staticmethod
    def _usage(value: TokenUsage) -> AnthropicUsage:
        if value.input_tokens is None or value.output_tokens is None:
            raise SystemXError(
                502,
                "system_x_backend_response_invalid",
                "Private inference backend omitted required usage",
            )
        return AnthropicUsage(
            input_tokens=value.input_tokens,
            output_tokens=value.output_tokens,
        )

    @staticmethod
    def _stop_reason(reason: str) -> str:
        mapping = {
            "completed": "end_turn",
            "output_limit": "max_tokens",
            "stop_sequence": "stop_sequence",
            "context_limit": "model_context_window_exceeded",
            "tool_call": "tool_use",
        }
        try:
            return mapping[reason]
        except KeyError as exc:
            raise SystemXError(
                502,
                "system_x_backend_response_invalid",
                "Private inference backend returned an unknown finish state",
            ) from exc

    def canonical_message_request(
        self,
        body: AnthropicMessageRequest,
    ) -> ChatRequest:
        return ChatRequest(
            model=body.model,
            messages=self.canonical_messages(body),
            max_output_tokens=body.max_tokens,
            temperature=body.temperature,
            stop=body.stop_sequences,
            tools=self._tools(body),
            tool_choice=self._tool_choice(body),
            output_format=self._output_format(body),
            stream=body.stream,
        )

    async def message(
        self, request_id: str, body: AnthropicMessageRequest
    ) -> AnthropicMessage:
        canonical = self.canonical_message_request(body)
        result = await self.inference.chat(request_id, canonical)
        if result.finish_reason == "stop_sequence":
            raise SystemXError(
                502,
                "system_x_backend_response_invalid",
                "Exact matched stop sequence is unavailable",
            )
        content = (
            [
                AnthropicOutputThinkingBlock(
                    thinking="\n".join(result.output.reasoning)
                )
            ]
            if result.output.reasoning
            else []
        )
        content.extend(
            [AnthropicOutputTextBlock(text=result.output.content)]
            if result.output.content
            else []
        )
        content.extend(
            AnthropicOutputToolUseBlock(
                id=anthropic_call_id(call.id),
                name=call.name,
                input=call.arguments,
            )
            for call in result.output.tool_calls
        )
        return AnthropicMessage(
            id=anthropic_message_id(request_id),
            content=content,
            model=result.model,
            stop_reason=self._stop_reason(result.finish_reason),
            stop_sequence=None,
            usage=self._usage(result.usage),
        )

    async def count_tokens(
        self, request_id: str, body: AnthropicCountTokensRequest
    ) -> AnthropicMessageTokensCount:
        canonical = ChatTokenCountRequest(
            model=body.model,
            operation="chat",
            messages=self.canonical_messages(body),
            tools=self._tools(body),
        )
        result = await self.inference.count_tokens(request_id, canonical)
        return AnthropicMessageTokensCount(input_tokens=result.input_tokens)
