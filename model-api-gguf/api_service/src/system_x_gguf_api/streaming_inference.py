"""One canonical private-to-public streaming inference subsystem."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from typing import Any, AsyncIterator, Literal, Protocol

import httpx

from .backend import BackendError, InferenceBackendLease
from .errors import SystemXError
from .inference_service import (
    InferenceService,
    PreparedChat,
    PreparedGenerate,
    PreparedResponses,
    SERVICE_TRANSACTION_ENV,
)
from .router_client import (
    PrivateRouterStream,
    RouterStreamOpenError,
)
from .response_normalizer import (
    ResponseNormalizationError,
    normalize_token_count,
)
from .schemas import ChatRequest, GenerateRequest, ResponsesRequest
from .sse import SSEProtocolError, ValidatedPrivateFrame, validate_private_frame
from .stream_control import ActiveStreamRegistry, StreamControl
from .stream_types import (
    ActiveStreamState,
    CanonicalStreamEvent,
    CanonicalStreamEventType,
    StreamOperation,
    StreamState,
)
from .tool_contract import (
    FunctionTool,
    ToolCall,
    ToolChoice,
    ToolChoiceFunction,
    ToolChoiceRequired,
    new_tool_call_id,
    validate_returned_tool_calls,
)


LOGGER = logging.getLogger("uvicorn.error")
MAXIMUM_ACCUMULATED_TEXT_BYTES = 4_194_304
MAXIMUM_TOOL_ARGUMENT_BYTES = 1_048_576
PrivateStreamKind = Literal["completion", "chat", "responses"]


class StreamNormalizationError(ValueError):
    """A private event stream cannot satisfy the canonical contract."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _append_bounded(
    parts: list[str],
    value: str,
    current_bytes: int,
    maximum_bytes: int,
    label: str,
) -> int:
    if not isinstance(value, str) or not value:
        raise StreamNormalizationError(f"invalid_{label}_delta")
    new_size = current_bytes + len(value.encode("utf-8"))
    if new_size > maximum_bytes:
        raise StreamNormalizationError(f"{label}_exceeded_bound")
    parts.append(value)
    return new_size


def _usage(
    raw: Any,
    input_name: str,
    output_name: str,
) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise StreamNormalizationError("invalid_usage")
    input_tokens = raw.get(input_name)
    output_tokens = raw.get(output_name)
    total_tokens = raw.get("total_tokens")
    if any(
        type(value) is not int or value < 0
        for value in (input_tokens, output_tokens, total_tokens)
    ):
        raise StreamNormalizationError("invalid_usage")
    if total_tokens != input_tokens + output_tokens:
        raise StreamNormalizationError("contradictory_usage")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _tool_error(exc: ValueError) -> StreamNormalizationError:
    code = getattr(exc, "code", "")
    if code == "system_x_tool_arguments_invalid":
        return StreamNormalizationError("tool_arguments_invalid")
    return StreamNormalizationError("tool_call_invalid")


@dataclass(slots=True)
class _ToolAccumulator:
    private_id: str
    public_id: str
    name: str
    argument_parts: list[str]
    argument_bytes: int = 0
    completed: bool = False

    def append(self, delta: str) -> None:
        self.argument_bytes = _append_bounded(
            self.argument_parts,
            delta,
            self.argument_bytes,
            MAXIMUM_TOOL_ARGUMENT_BYTES,
            "tool_arguments",
        )

    @property
    def argument_text(self) -> str:
        return "".join(self.argument_parts)

    def clear(self) -> None:
        self.argument_parts.clear()
        self.argument_bytes = 0


class _CanonicalNormalizer(Protocol):
    tool_completed: bool

    def accept(
        self, frame: ValidatedPrivateFrame
    ) -> list[CanonicalStreamEvent]: ...

    def clear(self) -> None: ...


class OpenAIPrivateStreamNormalizer:
    """Normalize pinned Completion or Chat chunks into canonical events."""

    def __init__(
        self,
        state: ActiveStreamState,
        kind: Literal["completion", "chat"],
        tools: list[FunctionTool],
        selected: ToolChoice | None,
    ) -> None:
        self.state = state
        self.kind = kind
        self.tools = tools
        self.selected = selected
        self.finish_reason: str | None = None
        self.verbose_truncated = False
        self.stop_sequence = False
        self.usage: dict[str, int] | None = None
        self.saw_text = False
        self.saw_reasoning = False
        self.pending_tool_private_id: str | None = None
        self.tool: _ToolAccumulator | None = None
        self.tool_completed = False
        self.done = False

    def _store_usage(self, raw: Any) -> None:
        if self.usage is not None:
            raise StreamNormalizationError("duplicate_usage")
        self.usage = _usage(raw, "prompt_tokens", "completion_tokens")

    def _accept_tool_delta(
        self, raw_calls: Any
    ) -> list[CanonicalStreamEvent]:
        if (
            self.kind != "chat"
            or not isinstance(raw_calls, list)
            or len(raw_calls) != 1
            or not isinstance(raw_calls[0], dict)
        ):
            raise StreamNormalizationError("tool_call_invalid")
        raw = raw_calls[0]
        if raw.get("index") != 0:
            raise StreamNormalizationError("parallel_tool_call_invalid")
        private_id = raw.get("id")
        raw_type = raw.get("type")
        function = raw.get("function")
        if function is not None and not isinstance(function, dict):
            raise StreamNormalizationError("tool_call_invalid")
        name = function.get("name") if isinstance(function, dict) else None
        arguments = (
            function.get("arguments") if isinstance(function, dict) else None
        )
        events: list[CanonicalStreamEvent] = []
        if self.tool is None:
            if private_id is not None:
                if (
                    not isinstance(private_id, str)
                    or not private_id
                    or raw_type != "function"
                    or self.pending_tool_private_id not in {None, private_id}
                ):
                    raise StreamNormalizationError("tool_call_invalid")
                self.pending_tool_private_id = private_id
            elif raw_type is not None:
                raise StreamNormalizationError("tool_call_invalid")
            if name is not None:
                if (
                    not isinstance(name, str)
                    or not name
                    or self.pending_tool_private_id is None
                ):
                    raise StreamNormalizationError("tool_call_invalid")
                self.tool = _ToolAccumulator(
                    self.pending_tool_private_id,
                    new_tool_call_id(),
                    name,
                    [],
                )
                events.append(
                    self.state.emit(
                        CanonicalStreamEventType.TOOL_CALL_ADDED,
                        index=0,
                        tool_call={
                            "id": self.tool.public_id,
                            "type": "function",
                            "name": self.tool.name,
                        },
                    )
                )
        else:
            if private_id not in {None, self.tool.private_id}:
                raise StreamNormalizationError("tool_call_identity_changed")
            if raw_type not in {None, "function"}:
                raise StreamNormalizationError("tool_call_invalid")
            if name not in {None, self.tool.name}:
                raise StreamNormalizationError("tool_call_name_changed")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise StreamNormalizationError("tool_arguments_invalid")
            if arguments:
                if self.tool is None:
                    raise StreamNormalizationError("tool_arguments_invalid")
                self.tool.append(arguments)
                events.append(
                    self.state.emit(
                        CanonicalStreamEventType.TOOL_CALL_ARGUMENTS_DELTA,
                        index=0,
                        tool_call_id=self.tool.public_id,
                        delta=arguments,
                    )
                )
        return events

    def _finish_tool(self) -> CanonicalStreamEvent:
        if self.tool is None or self.tool.completed:
            raise StreamNormalizationError("tool_call_invalid")
        raw_call = {
            "id": self.tool.private_id,
            "type": "function",
            "function": {
                "name": self.tool.name,
                "arguments": self.tool.argument_text,
            },
        }
        try:
            validated = validate_returned_tool_calls(
                [raw_call],
                self.tools,
                self.selected,
            )[0]
            public_call = ToolCall(
                id=self.tool.public_id,
                name=validated.name,
                arguments=validated.arguments,
            )
        except ValueError as exc:
            raise _tool_error(exc) from exc
        self.tool.completed = True
        self.tool_completed = True
        return self.state.emit(
            CanonicalStreamEventType.TOOL_CALL_DONE,
            index=0,
            tool_call={
                "id": public_call.id,
                "type": "function",
                "name": public_call.name,
                "arguments": public_call.arguments,
            },
        )

    def _accept_value(
        self, value: dict[str, Any]
    ) -> list[CanonicalStreamEvent]:
        if "error" in value:
            raise StreamNormalizationError("private_stream_error")
        verbose = value.get("__verbose")
        if verbose is not None:
            if not isinstance(verbose, dict):
                raise StreamNormalizationError("invalid_verbose_state")
            self.verbose_truncated = (
                self.verbose_truncated or verbose.get("truncated") is True
            )
            self.stop_sequence = (
                self.stop_sequence or verbose.get("stop_type") == "word"
            )
        raw_usage = value.get("usage")
        if raw_usage is not None:
            self._store_usage(raw_usage)
        choices = value.get("choices")
        if not isinstance(choices, list):
            raise StreamNormalizationError("invalid_choices")
        if not choices:
            if raw_usage is None:
                raise StreamNormalizationError("empty_choices_without_usage")
            return []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise StreamNormalizationError("invalid_choices")
        choice = choices[0]
        if choice.get("index") not in {None, 0}:
            raise StreamNormalizationError("multiple_choice_invalid")
        events: list[CanonicalStreamEvent] = []
        if self.kind == "completion":
            text = choice.get("text")
            if not isinstance(text, str):
                raise StreamNormalizationError("invalid_completion_delta")
            if text:
                self.saw_text = True
                events.append(
                    self.state.emit(
                        CanonicalStreamEventType.OUTPUT_TEXT_DELTA,
                        delta=text,
                    )
                )
        else:
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise StreamNormalizationError("invalid_chat_delta")
            role = delta.get("role")
            if role not in {None, "assistant"}:
                raise StreamNormalizationError("invalid_chat_role")
            reasoning = delta.get("reasoning_content")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise StreamNormalizationError("invalid_reasoning_delta")
                if reasoning:
                    self.saw_reasoning = True
                    events.append(
                        self.state.emit(
                            CanonicalStreamEventType.REASONING_DELTA,
                            delta=reasoning,
                        )
                    )
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise StreamNormalizationError("invalid_chat_delta")
                if content:
                    self.saw_text = True
                    events.append(
                        self.state.emit(
                            CanonicalStreamEventType.OUTPUT_TEXT_DELTA,
                            delta=content,
                        )
                    )
            raw_calls = delta.get("tool_calls")
            if raw_calls is not None:
                events.extend(self._accept_tool_delta(raw_calls))
        raw_finish = choice.get("finish_reason")
        if raw_finish is not None:
            if raw_finish not in {"stop", "length", "tool_calls"}:
                raise StreamNormalizationError("invalid_finish_reason")
            if self.finish_reason not in {None, raw_finish}:
                raise StreamNormalizationError("finish_reason_changed")
            self.finish_reason = raw_finish
            if raw_finish == "tool_calls" and not self.tool_completed:
                events.append(self._finish_tool())
        return events

    def _finish(self) -> list[CanonicalStreamEvent]:
        if self.done:
            raise StreamNormalizationError("duplicate_done")
        self.done = True
        if self.finish_reason is None:
            raise StreamNormalizationError("missing_finish_reason")
        if self.finish_reason == "tool_calls":
            if self.tool is None or not self.tool_completed:
                raise StreamNormalizationError("tool_call_invalid")
        elif self.tool is not None:
            raise StreamNormalizationError("tool_call_finish_mismatch")
        elif isinstance(self.selected, (ToolChoiceRequired, ToolChoiceFunction)):
            raise StreamNormalizationError("required_tool_call_missing")
        if self.tool is None and not self.saw_text:
            if self.saw_reasoning and self.finish_reason == "length":
                pass
            elif self.saw_reasoning:
                raise StreamNormalizationError("reasoning_only_output")
            else:
                raise StreamNormalizationError("empty_stream_output")
        if self.usage is None:
            raise StreamNormalizationError("missing_usage")
        events = [
            self.state.emit(
                CanonicalStreamEventType.USAGE,
                usage=self.usage,
            )
        ]
        if self.finish_reason == "tool_calls":
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.REQUIRES_ACTION,
                    status="requires_action",
                    finish_reason="tool_call",
                )
            )
        elif self.verbose_truncated:
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.INCOMPLETE,
                    status="incomplete",
                    finish_reason="context_limit",
                )
            )
        elif self.finish_reason == "length":
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.INCOMPLETE,
                    status="incomplete",
                    finish_reason="output_limit",
                )
            )
        else:
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.COMPLETED,
                    status="completed",
                    finish_reason=(
                        "stop_sequence" if self.stop_sequence else "completed"
                    ),
                )
            )
        return events

    def accept(
        self, frame: ValidatedPrivateFrame
    ) -> list[CanonicalStreamEvent]:
        if frame.heartbeat:
            return []
        if frame.done:
            return self._finish()
        if frame.value is None:
            raise StreamNormalizationError("empty_private_frame")
        return self._accept_value(frame.value)

    def clear(self) -> None:
        if self.tool is not None:
            self.tool.clear()
        self.usage = None


class ResponsesPrivateStreamNormalizer:
    """Normalize pinned typed Responses events into canonical events."""

    def __init__(
        self,
        state: ActiveStreamState,
        tools: list[FunctionTool],
        selected: ToolChoice,
        maximum_output_tokens: int,
    ) -> None:
        self.state = state
        self.tools = tools
        self.selected = selected
        self.maximum_output_tokens = maximum_output_tokens
        self.text_parts: list[str] = []
        self.text_bytes = 0
        self.reasoning_parts: list[str] = []
        self.reasoning_bytes = 0
        self.tool: _ToolAccumulator | None = None
        self.tool_completed = False
        self.terminal = False

    def _append_text(self, delta: str) -> None:
        self.text_bytes = _append_bounded(
            self.text_parts,
            delta,
            self.text_bytes,
            MAXIMUM_ACCUMULATED_TEXT_BYTES,
            "output_text",
        )

    def _append_reasoning(self, delta: str) -> None:
        self.reasoning_bytes = _append_bounded(
            self.reasoning_parts,
            delta,
            self.reasoning_bytes,
            MAXIMUM_ACCUMULATED_TEXT_BYTES,
            "reasoning",
        )

    @staticmethod
    def _item(value: dict[str, Any]) -> dict[str, Any]:
        item = value.get("item")
        if not isinstance(item, dict):
            raise StreamNormalizationError("invalid_output_item")
        return item

    def _start_tool(
        self, item: dict[str, Any]
    ) -> CanonicalStreamEvent:
        if self.tool is not None:
            raise StreamNormalizationError("parallel_tool_call_invalid")
        private_id = item.get("call_id")
        item_id = item.get("id")
        name = item.get("name")
        if (
            item.get("type") != "function_call"
            or item.get("status") != "in_progress"
            or not isinstance(private_id, str)
            or not private_id.startswith("call_")
            or not isinstance(item_id, str)
            or item_id != f"fc_{private_id.removeprefix('call_')}"
            or not isinstance(name, str)
            or not name
            or item.get("arguments") != ""
        ):
            raise StreamNormalizationError("tool_call_invalid")
        self.tool = _ToolAccumulator(
            private_id,
            new_tool_call_id(),
            name,
            [],
        )
        return self.state.emit(
            CanonicalStreamEventType.TOOL_CALL_ADDED,
            index=0,
            tool_call={
                "id": self.tool.public_id,
                "type": "function",
                "name": name,
            },
        )

    def _tool_delta(
        self, value: dict[str, Any]
    ) -> CanonicalStreamEvent:
        delta = value.get("delta")
        item_id = value.get("item_id")
        if (
            self.tool is None
            or not isinstance(item_id, str)
            or item_id != f"fc_{self.tool.private_id.removeprefix('call_')}"
            or not isinstance(delta, str)
            or not delta
        ):
            raise StreamNormalizationError("tool_arguments_invalid")
        self.tool.append(delta)
        return self.state.emit(
            CanonicalStreamEventType.TOOL_CALL_ARGUMENTS_DELTA,
            index=0,
            tool_call_id=self.tool.public_id,
            delta=delta,
        )

    def _finish_tool(
        self, item: dict[str, Any]
    ) -> CanonicalStreamEvent:
        if self.tool is None or self.tool.completed:
            raise StreamNormalizationError("tool_call_invalid")
        if (
            item.get("type") != "function_call"
            or item.get("status") != "completed"
            or item.get("call_id") != self.tool.private_id
            or item.get("id")
            != f"fc_{self.tool.private_id.removeprefix('call_')}"
            or item.get("name") != self.tool.name
            or item.get("arguments") != self.tool.argument_text
        ):
            raise StreamNormalizationError("tool_call_invalid")
        raw_call = {
            "id": self.tool.private_id,
            "type": "function",
            "function": {
                "name": self.tool.name,
                "arguments": self.tool.argument_text,
            },
        }
        try:
            validated = validate_returned_tool_calls(
                [raw_call],
                self.tools,
                self.selected,
            )[0]
            public_call = ToolCall(
                id=self.tool.public_id,
                name=validated.name,
                arguments=validated.arguments,
            )
        except ValueError as exc:
            raise _tool_error(exc) from exc
        self.tool.completed = True
        self.tool_completed = True
        return self.state.emit(
            CanonicalStreamEventType.TOOL_CALL_DONE,
            index=0,
            tool_call={
                "id": public_call.id,
                "type": "function",
                "name": public_call.name,
                "arguments": public_call.arguments,
            },
        )

    def _validate_done_item(self, item: dict[str, Any]) -> None:
        item_type = item.get("type")
        if item_type == "function_call":
            return
        if item_type == "reasoning":
            content = item.get("content")
            text = (
                content[0].get("text")
                if isinstance(content, list)
                and len(content) == 1
                and isinstance(content[0], dict)
                and content[0].get("type") == "reasoning_text"
                else None
            )
            if text != "".join(self.reasoning_parts):
                raise StreamNormalizationError("reasoning_done_mismatch")
            return
        if item_type == "message":
            content = item.get("content")
            text = (
                content[0].get("text")
                if isinstance(content, list)
                and len(content) == 1
                and isinstance(content[0], dict)
                and content[0].get("type") == "output_text"
                else None
            )
            if (
                item.get("role") != "assistant"
                or item.get("status") != "completed"
                or text != "".join(self.text_parts)
            ):
                raise StreamNormalizationError("output_item_done_mismatch")
            return
        raise StreamNormalizationError("invalid_output_item")

    def _terminal(
        self,
        response: dict[str, Any],
        *,
        incomplete: bool,
    ) -> list[CanonicalStreamEvent]:
        if self.terminal:
            raise StreamNormalizationError("duplicate_terminal")
        self.terminal = True
        usage = _usage(response.get("usage"), "input_tokens", "output_tokens")
        events = [
            self.state.emit(
                CanonicalStreamEventType.USAGE,
                usage=usage,
            )
        ]
        if self.tool is not None:
            if not self.tool_completed or incomplete:
                raise StreamNormalizationError("tool_call_invalid")
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.REQUIRES_ACTION,
                    status="requires_action",
                    finish_reason="tool_call",
                )
            )
            return events
        if isinstance(self.selected, (ToolChoiceRequired, ToolChoiceFunction)):
            raise StreamNormalizationError("required_tool_call_missing")
        if incomplete:
            detail = response.get("incomplete_details")
            reason = detail.get("reason") if isinstance(detail, dict) else None
            finish_reason = (
                "context_limit"
                if reason in {"context_length", "context_length_exceeded"}
                else "output_limit"
                if reason in {"max_output_tokens", "max_output_tokens_reached"}
                else "unknown"
            )
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.INCOMPLETE,
                    status="incomplete",
                    finish_reason=finish_reason,
                )
            )
            return events
        if self.text_parts:
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.COMPLETED,
                    status="completed",
                    finish_reason="completed",
                )
            )
            return events
        if (
            self.reasoning_parts
            and usage["output_tokens"] >= self.maximum_output_tokens
        ):
            events.append(
                self.state.emit(
                    CanonicalStreamEventType.INCOMPLETE,
                    status="incomplete",
                    finish_reason="output_limit",
                )
            )
            return events
        raise StreamNormalizationError(
            "reasoning_only_output"
            if self.reasoning_parts
            else "empty_stream_output"
        )

    def accept(
        self, frame: ValidatedPrivateFrame
    ) -> list[CanonicalStreamEvent]:
        if frame.heartbeat:
            return []
        if frame.done:
            if not self.terminal:
                raise StreamNormalizationError("responses_done_without_terminal")
            return []
        value = frame.value
        if value is None:
            raise StreamNormalizationError("empty_private_frame")
        event_type = value.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise StreamNormalizationError("missing_responses_event_type")
        if event_type in {"response.created", "response.in_progress"}:
            response = value.get("response")
            if not isinstance(response, dict) or response.get("status") != "in_progress":
                raise StreamNormalizationError("invalid_responses_start")
            return []
        if event_type == "response.reasoning_text.delta":
            delta = value.get("delta")
            if not isinstance(delta, str) or not delta:
                raise StreamNormalizationError("invalid_reasoning_delta")
            self._append_reasoning(delta)
            return [
                self.state.emit(
                    CanonicalStreamEventType.REASONING_DELTA,
                    delta=delta,
                )
            ]
        if event_type == "response.output_text.delta":
            delta = value.get("delta")
            if not isinstance(delta, str) or not delta:
                raise StreamNormalizationError("invalid_output_text_delta")
            self._append_text(delta)
            return [
                self.state.emit(
                    CanonicalStreamEventType.OUTPUT_TEXT_DELTA,
                    delta=delta,
                )
            ]
        if event_type == "response.output_item.added":
            item = self._item(value)
            if item.get("type") == "function_call":
                return [self._start_tool(item)]
            if item.get("type") not in {"message", "reasoning"}:
                raise StreamNormalizationError("invalid_output_item")
            return []
        if event_type == "response.function_call_arguments.delta":
            return [self._tool_delta(value)]
        if event_type == "response.output_text.done":
            if value.get("text") != "".join(self.text_parts):
                raise StreamNormalizationError("output_text_done_mismatch")
            return []
        if event_type == "response.content_part.added":
            part = value.get("part")
            if not isinstance(part, dict) or part.get("type") != "output_text":
                raise StreamNormalizationError("invalid_content_part")
            return []
        if event_type == "response.content_part.done":
            part = value.get("part")
            if (
                not isinstance(part, dict)
                or part.get("type") != "output_text"
                or part.get("text") != "".join(self.text_parts)
            ):
                raise StreamNormalizationError("content_part_done_mismatch")
            return []
        if event_type == "response.output_item.done":
            item = self._item(value)
            if item.get("type") == "function_call":
                return [self._finish_tool(item)]
            self._validate_done_item(item)
            return []
        if event_type in {"response.completed", "response.incomplete"}:
            response = value.get("response")
            expected = (
                "incomplete"
                if event_type == "response.incomplete"
                else "completed"
            )
            if not isinstance(response, dict) or response.get("status") != expected:
                raise StreamNormalizationError("invalid_responses_terminal")
            return self._terminal(
                response,
                incomplete=event_type == "response.incomplete",
            )
        if event_type in {"error", "response.failed"}:
            raise StreamNormalizationError("private_stream_error")
        raise StreamNormalizationError("unsupported_responses_event")

    def clear(self) -> None:
        self.text_parts.clear()
        self.reasoning_parts.clear()
        self.text_bytes = 0
        self.reasoning_bytes = 0
        if self.tool is not None:
            self.tool.clear()


@dataclass(slots=True)
class CanonicalStreamSession:
    """An already-open private response ready for one public encoder."""

    operation: StreamOperation
    private_kind: PrivateStreamKind
    private_endpoint: str
    control: StreamControl
    upstream: PrivateRouterStream
    lease: InferenceBackendLease
    exit_stack: AsyncExitStack
    normalizer: _CanonicalNormalizer
    inference: InferenceService
    capability_model_id: str
    protocol_surface: str
    input_tokens: int | None = None
    frame_count: int = 0
    data_bytes: int = 0
    _tool_capability_marked: bool = False
    _streaming_capability_marked: bool = False

    async def _mark_tool_capability(self) -> None:
        if self._tool_capability_marked:
            return
        await self.inference.catalogue.mark_capability_proven(
            self.capability_model_id,
            "tool_calling",
            self.control.state.request_id,
            self.inference._service_transaction_id(),
            self.lease.router_identity.transaction_id,
        )
        self._tool_capability_marked = True

    async def _mark_streaming_capability(self) -> None:
        if self._streaming_capability_marked:
            return
        try:
            await self.inference.catalogue.mark_capability_proven(
                self.capability_model_id,
                "streaming",
                self.control.state.request_id,
                self.inference._service_transaction_id(),
                self.lease.router_identity.transaction_id,
                (self.protocol_surface,),
            )
        except SystemXError:
            LOGGER.exception(
                "streaming capability evidence persistence failed "
                "request_id=%s protocol_surface=%s",
                self.control.state.request_id,
                self.protocol_surface,
            )
            return
        self._streaming_capability_marked = True

    @staticmethod
    def _failure_kind(exc: BaseException) -> tuple[str, str]:
        if isinstance(exc, httpx.TimeoutException):
            return (
                "system_x_backend_timeout",
                "Private streaming inference timed out",
            )
        if isinstance(exc, httpx.RequestError):
            return (
                "system_x_backend_unavailable",
                "Private streaming inference connection failed",
            )
        return (
            "system_x_backend_response_invalid",
            "Private streaming inference returned invalid events",
        )

    async def events(self) -> AsyncIterator[CanonicalStreamEvent]:
        digest = hashlib.sha256()
        terminal_observed = False
        usage_evidence: dict[str, int] | None = None
        try:
            yield self.control.state.start(self.operation)
            async for raw_frame in self.upstream.frames():
                self.frame_count += 1
                if raw_frame.data is not None:
                    encoded = raw_frame.data.encode("utf-8")
                    self.data_bytes += len(encoded)
                    digest.update(encoded)
                frame = validate_private_frame(raw_frame)
                events = self.normalizer.accept(frame)
                for event in events:
                    if event.type is CanonicalStreamEventType.USAGE:
                        usage_evidence = dict(event.payload["usage"])
                    if event.type is CanonicalStreamEventType.TOOL_CALL_DONE:
                        await self._mark_tool_capability()
                    if event.terminal:
                        await self._mark_streaming_capability()
                        self.inference.catalogue.mark_operation_proven(
                            self.capability_model_id,
                            self.operation,
                        )
                        if event.type is CanonicalStreamEventType.FAILED:
                            error = event.payload.get("error")
                            error_code = (
                                error.get("code")
                                if isinstance(error, dict)
                                else None
                            )
                            if not isinstance(error_code, str):
                                raise StreamNormalizationError(
                                    "invalid_stream_error_code"
                                )
                            operation_state = "failed"
                            finish_reason = None
                        else:
                            operation_state = str(event.payload["status"])
                            finish_reason = str(
                                event.payload["finish_reason"]
                            )
                            error_code = None
                        self.inference.operations.note_terminal(
                            self.control.state.request_id,
                            state=operation_state,
                            finish_reason=finish_reason,
                            input_tokens=(
                                usage_evidence["input_tokens"]
                                if usage_evidence is not None
                                else None
                            ),
                            output_tokens=(
                                usage_evidence["output_tokens"]
                                if usage_evidence is not None
                                else None
                            ),
                            error_code=error_code,
                        )
                        terminal_observed = True
                    yield event
            if not terminal_observed:
                raise StreamNormalizationError("private_stream_missing_terminal")
        except asyncio.CancelledError:
            raise
        except (
            SSEProtocolError,
            StreamNormalizationError,
            httpx.HTTPError,
        ) as exc:
            if self.control.state.state is StreamState.STREAMING:
                code, message = self._failure_kind(exc)
                failed = self.control.state.emit(
                    CanonicalStreamEventType.FAILED,
                    status="failed",
                    finish_reason="error",
                    error={"code": code, "message": message},
                )
                self.inference.operations.note_terminal(
                    self.control.state.request_id,
                    state="failed",
                    finish_reason=None,
                    input_tokens=(
                        usage_evidence["input_tokens"]
                        if usage_evidence is not None
                        else None
                    ),
                    output_tokens=(
                        usage_evidence["output_tokens"]
                        if usage_evidence is not None
                        else None
                    ),
                    error_code=code,
                )
                terminal_observed = True
                yield failed
            else:
                raise
        finally:
            self.normalizer.clear()
            await asyncio.shield(self.exit_stack.aclose())
            LOGGER.info(
                "private streaming result "
                "service_transaction=%s request_id=%s "
                "router_transaction=%s router_pid=%s router_start=%s "
                "router_model=%s endpoint=%s frames=%s data_bytes=%s "
                "data_sha256=%s state=%s",
                os.environ.get(SERVICE_TRANSACTION_ENV, "unavailable"),
                self.control.state.request_id,
                self.lease.router_identity.transaction_id,
                self.lease.router_identity.pid,
                self.lease.router_identity.process_start_identity,
                self.lease.router_model_id,
                self.private_endpoint,
                self.frame_count,
                self.data_bytes,
                digest.hexdigest(),
                self.control.state.state.value,
            )


class StreamingInferenceService:
    """Open one canonical private stream for every public protocol adapter."""

    def __init__(
        self,
        inference: InferenceService,
        registry: ActiveStreamRegistry,
    ) -> None:
        self.inference = inference
        self.registry = registry

    @staticmethod
    def _raise_stream_open(exc: RouterStreamOpenError) -> None:
        observation = exc.observation
        if observation.error == "timeout":
            raise SystemXError(
                504,
                "system_x_backend_timeout",
                "Private inference backend timed out",
                retryable=True,
            ) from exc
        if observation.error == "connection_failure":
            raise SystemXError(
                503,
                "system_x_backend_unavailable",
                "Private inference backend is unavailable",
                retryable=True,
            ) from exc
        raise SystemXError(
            502,
            "system_x_backend_response_invalid",
            "Private inference backend did not open an accepted stream",
        ) from exc

    async def _cleanup_open_failure(
        self,
        control: StreamControl,
        stack: AsyncExitStack,
        *,
        cancelled: bool,
    ) -> None:
        await asyncio.shield(stack.aclose())
        if control.state.state in {
            StreamState.BACKEND_OPENING,
            StreamState.STREAMING,
        }:
            if cancelled:
                await asyncio.shield(
                    control.cancel(
                        "route_task_cancelled",
                        cancel_producer=False,
                    )
                )
            elif control.state.state is StreamState.BACKEND_OPENING:
                control.state.fail_before_start()
            else:
                await control.cancel(
                    "stream_open_failure",
                    cancel_producer=False,
                )
        await asyncio.shield(control.finalize("stream_open_failure"))

    async def _open_session(
        self,
        *,
        request_id: str,
        public_endpoint: str,
        operation: StreamOperation,
        private_kind: PrivateStreamKind,
        private_endpoint: str,
        model_id: str,
        router_model_id: str,
        protocol_surface: str,
        private_context: Any,
        normalizer_factory: Any,
        preopen: Any = None,
    ) -> CanonicalStreamSession:
        state = ActiveStreamState(
            request_id,
            public_endpoint,
            model_id,
        )
        control = await self.registry.register(state)
        state.begin_backend()
        stack = AsyncExitStack()
        try:
            lease = await stack.enter_async_context(
                self.inference.backend.inference_session(
                    router_model_id,
                    lambda: self.inference.catalogue.verify(
                        normalizer_factory["snapshot"]
                    ),
                )
            )
            self.inference.operations.note_router(
                request_id,
                lease.router_identity.transaction_id,
            )
            input_tokens = (
                await preopen(lease) if preopen is not None else None
            )
            upstream = await stack.enter_async_context(private_context(lease))
            await control.attach_upstream(upstream)
            normalizer = normalizer_factory["build"](state)
            self.inference.operations.mark_streamed(request_id)
            return CanonicalStreamSession(
                operation=operation,
                private_kind=private_kind,
                private_endpoint=private_endpoint,
                control=control,
                upstream=upstream,
                lease=lease,
                exit_stack=stack,
                normalizer=normalizer,
                inference=self.inference,
                capability_model_id=model_id,
                protocol_surface=protocol_surface,
                input_tokens=input_tokens,
            )
        except asyncio.CancelledError:
            await self._cleanup_open_failure(
                control,
                stack,
                cancelled=True,
            )
            raise
        except RouterStreamOpenError as exc:
            await self._cleanup_open_failure(
                control,
                stack,
                cancelled=False,
            )
            self._raise_stream_open(exc)
        except BackendError as exc:
            await self._cleanup_open_failure(
                control,
                stack,
                cancelled=False,
            )
            self.inference._raise_backend(exc)
        except ResponseNormalizationError as exc:
            await self._cleanup_open_failure(
                control,
                stack,
                cancelled=False,
            )
            self.inference._raise_normalization(exc)
        except BaseException:
            await self._cleanup_open_failure(
                control,
                stack,
                cancelled=False,
            )
            raise
        raise AssertionError("unreachable stream-open path")

    async def open_generate(
        self,
        request_id: str,
        public_endpoint: str,
        request: GenerateRequest,
        *,
        protocol_surface: str,
    ) -> CanonicalStreamSession:
        prepared: PreparedGenerate = await self.inference.prepare_generate(
            request_id, request
        )
        snapshot = prepared.snapshot

        def private_context(lease: InferenceBackendLease) -> Any:
            return lease.router.completion_stream(
                lease.router_model_id,
                request.input,
                request.max_output_tokens,
                request.temperature,
                request.stop,
            )

        return await self._open_session(
            request_id=request_id,
            public_endpoint=public_endpoint,
            operation="generate",
            private_kind="completion",
            private_endpoint="/v1/completions",
            model_id=snapshot.public_model_id,
            router_model_id=snapshot.router_model_id,
            protocol_surface=protocol_surface,
            private_context=private_context,
            normalizer_factory={
                "snapshot": snapshot,
                "build": lambda state: OpenAIPrivateStreamNormalizer(
                    state,
                    "completion",
                    [],
                    None,
                ),
            },
        )

    async def open_chat(
        self,
        request_id: str,
        public_endpoint: str,
        request: ChatRequest,
        *,
        require_input_tokens: bool = False,
        protocol_surface: str,
    ) -> CanonicalStreamSession:
        if request.output_format is not None:
            raise SystemXError(
                422,
                "system_x_streaming_structured_output_unsupported",
                "Structured-output streaming is not available",
            )
        prepared: PreparedChat = await self.inference.prepare_chat(
            request_id, request
        )
        snapshot = prepared.snapshot

        def private_context(lease: InferenceBackendLease) -> Any:
            return lease.router.chat_completion_stream(
                lease.router_model_id,
                prepared.messages,
                request.max_output_tokens,
                request.temperature,
                request.stop,
                prepared.private_tools,
                prepared.private_choice,
                None,
                prepared.private_template_kwargs,
            )

        async def preopen(lease: InferenceBackendLease) -> int:
            observation = await lease.router.chat_input_tokens(
                lease.router_model_id,
                prepared.messages,
                prepared.private_tools,
                prepared.private_template_kwargs,
            )
            self.inference._record_private_result(
                request_id,
                "/v1/chat/completions/input_tokens",
                lease,
                observation,
            )
            return normalize_token_count(observation)

        return await self._open_session(
            request_id=request_id,
            public_endpoint=public_endpoint,
            operation="chat",
            private_kind="chat",
            private_endpoint="/v1/chat/completions",
            model_id=snapshot.public_model_id,
            router_model_id=snapshot.router_model_id,
            protocol_surface=protocol_surface,
            private_context=private_context,
            normalizer_factory={
                "snapshot": snapshot,
                "build": lambda state: OpenAIPrivateStreamNormalizer(
                    state,
                    "chat",
                    request.tools,
                    prepared.selected,
                ),
            },
            preopen=preopen if require_input_tokens else None,
        )

    async def open_responses(
        self,
        request_id: str,
        public_endpoint: str,
        request: ResponsesRequest,
        *,
        protocol_surface: str,
    ) -> CanonicalStreamSession:
        if request.output_format is not None:
            raise SystemXError(
                422,
                "system_x_streaming_structured_output_unsupported",
                "Structured-output streaming is not available",
            )
        prepared: PreparedResponses = await self.inference.prepare_responses(
            request_id, request
        )
        snapshot = prepared.snapshot

        def private_context(lease: InferenceBackendLease) -> Any:
            return lease.router.responses_stream(
                lease.router_model_id,
                prepared.private_input,
                request.max_output_tokens,
                request.instructions,
                request.temperature,
                prepared.private_tools,
                prepared.private_choice,
                None,
                prepared.private_template_kwargs,
            )

        return await self._open_session(
            request_id=request_id,
            public_endpoint=public_endpoint,
            operation="responses",
            private_kind="responses",
            private_endpoint="/v1/responses",
            model_id=snapshot.public_model_id,
            router_model_id=snapshot.router_model_id,
            protocol_surface=protocol_surface,
            private_context=private_context,
            normalizer_factory={
                "snapshot": snapshot,
                "build": lambda state: ResponsesPrivateStreamNormalizer(
                    state,
                    request.tools,
                    prepared.selected,
                    request.max_output_tokens,
                ),
            },
        )
