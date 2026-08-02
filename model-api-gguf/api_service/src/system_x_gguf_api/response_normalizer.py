"""Bounded, model-neutral normalization of pinned llama-server responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .router_client import RouterObservation
from .schemas import FinishReason, InferenceStatus, TokenUsage
from .tool_contract import (
    FunctionTool,
    PRIVATE_CALL_ID_PATTERN,
    StructuredOutputFormat,
    ToolCall,
    ToolContractError,
    ToolChoice,
    ToolChoiceFunction,
    ToolChoiceRequired,
    parse_structured_output,
    validate_returned_tool_calls,
    validate_tool_choice,
)
from .tool_schema import SchemaPolicyError


class ResponseNormalizationError(ValueError):
    """A private response cannot satisfy the System X output contract."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class NormalizedText:
    status: InferenceStatus
    finish_reason: FinishReason
    text: str
    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class NormalizedChat:
    status: InferenceStatus
    finish_reason: FinishReason
    role: str
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    structured: Any | None
    usage: TokenUsage
    reasoning_observed: bool
    reasoning: tuple[str, ...]


def _payload(observation: RouterObservation) -> dict[str, Any]:
    if observation.error is not None:
        raise ResponseNormalizationError(observation.error)
    if observation.status_code != 200:
        raise ResponseNormalizationError("private_http_error")
    if not isinstance(observation.json_value, dict):
        raise ResponseNormalizationError("response_not_object")
    return observation.json_value


def _optional_count(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ResponseNormalizationError(f"invalid_usage_{field}")
    return value


def _usage(
    payload: dict[str, Any],
    input_name: str,
    output_name: str,
) -> TokenUsage:
    raw = payload.get("usage")
    if raw is None:
        return TokenUsage()
    if not isinstance(raw, dict):
        raise ResponseNormalizationError("invalid_usage")
    input_tokens = _optional_count(raw.get(input_name), input_name)
    output_tokens = _optional_count(raw.get(output_name), output_name)
    total_tokens = _optional_count(raw.get("total_tokens"), "total_tokens")
    try:
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    except ValueError as exc:
        raise ResponseNormalizationError("contradictory_usage") from exc


def _finish(
    payload: dict[str, Any], choice: dict[str, Any]
) -> tuple[InferenceStatus, FinishReason]:
    raw_finish = choice.get("finish_reason")
    if not isinstance(raw_finish, str) or not raw_finish:
        raise ResponseNormalizationError("invalid_finish_reason")
    verbose = payload.get("__verbose")
    verbose_object = verbose if isinstance(verbose, dict) else {}
    if verbose_object.get("truncated") is True:
        return "incomplete", "context_limit"
    if raw_finish == "length":
        return "incomplete", "output_limit"
    if raw_finish == "stop":
        if verbose_object.get("stop_type") == "word":
            return "completed", "stop_sequence"
        return "completed", "completed"
    return "incomplete", "unknown"


def _first_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise ResponseNormalizationError("invalid_choices")
    return choices[0]


def normalize_completion(observation: RouterObservation) -> NormalizedText:
    payload = _payload(observation)
    choice = _first_choice(payload)
    text = choice.get("text")
    if not isinstance(text, str) or not text:
        raise ResponseNormalizationError("empty_completion_output")
    status, finish_reason = _finish(payload, choice)
    return NormalizedText(
        status=status,
        finish_reason=finish_reason,
        text=text,
        usage=_usage(payload, "prompt_tokens", "completion_tokens"),
    )


def normalize_chat(observation: RouterObservation) -> NormalizedChat:
    return normalize_chat_turn(observation, [], None, None)


def _tool_error(exc: ValueError) -> ResponseNormalizationError:
    if isinstance(exc, ToolContractError):
        if exc.code == "system_x_tool_arguments_invalid":
            return ResponseNormalizationError("tool_arguments_invalid")
        return ResponseNormalizationError("tool_call_invalid")
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, SchemaPolicyError):
            return ResponseNormalizationError("tool_arguments_invalid")
        cause = cause.__cause__
    return ResponseNormalizationError("tool_call_invalid")


def _reasoning_parts(message: dict[str, Any]) -> tuple[str, ...]:
    parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        value = message.get(field)
        if value is None:
            continue
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 1_048_576
        ):
            raise ResponseNormalizationError("invalid_reasoning_output")
        parts.append(value)
    return tuple(parts)


def normalize_chat_turn(
    observation: RouterObservation,
    tools: list[FunctionTool],
    tool_choice: ToolChoice | None,
    output_format: StructuredOutputFormat | None,
) -> NormalizedChat:
    payload = _payload(observation)
    choice = _first_choice(payload)
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ResponseNormalizationError("invalid_chat_message")
    reasoning = _reasoning_parts(message)
    reasoning_observed = bool(reasoning)
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ResponseNormalizationError("invalid_chat_content")
    if isinstance(content, str) and not content:
        content = None
    raw_calls = message.get("tool_calls")
    selected = validate_tool_choice(tool_choice, tools)
    if raw_calls is not None:
        if output_format is not None:
            raise ResponseNormalizationError("tool_and_output_format_conflict")
        if choice.get("finish_reason") != "tool_calls":
            raise ResponseNormalizationError("tool_call_invalid")
        try:
            calls = validate_returned_tool_calls(raw_calls, tools, selected)
        except ValueError as exc:
            raise _tool_error(exc) from exc
        return NormalizedChat(
            status="requires_action",
            finish_reason="tool_call",
            role="assistant",
            content=content,
            tool_calls=tuple(calls),
            structured=None,
            usage=_usage(payload, "prompt_tokens", "completion_tokens"),
            reasoning_observed=reasoning_observed,
            reasoning=reasoning,
        )
    status, finish_reason = _finish(payload, choice)
    if status == "incomplete":
        return NormalizedChat(
            status=status,
            finish_reason=finish_reason,
            role="assistant",
            content=content,
            tool_calls=(),
            structured=None,
            usage=_usage(payload, "prompt_tokens", "completion_tokens"),
            reasoning_observed=reasoning_observed,
            reasoning=reasoning,
        )
    if isinstance(selected, (ToolChoiceRequired, ToolChoiceFunction)):
        raise ResponseNormalizationError("required_tool_call_missing")
    if content is None:
        raise ResponseNormalizationError(
            "reasoning_only_output" if reasoning_observed else "empty_final_chat_output"
        )
    structured = None
    if output_format is not None:
        if status != "completed":
            return NormalizedChat(
                status=status,
                finish_reason=finish_reason,
                role="assistant",
                content=content,
                tool_calls=(),
                structured=None,
                usage=_usage(payload, "prompt_tokens", "completion_tokens"),
                reasoning_observed=reasoning_observed,
                reasoning=reasoning,
            )
        try:
            content, structured = parse_structured_output(output_format, content)
        except ValueError as exc:
            raise ResponseNormalizationError("structured_output_invalid") from exc
    return NormalizedChat(
        status=status,
        finish_reason=finish_reason,
        role="assistant",
        content=content,
        tool_calls=(),
        structured=structured,
        usage=_usage(payload, "prompt_tokens", "completion_tokens"),
        reasoning_observed=reasoning_observed,
        reasoning=reasoning,
    )


def _responses_finish(
    payload: dict[str, Any],
) -> tuple[InferenceStatus, FinishReason]:
    private_status = payload.get("status")
    if private_status == "completed":
        return "completed", "completed"
    if private_status == "incomplete":
        detail = payload.get("incomplete_details")
        reason = detail.get("reason") if isinstance(detail, dict) else None
        if reason in {"max_output_tokens", "max_output_tokens_reached"}:
            return "incomplete", "output_limit"
        if reason in {"context_length", "context_length_exceeded"}:
            return "incomplete", "context_limit"
        return "incomplete", "unknown"
    if isinstance(private_status, str) and private_status:
        return "incomplete", "unknown"
    raise ResponseNormalizationError("invalid_responses_status")


def _private_response_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or PRIVATE_CALL_ID_PATTERN.fullmatch(value) is None
    ):
        raise ResponseNormalizationError("invalid_responses_item_identity")
    return value


def normalize_responses(observation: RouterObservation) -> NormalizedText:
    turn = normalize_responses_turn(observation, [], None, None)
    if turn.content is None:
        raise ResponseNormalizationError("empty_responses_output")
    return NormalizedText(
        status=turn.status,
        finish_reason=turn.finish_reason,
        text=turn.content,
        usage=turn.usage,
    )


def normalize_responses_turn(
    observation: RouterObservation,
    tools: list[FunctionTool],
    tool_choice: ToolChoice | None,
    output_format: StructuredOutputFormat | None,
    maximum_output_tokens: int | None = None,
) -> NormalizedChat:
    payload = _payload(observation)
    output = payload.get("output")
    if not isinstance(output, list):
        raise ResponseNormalizationError("invalid_responses_output")
    status, finish_reason = _responses_finish(payload)
    usage = _usage(payload, "input_tokens", "output_tokens")
    text_parts: list[str] = []
    raw_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise ResponseNormalizationError("invalid_responses_output_item")
        if item.get("type") == "reasoning":
            _private_response_id(item.get("id"))
            reasoning_status = item.get("status")
            if (
                reasoning_status not in {"completed", "incomplete"}
                or (
                    status == "completed"
                    and reasoning_status != "completed"
                )
                or not isinstance(item.get("summary"), list)
                or (
                    "encrypted_content" in item
                    and not isinstance(item.get("encrypted_content"), str)
                )
            ):
                raise ResponseNormalizationError("invalid_reasoning_output")
            content = item.get("content")
            if not isinstance(content, list) or not content:
                raise ResponseNormalizationError("invalid_reasoning_output")
            for block in content:
                if (
                    not isinstance(block, dict)
                    or block.get("type") != "reasoning_text"
                    or not isinstance(block.get("text"), str)
                    or not block["text"].strip()
                    or len(block["text"]) > 1_048_576
                ):
                    raise ResponseNormalizationError(
                        "invalid_reasoning_output"
                    )
                reasoning_parts.append(block["text"])
                if len(reasoning_parts) > 8:
                    raise ResponseNormalizationError(
                        "invalid_reasoning_output"
                    )
            continue
        if item.get("type") == "function_call":
            private_id = _private_response_id(item.get("id"))
            call_id = _private_response_id(item.get("call_id"))
            if (
                status != "completed"
                or item.get("status") != "completed"
                or not private_id.startswith("fc_")
                or not call_id.startswith("call_")
                or not private_id.removeprefix("fc_")
                or private_id.removeprefix("fc_")
                != call_id.removeprefix("call_")
            ):
                raise ResponseNormalizationError("tool_call_invalid")
            raw_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    },
                }
            )
            continue
        if item.get("type") != "message":
            raise ResponseNormalizationError("invalid_responses_output_item")
        _private_response_id(item.get("id"))
        if (
            item.get("role") != "assistant"
            or item.get("status") not in {"completed", "incomplete"}
        ):
            raise ResponseNormalizationError("invalid_responses_message")
        content = item.get("content")
        if not isinstance(content, list) or not content:
            raise ResponseNormalizationError("invalid_responses_content")
        for block in content:
            if (
                not isinstance(block, dict)
                or block.get("type") != "output_text"
                or not set(block).issubset(
                    {"type", "text", "annotations", "logprobs"}
                )
            ):
                raise ResponseNormalizationError("invalid_responses_content")
            text = block.get("text")
            if not isinstance(text, str):
                raise ResponseNormalizationError("invalid_responses_text")
            if text:
                text_parts.append(text)
    text = "".join(text_parts)
    reasoning = tuple(reasoning_parts)
    reasoning_observed = bool(reasoning)
    if (
        status == "completed"
        and not text
        and not raw_calls
        and reasoning_observed
        and maximum_output_tokens is not None
        and usage.output_tokens is not None
        and usage.output_tokens >= maximum_output_tokens
    ):
        status = "incomplete"
        finish_reason = "output_limit"
    selected = validate_tool_choice(tool_choice, tools)
    if raw_calls:
        if output_format is not None:
            raise ResponseNormalizationError("tool_and_output_format_conflict")
        try:
            calls = validate_returned_tool_calls(raw_calls, tools, selected)
        except ValueError as exc:
            raise _tool_error(exc) from exc
        return NormalizedChat(
            status="requires_action",
            finish_reason="tool_call",
            role="assistant",
            content=text or None,
            tool_calls=tuple(calls),
            structured=None,
            usage=usage,
            reasoning_observed=reasoning_observed,
            reasoning=reasoning,
        )
    if status == "incomplete":
        return NormalizedChat(
            status=status,
            finish_reason=finish_reason,
            role="assistant",
            content=text or None,
            tool_calls=(),
            structured=None,
            usage=usage,
            reasoning_observed=reasoning_observed,
            reasoning=reasoning,
        )
    if isinstance(selected, (ToolChoiceRequired, ToolChoiceFunction)):
        raise ResponseNormalizationError("required_tool_call_missing")
    if not text:
        raise ResponseNormalizationError(
            "reasoning_only_output" if reasoning_observed else "empty_responses_output"
        )
    structured = None
    if output_format is not None and status == "completed":
        try:
            text, structured = parse_structured_output(output_format, text)
        except ValueError as exc:
            raise ResponseNormalizationError("structured_output_invalid") from exc
    return NormalizedChat(
        status=status,
        finish_reason=finish_reason,
        role="assistant",
        content=text,
        tool_calls=(),
        structured=structured,
        usage=usage,
        reasoning_observed=reasoning_observed,
        reasoning=reasoning,
    )


def normalize_token_count(
    observation: RouterObservation, *, tokenize: bool = False
) -> int:
    payload = _payload(observation)
    if tokenize:
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ResponseNormalizationError("invalid_token_array")
        count = len(tokens)
    else:
        count = payload.get("input_tokens")
        if type(count) is not int:
            raise ResponseNormalizationError("invalid_input_token_count")
    if count <= 0:
        raise ResponseNormalizationError("nonpositive_input_token_count")
    return count
