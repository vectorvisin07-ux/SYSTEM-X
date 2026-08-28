"""Canonical turn intent and private finalization policy."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

from .tool_contract import CanonicalMessage, ToolHistory


class TurnIntent(str, Enum):
    """Internal generation intent derived only from canonical request state."""

    NORMAL_TEXT = "normal_text"
    TOOL_SELECTION = "tool_selection"
    TOOL_RESULT_FINALIZATION = "tool_result_finalization"
    STRUCTURED_FINALIZATION = "structured_finalization"


FINAL_ANSWER_RESERVE_CONTRACT = "bounded_two_phase_non_stream"


def validate_final_answer_reserve(
    *,
    intent: TurnIntent,
    reasoning_mode: str | None,
    reserve_tokens: int | None,
    max_output_tokens: int,
    stream: bool,
) -> int | None:
    """Validate the explicit reserve contract before private forwarding."""

    if reserve_tokens is None:
        return None
    if intent is not TurnIntent.NORMAL_TEXT or reasoning_mode != "pro_extended":
        raise ValueError(
            "final_answer_reserve_tokens requires pro_extended normal text"
        )
    if stream:
        raise ValueError(
            "final_answer_reserve_tokens is unsupported for streaming"
        )
    if type(reserve_tokens) is not int or reserve_tokens < 1:
        raise ValueError(
            "final_answer_reserve_tokens must be at least one token"
        )
    if reserve_tokens >= max_output_tokens:
        raise ValueError(
            "final_answer_reserve_tokens must leave a reasoning allocation"
        )
    return reserve_tokens


def classify_turn_intent(
    *,
    history: ToolHistory,
    messages: Sequence[CanonicalMessage],
    has_tools: bool,
    has_output_format: bool,
) -> TurnIntent:
    """Classify a validated canonical turn without model-specific behavior."""

    if has_tools and has_output_format:
        raise ValueError("tools and structured output cannot be combined")
    if has_output_format:
        return TurnIntent.STRUCTURED_FINALIZATION
    latest_tool_result = max(
        (
            index
            for index, message in enumerate(messages)
            if message.role == "tool"
        ),
        default=-1,
    )
    latest_final_assistant = max(
        (
            index
            for index, message in enumerate(messages)
            if message.role == "assistant" and not message.tool_calls
        ),
        default=-1,
    )
    if history.resolved_ids and latest_tool_result > latest_final_assistant:
        if not has_tools:
            raise ValueError("resolved tool history requires tool definitions")
        return TurnIntent.TOOL_RESULT_FINALIZATION
    if has_tools:
        return TurnIntent.TOOL_SELECTION
    return TurnIntent.NORMAL_TEXT


def private_chat_template_kwargs(
    intent: TurnIntent,
    enable_thinking: bool | None = None,
) -> dict[str, Any] | None:
    """Return the one bounded private template override for finalization."""

    if intent in {
        TurnIntent.TOOL_RESULT_FINALIZATION,
        TurnIntent.STRUCTURED_FINALIZATION,
    }:
        return {"enable_thinking": False}
    if enable_thinking is not None:
        return {"enable_thinking": enable_thinking}
    return None


def retain_declared_tools(intent: TurnIntent) -> bool:
    """Retain client definitions while a tool result is being finalized."""

    return intent is TurnIntent.TOOL_RESULT_FINALIZATION
