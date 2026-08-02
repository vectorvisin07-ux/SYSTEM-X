"""Bounded incremental Server-Sent Events parsing for private streams."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import json
from typing import Any


DEFAULT_MAXIMUM_LINE_BYTES = 65_536
DEFAULT_MAXIMUM_FRAME_BYTES = 1_048_576


class SSEProtocolError(RuntimeError):
    """A private stream violated the bounded UTF-8/SSE/JSON contract."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class SSEFrame:
    event: str | None
    id: str | None
    data: str | None
    comments: tuple[str, ...]

    def json_value(self) -> Any:
        if self.data is None:
            raise SSEProtocolError("SSE frame does not contain a data field")
        try:
            return json.loads(
                self.data,
                object_pairs_hook=_unique_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise SSEProtocolError("SSE data is malformed JSON") from exc

    def json_object(self) -> dict[str, Any]:
        value = self.json_value()
        if not isinstance(value, dict):
            raise SSEProtocolError("SSE JSON data must be an object")
        return value


@dataclass(frozen=True, slots=True)
class ValidatedPrivateFrame:
    event: str | None
    id: str | None
    value: dict[str, Any] | None
    done: bool
    heartbeat: bool
    comments: tuple[str, ...]


def validate_private_frame(frame: SSEFrame) -> ValidatedPrivateFrame:
    """Validate one bounded llama-server frame before protocol normalization."""

    if frame.event is not None and (
        len(frame.event) > 128
        or any(ord(character) < 32 for character in frame.event)
    ):
        raise SSEProtocolError("private SSE event name is invalid")
    if frame.id is not None and (
        len(frame.id) > 512
        or any(ord(character) < 32 for character in frame.id)
    ):
        raise SSEProtocolError("private SSE event ID is invalid")
    if frame.data is None:
        if not frame.comments:
            raise SSEProtocolError("private SSE frame is empty")
        return ValidatedPrivateFrame(
            frame.event,
            frame.id,
            None,
            False,
            True,
            frame.comments,
        )
    if frame.data == "[DONE]":
        return ValidatedPrivateFrame(
            frame.event,
            frame.id,
            None,
            True,
            False,
            frame.comments,
        )
    value = frame.json_object()
    typed_event = value.get("type")
    if (
        frame.event not in {None, "", "message"}
        and isinstance(typed_event, str)
        and typed_event != frame.event
    ):
        raise SSEProtocolError("private SSE event field and JSON type disagree")
    return ValidatedPrivateFrame(
        frame.event,
        frame.id,
        value,
        False,
        False,
        frame.comments,
    )


class IncrementalSSEParser:
    """Decode and frame SSE incrementally without buffering a response body."""

    def __init__(
        self,
        *,
        maximum_line_bytes: int = DEFAULT_MAXIMUM_LINE_BYTES,
        maximum_frame_bytes: int = DEFAULT_MAXIMUM_FRAME_BYTES,
    ) -> None:
        if (
            type(maximum_line_bytes) is not int
            or maximum_line_bytes <= 0
            or type(maximum_frame_bytes) is not int
            or maximum_frame_bytes < maximum_line_bytes
        ):
            raise ValueError("SSE parser bounds are invalid")
        self.maximum_line_bytes = maximum_line_bytes
        self.maximum_frame_bytes = maximum_frame_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._line_buffer = ""
        self._frame_bytes = 0
        self._event: str | None = None
        self._id: str | None = None
        self._data: list[str] = []
        self._comments: list[str] = []
        self._frame_started = False
        self._finished = False

    def _line_size(self, line: str) -> int:
        return len(line.encode("utf-8"))

    def _reset_frame(self) -> None:
        self._frame_bytes = 0
        self._event = None
        self._id = None
        self._data.clear()
        self._comments.clear()
        self._frame_started = False

    def _dispatch(self) -> SSEFrame | None:
        if not self._frame_started:
            return None
        frame = SSEFrame(
            event=self._event,
            id=self._id,
            data="\n".join(self._data) if self._data else None,
            comments=tuple(self._comments),
        )
        self._reset_frame()
        return frame

    def _accept_line(self, raw_line: str) -> SSEFrame | None:
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        line_bytes = self._line_size(line)
        if line_bytes > self.maximum_line_bytes:
            raise SSEProtocolError("SSE line exceeds the configured bound")
        if not line:
            return self._dispatch()
        self._frame_started = True
        self._frame_bytes += line_bytes + 1
        if self._frame_bytes > self.maximum_frame_bytes:
            raise SSEProtocolError("SSE frame exceeds the configured bound")
        if line.startswith(":"):
            comment = line[1:]
            if comment.startswith(" "):
                comment = comment[1:]
            self._comments.append(comment)
            return None
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if not separator:
            value = ""
        if field == "event":
            if "\x00" in value:
                raise SSEProtocolError("SSE event contains NUL")
            self._event = value
        elif field == "id":
            if "\x00" in value:
                raise SSEProtocolError("SSE id contains NUL")
            self._id = value
        elif field == "data":
            self._data.append(value)
        return None

    def _accept_text(self, text: str) -> list[SSEFrame]:
        self._line_buffer += text
        frames: list[SSEFrame] = []
        while True:
            line_end = self._line_buffer.find("\n")
            if line_end < 0:
                if self._line_size(self._line_buffer) > self.maximum_line_bytes:
                    raise SSEProtocolError("SSE line exceeds the configured bound")
                break
            line = self._line_buffer[:line_end]
            self._line_buffer = self._line_buffer[line_end + 1 :]
            frame = self._accept_line(line)
            if frame is not None:
                frames.append(frame)
        return frames

    def feed(self, chunk: bytes) -> list[SSEFrame]:
        if self._finished:
            raise SSEProtocolError("SSE parser is already finished")
        if not isinstance(chunk, bytes):
            raise TypeError("SSE parser accepts bytes")
        try:
            text = self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise SSEProtocolError("SSE stream contains malformed UTF-8") from exc
        return self._accept_text(text)

    def finish(self) -> list[SSEFrame]:
        if self._finished:
            raise SSEProtocolError("SSE parser is already finished")
        self._finished = True
        try:
            text = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            self.clear()
            raise SSEProtocolError("SSE stream ended inside a UTF-8 sequence") from exc
        frames = self._accept_text(text)
        if self._line_buffer or self._frame_started:
            self.clear()
            raise SSEProtocolError("SSE stream ended before a frame delimiter")
        self.clear()
        return frames

    def clear(self) -> None:
        """Release all partial text and frame references."""

        self._line_buffer = ""
        self._reset_frame()
        self._finished = True
