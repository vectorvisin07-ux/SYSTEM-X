"""Internal HTTP client for the private llama-server router control plane."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import ipaddress
import json
import math
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from .sse import IncrementalSSEParser, SSEFrame


MAX_BODY_BYTES = 1024 * 1024
MAX_MODEL_ID_LENGTH = 1024
MAX_STATUS_LENGTH = 128
KNOWN_MODEL_STATES = {
    "unloaded",
    "loading",
    "loaded",
    "sleeping",
    "downloading",
    "failed",
}
PRIVATE_CHAT_TEMPLATE_KEYS = frozenset({"enable_thinking"})


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


@dataclass(frozen=True)
class RouterObservation:
    status_code: int | None
    body: str
    json_value: Any | None
    error: str | None


@dataclass(frozen=True)
class RouterModel:
    model_id: str
    status: str
    upstream_status: str
    source: str
    physical_path: str | None
    connected_paths: tuple[str, ...]
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RouterModelList:
    observation: RouterObservation
    models: tuple[RouterModel, ...]
    valid: bool


class RouterStreamOpenError(RuntimeError):
    """A private streamed request failed before an accepted SSE response."""

    def __init__(self, observation: RouterObservation) -> None:
        super().__init__(observation.error or "private_stream_open_failed")
        self.observation = observation


class PrivateRouterStream:
    """One manually streamed private response owned by one public request."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._parser = IncrementalSSEParser()
        self._iteration_started = False
        self._closed = False
        self._response_close_count = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def response_close_count(self) -> int:
        return self._response_close_count

    async def frames(self) -> AsyncIterator[SSEFrame]:
        if self._iteration_started:
            raise RuntimeError("private response may be iterated only once")
        if self._closed:
            raise RuntimeError("private response is closed")
        self._iteration_started = True
        try:
            async for chunk in self._response.aiter_bytes():
                for frame in self._parser.feed(chunk):
                    yield frame
            for frame in self._parser.finish():
                yield frame
        finally:
            self._parser.clear()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._parser.clear()
        await self._response.aclose()
        self._response_close_count += 1


def _validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str):
        raise ValueError("model id must be a string")
    if (
        not model_id
        or len(model_id) > MAX_MODEL_ID_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in model_id)
    ):
        raise ValueError("model id is invalid")
    return model_id


def _validate_text(value: str, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) == 0 for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _validate_max_tokens(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 1_048_576:
        raise ValueError("max tokens is invalid")
    return value


def _validate_temperature(value: float | None) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 2.0:
        raise ValueError("temperature is invalid")
    return normalized


def _validate_stop(stop: list[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if not isinstance(stop, list) or not 1 <= len(stop) <= 16:
        raise ValueError("stop list is invalid")
    return [_validate_text(item, "stop value", 256) for item in stop]


def _validate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not 1 <= len(messages) <= 256:
        raise ValueError("messages are invalid")
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message shape is invalid")
        role = message.get("role")
        if role in {"system", "user"} and set(message) == {"role", "content"}:
            result.append(
                {
                    "role": str(role),
                    "content": _validate_text(
                        message.get("content"), "message content", 1_048_576
                    ),
                }
            )
            continue
        if role == "assistant" and set(message).issubset(
            {"role", "content", "tool_calls", "reasoning_content"}
        ):
            content = message.get("content")
            calls = message.get("tool_calls")
            reasoning = message.get("reasoning_content")
            if content is not None and (
                not isinstance(content, str) or len(content) > 1_048_576
            ):
                raise ValueError("assistant content is invalid")
            if reasoning is not None and (
                not isinstance(reasoning, str)
                or not reasoning.strip()
                or len(reasoning) > 1_048_576
            ):
                raise ValueError("assistant reasoning is invalid")
            if calls is None:
                if not content and not reasoning:
                    raise ValueError("assistant message is empty")
                normalized_message = {
                    "role": "assistant",
                    "content": content,
                }
                if reasoning is not None:
                    normalized_message["reasoning_content"] = reasoning
                result.append(normalized_message)
                continue
            if not isinstance(calls, list) or not 1 <= len(calls) <= 8:
                raise ValueError("assistant tool calls are invalid")
            normalized_calls = []
            for call in calls:
                if (
                    not isinstance(call, dict)
                    or set(call) != {"id", "type", "function"}
                    or call.get("type") != "function"
                    or not isinstance(call.get("id"), str)
                    or not isinstance(call.get("function"), dict)
                    or set(call["function"]) != {"name", "arguments"}
                    or not isinstance(call["function"]["name"], str)
                    or not isinstance(call["function"]["arguments"], str)
                ):
                    raise ValueError("assistant tool-call shape is invalid")
                normalized_calls.append(call)
            normalized_message = {
                "role": "assistant",
                "content": content,
                "tool_calls": normalized_calls,
            }
            if reasoning is not None:
                normalized_message["reasoning_content"] = reasoning
            result.append(normalized_message)
            continue
        if role == "tool" and set(message).issubset(
            {"role", "content", "tool_call_id", "name"}
        ):
            if (
                not isinstance(message.get("content"), str)
                or not isinstance(message.get("tool_call_id"), str)
                or (
                    "name" in message
                    and not isinstance(message.get("name"), str)
                )
            ):
                raise ValueError("tool-result message is invalid")
            result.append(dict(message))
            continue
        raise ValueError("message role is invalid")
    return result


def _validate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list) or not 1 <= len(tools) <= 20:
        raise ValueError("private tools payload is invalid")
    encoded = json.dumps(tools, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 262_144:
        raise ValueError("private tools payload exceeds the bound")
    return tools


def _validate_tool_choice(value: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if value is None or (
        isinstance(value, str) and value in {"none", "auto", "required"}
    ):
        return value
    if (
        isinstance(value, dict)
        and value.get("type") == "function"
        and isinstance(value.get("function"), dict)
        and set(value["function"]) == {"name"}
        and isinstance(value["function"]["name"], str)
    ):
        return value
    raise ValueError("private tool choice is invalid")


def _validate_response_format(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("type") != "json_schema"
        or not isinstance(value.get("json_schema"), dict)
        or not isinstance(value["json_schema"].get("schema"), dict)
        or value["json_schema"].get("strict") is not True
    ):
        raise ValueError("private response format is invalid")
    return value


def _validate_chat_template_kwargs(
    value: dict[str, Any] | None,
) -> dict[str, bool] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != PRIVATE_CHAT_TEMPLATE_KEYS
        or type(value.get("enable_thinking")) is not bool
    ):
        raise ValueError("private chat template arguments are invalid")
    return {"enable_thinking": value["enable_thinking"]}


def _completion_body(
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    stop: list[str] | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _validate_model_id(model_id),
        "prompt": _validate_text(prompt, "prompt", 1_048_576),
        "max_tokens": _validate_max_tokens(max_tokens),
        "stream": stream,
        "verbose": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    normalized_temperature = _validate_temperature(temperature)
    if normalized_temperature is not None:
        body["temperature"] = normalized_temperature
    normalized_stop = _validate_stop(stop)
    if normalized_stop is not None:
        body["stop"] = normalized_stop
    return body


def _chat_completion_body(
    model_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    response_format: dict[str, Any] | None,
    chat_template_kwargs: dict[str, Any] | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _validate_model_id(model_id),
        "messages": _validate_messages(messages),
        "max_tokens": _validate_max_tokens(max_tokens),
        "stream": stream,
        "verbose": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    normalized_tools = _validate_tools(tools)
    normalized_choice = _validate_tool_choice(tool_choice)
    normalized_format = _validate_response_format(response_format)
    normalized_template_kwargs = _validate_chat_template_kwargs(
        chat_template_kwargs
    )
    if normalized_tools is not None and normalized_format is not None:
        raise ValueError("tools and response format cannot be combined")
    if normalized_tools is not None:
        body["tools"] = normalized_tools
        body["tool_choice"] = normalized_choice or "auto"
        body["parallel_tool_calls"] = False
        body["parse_tool_calls"] = True
    elif normalized_choice not in {None, "none"}:
        raise ValueError("tool choice requires private tools")
    if normalized_format is not None:
        body["response_format"] = normalized_format
    if normalized_template_kwargs is not None:
        body["chat_template_kwargs"] = normalized_template_kwargs
    normalized_temperature = _validate_temperature(temperature)
    if normalized_temperature is not None:
        body["temperature"] = normalized_temperature
    normalized_stop = _validate_stop(stop)
    if normalized_stop is not None:
        body["stop"] = normalized_stop
    return body


def _responses_body(
    model_id: str,
    input_value: str | list[dict[str, Any]],
    max_output_tokens: int,
    instructions: str | None,
    temperature: float | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    response_format: dict[str, Any] | None,
    chat_template_kwargs: dict[str, Any] | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    if isinstance(input_value, str):
        normalized_input: str | list[dict[str, Any]] = _validate_text(
            input_value, "response input", 1_048_576
        )
    elif isinstance(input_value, list):
        encoded = json.dumps(
            input_value, separators=(",", ":"), allow_nan=False
        ).encode()
        if not input_value or len(input_value) > 512 or len(encoded) > MAX_BODY_BYTES:
            raise ValueError("responses input items are invalid")
        normalized_input = input_value
    else:
        raise ValueError("responses input is invalid")
    body: dict[str, Any] = {
        "model": _validate_model_id(model_id),
        "input": normalized_input,
        "max_output_tokens": _validate_max_tokens(max_output_tokens),
        "stream": stream,
    }
    normalized_tools = _validate_tools(tools)
    normalized_choice = _validate_tool_choice(tool_choice)
    normalized_format = _validate_response_format(response_format)
    normalized_template_kwargs = _validate_chat_template_kwargs(
        chat_template_kwargs
    )
    if normalized_tools is not None and normalized_format is not None:
        raise ValueError("tools and response format cannot be combined")
    if normalized_tools is not None:
        body["tools"] = normalized_tools
        body["tool_choice"] = normalized_choice or "auto"
        body["parallel_tool_calls"] = False
    elif normalized_choice not in {None, "none"}:
        raise ValueError("tool choice requires private tools")
    if normalized_format is not None:
        body["response_format"] = normalized_format
    if normalized_template_kwargs is not None:
        body["chat_template_kwargs"] = normalized_template_kwargs
    if instructions is not None:
        body["instructions"] = _validate_text(
            instructions, "response instructions", 65_536
        )
    normalized_temperature = _validate_temperature(temperature)
    if normalized_temperature is not None:
        body["temperature"] = normalized_temperature
    return body


class RouterClient:
    """Expose only the authorized private router control operations."""

    def __init__(
        self,
        host: str,
        port: int,
        operation_timeout_seconds: float,
        *,
        inference_timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        address = ipaddress.ip_address(host)
        if not isinstance(address, ipaddress.IPv4Address) or not address.is_loopback:
            raise ValueError("router host must be an IPv4 loopback address")
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError("router port is invalid")
        if operation_timeout_seconds <= 0 or operation_timeout_seconds > 300:
            raise ValueError("router timeout is out of bounds")
        if inference_timeout_seconds is None:
            inference_timeout_seconds = operation_timeout_seconds
        if (
            inference_timeout_seconds <= 0
            or inference_timeout_seconds > 3600
        ):
            raise ValueError("router inference timeout is out of bounds")
        package_file = Path(__file__).resolve(strict=True)
        self._models_root = (
            package_file.parents[3] / "MODEL" / "SUPERMODEL"
        ).resolve(strict=False)
        self._control_timeout = httpx.Timeout(
            connect=min(10.0, operation_timeout_seconds),
            read=operation_timeout_seconds,
            write=operation_timeout_seconds,
            pool=min(10.0, operation_timeout_seconds),
        )
        self._inference_timeout = httpx.Timeout(
            connect=min(10.0, inference_timeout_seconds),
            read=inference_timeout_seconds,
            write=inference_timeout_seconds,
            pool=min(10.0, inference_timeout_seconds),
        )
        self._client = httpx.AsyncClient(
            base_url=f"http://{address.compressed}:{port}",
            timeout=self._control_timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            follow_redirects=False,
            transport=transport,
        )

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        *,
        inference: bool = False,
    ) -> RouterObservation:
        try:
            response = await self._client.request(
                method,
                path,
                json=json_body,
                params=params,
                timeout=(
                    self._inference_timeout
                    if inference
                    else self._control_timeout
                ),
            )
        except httpx.TimeoutException:
            return RouterObservation(None, "", None, "timeout")
        except httpx.RequestError:
            return RouterObservation(None, "", None, "connection_failure")
        content = response.content
        if len(content) > MAX_BODY_BYTES:
            return RouterObservation(
                response.status_code, "", None, "response_body_exceeded_bound"
            )
        try:
            body = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return RouterObservation(
                response.status_code, "", None, "malformed_utf8"
            )
        if 300 <= response.status_code < 400:
            return RouterObservation(
                response.status_code, body, None, "redirect_rejected"
            )
        try:
            parsed = json.loads(
                body,
                object_pairs_hook=_unique_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {value}")
                ),
            )
            error = None
        except (json.JSONDecodeError, ValueError):
            parsed = None
            error = "malformed_json"
        return RouterObservation(response.status_code, body, parsed, error)

    async def _rejected_stream_observation(
        self,
        response: httpx.Response,
        *,
        error_override: str | None = None,
    ) -> RouterObservation:
        content = bytearray()
        exceeded = False
        try:
            async for chunk in response.aiter_bytes():
                if len(content) + len(chunk) > MAX_BODY_BYTES:
                    exceeded = True
                    break
                content.extend(chunk)
        finally:
            await response.aclose()
        if exceeded:
            return RouterObservation(
                response.status_code,
                "",
                None,
                "response_body_exceeded_bound",
            )
        try:
            body = bytes(content).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return RouterObservation(
                response.status_code,
                "",
                None,
                error_override or "malformed_utf8",
            )
        try:
            parsed = json.loads(
                body,
                object_pairs_hook=_unique_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {value}")
                ),
            )
            error = error_override
        except (json.JSONDecodeError, ValueError):
            parsed = None
            error = error_override or "malformed_json"
        return RouterObservation(response.status_code, body, parsed, error)

    async def _open_stream(
        self,
        path: str,
        body: dict[str, Any],
    ) -> PrivateRouterStream:
        try:
            request = self._client.build_request(
                "POST",
                path,
                json=body,
                timeout=self._inference_timeout,
            )
            response = await self._client.send(
                request,
                stream=True,
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise RouterStreamOpenError(
                RouterObservation(None, "", None, "timeout")
            ) from exc
        except httpx.RequestError as exc:
            raise RouterStreamOpenError(
                RouterObservation(None, "", None, "connection_failure")
            ) from exc
        if response.status_code != 200:
            error_override = (
                "redirect_rejected"
                if 300 <= response.status_code < 400
                else None
            )
            raise RouterStreamOpenError(
                await self._rejected_stream_observation(
                    response,
                    error_override=error_override,
                )
            )
        content_type = response.headers.get("content-type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != "text/event-stream":
            raise RouterStreamOpenError(
                await self._rejected_stream_observation(
                    response,
                    error_override="stream_content_type_rejected",
                )
            )
        return PrivateRouterStream(response)

    @asynccontextmanager
    async def _stream_request(
        self,
        path: str,
        body: dict[str, Any],
    ) -> AsyncIterator[PrivateRouterStream]:
        stream = await self._open_stream(path, body)
        try:
            yield stream
        finally:
            await stream.aclose()

    async def health(self) -> RouterObservation:
        return await self._request("GET", "/health")

    def _contained_path(self, supplied: str, *, strict: bool = True) -> str:
        candidate = Path(supplied)
        if not candidate.is_absolute():
            candidate = self._models_root / candidate
        if candidate.is_symlink():
            raise ValueError("model path is a symlink")
        canonical = candidate.resolve(strict=strict)
        canonical.relative_to(self._models_root)
        return str(canonical)

    def _entry_paths(
        self,
        entry: dict[str, Any],
        status_object: dict[str, Any],
        *,
        allow_missing: bool = False,
    ) -> tuple[str | None, tuple[str, ...]]:
        supplied: list[str] = []
        top_level = entry.get("path")
        if top_level is not None:
            if not isinstance(top_level, str) or not top_level:
                raise ValueError("model path is invalid")
            supplied.append(top_level)
        arguments = status_object.get("args", [])
        if not isinstance(arguments, list) or not all(
            isinstance(item, str) for item in arguments
        ):
            raise ValueError("model status args are invalid")
        path_options = {
            "--model",
            "-m",
            "--mmproj",
            "-mm",
            "--model-draft",
            "--model-vocoder",
        }
        for index, argument in enumerate(arguments[:-1]):
            if argument in path_options:
                supplied.append(arguments[index + 1])
        preset = status_object.get("preset")
        if preset is not None:
            if not isinstance(preset, str) or len(preset) > MAX_BODY_BYTES:
                raise ValueError("model preset is invalid")
            preset_path_keys = {
                "model",
                "m",
                "mmproj",
                "mm",
                "model-draft",
                "model-vocoder",
            }
            for line in preset.splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip().lower() in preset_path_keys:
                    normalized_value = value.strip().strip("\"'")
                    if normalized_value:
                        supplied.append(normalized_value)
        connected = []
        for value in supplied:
            try:
                canonical = self._contained_path(value)
            except FileNotFoundError:
                if not allow_missing:
                    raise
                canonical = self._contained_path(value, strict=False)
            if canonical not in connected:
                connected.append(canonical)
        return (
            connected[0] if connected else None,
            tuple(connected[1:]),
        )

    @staticmethod
    def _modalities(
        architecture: Any,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if architecture is None:
            return (), ()
        if not isinstance(architecture, dict):
            raise ValueError("model architecture is invalid")
        normalized = []
        for key in ("input_modalities", "output_modalities"):
            values = architecture.get(key, [])
            if not isinstance(values, list) or not all(
                isinstance(value, str)
                and value
                and len(value) <= 64
                and not any(ord(character) < 32 for character in value)
                for value in values
            ):
                raise ValueError(f"model architecture {key} is invalid")
            normalized.append(tuple(sorted(set(values))))
        return normalized[0], normalized[1]

    async def list_models(self, reload: bool = False) -> RouterModelList:
        observation = await self._request(
            "GET", "/models", params={"reload": "1"} if reload else None
        )
        if (
            observation.status_code != 200
            or observation.error is not None
            or not isinstance(observation.json_value, dict)
            or not isinstance(observation.json_value.get("data"), list)
        ):
            return RouterModelList(observation, (), False)
        accepted: list[RouterModel] = []
        try:
            for entry in observation.json_value["data"]:
                if not isinstance(entry, dict):
                    raise ValueError("model entry is not an object")
                model_id = _validate_model_id(entry.get("id"))
                status_object = entry.get("status")
                if not isinstance(status_object, dict):
                    raise ValueError("model status is not an object")
                upstream_status = status_object.get("value")
                if (
                    not isinstance(upstream_status, str)
                    or not upstream_status
                    or len(upstream_status) > MAX_STATUS_LENGTH
                ):
                    raise ValueError("model status value is invalid")
                normalized_status = (
                    upstream_status
                    if upstream_status in KNOWN_MODEL_STATES
                    else "unknown"
                )
                source = entry.get("source")
                if (
                    not isinstance(source, str)
                    or not source
                    or len(source) > MAX_STATUS_LENGTH
                ):
                    raise ValueError("model source is invalid")
                physical_path, connected_paths = self._entry_paths(
                    entry,
                    status_object,
                    allow_missing=upstream_status == "unloaded",
                )
                input_modalities, output_modalities = self._modalities(
                    entry.get("architecture")
                )
                accepted.append(
                    RouterModel(
                        model_id=model_id,
                        status=normalized_status,
                        upstream_status=upstream_status,
                        source=source,
                        physical_path=physical_path,
                        connected_paths=connected_paths,
                        input_modalities=input_modalities,
                        output_modalities=output_modalities,
                        raw=dict(entry),
                    )
                )
        except (OSError, ValueError):
            return RouterModelList(observation, (), False)
        return RouterModelList(observation, tuple(accepted), True)

    async def load_model(self, model_id: str) -> RouterObservation:
        return await self._request(
            "POST", "/models/load", {"model": _validate_model_id(model_id)}
        )

    async def unload_model(self, model_id: str) -> RouterObservation:
        return await self._request(
            "POST", "/models/unload", {"model": _validate_model_id(model_id)}
        )

    async def get_props(
        self, model_id: str, autoload: bool = False
    ) -> RouterObservation:
        return await self._request(
            "GET",
            "/props",
            params={
                "model": _validate_model_id(model_id),
                "autoload": "true" if autoload else "false",
            },
        )

    async def completion(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> RouterObservation:
        body = _completion_body(
            model_id,
            prompt,
            max_tokens,
            temperature,
            stop,
            stream=False,
        )
        return await self._request(
            "POST", "/v1/completions", body, inference=True
        )

    @asynccontextmanager
    async def completion_stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[PrivateRouterStream]:
        body = _completion_body(
            model_id,
            prompt,
            max_tokens,
            temperature,
            stop,
            stream=True,
        )
        async with self._stream_request("/v1/completions", body) as stream:
            yield stream

    async def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> RouterObservation:
        body = _chat_completion_body(
            model_id,
            messages,
            max_tokens,
            temperature,
            stop,
            tools,
            tool_choice,
            response_format,
            chat_template_kwargs,
            stream=False,
        )
        return await self._request(
            "POST", "/v1/chat/completions", body, inference=True
        )

    @asynccontextmanager
    async def chat_completion_stream(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[PrivateRouterStream]:
        body = _chat_completion_body(
            model_id,
            messages,
            max_tokens,
            temperature,
            stop,
            tools,
            tool_choice,
            response_format,
            chat_template_kwargs,
            stream=True,
        )
        async with self._stream_request("/v1/chat/completions", body) as stream:
            yield stream

    async def responses(
        self,
        model_id: str,
        input_value: str | list[dict[str, Any]],
        max_output_tokens: int,
        instructions: str | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> RouterObservation:
        body = _responses_body(
            model_id,
            input_value,
            max_output_tokens,
            instructions,
            temperature,
            tools,
            tool_choice,
            response_format,
            chat_template_kwargs,
            stream=False,
        )
        return await self._request(
            "POST", "/v1/responses", body, inference=True
        )

    @asynccontextmanager
    async def responses_stream(
        self,
        model_id: str,
        input_value: str | list[dict[str, Any]],
        max_output_tokens: int,
        instructions: str | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[PrivateRouterStream]:
        body = _responses_body(
            model_id,
            input_value,
            max_output_tokens,
            instructions,
            temperature,
            tools,
            tool_choice,
            response_format,
            chat_template_kwargs,
            stream=True,
        )
        async with self._stream_request("/v1/responses", body) as stream:
            yield stream

    async def tokenize(
        self, model_id: str, content: str
    ) -> RouterObservation:
        return await self._request(
            "POST",
            "/tokenize",
            {
                "model": _validate_model_id(model_id),
                "content": _validate_text(content, "tokenize content", 1_048_576),
                "add_special": False,
                "parse_special": True,
                "with_pieces": False,
            },
            inference=True,
        )

    async def chat_input_tokens(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> RouterObservation:
        body: dict[str, Any] = {
            "model": _validate_model_id(model_id),
            "messages": _validate_messages(messages),
        }
        normalized_tools = _validate_tools(tools)
        normalized_template_kwargs = _validate_chat_template_kwargs(
            chat_template_kwargs
        )
        if normalized_tools is not None:
            body["tools"] = normalized_tools
            body["parallel_tool_calls"] = False
        if normalized_template_kwargs is not None:
            body["chat_template_kwargs"] = normalized_template_kwargs
        return await self._request(
            "POST",
            "/v1/chat/completions/input_tokens",
            body,
            inference=True,
        )

    async def responses_input_tokens(
        self,
        model_id: str,
        input_value: str | list[dict[str, Any]],
        instructions: str | None = None,
    ) -> RouterObservation:
        if isinstance(input_value, str):
            normalized_input: str | list[dict[str, Any]] = _validate_text(
                input_value, "responses token input", 1_048_576
            )
        elif isinstance(input_value, list):
            encoded = json.dumps(
                input_value, separators=(",", ":"), allow_nan=False
            ).encode()
            if (
                not input_value
                or len(input_value) > 512
                or len(encoded) > MAX_BODY_BYTES
            ):
                raise ValueError("responses token input items are invalid")
            normalized_input = input_value
        else:
            raise ValueError("responses token input is invalid")
        body: dict[str, Any] = {
            "model": _validate_model_id(model_id),
            "input": normalized_input,
        }
        if instructions is not None:
            body["instructions"] = _validate_text(
                instructions, "response instructions", 65_536
            )
        return await self._request(
            "POST", "/v1/responses/input_tokens", body, inference=True
        )

    async def aclose(self) -> None:
        await self._client.aclose()
