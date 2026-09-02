"""Thin first-party interactive client for the retained System X API."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any, Callable

SYSTEM_MESSAGE = "Give the final answer in normal assistant content. Do not leave the answer only in reasoning content."

def _root() -> Path:
    return Path(__file__).resolve().parents[2]

def _receipt() -> dict[str, Any]:
    path = _root() / "INSPECTOR/RUNTIME/status/api-connection.json"
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode & 63):
        raise RuntimeError("System X connection is not ready for chat")
    data = json.loads(path.read_text(encoding="utf-8"))
    service = data.get("service", {})
    connection = data.get("connections", {}).get("openai_compatible", {})
    if data.get("schema_version") != "system-x.inspector-api-connection-receipt.v1" or service.get("service_readiness") != "READY" or not service.get("service_available") or not isinstance(connection.get("base_url"), str):
        raise RuntimeError("System X chat endpoint is unavailable")
    return data

def _key() -> str:
    path = _root() / "model-api-gguf/RUNTIME/api/auth/handoff/local-primary.key"
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode & 63):
        raise RuntimeError("System X chat credentials are unavailable")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("System X chat credentials are unavailable")
    return value

def _request(base: str, key: str, messages: list[dict[str, str]]) -> str:
    body = json.dumps({"model": "default", "messages": messages, "stream": False, "max_tokens": 1024}, separators=(",", ":")).encode()
    request = urllib.request.Request(base + "/chat/completions", data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError("System X chat request failed")
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("System X chat request failed") from exc
    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content")
    finish = choices[0].get("finish_reason") if choices else None
    if not isinstance(content, str) or not content.strip() or finish in {"length", "output_limit"}:
        raise RuntimeError("System X returned no completed final answer")
    return content

def _status_text(receipt: dict[str, Any]) -> str:
    return "State: READY\nModel: default\nHealth: READY"

def _info_text(receipt: dict[str, Any]) -> str:
    model = receipt.get("model", {})
    caps = receipt.get("capabilities", {})
    return "\n".join(("Model: " + str(model.get("default_alias", "default")), "Resolved model: " + str(model.get("resolved_immutable_model_id", "available")), "Architecture: " + str(model.get("physical_architecture", "available")), "Context window: " + str(model.get("context_window_tokens", caps.get("context_window_tokens", "available"))), "Protocol: openai_compatible", "Streaming: " + str(caps.get("streaming", "not_tested"))))

def run_chat(input_stream: Any = None, output_stream: Any = None, *, request: Callable[[str, str, list[dict[str, str]]], str] = _request) -> int:
    inp = input_stream or sys.stdin
    out = output_stream or sys.stdout
    try:
        receipt = _receipt()
        key = _key()
    except (OSError, RuntimeError, json.JSONDecodeError):
        print("System X is not currently ready for chat.", file=out)
        return 2
    print("SYSTEM X LOCAL CHAT\n" + _status_text(receipt) + "\nType /help for commands.", file=out)
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    while True:
        try:
            print("you>", end=" ", flush=True, file=out)
            line = inp.readline()
        except KeyboardInterrupt:
            print("\nChat ended.", file=out)
            return 0
        if line == "":
            return 0
        text = line.rstrip("\r\n")
        if not text:
            continue
        if text == "/exit":
            return 0
        if text == "/help":
            print("/help  /new  /clear  /status  /info  /exit", file=out)
            continue
        if text == "/new":
            history = [{"role": "system", "content": SYSTEM_MESSAGE}]
            print("Conversation cleared.", file=out)
            continue
        if text == "/clear":
            if getattr(out, "isatty", lambda: False)():
                print(chr(27) + "[2J" + chr(27) + "[H", end="", file=out)
            else:
                print("---", file=out)
            continue
        if text == "/status":
            print(_status_text(receipt), file=out)
            continue
        if text == "/info":
            print(_info_text(receipt), file=out)
            continue
        candidate = history + [{"role": "user", "content": text}]
        try:
            answer = request(receipt["connections"]["openai_compatible"]["base_url"], key, candidate)
        except (RuntimeError, OSError):
            print("System X chat request failed.", file=out)
            continue
        history.extend(({"role": "user", "content": text}, {"role": "assistant", "content": answer}))
        print("assistant> " + answer, file=out)
