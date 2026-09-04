"""Bounded, deterministic data-plane primitives used at the request boundary."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Deque


class RequestState(StrEnum):
    RECEIVED="RECEIVED"; QUEUED="QUEUED"; ADMITTED="ADMITTED"; STREAMING="STREAMING"; CANCELLING="CANCELLING"; COMPLETED="COMPLETED"; CANCELLED="CANCELLED"; FAILED="FAILED"
class StreamState(StrEnum):
    OPEN="OPEN"; TERMINAL="TERMINAL"

_REQUEST_TRANSITIONS = {RequestState.RECEIVED:{RequestState.QUEUED,RequestState.FAILED},RequestState.QUEUED:{RequestState.ADMITTED,RequestState.CANCELLED,RequestState.FAILED},RequestState.ADMITTED:{RequestState.STREAMING,RequestState.CANCELLING,RequestState.FAILED},RequestState.STREAMING:{RequestState.CANCELLING,RequestState.COMPLETED,RequestState.FAILED},RequestState.CANCELLING:{RequestState.CANCELLED,RequestState.COMPLETED,RequestState.FAILED}}

@dataclass
class Request:
    identity: str
    state: RequestState = RequestState.RECEIVED
    terminal_events: int = 0
    admission_count: int = 0
    def transition(self, state: RequestState) -> None:
        if state not in _REQUEST_TRANSITIONS.get(self.state, set()): raise ValueError(f"ILLEGAL_REQUEST_TRANSITION:{self.state}->{state}")
        self.state = state
        if state in (RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED): self.terminal_events += 1

class AdmissionRegistry:
    def __init__(self): self._seen: dict[str, Request] = {}; self._lock=Lock()
    def admit_once(self, identity: str) -> Request:
        with self._lock:
            if identity in self._seen:
                request=self._seen[identity]
                if request.admission_count: return request
            request=self._seen.setdefault(identity, Request(identity)); request.admission_count += 1; return request

class FairScheduler:
    def __init__(self, capacity: int = 1, queue_capacity: int = 8):
        if capacity < 1 or queue_capacity < 0: raise ValueError("invalid scheduler bounds")
        self.capacity=capacity; self.queue_capacity=queue_capacity; self._active=0; self._queue: Deque[Request]=deque(); self._lock=Lock()
    def submit(self, request: Request) -> str:
        with self._lock:
            if request.state is RequestState.RECEIVED:
                request.transition(RequestState.QUEUED)
            if self._active < self.capacity: self._active += 1; request.transition(RequestState.ADMITTED); return "ADMITTED"
            if len(self._queue) >= self.queue_capacity: return "BUSY"
            if request.state is not RequestState.QUEUED:
                request.transition(RequestState.QUEUED)
            self._queue.append(request); return "QUEUED"
    def release(self) -> Request | None:
        with self._lock:
            if self._active: self._active -= 1
            if self._queue:
                request=self._queue.popleft(); self._active += 1; request.transition(RequestState.ADMITTED); return request
            return None

class CancellationToken:
    def __init__(self): self.cancelled=False
    def cancel(self): self.cancelled=True

class StreamMachine:
    def __init__(self): self.state=StreamState.OPEN; self.frames=0
    def frame(self, _value: str) -> None:
        if self.state is StreamState.TERMINAL: raise ValueError("FRAME_AFTER_TERMINAL")
        self.frames += 1
    def terminal(self, kind: str) -> None:
        if kind not in {"completed","cancelled","failed"}: raise ValueError("INVALID_TERMINAL")
        if self.state is StreamState.TERMINAL: raise ValueError("DUPLICATE_TERMINAL")
        self.state=StreamState.TERMINAL
