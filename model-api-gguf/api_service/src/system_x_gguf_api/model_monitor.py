"""Filesystem watcher and authoritative periodic reconciliation triggers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from watchfiles import awatch


ReconcileCallback = Callable[[str, bool], Awaitable[None]]
DegradedCallback = Callable[[str], Awaitable[None]]
RecoveredCallback = Callable[[str], Awaitable[None]]


class RegistryMonitor:
    """Own independent watcher and periodic trigger tasks."""

    def __init__(
        self,
        model_root: Path,
        debounce_milliseconds: int,
        reconcile_interval_seconds: float,
        request_reconcile: ReconcileCallback,
        mark_degraded: DegradedCallback,
        mark_recovered: RecoveredCallback | None = None,
        *,
        watcher_factory: Callable[..., Any] = awatch,
    ) -> None:
        self.model_root = model_root
        self.debounce_milliseconds = debounce_milliseconds
        self.reconcile_interval_seconds = reconcile_interval_seconds
        self.request_reconcile = request_reconcile
        self.mark_degraded = mark_degraded
        self.mark_recovered = mark_recovered
        self.watcher_factory = watcher_factory
        self.stop_event = asyncio.Event()
        self.watcher_task: asyncio.Task[None] | None = None
        self.periodic_task: asyncio.Task[None] | None = None
        self.watcher_event_count = 0
        self.periodic_tick_count = 0
        self.watcher_failure_count = 0

    async def start_watcher(self) -> None:
        if self.watcher_task is not None:
            raise RuntimeError("registry watcher is already started")
        self.watcher_task = asyncio.create_task(
            self._watcher_loop(), name="system-x-registry-watcher"
        )
        await asyncio.sleep(0)

    async def start_periodic(self) -> None:
        if self.periodic_task is not None:
            raise RuntimeError("periodic reconciler is already started")
        self.periodic_task = asyncio.create_task(
            self._periodic_loop(), name="system-x-registry-periodic"
        )
        await asyncio.sleep(0)

    async def _watcher_loop(self) -> None:
        backoff = 0.25
        while not self.stop_event.is_set():
            try:
                async for changes in self.watcher_factory(
                    self.model_root,
                    debounce=self.debounce_milliseconds,
                    stop_event=self.stop_event,
                    recursive=True,
                    ignore_permission_denied=False,
                    raise_interrupt=False,
                ):
                    if self.stop_event.is_set():
                        return
                    if changes:
                        self.watcher_event_count += 1
                        if self.mark_recovered is not None:
                            await self.mark_recovered("watcher_failure")
                        await self.request_reconcile("watcher", False)
                if self.stop_event.is_set():
                    return
                raise RuntimeError("filesystem watcher ended unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.watcher_failure_count += 1
                await self.mark_degraded(
                    f"watcher_failure:{type(exc).__name__}"
                )
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=backoff)
                except TimeoutError:
                    backoff = min(backoff * 2.0, 5.0)

    async def _periodic_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=self.reconcile_interval_seconds,
                )
            except TimeoutError:
                self.periodic_tick_count += 1
                await self.request_reconcile("periodic", False)

    async def shutdown(self) -> dict[str, Any]:
        self.stop_event.set()
        tasks = [
            task
            for task in (self.watcher_task, self.periodic_task)
            if task is not None
        ]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=5.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if not task.cancelled():
                    exception = task.exception()
                    if exception is not None:
                        raise exception
        result = {
            "watcher_task_done": (
                self.watcher_task is None or self.watcher_task.done()
            ),
            "periodic_task_done": (
                self.periodic_task is None or self.periodic_task.done()
            ),
            "watcher_event_count": self.watcher_event_count,
            "periodic_tick_count": self.periodic_tick_count,
            "watcher_failure_count": self.watcher_failure_count,
        }
        self.watcher_task = None
        self.periodic_task = None
        return result
