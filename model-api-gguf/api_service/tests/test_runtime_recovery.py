"""Isolated production-code tests for bounded API lifespan recovery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from system_x_gguf_api.backend import (
    BackendError,
    RouterIdentity,
)
from system_x_gguf_api.controller_client import ControllerResult
from system_x_gguf_api.runtime_recovery import RuntimeRecoveryCoordinator
from system_x_gguf_api.warm_model import WarmStatus


PROFILE_IDENTITY = "sha256:" + ("a" * 64)


def controller_result(
    *,
    ok: bool = True,
    reason_code: str = "OK",
    exit_status: int = 0,
) -> ControllerResult:
    return ControllerResult(
        operation="status",
        ok=ok,
        reason_code=reason_code,
        message="fixture",
        data={
            "active": ok,
            "active_state_consistent": True,
            "transaction_id": "router-tx",
            "pid": 41001,
            "pgid": 41001,
            "sid": 41001,
            "launch_mode": "router",
        },
        stderr="",
        exit_status=exit_status,
    )


class FakeController:
    def __init__(self) -> None:
        self.result = controller_result()

    async def status(self) -> ControllerResult:
        return self.result


class FakeBackend:
    def __init__(self) -> None:
        self.controller = FakeController()
        self.identity = RouterIdentity(
            "router-tx", 41001, 41001, 41001, "start-41001"
        )
        self.recover_calls = 0
        self.recover_error: BaseException | None = None
        self.transient_recover_failures = 0

    def _status_matches_identity(self, data: dict[str, object]) -> bool:
        return bool(
            data.get("active") is True
            and data.get("active_state_consistent") is True
            and data.get("transaction_id") == self.identity.transaction_id
            and data.get("pid") == self.identity.pid
            and data.get("pgid") == self.identity.pgid
            and data.get("sid") == self.identity.sid
            and data.get("launch_mode") == "router"
        )

    async def recover_router(self) -> RouterIdentity:
        self.recover_calls += 1
        if self.transient_recover_failures > 0:
            self.transient_recover_failures -= 1
            raise BackendError(
                "branch controller plan failed: ENDPOINT_IN_USE"
            )
        if self.recover_error is not None:
            raise self.recover_error
        self.identity = RouterIdentity(
            f"router-tx-{self.recover_calls}",
            42000 + self.recover_calls,
            42000 + self.recover_calls,
            42000 + self.recover_calls,
            f"start-{self.recover_calls}",
        )
        return self.identity


class FakeWarmModel:
    def __init__(self) -> None:
        self.registry = SimpleNamespace(
            public_summary=self._public_summary
        )
        self.default_alias_ready = True
        self.status = WarmStatus(
            "READY", "default", None, None, "2026-01-02T03:04:05Z"
        )
        self.observe_result = self.status
        self.recover_calls = 0
        self.recover_error: BaseException | None = None
        self.transient_recover_failures = 0

    async def _public_summary(self) -> SimpleNamespace:
        return SimpleNamespace(
            registry_status="ready",
            default_alias_ready=self.default_alias_ready,
            default_alias_model_id=(
                "sx-ready" if self.default_alias_ready else None
            ),
        )

    async def observe_once(self) -> WarmStatus:
        return self.observe_result

    async def recover_current_target(self) -> WarmStatus:
        self.recover_calls += 1
        if self.transient_recover_failures > 0:
            self.transient_recover_failures -= 1
            raise BackendError("fixture transient warm adoption")
        if self.recover_error is not None:
            raise self.recover_error
        self.status = WarmStatus(
            "READY", "default", None, None, "2026-01-02T03:04:06Z"
        )
        self.observe_result = self.status
        return self.status


def fixture_settings(
    desired_path: Path,
    *,
    enabled: bool = True,
    attempts: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        automatic_recovery_enabled=enabled,
        service_control_profile_identity=PROFILE_IDENTITY,
        service_control_desired_state_path=str(desired_path),
        recovery_delay_initial_seconds=0.0,
        recovery_delay_maximum_seconds=0.01,
        recovery_delay_multiplier=2.0,
        recovery_maximum_attempts_in_window=attempts,
        recovery_attempt_window_seconds=60.0,
        recovery_stable_reset_seconds=1.0,
        private_backend_poll_interval_seconds=0.01,
        private_backend_model_timeout_seconds=1.0,
    )


def write_desired(path: Path, state: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "system-x.service-desired-state.v1",
                "profile_identity": PROFILE_IDENTITY,
                "desired_state": state,
                "generation": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class RuntimeRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.desired_path = self.root / "desired-state.json"
        write_desired(self.desired_path, "RUNNING")
        self.backend = FakeBackend()
        self.warm = FakeWarmModel()

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    def coordinator(
        self, *, enabled: bool = True, attempts: int = 3
    ) -> RuntimeRecoveryCoordinator:
        return RuntimeRecoveryCoordinator(
            fixture_settings(
                self.desired_path, enabled=enabled, attempts=attempts
            ),
            self.backend,
            self.warm,
            recovery_root=self.root / "recovery",
        )

    async def test_private_listener_loss_recovers_router_and_exact_warm_target(
        self,
    ) -> None:
        coordinator = self.coordinator()
        self.backend.controller.result = controller_result(
            ok=False,
            reason_code="PRIVATE_LISTENER_LOST",
            exit_status=3,
        )
        await coordinator.observe_once()
        self.assertEqual(self.backend.recover_calls, 1)
        self.assertEqual(self.warm.recover_calls, 1)
        self.assertEqual(
            coordinator.public_status["recovery_state"], "RECOVERED"
        )
        transactions = list(
            (self.root / "recovery/transactions").glob("ar-*.json")
        )
        self.assertEqual(len(transactions), 1)
        record = json.loads(transactions[0].read_text(encoding="utf-8"))
        self.assertEqual(
            record["selected_action"], "CONTROLLER_ROUTER_RESTART"
        )
        self.assertEqual(record["outcome"], "RECOVERED")

    async def test_model_health_loss_reloads_without_router_restart(
        self,
    ) -> None:
        coordinator = self.coordinator()
        self.warm.observe_result = WarmStatus(
            "DEGRADED",
            "default",
            None,
            "default_target_private_health_failed",
            "2026-01-02T03:04:05Z",
        )
        await coordinator.observe_once()
        self.assertEqual(self.backend.recover_calls, 0)
        self.assertEqual(self.warm.recover_calls, 1)
        self.assertEqual(
            coordinator.public_status["recovery_state"], "RECOVERED"
        )

    async def test_waiting_and_candidate_do_not_attempt_recovery(self) -> None:
        coordinator = self.coordinator()
        self.warm.default_alias_ready = False
        for state, reason in (
            ("WAITING_FOR_MODEL", "NO_READY_MODEL"),
            ("MODEL_CANDIDATE_LOADING", "MODEL_CANDIDATE_LOADING"),
        ):
            with self.subTest(state=state):
                self.warm.observe_result = WarmStatus(
                    state,
                    "default",
                    None,
                    reason,
                    "2026-01-02T03:04:05Z",
                )
                await coordinator.observe_once()
                self.assertEqual(self.backend.recover_calls, 0)
                self.assertEqual(self.warm.recover_calls, 0)
                self.assertEqual(
                    coordinator.public_status["recovery_state"], "IDLE"
                )
                self.assertEqual(
                    coordinator.public_status["current_attempt"], 0
                )
                self.assertEqual(
                    coordinator.public_status["attempts_in_window"], 0
                )
        self.assertFalse(
            (self.root / "recovery/transactions").exists()
        )

    async def test_degraded_without_expected_model_stays_idle(self) -> None:
        coordinator = self.coordinator()
        self.warm.default_alias_ready = False
        self.warm.observe_result = WarmStatus(
            "DEGRADED",
            "default",
            None,
            "registry_unavailable",
            "2026-01-02T03:04:05Z",
        )
        await coordinator.observe_once()
        self.assertEqual(self.warm.recover_calls, 0)
        self.assertEqual(
            coordinator.public_status["recovery_state"], "IDLE"
        )
        self.assertEqual(
            coordinator.public_status["primary_reason_code"],
            "NO_EXPECTED_MODEL",
        )

    async def test_transient_endpoint_reuse_stays_in_one_attempt(
        self,
    ) -> None:
        coordinator = self.coordinator()
        self.backend.controller.result = controller_result(
            ok=False,
            reason_code="PRIVATE_LISTENER_LOST",
            exit_status=3,
        )
        self.backend.transient_recover_failures = 2
        await coordinator.observe_once()
        self.assertEqual(self.backend.recover_calls, 3)
        self.assertEqual(self.warm.recover_calls, 1)
        self.assertEqual(
            coordinator.public_status["attempts_in_window"], 1
        )
        self.assertEqual(
            coordinator.public_status["recovery_state"], "RECOVERED"
        )

    async def test_transient_warm_adoption_stays_in_one_attempt(
        self,
    ) -> None:
        coordinator = self.coordinator()
        self.warm.observe_result = WarmStatus(
            "DEGRADED",
            "default",
            None,
            "health_lost",
            "2026-01-02T03:04:05Z",
        )
        self.warm.transient_recover_failures = 2
        await coordinator.observe_once()
        self.assertEqual(self.warm.recover_calls, 3)
        self.assertEqual(
            coordinator.public_status["attempts_in_window"], 1
        )
        self.assertEqual(
            coordinator.public_status["recovery_state"], "RECOVERED"
        )

    async def test_stopped_and_disabled_state_precede_every_recovery(
        self,
    ) -> None:
        write_desired(self.desired_path, "STOPPED")
        stopped = self.coordinator()
        self.backend.controller.result = controller_result(
            ok=False, reason_code="PRIVATE_LISTENER_LOST", exit_status=3
        )
        await stopped.observe_once()
        self.assertEqual(self.backend.recover_calls, 0)
        self.assertEqual(
            stopped.public_status["primary_reason_code"],
            "DESIRED_STATE_STOPPED",
        )

        disabled = self.coordinator(enabled=False)
        await disabled.startup()
        self.assertEqual(self.backend.recover_calls, 0)
        self.assertEqual(
            disabled.public_status["primary_reason_code"],
            "AUTOMATIC_RECOVERY_DISABLED",
        )

    async def test_rapid_failures_persist_fail_closed_latch(self) -> None:
        coordinator = self.coordinator(attempts=2)
        self.warm.observe_result = WarmStatus(
            "DEGRADED",
            "default",
            None,
            "health_lost",
            "2026-01-02T03:04:05Z",
        )
        self.warm.recover_error = BackendError(
            "private router ownership identity changed"
        )
        await coordinator.observe_once()
        await coordinator.observe_once()
        self.assertEqual(self.warm.recover_calls, 2)
        self.assertEqual(
            coordinator.public_status["recovery_state"], "FAIL_CLOSED"
        )
        latch = self.root / "recovery/fail-closed/api-runtime.json"
        self.assertTrue(latch.is_file())
        await coordinator.observe_once()
        self.assertEqual(self.warm.recover_calls, 2)

    async def test_shutdown_fence_prevents_late_recovery(self) -> None:
        coordinator = self.coordinator()
        await coordinator.shutdown()
        self.backend.controller.result = controller_result(
            ok=False, reason_code="PRIVATE_LISTENER_LOST", exit_status=3
        )
        await coordinator.observe_once()
        self.assertEqual(self.backend.recover_calls, 0)
        self.assertEqual(
            coordinator.public_status["recovery_state"], "STOPPED"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
