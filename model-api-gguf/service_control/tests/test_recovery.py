"""Consolidated isolated tests for the persistent recovery state machine."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from service_control.recovery import (
    RecoveryError,
    RecoveryPolicy,
    RecoveryStore,
)


PROFILE_IDENTITY = "sha256:" + ("b" * 64)
OBSERVATION = {
    "api_identity": {"pid": 41001, "start": "start-a"},
    "router_identity": {"pid": 42001, "start": "start-b"},
    "model_child": {"pid": 43001, "start": "start-c"},
}


def policy(
    *,
    enabled: bool = True,
    maximum_attempts: int = 3,
) -> RecoveryPolicy:
    return RecoveryPolicy(
        automatic_recovery_enabled=enabled,
        initial_delay_seconds=0.01,
        maximum_delay_seconds=1.0,
        delay_multiplier=2.0,
        maximum_attempts_in_window=maximum_attempts,
        attempt_window_seconds=60.0,
        stable_reset_seconds=1.0,
    )


class RecoveryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "recovery"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def store(
        self, *, enabled: bool = True, maximum_attempts: int = 3
    ) -> RecoveryStore:
        return RecoveryStore(
            self.root,
            PROFILE_IDENTITY,
            policy(
                enabled=enabled, maximum_attempts=maximum_attempts
            ),
        )

    def test_stopped_and_disabled_precede_recovery(self) -> None:
        store = self.store()
        self.assertIsNone(
            store.begin(
                reason_code="API_PROCESS_LOST",
                desired_state="STOPPED",
                desired_generation=2,
                observation=OBSERVATION,
                selected_action="CONTROLLER_OWNED_API_STACK_RESTART",
            )
        )
        self.assertEqual(
            store.public_status()["primary_reason_code"],
            "DESIRED_STATE_STOPPED",
        )

        disabled = RecoveryStore(
            Path(self.temporary.name) / "disabled",
            PROFILE_IDENTITY,
            policy(enabled=False),
        )
        self.assertIsNone(
            disabled.begin(
                reason_code="API_PROCESS_LOST",
                desired_state="RUNNING",
                desired_generation=1,
                observation=OBSERVATION,
                selected_action="CONTROLLER_OWNED_API_STACK_RESTART",
            )
        )
        self.assertEqual(
            disabled.public_status()["primary_reason_code"],
            "AUTOMATIC_RECOVERY_DISABLED",
        )

    def test_backoff_history_and_rapid_loop_fail_closed(self) -> None:
        store = self.store(maximum_attempts=3)
        delays: list[float] = []
        for ordinal in range(1, 4):
            attempt = store.begin(
                reason_code="API_PROCESS_LOST",
                desired_state="RUNNING",
                desired_generation=1,
                observation=OBSERVATION,
                selected_action="CONTROLLER_OWNED_API_STACK_RESTART",
            )
            self.assertIsNotNone(attempt)
            assert attempt is not None
            self.assertEqual(attempt.attempt_ordinal, ordinal)
            delays.append(attempt.delay_seconds)
            status = store.complete(
                attempt,
                desired_state="RUNNING",
                desired_generation=1,
                outcome="FAILED",
                observation=OBSERVATION,
                error_category="fixture_failure",
            )
        self.assertEqual(delays, [0.01, 0.02, 0.04])
        self.assertEqual(status["recovery_state"], "FAIL_CLOSED")
        self.assertTrue(store.paths.active_latch.is_file())
        self.assertIsNone(
            store.begin(
                reason_code="API_PROCESS_LOST",
                desired_state="RUNNING",
                desired_generation=1,
                observation=OBSERVATION,
                selected_action="CONTROLLER_OWNED_API_STACK_RESTART",
            )
        )
        history = [
            json.loads(line)
            for line in store.paths.history.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(
            len([item for item in history if item["event"] == "ATTEMPT"]),
            3,
        )
        transactions = list(store.paths.transaction_dir.glob("rc-*.json"))
        self.assertEqual(len(transactions), 3)

    def test_immediate_ownership_fail_closed_and_clean_stopped_reset(
        self,
    ) -> None:
        store = self.store()
        status = store.fail_closed_now(
            reason_code="OWNERSHIP_UNCERTAIN",
            desired_state="RUNNING",
            desired_generation=1,
            observation=OBSERVATION,
        )
        self.assertEqual(status["recovery_state"], "FAIL_CLOSED")
        with self.assertRaises(RecoveryError):
            store.reset_fail_closed(
                desired_state="RUNNING",
                desired_generation=1,
                owned_runtime_absent=True,
                listeners_absent=True,
            )
        with self.assertRaises(RecoveryError):
            store.reset_fail_closed(
                desired_state="STOPPED",
                desired_generation=2,
                owned_runtime_absent=False,
                listeners_absent=True,
            )
        reset = store.reset_fail_closed(
            desired_state="STOPPED",
            desired_generation=2,
            owned_runtime_absent=True,
            listeners_absent=True,
        )
        self.assertTrue(reset["reset"])
        self.assertFalse(store.paths.active_latch.exists())
        self.assertTrue(
            list(store.paths.fail_closed_dir.glob("reset-*.json"))
        )
        self.assertIn(
            '"event":"FAIL_CLOSED_RESET"',
            store.paths.history.read_text(encoding="utf-8"),
        )

    def test_partial_transaction_records_remain_fail_closed(self) -> None:
        store = self.store()
        attempt = store.begin(
            reason_code="PARTIAL_STARTUP",
            desired_state="RUNNING",
            desired_generation=1,
            observation=OBSERVATION,
            selected_action="CONTROLLER_RECONCILIATION",
        )
        assert attempt is not None
        store.transition(
            attempt,
            desired_state="RUNNING",
            desired_generation=1,
            recovery_state="RECONCILING",
            observation=OBSERVATION,
            controller_result={
                "operation": "reconcile",
                "ok": True,
                "unrelated_process_signaled": False,
            },
        )
        transaction = json.loads(
            store.paths.transaction(
                attempt.recovery_transaction_id
            ).read_text(encoding="utf-8")
        )
        self.assertIsNone(transaction["outcome"])
        self.assertEqual(
            transaction["readiness_transitions"][-1]["state"],
            "RECONCILING",
        )
        resumed = RecoveryStore(
            self.root, PROFILE_IDENTITY, policy()
        )
        self.assertEqual(resumed._attempts_in_window(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
