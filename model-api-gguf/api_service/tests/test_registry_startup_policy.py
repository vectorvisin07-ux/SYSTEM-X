from __future__ import annotations

import os

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from system_x_gguf_api.application import create_application
from system_x_gguf_api.settings import ServiceSettings
from system_x_gguf_api.warm_model import WarmModelCoordinator


def settings_for(policy: str, *, registry: bool = False, recovery: bool = False) -> ServiceSettings:
    backend = policy in {"router_control", "registry_control"}
    return ServiceSettings(
        authentication_enabled=False,
        public_port=58080,
        private_backend_port=58081,
        private_backend_enabled=backend,
        registry_enabled=registry,
        startup_model_policy=policy,
        automatic_recovery_enabled=recovery,
    )


class RegistryStartupPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_startup_policies_preserve_existing_modes_and_add_registry_control(self) -> None:
        self.assertEqual(settings_for("always_warm").startup_model_policy, "always_warm")
        self.assertFalse(settings_for("api_only").private_backend_enabled)
        self.assertFalse(settings_for("router_control").registry_enabled)
        self.assertTrue(settings_for("registry_control", registry=True).registry_enabled)
        with self.assertRaisesRegex(ValueError, "registry_enabled must be true"):
            settings_for("registry_control")
        with self.assertRaisesRegex(ValueError, "automatic_recovery_enabled must be false"):
            settings_for("registry_control", registry=True, recovery=True)

    async def test_registry_control_lifespan_starts_registry_without_warm_or_recovery(self) -> None:
        events: list[str] = []
        backend = MagicMock()
        backend.startup = AsyncMock(side_effect=lambda: events.append("backend_start"))
        backend.shutdown = AsyncMock(side_effect=lambda: events.append("backend_stop"))
        registry = MagicMock()
        registry.startup = AsyncMock(side_effect=lambda: events.append("registry_start"))
        registry.shutdown = AsyncMock(side_effect=lambda: events.append("registry_stop"))
        warm = MagicMock()
        warm.startup = AsyncMock(side_effect=lambda **_: events.append("warm_start"))
        warm.shutdown = AsyncMock(side_effect=lambda: events.append("warm_stop"))
        recovery = MagicMock()
        recovery.startup = AsyncMock(side_effect=lambda: events.append("recovery_start"))
        recovery.shutdown = AsyncMock(side_effect=lambda: events.append("recovery_stop"))
        settings = settings_for("registry_control", registry=True)
        backend.settings = settings
        with (
            patch("system_x_gguf_api.application.BackendCoordinator", return_value=backend),
            patch("system_x_gguf_api.application.ModelRegistry", return_value=registry),
            patch("system_x_gguf_api.application.WarmModelCoordinator", return_value=warm),
            patch("system_x_gguf_api.application.RuntimeRecoveryCoordinator", return_value=recovery),
        ):
            application = create_application(settings)
            with patch.dict("os.environ", {"SYSTEM_X_API_SERVICE_TRANSACTION_ID": "tx-registry-test"}):
                async with application.router.lifespan_context(application):
                    self.assertEqual(application.state.settings.startup_model_policy, "registry_control")
        self.assertEqual(events, ["backend_start", "registry_start", "registry_stop", "backend_stop"])
        warm.startup.assert_not_awaited()
        recovery.startup.assert_not_awaited()

    async def test_non_always_warm_observer_does_not_adopt_model(self) -> None:
        registry = MagicMock()
        registry.public_summary = AsyncMock()
        catalogue = MagicMock()
        backend = MagicMock()
        backend.ensure_warm_model = AsyncMock()
        warm = WarmModelCoordinator(
            settings_for("registry_control", registry=True),
            catalogue,
            registry,
            backend,
        )
        status = await warm.observe_once()
        self.assertEqual(status.service_readiness_state, "WAITING_FOR_MODEL")
        self.assertEqual(status.reason_code, "startup_policy_unloaded")
        registry.public_summary.assert_not_awaited()
        backend.ensure_warm_model.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
