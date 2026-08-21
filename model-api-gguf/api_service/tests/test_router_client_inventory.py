"""Regression coverage for router inventory handling of retired model paths."""

from __future__ import annotations

from pathlib import Path
import unittest

from system_x_gguf_api.router_client import RouterClient, RouterObservation


class RouterClientInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_unloaded_missing_path_keeps_inventory_valid(self) -> None:
        client = RouterClient("127.0.0.1", 54037, 1.0)
        try:
            missing_path = client._models_root / (
                "router-client-stale-missing-regression.gguf"
            )
            payload = {
                "data": [
                    {
                        "id": "candidate-stale",
                        "status": {
                            "value": "unloaded",
                            "args": ["--model", str(missing_path)],
                        },
                        "source": "models_dir",
                        "architecture": {
                            "input_modalities": ["text"],
                            "output_modalities": ["text"],
                        },
                    },
                    {
                        "id": "candidate-safe",
                        "status": {"value": "loaded", "args": []},
                        "source": "models_dir",
                        "architecture": {
                            "input_modalities": ["text"],
                            "output_modalities": ["text"],
                        },
                    },
                ]
            }

            async def fake_request(
                method: str,
                path: str,
                *,
                params: dict[str, str] | None = None,
                body: dict[str, object] | None = None,
            ) -> RouterObservation:
                return RouterObservation(200, "", payload, None)

            client._request = fake_request  # type: ignore[method-assign]
            result = await client.list_models()

            self.assertTrue(result.valid)
            self.assertEqual(len(result.models), 2)
            self.assertEqual(
                result.models[0].physical_path, str(missing_path)
            )
            self.assertEqual(result.models[0].upstream_status, "unloaded")
        finally:
            await client._client.aclose()

    async def test_loaded_missing_path_still_invalidates_inventory(self) -> None:
        client = RouterClient("127.0.0.1", 54037, 1.0)
        try:
            missing_path = client._models_root / (
                "router-client-loaded-missing-regression.gguf"
            )
            payload = {
                "data": [
                    {
                        "id": "candidate-loaded-missing",
                        "status": {
                            "value": "loaded",
                            "args": ["--model", str(missing_path)],
                        },
                        "source": "models_dir",
                    }
                ]
            }

            async def fake_request(
                method: str,
                path: str,
                *,
                params: dict[str, str] | None = None,
                body: dict[str, object] | None = None,
            ) -> RouterObservation:
                return RouterObservation(200, "", payload, None)

            client._request = fake_request  # type: ignore[method-assign]
            result = await client.list_models()

            self.assertFalse(result.valid)
            self.assertEqual(result.models, ())
        finally:
            await client._client.aclose()


if __name__ == "__main__":
    unittest.main()
