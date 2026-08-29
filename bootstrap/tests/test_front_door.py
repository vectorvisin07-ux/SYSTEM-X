"""Focused tests for the public System X boundary."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys
import unittest
from unittest.mock import patch

from system_x_bootstrap import front_door

ROOT = Path(__file__).resolve().parents[2]


class FrontDoorTests(unittest.TestCase):
    def test_closed_operation_set(self) -> None:
        self.assertEqual(front_door.PUBLIC_OPERATIONS, ("install", "status", "connection", "doctor", "help"))

    def test_help_has_required_shape_and_snapshot(self) -> None:
        result = front_door._help()
        self.assertTrue(result["ok"])
        self.assertIn("system-x install", result["message"])
        self.assertIn("OpenClaw is not required", result["message"])
        self.assertEqual(result["child_result_identities"], [])
        self.assertEqual(set(result), {"schema_version", "operation", "ok", "reason_code", "message", "installation_state", "service_state", "readiness_state", "model_state", "connection_state", "recommended_model", "child_result_identities", "timestamp_utc"})

    def test_unknown_operation_is_clean_json(self) -> None:
        result = front_door._result("unknown", ok=False, reason_code="UNKNOWN_OPERATION", message="bad", installation_state="SOURCE_ONLY", service_state="STOPPED", readiness_state="WAITING_FOR_MODEL", model_state="ABSENT", connection_state="NOT_READY", recommended_model=None, child_result_identities=())
        self.assertFalse(result["ok"])

    def test_child_identity_drops_details_and_secret_like_values(self) -> None:
        run = front_door.ChildRun(("child",), 0, False, {"operation": "status", "status": "ok", "details": {"raw_api_key": "never"}, "paths": {"physical_gguf_path": "never"}}, True)
        identity = front_door._safe_identity(run)
        self.assertNotIn("details", identity)
        self.assertNotIn("paths", identity)
        self.assertNotIn("raw_api_key", json.dumps(identity))

    def test_install_delegates_once_without_model_arguments(self) -> None:
        payload = {"schema": "system-x.bootstrap.ordered-results.v1", "results": [{"operation": "reconstruct", "status": "ok", "state": "WAITING_FOR_MODEL", "details": {"model": "bad"}}]}
        fake = front_door.ChildRun(("bootstrap",), 0, False, payload, True)
        with patch.object(front_door, "_run_child", return_value=fake) as runner:
            result = front_door._install(ROOT)
        runner.assert_called_once()
        argv = runner.call_args.args[0]
        self.assertIn("reconstruct", argv)
        self.assertIn("--authorize", argv)
        self.assertNotIn("--model", argv)
        self.assertNotIn("--candidate", argv)
        self.assertEqual(result["readiness_state"], "WAITING_FOR_MODEL")

    def test_status_is_read_only_and_bounded(self) -> None:
        def fake(argv, *, cwd, timeout):
            if "run_bootstrap.py" in str(argv):
                value = {"schema": "system-x.bootstrap.result.v1", "operation": "status", "status": "ok", "state": "CLONED"}
            elif "run_inspector.py" in str(argv):
                value = {"operation": "status", "ok": True, "reason_code": "OK", "data": {}}
            else:
                value = {"ok": True, "registered": False, "active": False, "enabled": False}
            return front_door.ChildRun(tuple(str(x) for x in argv), 0, False, value, True)
        with patch.object(front_door, "_run_child", side_effect=fake) as runner:
            result = front_door._status(ROOT)
        self.assertEqual(result["installation_state"], "SOURCE_ONLY")
        self.assertGreaterEqual(runner.call_count, 1)

    def test_connection_delegates_once_and_suppresses_receipt_secrets(self) -> None:
        receipt = {"receipt_id": "connection-20260829T170000000000Z-0123456789abcdef", "receipt_identity": "sha256:" + "a" * 64, "receipt_source": "DEPLOY_GGUF", "service": {"public_origin": "http://127.0.0.1:1234", "service_readiness": "READY", "model_service_state": "READY", "inference_ready": True, "service_available": True, "desired_state": "RUNNING", "always_on": True, "authentication_required": True}, "connections": {}, "model": {"recommended_reference": "default", "model_state": "ready", "warm": True}, "authentication": {"required": True, "accepted_schemes": ["x-api-key", "Authorization Bearer"], "non_secret_key_id": "a" * 32, "raw_api_key_returned": False}, "capabilities": {}, "proof": {}, "lifecycle": {}}
        fake = front_door.ChildRun(("inspector",), 0, False, {"operation": "show-connection", "ok": True, "data": {"result_class": "CONNECTION_READY", "reason_code": "CONNECTION_READY", "receipt": receipt}}, True)
        with patch.object(front_door, "_run_child", return_value=fake) as runner:
            result = front_door._connection(ROOT)
        runner.assert_called_once()
        self.assertEqual(result["connection_state"], "READY")
        self.assertNotIn("never", json.dumps(result))
        self.assertIs(result["connection"]["authentication"]["raw_api_key_returned"], False)
        self.assertNotIn("physical_gguf_path", json.dumps(result))

    def test_launcher_is_regular_executable_and_caller_independent(self) -> None:
        launcher = ROOT / "system-x"
        mode = launcher.stat().st_mode
        self.assertFalse(launcher.is_symlink())
        self.assertTrue(stat.S_ISREG(mode))
        self.assertEqual(stat.S_IMODE(mode), 0o755)
        completed = subprocess.run([sys.executable, "-B", "-S", str(launcher), "help"], cwd="/tmp", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(json.loads(completed.stdout)["operation"], "help")
        self.assertEqual(completed.stderr, "")

    def test_source_boundary_has_no_direct_manager_or_controller_invocation(self) -> None:
        source = (ROOT / "bootstrap" / "system_x_bootstrap" / "front_door.py").read_text(encoding="utf-8")
        self.assertNotIn("shell=True", source)
        self.assertNotIn("systemctl", source)
        self.assertNotIn("llama-server", source)

    def test_schema_is_closed_and_loadable(self) -> None:
        schema = json.loads((ROOT / "bootstrap" / "schemas" / "front-door-result.schema.json").read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["$id"], "system-x.front-door-result.v1")


if __name__ == "__main__":
    unittest.main()
