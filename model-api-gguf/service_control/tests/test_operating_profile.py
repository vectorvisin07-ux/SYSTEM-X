"""Consolidated operating-profile tests.

All stateful fixtures are created below a TemporaryDirectory.  The production
RUNTIME tree is never initialized by this suite.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock


BRANCH_ROOT = Path(__file__).resolve().parents[2]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from service_control import operating_profile as op  # noqa: E402


FIXED_UTC = "2026-01-02T03:04:05.000006Z"


def unused_loopback_ports() -> tuple[int, int]:
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return sockets[0].getsockname()[1], sockets[1].getsockname()[1]
    finally:
        for item in sockets:
            item.close()


def valid_profile(
    public_port: int,
    private_port: int,
    *,
    automatic_recovery_enabled: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": op.OPERATING_PROFILE_SCHEMA,
        "public_endpoint": {"host": "127.0.0.1", "port": public_port},
        "private_router_endpoint": {
            "host": "127.0.0.1",
            "port": private_port,
        },
        "default_model_alias": "default",
        "startup_model_policy": "always_warm",
        "automatic_recovery_enabled": automatic_recovery_enabled,
        "graceful_shutdown": {"enabled": True, "timeout_seconds": 30},
        "recovery_delay": {
            "initial_seconds": 0.25,
            "maximum_seconds": 30,
            "multiplier": 2,
        },
    }


def write_json(path: Path, value: object, *, sort_keys: bool = False) -> None:
    path.write_text(
        json.dumps(value, sort_keys=sort_keys, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class ProfileValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.public_port, self.private_port = unused_loopback_ports()
        self.value = valid_profile(self.public_port, self.private_port)

    def assert_invalid(
        self, value: object, expected_reason: str | None = None
    ) -> None:
        with self.assertRaises(op.ServiceControlError) as caught:
            op.validate_operating_profile(value)
        if expected_reason is not None:
            self.assertEqual(caught.exception.reason_code, expected_reason)

    def test_valid_profile_and_identity(self) -> None:
        profile = op.validate_operating_profile(self.value)
        self.assertEqual(profile.default_model_alias, "default")
        self.assertEqual(profile.startup_model_policy, "always_warm")
        self.assertTrue(profile.automatic_recovery_enabled)
        self.assertEqual(profile.recovery_maximum_attempts_in_window, 3)
        self.assertEqual(profile.recovery_attempt_window_seconds, 60)
        self.assertEqual(profile.recovery_stable_reset_seconds, 30)
        self.assertRegex(profile.identity, r"\Asha256:[0-9a-f]{64}\Z")
        self.assertEqual(
            json.loads(profile.canonical_json),
            profile.as_dict(),
        )

        disabled = copy.deepcopy(self.value)
        disabled["automatic_recovery_enabled"] = False
        self.assertFalse(
            op.validate_operating_profile(disabled).automatic_recovery_enabled
        )

    def test_identity_is_logical_and_stable(self) -> None:
        first = op.validate_operating_profile(self.value)
        reordered = dict(reversed(list(self.value.items())))
        reordered["public_endpoint"] = {
            "port": self.public_port,
            "host": "127.0.0.1",
        }
        second = op.validate_operating_profile(reordered)
        self.assertEqual(first.identity, second.identity)

        numeric_forms = copy.deepcopy(self.value)
        numeric_forms["graceful_shutdown"]["timeout_seconds"] = 30.0
        numeric_forms["recovery_delay"]["maximum_seconds"] = 30.0
        self.assertEqual(
            first.identity,
            op.validate_operating_profile(numeric_forms).identity,
        )

        changed = copy.deepcopy(self.value)
        changed["default_model_alias"] = "default-v2"
        self.assertNotEqual(
            first.identity,
            op.validate_operating_profile(changed).identity,
        )

    def test_file_whitespace_and_key_order_do_not_change_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "compact.json"
            pretty = root / "pretty.json"
            compact.write_text(
                json.dumps(self.value, separators=(",", ":")),
                encoding="utf-8",
            )
            pretty.write_text(
                json.dumps(
                    dict(reversed(list(self.value.items()))),
                    indent=4,
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                op.load_operating_profile(compact).identity,
                op.load_operating_profile(pretty).identity,
            )

    def test_endpoint_matrix(self) -> None:
        ipv6 = copy.deepcopy(self.value)
        ipv6["public_endpoint"]["host"] = "0:0:0:0:0:0:0:1"
        profile = op.validate_operating_profile(ipv6)
        self.assertEqual(profile.public_endpoint.host, "::1")

        for bad_host in ("localhost", "0.0.0.0", "192.0.2.1", ""):
            with self.subTest(host=bad_host):
                candidate = copy.deepcopy(self.value)
                candidate["public_endpoint"]["host"] = bad_host
                self.assert_invalid(candidate)

        for bad_port in (0, 65_536, -1, True, 3.5, "1234"):
            with self.subTest(port=bad_port):
                candidate = copy.deepcopy(self.value)
                candidate["public_endpoint"]["port"] = bad_port
                self.assert_invalid(candidate, "invalid_endpoint_port")

        duplicate = copy.deepcopy(self.value)
        duplicate["private_router_endpoint"] = copy.deepcopy(
            duplicate["public_endpoint"]
        )
        self.assert_invalid(duplicate, "duplicate_endpoint")

    def test_alias_matrix(self) -> None:
        invalid_aliases = (
            "",
            " " ,
            " default",
            "default ",
            "../model",
            r"folder\model",
            "bad\nalias",
            "x" * (op.MAX_ALIAS_CHARACTERS + 1),
        )
        for alias in invalid_aliases:
            with self.subTest(alias=repr(alias)):
                candidate = copy.deepcopy(self.value)
                candidate["default_model_alias"] = alias
                self.assert_invalid(candidate, "invalid_model_alias")

    def test_policy_and_numeric_matrix(self) -> None:
        wrong_policy = copy.deepcopy(self.value)
        wrong_policy["startup_model_policy"] = "lazy"
        self.assert_invalid(
            wrong_policy, "unsupported_startup_model_policy"
        )

        wrong_boolean = copy.deepcopy(self.value)
        wrong_boolean["automatic_recovery_enabled"] = 1
        self.assert_invalid(wrong_boolean, "invalid_boolean")

        for number in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(number=number):
                candidate = copy.deepcopy(self.value)
                candidate["recovery_delay"]["initial_seconds"] = number
                self.assert_invalid(candidate, "non_finite_number")

        negative = copy.deepcopy(self.value)
        negative["recovery_delay"]["initial_seconds"] = -0.1
        self.assert_invalid(negative, "number_out_of_range")

        zero_maximum = copy.deepcopy(self.value)
        zero_maximum["recovery_delay"]["maximum_seconds"] = 0
        self.assert_invalid(zero_maximum, "number_out_of_range")

        reversed_delay = copy.deepcopy(self.value)
        reversed_delay["recovery_delay"]["initial_seconds"] = 31
        self.assert_invalid(
            reversed_delay, "invalid_recovery_delay_relation"
        )

        low_multiplier = copy.deepcopy(self.value)
        low_multiplier["recovery_delay"]["multiplier"] = 0.99
        self.assert_invalid(low_multiplier, "number_out_of_range")

        high_multiplier = copy.deepcopy(self.value)
        high_multiplier["recovery_delay"]["multiplier"] = (
            op.MAX_RECOVERY_MULTIPLIER + 0.01
        )
        self.assert_invalid(high_multiplier, "number_out_of_range")

        for timeout in (0, -1, op.MAX_GRACEFUL_TIMEOUT_SECONDS + 1):
            with self.subTest(timeout=timeout):
                candidate = copy.deepcopy(self.value)
                candidate["graceful_shutdown"]["timeout_seconds"] = timeout
                self.assert_invalid(candidate, "number_out_of_range")

        for field, value in (
            ("maximum_attempts_in_window", 0),
            ("attempt_window_seconds", 0),
            ("stable_reset_seconds", 0),
        ):
            with self.subTest(recovery_loop_field=field):
                candidate = copy.deepcopy(self.value)
                candidate["recovery_loop"] = {
                    "maximum_attempts_in_window": 3,
                    "attempt_window_seconds": 60,
                    "stable_reset_seconds": 30,
                }
                candidate["recovery_loop"][field] = value
                self.assert_invalid(
                    candidate,
                    (
                        "invalid_recovery_loop"
                        if field == "maximum_attempts_in_window"
                        else "number_out_of_range"
                    ),
                )

    def test_shape_and_json_file_failures(self) -> None:
        unknown = copy.deepcopy(self.value)
        unknown["future_supervisor"] = {}
        self.assert_invalid(unknown, "unknown_field")

        unknown_nested = copy.deepcopy(self.value)
        unknown_nested["recovery_delay"]["jitter"] = 1
        self.assert_invalid(unknown_nested, "unknown_field")

        missing = copy.deepcopy(self.value)
        del missing["default_model_alias"]
        self.assert_invalid(missing, "missing_field")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_path = root / "missing.json"
            with self.assertRaises(op.ServiceControlError) as caught:
                op.load_operating_profile(missing_path)
            self.assertEqual(caught.exception.reason_code, "missing_file")

            malformed = root / "malformed.json"
            malformed.write_text('{"broken":', encoding="utf-8")
            with self.assertRaises(op.ServiceControlError) as caught:
                op.load_operating_profile(malformed)
            self.assertEqual(caught.exception.reason_code, "invalid_json")

            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":"x","schema_version":"y"}',
                encoding="utf-8",
            )
            with self.assertRaises(op.ServiceControlError) as caught:
                op.load_operating_profile(duplicate)
            self.assertEqual(caught.exception.reason_code, "duplicate_json_key")

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text(
                json.dumps(self.value).replace("0.25", "NaN"),
                encoding="utf-8",
            )
            with self.assertRaises(op.ServiceControlError) as caught:
                op.load_operating_profile(nonfinite)
            self.assertEqual(caught.exception.reason_code, "non_finite_number")

    def test_symlink_profile_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual.json"
            link = root / "link.json"
            write_json(actual, self.value)
            try:
                link.symlink_to(actual)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            with self.assertRaises(op.ServiceControlError) as caught:
                op.load_operating_profile(link)
            self.assertEqual(caught.exception.reason_code, "symlink_rejected")

    def test_schema_documents_parse_and_match_source_identity(self) -> None:
        source_dir = Path(op.__file__).resolve().parent
        profile_schema = json.loads(
            (source_dir / "operating-profile.schema.json").read_text(
                encoding="utf-8"
            )
        )
        desired_schema = json.loads(
            (source_dir / "desired-state.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(profile_schema["$id"], op.OPERATING_PROFILE_SCHEMA)
        self.assertEqual(desired_schema["$id"], op.DESIRED_STATE_SCHEMA)
        self.assertFalse(profile_schema["additionalProperties"])
        self.assertFalse(desired_schema["additionalProperties"])


class DesiredStateValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        public_port, private_port = unused_loopback_ports()
        self.profile = op.validate_operating_profile(
            valid_profile(public_port, private_port)
        )

    def test_initialize_update_show_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "desired-state.json"
            first = op.initialize_desired_state(
                self.profile,
                path,
                "RUNNING",
                updated_utc=FIXED_UTC,
            )
            self.assertEqual(first.desired_state, "RUNNING")
            self.assertEqual(first.generation, 1)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                op.load_desired_state(path, self.profile.identity),
                first,
            )

            second = op.set_desired_state(
                self.profile,
                "STOPPED",
                path,
                expected_generation=1,
                updated_utc="2026-01-02T03:04:06.000007Z",
            )
            self.assertEqual(second.desired_state, "STOPPED")
            self.assertEqual(second.generation, 2)
            self.assertEqual(self.profile.identity, second.profile_identity)

    def test_initialization_refuses_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "desired-state.json"
            path.write_text("sentinel", encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaises(op.ServiceControlError) as caught:
                op.initialize_desired_state(self.profile, path)
            self.assertEqual(
                caught.exception.reason_code, "desired_state_already_exists"
            )
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_request_and_stale_generation_do_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "desired-state.json"
            op.initialize_desired_state(
                self.profile, path, updated_utc=FIXED_UTC
            )
            before = path.read_bytes()
            with self.assertRaises(op.ServiceControlError) as caught:
                op.set_desired_state(self.profile, "PAUSED", path)
            self.assertEqual(caught.exception.reason_code, "invalid_desired_state")
            self.assertEqual(path.read_bytes(), before)

            with self.assertRaises(op.ServiceControlError) as caught:
                op.set_desired_state(
                    self.profile,
                    "RUNNING",
                    path,
                    expected_generation=2,
                )
            self.assertEqual(
                caught.exception.reason_code, "stale_expected_generation"
            )
            self.assertEqual(path.read_bytes(), before)

    def test_profile_identity_mismatch_does_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "desired-state.json"
            op.initialize_desired_state(
                self.profile, path, updated_utc=FIXED_UTC
            )
            before = path.read_bytes()
            changed = self.profile.as_dict()
            changed["default_model_alias"] = "other"
            other_profile = op.validate_operating_profile(changed)
            with self.assertRaises(op.ServiceControlError) as caught:
                op.set_desired_state(other_profile, "RUNNING", path)
            self.assertEqual(
                caught.exception.reason_code, "profile_identity_mismatch"
            )
            self.assertEqual(path.read_bytes(), before)

    def test_desired_state_shape_matrix(self) -> None:
        base = {
            "schema_version": op.DESIRED_STATE_SCHEMA,
            "profile_identity": self.profile.identity,
            "desired_state": "RUNNING",
            "generation": 1,
            "updated_utc": FIXED_UTC,
        }
        invalid_cases = []
        for key, value in (
            ("desired_state", "PAUSED"),
            ("generation", 0),
            ("generation", True),
            ("profile_identity", "sha256:ABC"),
            ("updated_utc", "2026-01-02T03:04:05Z"),
        ):
            candidate = copy.deepcopy(base)
            candidate[key] = value
            invalid_cases.append(candidate)
        unknown = copy.deepcopy(base)
        unknown["pid"] = 123
        invalid_cases.append(unknown)
        for candidate in invalid_cases:
            with self.subTest(candidate=candidate):
                with self.assertRaises(op.ServiceControlError):
                    op.validate_desired_state(candidate)

    def test_state_target_and_directory_symlinks_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual.json"
            actual.write_text("sentinel", encoding="utf-8")
            link = root / "state.json"
            try:
                link.symlink_to(actual)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            with self.assertRaises(op.ServiceControlError) as caught:
                op.initialize_desired_state(self.profile, link)
            self.assertEqual(caught.exception.reason_code, "symlink_rejected")
            self.assertEqual(actual.read_text(encoding="utf-8"), "sentinel")

            real_directory = root / "real"
            real_directory.mkdir()
            linked_directory = root / "linked"
            linked_directory.symlink_to(real_directory, target_is_directory=True)
            with self.assertRaises(op.ServiceControlError) as caught:
                op.initialize_desired_state(
                    self.profile, linked_directory / "state.json"
                )
            self.assertEqual(caught.exception.reason_code, "symlink_rejected")


class PathContractTests(unittest.TestCase):
    def test_default_paths_are_self_relative_and_not_created(self) -> None:
        source_dir = Path(op.__file__).resolve().parent
        branch_root = source_dir.parent
        expected_runtime = branch_root / "RUNTIME" / "service_control"
        self.assertEqual(
            op.DEFAULT_PROFILE_PATH,
            expected_runtime / "operating-profile.json",
        )
        self.assertEqual(
            op.DEFAULT_DESIRED_STATE_PATH,
            expected_runtime / "desired-state.json",
        )

    def test_explicit_profile_and_state_paths(self) -> None:
        public_port, private_port = unused_loopback_ports()
        value = valid_profile(public_port, private_port)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_path = root / "custom-profile.json"
            state_path = root / "custom-state.json"
            write_json(profile_path, value)
            profile = op.load_operating_profile(profile_path)
            op.initialize_desired_state(
                profile,
                state_path,
                "STOPPED",
                updated_utc=FIXED_UTC,
            )
            self.assertEqual(
                op.load_desired_state(
                    state_path,
                    expected_profile_identity=profile.identity,
                ).generation,
                1,
            )


class AtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        public_port, private_port = unused_loopback_ports()
        self.profile = op.validate_operating_profile(
            valid_profile(public_port, private_port)
        )

    def test_replace_failure_preserves_old_state_and_cleans_owned_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "desired-state.json"
            op.initialize_desired_state(
                self.profile, path, updated_utc=FIXED_UTC
            )
            before = path.read_bytes()
            unrelated = root / ".unrelated.tmp"
            unrelated.write_text("retain", encoding="utf-8")
            with mock.patch.object(
                op.os,
                "replace",
                side_effect=OSError("injected replace failure"),
            ):
                with self.assertRaises(op.ServiceControlError) as caught:
                    op.set_desired_state(self.profile, "RUNNING", path)
            self.assertEqual(
                caught.exception.reason_code, "atomic_state_write_failed"
            )
            self.assertIn("injected replace failure", caught.exception.message)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "retain")
            self.assertEqual(
                list(root.glob(f".{path.name}.*.tmp")),
                [],
            )

    def test_atomic_visibility_and_stale_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "desired-state.json"
            op.initialize_desired_state(
                self.profile, path, updated_utc=FIXED_UTC
            )
            observed: list[tuple[str, int]] = []
            failures: list[BaseException] = []
            start = threading.Barrier(2)
            stop = threading.Event()

            def reader() -> None:
                try:
                    start.wait()
                    while not stop.is_set():
                        state = op.load_desired_state(
                            path, self.profile.identity
                        )
                        observed.append((state.desired_state, state.generation))
                    state = op.load_desired_state(path, self.profile.identity)
                    observed.append((state.desired_state, state.generation))
                except BaseException as exc:
                    failures.append(exc)

            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            start.wait()
            expected_generation = 1
            for index in range(20):
                requested = "RUNNING" if index % 2 == 0 else "STOPPED"
                result = op.set_desired_state(
                    self.profile,
                    requested,
                    path,
                    expected_generation=expected_generation,
                )
                expected_generation += 1
                self.assertEqual(result.generation, expected_generation)
            stop.set()
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(observed)
            self.assertTrue(
                all(state in op.DESIRED_STATES for state, _ in observed)
            )
            self.assertTrue(
                all(1 <= generation <= 21 for _, generation in observed)
            )
            final_bytes = path.read_bytes()
            final = op.load_desired_state(path, self.profile.identity)
            self.assertEqual(final.generation, 21)
            with self.assertRaises(op.ServiceControlError) as caught:
                op.set_desired_state(
                    self.profile,
                    "RUNNING",
                    path,
                    expected_generation=20,
                )
            self.assertEqual(
                caught.exception.reason_code, "stale_expected_generation"
            )
            self.assertEqual(path.read_bytes(), final_bytes)
            self.assertEqual(list(root.glob(f".{path.name}.*.tmp")), [])


class MachineInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        public_port, private_port = unused_loopback_ports()
        self.profile_value = valid_profile(public_port, private_port)
        self.profile_path = self.root / "operating-profile.json"
        self.state_path = self.root / "desired-state.json"
        write_json(self.profile_path, self.profile_value)
        source_program = Path(op.__file__).resolve()
        isolated_service_control = self.root / "service_control"
        isolated_service_control.mkdir()
        self.program = isolated_service_control / source_program.name
        self.program.write_bytes(source_program.read_bytes())
        self.isolated_default_profile = (
            self.root
            / "RUNTIME"
            / "service_control"
            / "operating-profile.json"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_cli(
        self, operation: str, *arguments: str
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        completed = subprocess.run(
            [sys.executable, str(self.program), operation, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 1, completed)
        result = json.loads(lines[0])
        self.assertEqual(result["schema_version"], op.RESULT_SCHEMA)
        self.assertEqual(result["operation"], operation)
        self.assertNotIn("Traceback", completed.stderr)
        return completed, result

    @property
    def explicit_paths(self) -> tuple[str, ...]:
        return (
            "--profile",
            str(self.profile_path),
            "--state-path",
            str(self.state_path),
        )

    def test_five_valid_operations(self) -> None:
        completed, result = self.run_cli(
            "validate-profile", "--profile", str(self.profile_path)
        )
        self.assertEqual(completed.returncode, 0)
        self.assertTrue(result["ok"])
        identity = result["profile_identity"]

        completed, result = self.run_cli(
            "show-profile", "--profile", str(self.profile_path)
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result["profile"]["default_model_alias"], "default")

        completed, result = self.run_cli(
            "initialize-desired-state",
            *self.explicit_paths,
            "--state",
            "RUNNING",
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result["generation"], 1)
        self.assertEqual(result["profile_identity"], identity)

        completed, result = self.run_cli(
            "show-desired-state", *self.explicit_paths
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result["desired_state"], "RUNNING")

        completed, result = self.run_cli(
            "set-desired-state",
            *self.explicit_paths,
            "--state",
            "STOPPED",
            "--expected-generation",
            "1",
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result["desired_state"], "STOPPED")
        self.assertEqual(result["generation"], 2)

    def test_failure_exit_class_and_default_paths(self) -> None:
        completed, result = self.run_cli("validate-profile")
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason_code"], "missing_file")
        self.assertEqual(
            result["resolved_paths"]["profile"],
            str(self.isolated_default_profile),
        )

        completed, result = self.run_cli("unknown-operation")
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason_code"], "invalid_arguments")

    def test_profile_mismatch_and_stale_generation(self) -> None:
        completed, _ = self.run_cli(
            "initialize-desired-state", *self.explicit_paths
        )
        self.assertEqual(completed.returncode, 0)

        completed, result = self.run_cli(
            "set-desired-state",
            *self.explicit_paths,
            "--state",
            "RUNNING",
            "--expected-generation",
            "7",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(
            result["reason_code"], "stale_expected_generation"
        )
        before = self.state_path.read_bytes()

        changed = copy.deepcopy(self.profile_value)
        changed["default_model_alias"] = "changed"
        write_json(self.profile_path, changed)
        completed, result = self.run_cli(
            "show-desired-state", *self.explicit_paths
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(result["reason_code"], "profile_identity_mismatch")
        self.assertEqual(self.state_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
