"""Regression coverage for the named WSL route with a space-bearing trial path."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from system_x_bootstrap.command import InstallationUser, build_elevated_reconstruct_argv


class SpaceSafeReconstructTests(unittest.TestCase):
    def test_wsl_route_quotes_space_bearing_repository_and_entrypoint(self) -> None:
        user = InstallationUser.from_name("user")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trial parent with space" / "source"
            (root / "bootstrap").mkdir(parents=True)
            entrypoint = root / "bootstrap" / "run_bootstrap.py"
            entrypoint.write_text("# fixture\n", encoding="utf-8")
            with mock.patch("system_x_bootstrap.command._wsl_handoff_available", return_value=True), mock.patch("system_x_bootstrap.command._validated_wsl_distribution", return_value="Ubuntu-26.04"):
                argv = build_elevated_reconstruct_argv(root, user)
        self.assertEqual(argv[0], "/init")
        self.assertIn(f"--cd \"{root}\"", argv[-1])
        self.assertIn(f"-B \"{entrypoint}\"", argv[-1])
        self.assertNotIn("unsafe token", argv[-1])


if __name__ == "__main__":
    unittest.main()
