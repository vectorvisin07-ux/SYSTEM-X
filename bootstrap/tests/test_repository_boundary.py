"""Repository product-source boundary regression tests."""
from __future__ import annotations
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR_PREFIX = "model-api-gguf/llama.cpp/"
PATH_PATTERN = re.compile("(?:" + "re" + "build[_\\.-]?[0-9]|" + "mi" + "ni[_\\. -]?[0-9]|" + "mac" + "ro[_\\. -]?[0-9]|" + "sys" + "min|" + "x\\.[e5]|" + "x\\.?er(?:[_\\. -]?[0-9])?|" + "open" + "claw" + ")", re.IGNORECASE)
TEXT_PATTERN = re.compile("(?:" + "re" + "build[_\\. ][0-9]+|" + "mi" + "ni[_\\. ]?[0-9]+|" + "mac" + "ro[_\\. ]?[0-9]+|" + "sys" + "min|" + "(?<![a-z0-9_-])x\\.[e5]\\s+(?:workflow|position)|" + "x\\.?er[0-9]+|" + "open" + "claw(?:.*(?:packet|output)|[\\/].*workspace)" + ")", re.IGNORECASE)
GENERATED = (".system-x-bootstrap-state/", "INSPECTOR/.venv/", "INSPECTOR/RUNTIME/", "INSPECTOR/capabilities/", "model-api-gguf/api_service/.venv/", "model-api-gguf/llama.cpp/build/", "model-api-gguf/RUNTIME/", "model-api-gguf/MODEL/")

class RepositoryBoundary(unittest.TestCase):
    @classmethod
    def tracked(cls) -> list[str]:
        return subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")[:-1]

    def test_functional_names_and_text(self) -> None:
        paths = self.tracked()
        first_party = [p for p in paths if not p.startswith(VENDOR_PREFIX)]
        self.assertEqual([], [p for p in first_party if PATH_PATTERN.search(p)])
        matches = []
        for rel in first_party:
            data = (ROOT / rel).read_bytes()
            if b"\0" in data[:4096]:
                continue
            text = data.decode("utf-8")
            if TEXT_PATTERN.search(text):
                matches.append(rel)
        self.assertEqual([], matches)

    def test_generated_state_is_untracked(self) -> None:
        paths = self.tracked()
        self.assertFalse(any(any(p == root[:-1] or p.startswith(root) for root in GENERATED) for p in paths))
        self.assertFalse(any(p.startswith("docs/rebuild/receipts/") for p in paths))
        self.assertNotIn("SYSTEM_X_EVIDENCE_INDEX.json", paths)

    def test_vendor_exception_is_exact(self) -> None:
        paths = self.tracked()
        vendor = [p for p in paths if p.startswith(VENDOR_PREFIX)]
        self.assertEqual(3242, len(vendor))
        self.assertEqual("5d4b137293d12c375e876a54dedd3444a6d28772", subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD:model-api-gguf/llama.cpp"]).decode().strip())

if __name__ == "__main__":
    unittest.main()
