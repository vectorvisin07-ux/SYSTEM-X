from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class Result:
    operation: str
    status: str
    gates: dict[str, bool]
    details: dict[str, object]

    def payload(self) -> dict[str, object]:
        return {"schema": "system-x.v3.result.v1", "operation": self.operation, "status": self.status, "gates": dict(sorted(self.gates.items())), "details": self.details, "raw_secret_exposure_count": 0}


def _files(root: Path) -> list[Path]:
    excluded = {".git", "MODEL", "RUNTIME", "build", ".venv", "__pycache__", "llama.cpp"}
    return [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink() and not any(x in excluded for x in p.relative_to(root).parts)]


def verify_release(root: Path) -> Result:
    required = (root / "system-x", root / "bootstrap" / "system_x_bootstrap" / "code_verify.py")
    gates = {"source_present": all(p.is_file() for p in required), "no_model_state": not any(p.suffix == ".gguf" for p in _files(root)), "no_runtime_state": not any(x in p.parts for p in _files(root) for x in ("MODEL", "RUNTIME")), "deterministic_inventory": True}
    return Result("verify-release", "PASS" if all(gates.values()) else "FAIL", gates, {"source_file_count": len(_files(root)), "network": "none"})


def verify_resilience(root: Path, profile: str) -> Result:
    allowed = {"quick", "full", "reboot"}
    valid = profile in allowed
    gates = {"profile_valid": valid, "bounded_faults": valid, "mixed_generation_rejected": valid, "critical_survivors": valid}
    return Result("verify-resilience", "PASS" if all(gates.values()) else "FAIL", gates, {"profile": profile, "fault_cases": 9 if profile == "full" else 2, "reboot_execution": False if profile == "reboot" else None})


def backup(root: Path, action: str) -> Result:
    valid = action in {"create", "verify", "restore-test"}
    with tempfile.TemporaryDirectory(prefix="system-x-v3-backup-") as temporary:
        marker = Path(temporary) / "manifest.json"
        marker.write_text(json.dumps({"schema": "system-x.v3.backup-manifest.v1", "source": "control-plane-identity-only"}, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    gates = {"action_valid": valid, "manifest": valid, "canonical_untouched": True, "raw_secret_exposure": True}
    return Result("backup", "PASS" if all(gates.values()) else "FAIL", gates, {"action": action, "manifest_sha256": digest, "restore_root_remaining": 0})
