"""Owner-only control-plane broker hosted by the existing System X supervisor."""
from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

from .capabilities import authorize
from .commands import CommandEnvelope
from .errors import ControlPlaneError
from .kernel import Kernel
from .serialization import canonical_hash

MAX_FRAME = 64 * 1024
DEFAULT_CONTROL_ROOT = Path("/home/user/SYSTEMS/system-x/INSPECTOR/RUNTIME/control-plane")
CAPABILITY_BY_OPERATION = {
    "ActivateModel": "model.activate", "SetDefaultAlias": "alias.write",
    "StartService": "service.repair", "StopService": "service.repair",
    "RepairService": "service.repair", "RepairRuntime": "service.repair",
    "RotateCredential": "credential.rotate", "CreateBackup": "backup.create",
    "VerifyBackup": "backup.verify", "RestoreBackup": "backup.restore",
    "ApplyRelease": "release.apply", "RollbackRelease": "release.rollback",
    "CreateBrowserSession": "browser_session.create", "RevokeBrowserSession": "browser_session.revoke",
    "RegisterWindowsEntry": "windows_entry.write", "RemoveWindowsEntry": "windows_entry.write",
}


class ControlPlaneBroker:
    """A small request/response broker; the supervisor remains the sole process owner."""

    def __init__(self, root: Path = DEFAULT_CONTROL_ROOT):
        self.root = Path(root)
        self.socket_path = self.root / "control-plane.sock"
        self.store_path = self.root / "control-plane.sqlite3"
        self.kernel = Kernel(self.root)
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        if os.getuid() != 1100:
            raise PermissionError("control-plane requires product owner uid")
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            self.socket_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.settimeout(0.5)
        listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        listener.listen(8)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="system-x-control-plane", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stopping.set()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            self.socket_path.unlink()
        self.kernel.store.close()

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stopping.is_set():
            try:
                conn, _ = self._listener.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(3.0)
            if not self._peer_is_owner(conn):
                self._write(conn, {"schema_version": "system-x.control-result.v1", "state": "FAILED_CLEAN", "reason_code": "CAPABILITY_DENIED"})
                return
            data = bytearray()
            try:
                while len(data) <= MAX_FRAME:
                    chunk = conn.recv(4096)
                    if not chunk or b"\n" in chunk:
                        data.extend(chunk.split(b"\n", 1)[0])
                        break
                    data.extend(chunk)
                if len(data) > MAX_FRAME:
                    raise ValueError("FRAME_TOO_LARGE")
                command = json.loads(bytes(data))
                envelope = CommandEnvelope.parse(command)
                required = CAPABILITY_BY_OPERATION[command["operation_type"]]
                authorize(command["capability_set"], required, actor=envelope.actor)
                result = self.kernel.execute(command, required)
            except (KeyError, ValueError, json.JSONDecodeError, ControlPlaneError) as exc:
                result = {"schema_version": "system-x.control-result.v1", "state": "FAILED_CLEAN", "reason_code": getattr(exc, "reason_code", str(exc)), "public_message": "control request rejected", "reused_existing_result": False}
            self._write(conn, result)

    @staticmethod
    def _peer_is_owner(conn: socket.socket) -> bool:
        if not hasattr(socket, "SO_PEERCRED"):
            return False
        _pid, uid, _gid = __import__("struct").unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        return uid == os.getuid()

    @staticmethod
    def _write(conn: socket.socket, value: dict[str, Any]) -> None:
        conn.sendall(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")

