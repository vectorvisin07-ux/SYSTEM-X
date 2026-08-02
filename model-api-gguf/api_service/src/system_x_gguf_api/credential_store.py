"""Protected local credential store for private System X API authentication."""

from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import os
from pathlib import Path
import secrets
import sqlite3
import stat
from typing import Any, Iterator

from .credential_types import (
    CREDENTIAL_SCHEMA_IDENTITY,
    CREDENTIAL_SCHEMA_VERSION,
    KEY_ID_PATTERN,
    VERIFIER_ALGORITHM,
    CredentialVerification,
    compute_verifier,
    execute_dummy_verification,
    generate_api_key,
    parse_api_key,
    verifier_matches,
)


DATABASE_BUSY_TIMEOUT_MILLISECONDS = 5_000
PEPPER_BYTES = 32
AUTH_ROOT_MODE = 0o700
AUTH_FILE_MODE = 0o600
PROHIBITED_COLUMN_FRAGMENTS = frozenset(
    {
        "raw_key",
        "secret",
        "pepper",
        "quota",
        "request_count",
        "token_budget",
        "rate_limit",
        "concurrency_limit",
        "billing",
        "balance",
    }
)


class CredentialStoreError(RuntimeError):
    """Credential state is invalid, unsafe, or unavailable."""


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def default_auth_root() -> Path:
    """Derive the fixed authentication root from this physical branch."""

    branch_root = Path(__file__).resolve(strict=True).parents[3]
    return branch_root / "RUNTIME" / "api" / "auth"


def _assert_native_absolute(path: Path) -> None:
    if not path.is_absolute():
        raise CredentialStoreError("credential path must be absolute")
    mounted_windows_root = Path("/mnt")
    if path == mounted_windows_root or mounted_windows_root in path.parents:
        raise CredentialStoreError("credential path may not use a Windows mount")


def _assert_no_symlink_components(path: Path) -> None:
    _assert_native_absolute(path)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise CredentialStoreError(
                f"credential path contains a symlink component: {current.name}"
            )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _validate_directory(path: Path, expected_mode: int) -> None:
    _assert_no_symlink_components(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CredentialStoreError(
            f"required credential directory is unavailable: {path.name}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CredentialStoreError(
            f"credential path is not a physical directory: {path.name}"
        )
    if stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise CredentialStoreError(
            f"credential directory mode is invalid: {path.name}"
        )


def _validate_file(
    path: Path,
    expected_mode: int,
    *,
    exact_size: int | None = None,
) -> None:
    _assert_no_symlink_components(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CredentialStoreError(
            f"required credential file is unavailable: {path.name}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CredentialStoreError(
            f"credential path is not a physical regular file: {path.name}"
        )
    if stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise CredentialStoreError(f"credential file mode is invalid: {path.name}")
    if exact_size is not None and metadata.st_size != exact_size:
        raise CredentialStoreError(f"credential file size is invalid: {path.name}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, data: bytes) -> None:
    _assert_no_symlink_components(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, AUTH_FILE_MODE)
    complete = False
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise CredentialStoreError("exclusive credential write did not progress")
            written += count
        os.fchmod(descriptor, AUTH_FILE_MODE)
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    _fsync_directory(path.parent)


class CredentialStore:
    """Own one physical v1 credential database and separate pepper."""

    def __init__(self, auth_root: Path | str | None = None) -> None:
        self.auth_root = Path(auth_root or default_auth_root())
        _assert_native_absolute(self.auth_root)
        self.database_path = self.auth_root / "credentials.sqlite3"
        self.pepper_path = self.auth_root / "pepper.bin"
        self.handoff_root = self.auth_root / "handoff"

    def _assert_containment(self) -> None:
        _assert_no_symlink_components(self.auth_root)
        for path in (
            self.database_path,
            self.pepper_path,
            self.handoff_root,
        ):
            if path.parent != self.auth_root:
                raise CredentialStoreError("credential path escaped authentication root")

    def _create_layout(self) -> None:
        self._assert_containment()
        parent = self.auth_root.parent
        _validate_directory(parent, _mode(parent))
        if not self.auth_root.exists():
            os.mkdir(self.auth_root, AUTH_ROOT_MODE)
            _fsync_directory(parent)
        _validate_directory(self.auth_root, AUTH_ROOT_MODE)
        if not self.handoff_root.exists():
            os.mkdir(self.handoff_root, AUTH_ROOT_MODE)
            _fsync_directory(self.auth_root)
        _validate_directory(self.handoff_root, AUTH_ROOT_MODE)

    def _create_pepper_once(self) -> bool:
        if self.pepper_path.exists():
            _validate_file(
                self.pepper_path,
                AUTH_FILE_MODE,
                exact_size=PEPPER_BYTES,
            )
            return False
        _write_exclusive(self.pepper_path, secrets.token_bytes(PEPPER_BYTES))
        _validate_file(
            self.pepper_path,
            AUTH_FILE_MODE,
            exact_size=PEPPER_BYTES,
        )
        _fsync_directory(self.auth_root)
        return True

    def _reserve_database_once(self) -> bool:
        if self.database_path.exists():
            _validate_file(self.database_path, AUTH_FILE_MODE)
            return False
        descriptor = os.open(
            self.database_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            AUTH_FILE_MODE,
        )
        os.fchmod(descriptor, AUTH_FILE_MODE)
        os.close(descriptor)
        _fsync_directory(self.auth_root)
        return True

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        _validate_file(self.database_path, AUTH_FILE_MODE)
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=rw",
            uri=True,
            isolation_level=None,
            timeout=DATABASE_BUSY_TIMEOUT_MILLISECONDS / 1_000,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                f"PRAGMA busy_timeout = {DATABASE_BUSY_TIMEOUT_MILLISECONDS}"
            )
            journal_mode = str(
                connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            ).lower()
            if journal_mode != "wal":
                raise CredentialStoreError("credential database WAL mode is unavailable")
            connection.execute("PRAGMA synchronous = FULL")
            yield connection
        finally:
            connection.close()
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{self.database_path}{suffix}")
                if sidecar.exists() and not sidecar.is_symlink():
                    os.chmod(sidecar, AUTH_FILE_MODE)

    def _initialize_schema(self, connection: sqlite3.Connection) -> bool:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected = {"credential_metadata", "api_keys"}
        if tables:
            if tables != expected:
                raise CredentialStoreError("credential database table set is invalid")
            self._validate_schema(connection)
            return False
        timestamp = utc_now()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE credential_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE TABLE api_keys (
                    key_id TEXT PRIMARY KEY
                        CHECK (
                            length(key_id) = 32
                            AND key_id NOT GLOB '*[^0-9a-f]*'
                        ),
                    label TEXT NOT NULL
                        CHECK (length(label) BETWEEN 1 AND 128),
                    verifier BLOB NOT NULL
                        CHECK (typeof(verifier) = 'blob' AND length(verifier) = 32),
                    verifier_algorithm TEXT NOT NULL
                        CHECK (verifier_algorithm = 'system-x-hmac-sha256-v1'),
                    status TEXT NOT NULL
                        CHECK (status IN ('ACTIVE', 'REVOKED')),
                    created_utc TEXT NOT NULL,
                    revoked_utc TEXT,
                    updated_utc TEXT NOT NULL,
                    CHECK (
                        (status = 'ACTIVE' AND revoked_utc IS NULL)
                        OR
                        (status = 'REVOKED' AND revoked_utc IS NOT NULL)
                    )
                ) WITHOUT ROWID
                """
            )
            connection.executemany(
                "INSERT INTO credential_metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_identity", CREDENTIAL_SCHEMA_IDENTITY),
                    ("schema_version", str(CREDENTIAL_SCHEMA_VERSION)),
                    ("created_utc", timestamp),
                    ("last_migration_utc", timestamp),
                ),
            )
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        return True

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if tables != {"credential_metadata", "api_keys"}:
            raise CredentialStoreError("credential database table set is invalid")
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute(
                "SELECT key, value FROM credential_metadata"
            )
        }
        required_metadata = {
            "schema_identity",
            "schema_version",
            "created_utc",
            "last_migration_utc",
        }
        if set(metadata) != required_metadata:
            raise CredentialStoreError("credential metadata record set is invalid")
        if metadata["schema_identity"] != CREDENTIAL_SCHEMA_IDENTITY:
            raise CredentialStoreError("credential schema identity is invalid")
        if metadata["schema_version"] != str(CREDENTIAL_SCHEMA_VERSION):
            raise CredentialStoreError("credential schema version is invalid")
        columns = {
            str(row["name"]): str(row["type"]).upper()
            for row in connection.execute("PRAGMA table_info(api_keys)")
        }
        expected_columns = {
            "key_id": "TEXT",
            "label": "TEXT",
            "verifier": "BLOB",
            "verifier_algorithm": "TEXT",
            "status": "TEXT",
            "created_utc": "TEXT",
            "revoked_utc": "TEXT",
            "updated_utc": "TEXT",
        }
        if columns != expected_columns:
            raise CredentialStoreError("credential key-column contract is invalid")
        lowered = {name.lower() for name in columns}
        if any(
            fragment in name
            for name in lowered
            for fragment in PROHIBITED_COLUMN_FRAGMENTS
        ):
            raise CredentialStoreError("credential database has a prohibited column")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if [str(row[0]) for row in integrity] != ["ok"]:
            raise CredentialStoreError("credential database integrity check failed")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise CredentialStoreError("credential database foreign-key check failed")
        if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise CredentialStoreError("credential database foreign keys are disabled")

    def initialize(self) -> dict[str, Any]:
        """Create missing protected state once and validate the complete store."""

        self._create_layout()
        pepper_created = self._create_pepper_once()
        database_created = self._reserve_database_once()
        try:
            with self._connection() as connection:
                schema_created = self._initialize_schema(connection)
                self._validate_schema(connection)
        except BaseException:
            if database_created:
                for suffix in ("", "-wal", "-shm"):
                    path = Path(f"{self.database_path}{suffix}")
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
            raise
        inspection = self.inspect()
        return {
            **inspection,
            "pepper_created": pepper_created,
            "database_created": database_created,
            "schema_created": schema_created,
        }

    def _read_pepper(self) -> bytes:
        _validate_file(
            self.pepper_path,
            AUTH_FILE_MODE,
            exact_size=PEPPER_BYTES,
        )
        descriptor = os.open(self.pepper_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            pepper = os.read(descriptor, PEPPER_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(pepper) != PEPPER_BYTES:
            raise CredentialStoreError("credential pepper length is invalid")
        return pepper

    @staticmethod
    def _validate_label(label: str) -> str:
        if (
            not isinstance(label, str)
            or not 1 <= len(label) <= 128
            or label.strip() != label
            or any(ord(character) < 0x20 for character in label)
        ):
            raise CredentialStoreError("credential label is invalid")
        return label

    def _validate_handoff_output(self, output_file: Path | str) -> Path:
        output = Path(output_file)
        _assert_native_absolute(output)
        _validate_directory(self.handoff_root, AUTH_ROOT_MODE)
        _assert_no_symlink_components(output.parent)
        if output.parent.resolve(strict=True) != self.handoff_root.resolve(strict=True):
            raise CredentialStoreError(
                "credential handoff output must be a direct child of handoff"
            )
        if os.path.lexists(output):
            raise CredentialStoreError("credential handoff output already exists")
        return output

    def issue(self, label: str, output_file: Path | str) -> dict[str, Any]:
        """Issue one key and write its only raw copy to an exclusive handoff."""

        self.inspect()
        clean_label = self._validate_label(label)
        output = self._validate_handoff_output(output_file)
        pepper = self._read_pepper()
        handoff_created = False
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                generated = generate_api_key()
                while (
                    connection.execute(
                        "SELECT 1 FROM api_keys WHERE key_id = ?",
                        (generated.key_id,),
                    ).fetchone()
                    is not None
                ):
                    generated = generate_api_key()
                timestamp = utc_now()
                verifier = compute_verifier(pepper, generated.raw_key)
                connection.execute(
                    """
                    INSERT INTO api_keys(
                        key_id, label, verifier, verifier_algorithm, status,
                        created_utc, revoked_utc, updated_utc
                    ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, NULL, ?)
                    """,
                    (
                        generated.key_id,
                        clean_label,
                        verifier,
                        VERIFIER_ALGORITHM,
                        timestamp,
                        timestamp,
                    ),
                )
                _write_exclusive(
                    output,
                    generated.raw_key.encode("utf-8") + b"\n",
                )
                handoff_created = True
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                if handoff_created:
                    try:
                        output.unlink()
                    except FileNotFoundError:
                        pass
                raise
        _validate_file(output, AUTH_FILE_MODE)
        return {
            "key_id": generated.key_id,
            "label": clean_label,
            "status": "ACTIVE",
            "created_utc": timestamp,
            "output_file": str(output),
        }

    def list_keys(self) -> list[dict[str, Any]]:
        """List only non-secret credential metadata."""

        self.inspect()
        with self._connection() as connection:
            return [
                {
                    "key_id": str(row["key_id"]),
                    "label": str(row["label"]),
                    "status": str(row["status"]),
                    "created_utc": str(row["created_utc"]),
                    "revoked_utc": (
                        str(row["revoked_utc"])
                        if row["revoked_utc"] is not None
                        else None
                    ),
                }
                for row in connection.execute(
                    """
                    SELECT key_id, label, status, created_utc, revoked_utc
                    FROM api_keys
                    ORDER BY created_utc, key_id
                    """
                )
            ]

    def revoke(self, key_id: str) -> dict[str, Any]:
        """Irreversibly revoke one active key without deleting its verifier."""

        if KEY_ID_PATTERN.fullmatch(key_id) is None:
            raise CredentialStoreError("credential key identity is invalid")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT key_id, label, status, created_utc, revoked_utc
                    FROM api_keys WHERE key_id = ?
                    """,
                    (key_id,),
                ).fetchone()
                if row is None:
                    raise CredentialStoreError("credential key identity was not found")
                if str(row["status"]) != "ACTIVE":
                    raise CredentialStoreError("credential key is already revoked")
                timestamp = utc_now()
                connection.execute(
                    """
                    UPDATE api_keys
                    SET status = 'REVOKED', revoked_utc = ?, updated_utc = ?
                    WHERE key_id = ? AND status = 'ACTIVE'
                    """,
                    (timestamp, timestamp, key_id),
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return {
            "key_id": str(row["key_id"]),
            "label": str(row["label"]),
            "status": "REVOKED",
            "created_utc": str(row["created_utc"]),
            "revoked_utc": timestamp,
        }

    def inspect(self, *, require_active: bool = False) -> dict[str, Any]:
        """Validate containment, permissions, schema, and database integrity."""

        self._assert_containment()
        _validate_directory(self.auth_root, AUTH_ROOT_MODE)
        _validate_directory(self.handoff_root, AUTH_ROOT_MODE)
        _validate_file(
            self.pepper_path,
            AUTH_FILE_MODE,
            exact_size=PEPPER_BYTES,
        )
        pepper = self._read_pepper()
        if len(pepper) != PEPPER_BYTES:
            raise CredentialStoreError("credential pepper read is incomplete")
        del pepper
        _validate_file(self.database_path, AUTH_FILE_MODE)
        with self._connection() as connection:
            self._validate_schema(connection)
            counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM api_keys GROUP BY status"
                )
            }
        active_count = counts.get("ACTIVE", 0)
        revoked_count = counts.get("REVOKED", 0)
        if require_active and active_count < 1:
            raise CredentialStoreError("credential store has no active key")
        return {
            "schema_identity": CREDENTIAL_SCHEMA_IDENTITY,
            "schema_version": CREDENTIAL_SCHEMA_VERSION,
            "auth_root": str(self.auth_root),
            "auth_root_mode": f"{_mode(self.auth_root):04o}",
            "database_path": str(self.database_path),
            "database_mode": f"{_mode(self.database_path):04o}",
            "pepper_path": str(self.pepper_path),
            "pepper_mode": f"{_mode(self.pepper_path):04o}",
            "pepper_length": PEPPER_BYTES,
            "handoff_root": str(self.handoff_root),
            "handoff_mode": f"{_mode(self.handoff_root):04o}",
            "active_key_count": active_count,
            "revoked_key_count": revoked_count,
            "integrity_check": "ok",
            "foreign_key_check": "ok",
            "raw_key_columns_absent": True,
        }

    def verify(self, supplied: str) -> CredentialVerification:
        """Verify one supplied key without retaining it in the result."""

        pepper = self._read_pepper()
        parsed = parse_api_key(supplied)
        if parsed is None:
            execute_dummy_verification(pepper, supplied)
            return CredentialVerification(False, "malformed")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT key_id, label, verifier, verifier_algorithm, status
                FROM api_keys WHERE key_id = ?
                """,
                (parsed.key_id,),
            ).fetchone()
        if row is None:
            execute_dummy_verification(pepper, supplied)
            return CredentialVerification(False, "unknown")
        if str(row["verifier_algorithm"]) != VERIFIER_ALGORITHM:
            raise CredentialStoreError("credential verifier algorithm is invalid")
        candidate = compute_verifier(pepper, supplied)
        matched = verifier_matches(bytes(row["verifier"]), candidate)
        if not matched:
            return CredentialVerification(False, "verifier_mismatch")
        if str(row["status"]) != "ACTIVE":
            return CredentialVerification(False, "revoked")
        return CredentialVerification(
            True,
            "accepted",
            key_id=str(row["key_id"]),
            label=str(row["label"]),
        )
