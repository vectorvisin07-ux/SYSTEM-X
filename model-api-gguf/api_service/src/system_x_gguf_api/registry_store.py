"""Atomic branch-local SQLite persistence for the GGUF model registry."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any, Callable, TypeVar
import uuid

from .registry_types import (
    ArtifactBundleEvidence,
    CapabilityEvidence,
    HashCache,
    ModelState,
    PhysicalIdentity,
    REGISTRY_SCHEMA_IDENTITY,
    REGISTRY_SCHEMA_VERSION,
    RouterModelEvidence,
    canonical_json,
    utc_now,
)


_T = TypeVar("_T")

MODEL_STATES_SQL = (
    "'DISCOVERED','PENDING_STABILITY','VALIDATING','REGISTERED','PROBING',"
    "'READY','UNAVAILABLE','REJECTED','REPLACED','REMOVED'"
)
ROLE_HINTS_SQL = (
    "'primary','shard','mmproj_sidecar','mtp_sidecar','other_gguf'"
)
ALIAS_KINDS_SQL = "'default','manual','compatibility'"
STREAMING_PROTOCOL_SURFACES = frozenset(
    {
        "system-x.streaming.v1",
        "system-x.openai-streaming.v1",
        "system-x.anthropic-streaming.v1",
    }
)

ALLOWED_TRANSITIONS = {
    ModelState.DISCOVERED: {
        ModelState.PENDING_STABILITY,
        ModelState.VALIDATING,
    },
    ModelState.PENDING_STABILITY: {ModelState.VALIDATING},
    ModelState.VALIDATING: {
        ModelState.REGISTERED,
        ModelState.REJECTED,
    },
    ModelState.REGISTERED: {ModelState.PROBING, ModelState.READY},
    ModelState.PROBING: {ModelState.READY, ModelState.UNAVAILABLE},
    ModelState.READY: {
        ModelState.REPLACED,
        ModelState.REMOVED,
        ModelState.UNAVAILABLE,
    },
    ModelState.UNAVAILABLE: {ModelState.PROBING},
    ModelState.REPLACED: {ModelState.REMOVED},
}


class RegistryStoreError(RuntimeError):
    """A bounded schema, transaction, or persistence failure."""


class AliasTransactionConflict(RegistryStoreError):
    """A compare-and-swap predicate changed or cannot be authenticated."""


class RegistryStore:
    """Serialize every mutation through one asyncio writer lock."""

    def __init__(self, database_path: Path, busy_timeout_milliseconds: int) -> None:
        if not 100 <= busy_timeout_milliseconds <= 60_000:
            raise ValueError("SQLite busy timeout is out of bounds")
        self.database_path = database_path
        self.busy_timeout_milliseconds = busy_timeout_milliseconds
        self._writer_lock = asyncio.Lock()
        self._initialized = False

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            connection = sqlite3.connect(
                f"file:{self.database_path}?mode=ro",
                uri=True,
                isolation_level=None,
                timeout=self.busy_timeout_milliseconds / 1000.0,
            )
        else:
            connection = sqlite3.connect(
                self.database_path,
                isolation_level=None,
                timeout=self.busy_timeout_milliseconds / 1000.0,
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f"PRAGMA busy_timeout = {self.busy_timeout_milliseconds}"
        )
        if not read_only:
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                connection.close()
                raise RegistryStoreError("SQLite did not enter WAL mode")
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    @staticmethod
    def _schema_sql() -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS registry_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifact_bundles (
            bundle_id TEXT PRIMARY KEY,
            bundle_sha256 TEXT UNIQUE NOT NULL
                CHECK(length(bundle_sha256) = 64),
            bundle_kind TEXT NOT NULL
                CHECK(bundle_kind IN ('single_file','directory_bundle')),
            file_count INTEGER NOT NULL CHECK(file_count >= 1),
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 24),
            manifest_json TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifact_files (
            bundle_id TEXT NOT NULL REFERENCES artifact_bundles(bundle_id),
            relative_path TEXT NOT NULL,
            file_sha256 TEXT NOT NULL CHECK(length(file_sha256) = 64),
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 24),
            gguf_version INTEGER NOT NULL CHECK(gguf_version IN (2,3)),
            tensor_count INTEGER NOT NULL CHECK(tensor_count >= 0),
            metadata_kv_count INTEGER NOT NULL CHECK(metadata_kv_count >= 0),
            role_hint TEXT NOT NULL CHECK(role_hint IN ({ROLE_HINTS_SQL})),
            PRIMARY KEY (bundle_id, relative_path)
        );
        CREATE TABLE IF NOT EXISTS artifact_locations (
            relative_root TEXT PRIMARY KEY,
            current_bundle_id TEXT REFERENCES artifact_bundles(bundle_id),
            present INTEGER NOT NULL CHECK(present IN (0,1)),
            physical_manifest_json TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_versions (
            model_version_id TEXT PRIMARY KEY,
            bundle_id TEXT NOT NULL REFERENCES artifact_bundles(bundle_id),
            router_model_id TEXT NOT NULL,
            router_source TEXT NOT NULL,
            display_name TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ({MODEL_STATES_SQL})),
            router_metadata_json TEXT NOT NULL,
            router_metadata_sha256 TEXT NOT NULL
                CHECK(length(router_metadata_sha256) = 64),
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            UNIQUE (bundle_id, router_model_id)
        );
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT PRIMARY KEY,
            model_version_id TEXT NOT NULL
                REFERENCES model_versions(model_version_id),
            alias_kind TEXT NOT NULL CHECK(alias_kind IN ({ALIAS_KINDS_SQL})),
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_version_locations (
            model_version_id TEXT PRIMARY KEY
                REFERENCES model_versions(model_version_id),
            relative_root TEXT NOT NULL
                REFERENCES artifact_locations(relative_root),
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS alias_bindings (
            alias TEXT PRIMARY KEY
                REFERENCES aliases(alias) ON DELETE CASCADE,
            relative_root TEXT NOT NULL
                REFERENCES artifact_locations(relative_root),
            promotion_policy TEXT NOT NULL
                CHECK(promotion_policy = 'on_ready_same_location'),
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS capability_manifests (
            model_version_id TEXT PRIMARY KEY
                REFERENCES model_versions(model_version_id),
            manifest_json TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
            props_payload_sha256 TEXT
                CHECK(props_payload_sha256 IS NULL OR length(props_payload_sha256) = 64),
            observed_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifact_rejections (
            relative_path TEXT PRIMARY KEY,
            reason_code TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS registry_events (
            event_id TEXT PRIMARY KEY,
            generation INTEGER NOT NULL CHECK(generation >= 1),
            event_type TEXT NOT NULL,
            subject_id TEXT,
            detail_json TEXT NOT NULL,
            created_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS model_versions_bundle_idx
            ON model_versions(bundle_id);
        CREATE INDEX IF NOT EXISTS model_versions_state_idx
            ON model_versions(state);
        CREATE INDEX IF NOT EXISTS model_version_locations_root_idx
            ON model_version_locations(relative_root);
        CREATE INDEX IF NOT EXISTS alias_bindings_root_idx
            ON alias_bindings(relative_root);
        CREATE INDEX IF NOT EXISTS registry_events_generation_idx
            ON registry_events(generation);
        """

    @staticmethod
    def _migrate_v1_to_v2(
        connection: sqlite3.Connection,
        now: str,
    ) -> None:
        versions = connection.execute(
            "SELECT model_version_id,bundle_id FROM model_versions ORDER BY model_version_id"
        ).fetchall()
        for version in versions:
            locations = connection.execute(
                """
                SELECT relative_root
                FROM artifact_locations
                WHERE current_bundle_id=?
                ORDER BY relative_root
                """,
                (version["bundle_id"],),
            ).fetchall()
            if len(locations) != 1:
                raise RegistryStoreError(
                    "v1 model version cannot be mapped to exactly one logical location"
                )
            connection.execute(
                """
                INSERT INTO model_version_locations(
                    model_version_id,relative_root,created_utc,updated_utc
                ) VALUES (?,?,?,?)
                """,
                (
                    version["model_version_id"],
                    locations[0]["relative_root"],
                    now,
                    now,
                ),
            )
        default_alias = connection.execute(
            """
            SELECT a.alias,a.model_version_id,mvl.relative_root
            FROM aliases AS a
            LEFT JOIN model_version_locations AS mvl
              ON mvl.model_version_id=a.model_version_id
            LEFT JOIN artifact_locations AS al
              ON al.relative_root=mvl.relative_root
             AND al.present=1
            WHERE a.alias='default'
              AND al.relative_root IS NOT NULL
            """
        ).fetchall()
        if len(default_alias) != 1:
            raise RegistryStoreError(
                "v1 default alias cannot be mapped to exactly one present location"
            )
        connection.execute(
            """
            INSERT INTO alias_bindings(
                alias,relative_root,promotion_policy,created_utc,updated_utc
            ) VALUES (?,?,?,?,?)
            """,
            (
                default_alias[0]["alias"],
                default_alias[0]["relative_root"],
                "on_ready_same_location",
                now,
                now,
            ),
        )

    def _initialize_sync(
        self,
        inject_migration_failure_before_commit: bool = False,
    ) -> dict[str, Any]:
        parent = self.database_path.parent
        parent_info = parent.lstat()
        if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
            raise RegistryStoreError("database root is not a direct directory")
        if stat.S_IMODE(parent_info.st_mode) != 0o700:
            raise RegistryStoreError("database root mode is not 0700")
        created = False
        if os.path.lexists(self.database_path):
            info = self.database_path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise RegistryStoreError("database target is not a direct regular file")
        else:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(self.database_path, flags, 0o600)
            os.fsync(descriptor)
            os.close(descriptor)
            created = True
        os.chmod(self.database_path, 0o600)
        connection = self._connect()
        try:
            connection.executescript("BEGIN IMMEDIATE;\n" + self._schema_sql())
            try:
                now = utc_now()
                metadata_before = dict(
                    connection.execute(
                        "SELECT key,value FROM registry_metadata"
                    ).fetchall()
                )
                migrated_from: int | None = None
                if metadata_before:
                    try:
                        previous_version = int(metadata_before["schema_version"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RegistryStoreError(
                            "registry schema metadata is invalid"
                        ) from exc
                    previous_identity = metadata_before.get("schema_identity")
                    if (
                        previous_version == 1
                        and previous_identity == "system-x.gguf-model-registry.v1"
                    ):
                        self._migrate_v1_to_v2(connection, now)
                        migrated_from = 1
                        if inject_migration_failure_before_commit:
                            raise RegistryStoreError(
                                "injected migration failure before commit"
                            )
                        connection.execute(
                            """
                            UPDATE registry_metadata
                            SET value=?
                            WHERE key='schema_identity'
                            """,
                            (REGISTRY_SCHEMA_IDENTITY,),
                        )
                        connection.execute(
                            """
                            UPDATE registry_metadata
                            SET value=?
                            WHERE key='schema_version'
                            """,
                            (str(REGISTRY_SCHEMA_VERSION),),
                        )
                        connection.execute(
                            """
                            UPDATE registry_metadata
                            SET value=?
                            WHERE key='last_migration_utc'
                            """,
                            (now,),
                        )
                    elif (
                        previous_version != REGISTRY_SCHEMA_VERSION
                        or previous_identity != REGISTRY_SCHEMA_IDENTITY
                    ):
                        raise RegistryStoreError(
                            "registry schema identity/version mismatch"
                        )
                required = {
                    "schema_identity": REGISTRY_SCHEMA_IDENTITY,
                    "schema_version": str(REGISTRY_SCHEMA_VERSION),
                    "created_utc": now,
                    "last_migration_utc": now,
                    "registry_generation": "0",
                    "last_reconcile_utc": "",
                }
                for key, value in required.items():
                    connection.execute(
                        "INSERT OR IGNORE INTO registry_metadata(key,value) VALUES (?,?)",
                        (key, value),
                    )
                if inject_migration_failure_before_commit and migrated_from is None:
                    raise RegistryStoreError(
                        "migration failure injection requires a v1 database"
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            metadata = dict(
                connection.execute(
                    "SELECT key,value FROM registry_metadata"
                ).fetchall()
            )
            if (
                metadata.get("schema_identity") != REGISTRY_SCHEMA_IDENTITY
                or metadata.get("schema_version") != str(REGISTRY_SCHEMA_VERSION)
            ):
                raise RegistryStoreError("registry schema identity/version mismatch")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
            if integrity != "ok" or foreign_keys != 1 or str(journal_mode).lower() != "wal":
                raise RegistryStoreError("SQLite initialization invariants failed")
            return {
                "created": created,
                "database_path": str(self.database_path),
                "database_mode": oct(stat.S_IMODE(self.database_path.stat().st_mode)),
                "integrity_check": integrity,
                "foreign_keys": foreign_keys,
                "journal_mode": journal_mode,
                "synchronous": synchronous,
                "busy_timeout": connection.execute(
                    "PRAGMA busy_timeout"
                ).fetchone()[0],
                "schema_identity": metadata["schema_identity"],
                "schema_version": int(metadata["schema_version"]),
                "migrated_from": migrated_from,
            }
        finally:
            connection.close()

    async def initialize(
        self,
        *,
        inject_migration_failure_before_commit: bool = False,
    ) -> dict[str, Any]:
        async with self._writer_lock:
            result = await asyncio.to_thread(
                self._initialize_sync,
                inject_migration_failure_before_commit,
            )
            self._initialized = True
            return result

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RegistryStoreError("registry store is not initialized")

    @staticmethod
    def _next_generation(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM registry_metadata WHERE key='registry_generation'"
        ).fetchone()
        if row is None:
            raise RegistryStoreError("registry generation metadata is missing")
        generation = int(row[0]) + 1
        connection.execute(
            "UPDATE registry_metadata SET value=? WHERE key='registry_generation'",
            (str(generation),),
        )
        return generation

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        generation: int,
        event_type: str,
        subject_id: str | None,
        detail: dict[str, Any],
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO registry_events(
                event_id,generation,event_type,subject_id,detail_json,created_utc
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                f"event-{uuid.uuid4()}",
                generation,
                event_type,
                subject_id,
                canonical_json(detail),
                now,
            ),
        )

    def _write_sync(self, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = callback(connection)
                connection.execute("COMMIT")
                return result
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    async def _write(self, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        self._require_initialized()
        async with self._writer_lock:
            return await asyncio.to_thread(self._write_sync, callback)

    def _read_sync(self, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        connection = self._connect(read_only=True)
        try:
            connection.execute("BEGIN")
            try:
                result = callback(connection)
                connection.execute("COMMIT")
                return result
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    async def _read(self, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        self._require_initialized()
        return await asyncio.to_thread(self._read_sync, callback)

    async def register_bundle(
        self,
        bundle: ArtifactBundleEvidence,
        router: RouterModelEvidence,
        model_version_id: str,
        *,
        inject_failure_before_commit: bool = False,
    ) -> dict[str, Any]:
        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            now = utc_now()
            existing_location = connection.execute(
                """
                SELECT current_bundle_id,present,physical_manifest_json
                FROM artifact_locations WHERE relative_root=?
                """,
                (bundle.relative_root,),
            ).fetchone()
            existing_model = connection.execute(
                """
                SELECT
                    mv.model_version_id,
                    mv.state,
                    mv.router_metadata_sha256,
                    mvl.relative_root
                FROM model_versions AS mv
                LEFT JOIN model_version_locations AS mvl
                  ON mvl.model_version_id=mv.model_version_id
                WHERE mv.model_version_id=?
                """,
                (model_version_id,),
            ).fetchone()
            same_location = bool(
                existing_location
                and existing_location["current_bundle_id"] == bundle.bundle_id
                and existing_location["present"] == 1
                and existing_location["physical_manifest_json"]
                == canonical_json(bundle.physical_manifest)
            )
            same_model = bool(
                existing_model
                and existing_model["model_version_id"] == model_version_id
                and existing_model["relative_root"] == bundle.relative_root
                and existing_model["router_metadata_sha256"]
                == router.metadata_sha256
                and existing_model["state"]
                not in {ModelState.REMOVED.value, ModelState.REPLACED.value}
            )
            if same_location and same_model:
                connection.execute(
                    "UPDATE artifact_bundles SET last_seen_utc=? WHERE bundle_id=?",
                    (now, bundle.bundle_id),
                )
                connection.execute(
                    "UPDATE artifact_locations SET last_seen_utc=? WHERE relative_root=?",
                    (now, bundle.relative_root),
                )
                return {
                    "changed": False,
                    "new_version": False,
                    "model_version_id": model_version_id,
                    "generation": int(
                        connection.execute(
                            "SELECT value FROM registry_metadata "
                            "WHERE key='registry_generation'"
                        ).fetchone()[0]
                    ),
                }
            if (
                same_location
                and existing_model
                and existing_model["model_version_id"] == model_version_id
                and existing_model["relative_root"] == bundle.relative_root
                and existing_model["state"]
                not in {ModelState.REMOVED.value, ModelState.REPLACED.value}
            ):
                connection.execute(
                    """
                    UPDATE model_versions SET
                        router_source=?,display_name=?,router_metadata_json=?,
                        router_metadata_sha256=?,updated_utc=?
                    WHERE model_version_id=?
                    """,
                    (
                        router.router_source,
                        router.display_name,
                        router.metadata_json,
                        router.metadata_sha256,
                        now,
                        model_version_id,
                    ),
                )
                connection.execute(
                    "UPDATE artifact_bundles SET last_seen_utc=? WHERE bundle_id=?",
                    (now, bundle.bundle_id),
                )
                connection.execute(
                    "UPDATE artifact_locations SET last_seen_utc=? WHERE relative_root=?",
                    (now, bundle.relative_root),
                )
                generation = self._next_generation(connection)
                self._insert_event(
                    connection,
                    generation,
                    "router_metadata_updated",
                    model_version_id,
                    {"router_status": router.router_status},
                    now,
                )
                return {
                    "changed": True,
                    "new_version": False,
                    "model_version_id": model_version_id,
                    "generation": generation,
                    "state_preserved": existing_model["state"],
                }

            connection.execute(
                """
                INSERT OR IGNORE INTO artifact_bundles(
                    bundle_id,bundle_sha256,bundle_kind,file_count,size_bytes,
                    manifest_json,first_seen_utc,last_seen_utc
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    bundle.bundle_id,
                    bundle.bundle_sha256,
                    bundle.bundle_kind.value,
                    bundle.file_count,
                    bundle.size_bytes,
                    canonical_json(bundle.content_manifest()),
                    now,
                    now,
                ),
            )
            for item in bundle.files:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO artifact_files(
                        bundle_id,relative_path,file_sha256,size_bytes,gguf_version,
                        tensor_count,metadata_kv_count,role_hint
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        bundle.bundle_id,
                        item.relative_path,
                        item.file_sha256,
                        item.size_bytes,
                        item.gguf_version,
                        item.tensor_count,
                        item.metadata_kv_count,
                        item.role_hint.value,
                    ),
                )
            old_bundle_id = (
                existing_location["current_bundle_id"]
                if existing_location is not None
                else None
            )
            connection.execute(
                """
                INSERT INTO artifact_locations(
                    relative_root,current_bundle_id,present,physical_manifest_json,
                    first_seen_utc,last_seen_utc
                ) VALUES (?,?,1,?,?,?)
                ON CONFLICT(relative_root) DO UPDATE SET
                    current_bundle_id=excluded.current_bundle_id,
                    present=1,
                    physical_manifest_json=excluded.physical_manifest_json,
                    last_seen_utc=excluded.last_seen_utc
                """,
                (
                    bundle.relative_root,
                    bundle.bundle_id,
                    canonical_json(bundle.physical_manifest),
                    now,
                    now,
                ),
            )
            replaced_count = 0
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO model_versions(
                    model_version_id,bundle_id,router_model_id,router_source,
                    display_name,state,router_metadata_json,router_metadata_sha256,
                    created_utc,updated_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    model_version_id,
                    bundle.bundle_id,
                    router.router_model_id,
                    router.router_source,
                    router.display_name,
                    ModelState.REGISTERED.value,
                    router.metadata_json,
                    router.metadata_sha256,
                    now,
                    now,
                ),
            ).rowcount
            revived_terminal = bool(
                not inserted
                and existing_model is not None
                and existing_model["relative_root"] == bundle.relative_root
                and existing_model["state"]
                in {
                    ModelState.REMOVED.value,
                    ModelState.REPLACED.value,
                }
            )
            if not inserted:
                preserved_state = (
                    existing_model["state"]
                    if existing_model is not None
                    and existing_model["relative_root"] == bundle.relative_root
                    and existing_model["state"]
                    not in {
                        ModelState.REMOVED.value,
                        ModelState.REPLACED.value,
                    }
                    else ModelState.REGISTERED.value
                )
                connection.execute(
                    """
                    UPDATE model_versions SET
                        router_source=?,display_name=?,state=?,
                        router_metadata_json=?,router_metadata_sha256=?,updated_utc=?
                    WHERE model_version_id=?
                    """,
                    (
                        router.router_source,
                        router.display_name,
                        preserved_state,
                        router.metadata_json,
                        router.metadata_sha256,
                        now,
                        model_version_id,
                    ),
                )
            if revived_terminal:
                connection.execute(
                    "DELETE FROM capability_manifests WHERE model_version_id=?",
                    (model_version_id,),
                )
            location_membership = connection.execute(
                """
                SELECT relative_root
                FROM model_version_locations
                WHERE model_version_id=?
                """,
                (model_version_id,),
            ).fetchone()
            if (
                location_membership is not None
                and location_membership["relative_root"] != bundle.relative_root
            ):
                raise RegistryStoreError(
                    "model version already belongs to another logical location"
                )
            connection.execute(
                """
                INSERT INTO model_version_locations(
                    model_version_id,relative_root,created_utc,updated_utc
                ) VALUES (?,?,?,?)
                ON CONFLICT(model_version_id) DO UPDATE SET
                    updated_utc=excluded.updated_utc
                """,
                (model_version_id, bundle.relative_root, now, now),
            )
            connection.execute(
                "DELETE FROM artifact_rejections WHERE relative_path=?",
                (bundle.relative_root,),
            )
            if inject_failure_before_commit:
                raise RuntimeError("injected pre-commit registry failure")
            generation = self._next_generation(connection)
            event_type = (
                "replacement_candidate_registered"
                if old_bundle_id and old_bundle_id != bundle.bundle_id
                else "model_registered"
                if inserted
                else "model_registration_refreshed"
            )
            self._insert_event(
                connection,
                generation,
                event_type,
                model_version_id,
                {
                    "bundle_id": bundle.bundle_id,
                    "relative_root": bundle.relative_root,
                    "file_count": bundle.file_count,
                    "replaced_model_count": replaced_count,
                    "replacement_deferred_until_ready": bool(
                        old_bundle_id and old_bundle_id != bundle.bundle_id
                    ),
                    "capability_reprobe_required": revived_terminal,
                },
                now,
            )
            return {
                "changed": True,
                "new_version": bool(inserted),
                "model_version_id": model_version_id,
                "generation": generation,
                "replaced_model_count": replaced_count,
            }

        return await self._write(transaction)

    async def transition_state(
        self,
        model_version_id: str,
        state: ModelState,
        event_type: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            row = connection.execute(
                "SELECT state FROM model_versions WHERE model_version_id=?",
                (model_version_id,),
            ).fetchone()
            if row is None:
                raise RegistryStoreError("model version does not exist")
            current = ModelState(row["state"])
            if current == state:
                generation = int(
                    connection.execute(
                        "SELECT value FROM registry_metadata "
                        "WHERE key='registry_generation'"
                    ).fetchone()[0]
                )
                return {"changed": False, "generation": generation, "state": state.value}
            if state not in ALLOWED_TRANSITIONS.get(current, set()):
                raise RegistryStoreError(
                    f"invalid model-state transition: {current.value}->{state.value}"
                )
            now = utc_now()
            connection.execute(
                "UPDATE model_versions SET state=?,updated_utc=? "
                "WHERE model_version_id=?",
                (state.value, now, model_version_id),
            )
            generation = self._next_generation(connection)
            self._insert_event(
                connection,
                generation,
                event_type,
                model_version_id,
                {"from": current.value, "to": state.value, **(detail or {})},
                now,
            )
            return {"changed": True, "generation": generation, "state": state.value}

        return await self._write(transaction)

    async def store_capability_ready(
        self,
        capability: CapabilityEvidence,
        default_alias: str,
        *,
        inject_failure_before_commit: bool = False,
    ) -> dict[str, Any]:
        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            row = connection.execute(
                """
                SELECT
                    mv.state,
                    mv.bundle_id,
                    mvl.relative_root,
                    al.current_bundle_id,
                    al.present
                FROM model_versions AS mv
                JOIN model_version_locations AS mvl
                  ON mvl.model_version_id=mv.model_version_id
                JOIN artifact_locations AS al
                  ON al.relative_root=mvl.relative_root
                WHERE mv.model_version_id=?
                """,
                (capability.model_version_id,),
            ).fetchone()
            if row is None:
                raise RegistryStoreError("capability model version does not exist")
            if (
                row["present"] != 1
                or row["current_bundle_id"] != row["bundle_id"]
            ):
                raise RegistryStoreError(
                    "capability model is not current at its logical location"
                )
            current = ModelState(row["state"])
            existing = connection.execute(
                """
                SELECT manifest_sha256,props_payload_sha256
                FROM capability_manifests WHERE model_version_id=?
                """,
                (capability.model_version_id,),
            ).fetchone()
            if current not in {
                ModelState.REGISTERED,
                ModelState.PROBING,
                ModelState.UNAVAILABLE,
                ModelState.READY,
            }:
                raise RegistryStoreError(
                    f"capability cannot make state ready from {current.value}"
                )
            alias_count = connection.execute(
                "SELECT count(*) FROM aliases"
            ).fetchone()[0]
            bound_before = connection.execute(
                """
                SELECT a.model_version_id
                FROM alias_bindings AS ab
                JOIN aliases AS a ON a.alias=ab.alias
                WHERE ab.relative_root=?
                  AND ab.promotion_policy='on_ready_same_location'
                """,
                (row["relative_root"],),
            ).fetchall()
            same_manifest = bool(
                existing
                and existing["manifest_sha256"] == capability.manifest_sha256
                and existing["props_payload_sha256"]
                == capability.props_payload_sha256
            )
            if (
                current == ModelState.READY
                and same_manifest
                and alias_count > 0
                and all(
                    alias["model_version_id"] == capability.model_version_id
                    for alias in bound_before
                )
            ):
                generation = int(
                    connection.execute(
                        "SELECT value FROM registry_metadata "
                        "WHERE key='registry_generation'"
                    ).fetchone()[0]
                )
                return {
                    "changed": False,
                    "generation": generation,
                    "alias_created": False,
                    "promoted_aliases": [],
                    "replaced_model_count": 0,
                }
            now = utc_now()
            connection.execute(
                """
                INSERT INTO capability_manifests(
                    model_version_id,manifest_json,manifest_sha256,
                    props_payload_sha256,observed_utc
                ) VALUES (?,?,?,?,?)
                ON CONFLICT(model_version_id) DO UPDATE SET
                    manifest_json=excluded.manifest_json,
                    manifest_sha256=excluded.manifest_sha256,
                    props_payload_sha256=excluded.props_payload_sha256,
                    observed_utc=excluded.observed_utc
                """,
                (
                    capability.model_version_id,
                    capability.manifest_json,
                    capability.manifest_sha256,
                    capability.props_payload_sha256,
                    capability.observed_utc,
                ),
            )
            connection.execute(
                "UPDATE model_versions SET state=?,updated_utc=? "
                "WHERE model_version_id=?",
                (ModelState.READY.value, now, capability.model_version_id),
            )
            alias_created = False
            if alias_count == 0:
                connection.execute(
                    """
                    INSERT INTO aliases(
                        alias,model_version_id,alias_kind,created_utc,updated_utc
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        default_alias,
                        capability.model_version_id,
                        "default",
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO alias_bindings(
                        alias,relative_root,promotion_policy,created_utc,updated_utc
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        default_alias,
                        row["relative_root"],
                        "on_ready_same_location",
                        now,
                        now,
                    ),
                )
                alias_created = True
            bound_aliases = connection.execute(
                """
                SELECT
                    ab.alias,
                    ab.promotion_policy,
                    a.model_version_id AS old_model_version_id
                FROM alias_bindings AS ab
                JOIN aliases AS a ON a.alias=ab.alias
                WHERE ab.relative_root=?
                  AND ab.promotion_policy='on_ready_same_location'
                ORDER BY ab.alias
                """,
                (row["relative_root"],),
            ).fetchall()
            aliases_to_move = [
                alias
                for alias in bound_aliases
                if alias["old_model_version_id"] != capability.model_version_id
            ]
            old_targets = sorted(
                {
                    str(alias["old_model_version_id"])
                    for alias in aliases_to_move
                }
            )
            for alias in aliases_to_move:
                connection.execute(
                    """
                    UPDATE aliases
                    SET model_version_id=?,updated_utc=?
                    WHERE alias=?
                    """,
                    (
                        capability.model_version_id,
                        now,
                        alias["alias"],
                    ),
                )
                connection.execute(
                    """
                    UPDATE alias_bindings
                    SET updated_utc=?
                    WHERE alias=?
                    """,
                    (now, alias["alias"]),
                )
            replaced_count = 0
            for old_target in old_targets:
                replaced_count += connection.execute(
                    """
                    UPDATE model_versions
                    SET state='REPLACED',updated_utc=?
                    WHERE model_version_id=?
                      AND state IN ('READY','UNAVAILABLE')
                      AND EXISTS (
                        SELECT 1
                        FROM model_version_locations
                        WHERE model_version_id=?
                          AND relative_root=?
                      )
                    """,
                    (
                        now,
                        old_target,
                        old_target,
                        row["relative_root"],
                    ),
                ).rowcount
            if inject_failure_before_commit:
                raise RuntimeError("injected promotion failure before commit")
            generation = self._next_generation(connection)
            self._insert_event(
                connection,
                generation,
                "replacement_ready" if bound_aliases else "capability_ready",
                capability.model_version_id,
                {
                    "from": current.value,
                    "to": ModelState.READY.value,
                    "default_alias_created": alias_created,
                    "replaced_model_count": replaced_count,
                },
                now,
            )
            for alias in bound_aliases:
                if (
                    alias["old_model_version_id"] != capability.model_version_id
                    or alias_created
                ):
                    self._insert_event(
                        connection,
                        generation,
                        "alias_promoted",
                        str(alias["alias"]),
                        {
                            "old_model_version_id": alias["old_model_version_id"],
                            "new_model_version_id": capability.model_version_id,
                            "alias": alias["alias"],
                            "promotion_policy": alias["promotion_policy"],
                            "registry_generation": generation,
                        },
                        now,
                    )
            return {
                "changed": True,
                "generation": generation,
                "alias_created": alias_created,
                "promoted_aliases": [
                    str(alias["alias"]) for alias in aliases_to_move
                ],
                "replaced_model_count": replaced_count,
            }

        return await self._write(transaction)

    async def create_bound_alias(
        self,
        alias: str,
        model_version_id: str,
        *,
        alias_kind: str = "manual",
        promotion_policy: str = "on_ready_same_location",
    ) -> dict[str, Any]:
        if (
            not alias
            or len(alias) > 64
            or alias[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
                for character in alias
            )
        ):
            raise RegistryStoreError("alias is invalid")
        if alias_kind not in {"default", "manual", "compatibility"}:
            raise RegistryStoreError("alias kind is invalid")
        if promotion_policy != "on_ready_same_location":
            raise RegistryStoreError("promotion policy is invalid")

        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            target = connection.execute(
                """
                SELECT
                    mv.state,
                    mv.bundle_id,
                    mvl.relative_root,
                    al.current_bundle_id,
                    al.present,
                    cm.model_version_id AS capability_model_version_id
                FROM model_versions AS mv
                JOIN model_version_locations AS mvl
                  ON mvl.model_version_id=mv.model_version_id
                JOIN artifact_locations AS al
                  ON al.relative_root=mvl.relative_root
                LEFT JOIN capability_manifests AS cm
                  ON cm.model_version_id=mv.model_version_id
                WHERE mv.model_version_id=?
                """,
                (model_version_id,),
            ).fetchone()
            if (
                target is None
                or target["state"] != ModelState.READY.value
                or target["present"] != 1
                or target["current_bundle_id"] != target["bundle_id"]
                or target["capability_model_version_id"] != model_version_id
            ):
                raise RegistryStoreError(
                    "bound alias target is not a current ready model"
                )
            existing = connection.execute(
                """
                SELECT
                    a.model_version_id,
                    a.alias_kind,
                    ab.relative_root,
                    ab.promotion_policy
                FROM aliases AS a
                LEFT JOIN alias_bindings AS ab ON ab.alias=a.alias
                WHERE a.alias=?
                """,
                (alias,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["model_version_id"] == model_version_id
                    and existing["alias_kind"] == alias_kind
                    and existing["relative_root"] == target["relative_root"]
                    and existing["promotion_policy"] == promotion_policy
                ):
                    generation = int(
                        connection.execute(
                            "SELECT value FROM registry_metadata "
                            "WHERE key='registry_generation'"
                        ).fetchone()[0]
                    )
                    return {"changed": False, "generation": generation}
                raise RegistryStoreError("alias already exists with another binding")
            now = utc_now()
            connection.execute(
                """
                INSERT INTO aliases(
                    alias,model_version_id,alias_kind,created_utc,updated_utc
                ) VALUES (?,?,?,?,?)
                """,
                (alias, model_version_id, alias_kind, now, now),
            )
            connection.execute(
                """
                INSERT INTO alias_bindings(
                    alias,relative_root,promotion_policy,created_utc,updated_utc
                ) VALUES (?,?,?,?,?)
                """,
                (
                    alias,
                    target["relative_root"],
                    promotion_policy,
                    now,
                    now,
                ),
            )
            generation = self._next_generation(connection)
            self._insert_event(
                connection,
                generation,
                "alias_binding_created",
                alias,
                {
                    "alias": alias,
                    "model_version_id": model_version_id,
                    "promotion_policy": promotion_policy,
                },
                now,
            )
            return {"changed": True, "generation": generation}

        return await self._write(transaction)

    async def remove_alias(self, alias: str) -> dict[str, Any]:
        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            existing = connection.execute(
                "SELECT model_version_id FROM aliases WHERE alias=?",
                (alias,),
            ).fetchone()
            if existing is None:
                generation = int(
                    connection.execute(
                        "SELECT value FROM registry_metadata "
                        "WHERE key='registry_generation'"
                    ).fetchone()[0]
                )
                return {"changed": False, "generation": generation}
            now = utc_now()
            connection.execute("DELETE FROM aliases WHERE alias=?", (alias,))
            generation = self._next_generation(connection)
            self._insert_event(
                connection,
                generation,
                "alias_removed",
                alias,
                {
                    "alias": alias,
                    "model_version_id": existing["model_version_id"],
                },
                now,
            )
            return {"changed": True, "generation": generation}

        return await self._write(transaction)

    async def compare_and_swap_default_alias(
        self,
        *,
        action: str,
        promotion_transaction_id: str,
        alias: str,
        expected_current_target: str | None,
        new_target: str | None,
        expected_registry_generation: int,
        target_artifact_version_id: str | None,
        target_capability_manifest_identity: str | None,
        target_relative_root: str | None,
        promotion_alias_event_identity: str | None = None,
    ) -> dict[str, Any]:
        """Atomically promote or roll back the private ``default`` alias.

        This is deliberately narrower than the automatic same-location alias
        policy.  It is an administrative, transaction-correlated CAS used only
        after the Inspector has authenticated a qualification result.
        """

        def bounded_identifier(value: object, name: str, maximum: int = 256) -> str:
            if (
                not isinstance(value, str)
                or not value
                or len(value) > maximum
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for character in value)
            ):
                raise RegistryStoreError(f"alias transaction {name} is invalid")
            return value

        if action not in {"promote", "rollback", "clear"}:
            raise RegistryStoreError("alias transaction action is invalid")
        transaction_id = bounded_identifier(
            promotion_transaction_id,
            "promotion transaction ID",
            160,
        )
        if alias != "default":
            raise RegistryStoreError("alias transaction is restricted to default")
        if expected_current_target is not None:
            expected_current_target = bounded_identifier(
                expected_current_target,
                "expected current target",
            )
        if new_target is not None:
            new_target = bounded_identifier(new_target, "new target")
        if new_target is None and (
            action not in {"rollback", "clear"}
            or expected_current_target is None
        ):
            raise RegistryStoreError(
                "only first-model rollback may remove default"
            )
        if expected_current_target == new_target:
            raise RegistryStoreError("alias transaction targets are identical")
        if (
            isinstance(expected_registry_generation, bool)
            or not isinstance(expected_registry_generation, int)
            or expected_registry_generation < 0
        ):
            raise RegistryStoreError(
                "alias transaction expected registry generation is invalid"
            )
        if new_target is None:
            if any(
                value is not None
                for value in (
                    target_artifact_version_id,
                    target_capability_manifest_identity,
                    target_relative_root,
                )
            ):
                raise RegistryStoreError(
                    "first-model rollback target identity must be absent"
                )
        else:
            target_artifact_version_id = bounded_identifier(
                target_artifact_version_id,
                "target artifact version",
            )
            if (
                not target_artifact_version_id.startswith("bundle-")
                or len(target_artifact_version_id) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in target_artifact_version_id[7:]
                )
            ):
                raise RegistryStoreError(
                    "alias transaction target artifact version is invalid"
                )
            if (
                not isinstance(target_capability_manifest_identity, str)
                or not target_capability_manifest_identity.startswith("sha256:")
                or len(target_capability_manifest_identity) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in target_capability_manifest_identity[7:]
                )
            ):
                raise RegistryStoreError(
                    "alias transaction target capability identity is invalid"
                )
            if (
                not isinstance(target_relative_root, str)
                or not target_relative_root
                or len(target_relative_root) > 512
                or target_relative_root.startswith("/")
                or "\x00" in target_relative_root
                or any(part in {"", ".", ".."} for part in Path(target_relative_root).parts)
            ):
                raise RegistryStoreError(
                    "alias transaction target relative root is invalid"
                )
        if action in {"promote", "clear"}:
            if promotion_alias_event_identity is not None:
                raise RegistryStoreError(
                    "promotion or clear must not supply a prior alias event identity"
                )
        else:
            if (
                not isinstance(promotion_alias_event_identity, str)
                or not promotion_alias_event_identity.startswith("sha256:")
                or len(promotion_alias_event_identity) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in promotion_alias_event_identity[7:]
                )
            ):
                raise RegistryStoreError(
                    "rollback promotion alias event identity is invalid"
                )

        event_basis = {
            "action": action,
            "alias": alias,
            "expected_current_target": expected_current_target,
            "expected_registry_generation": expected_registry_generation,
            "new_target": new_target,
            "promotion_alias_event_identity": promotion_alias_event_identity,
            "promotion_transaction_id": transaction_id,
            "target_artifact_version_id": target_artifact_version_id,
            "target_capability_manifest_identity": (
                target_capability_manifest_identity
            ),
            "target_relative_root": target_relative_root,
        }
        alias_event_identity = "sha256:" + hashlib.sha256(
            canonical_json(event_basis).encode("utf-8")
        ).hexdigest()

        def current_generation(connection: sqlite3.Connection) -> int:
            row = connection.execute(
                """
                SELECT value FROM registry_metadata
                WHERE key='registry_generation'
                """
            ).fetchone()
            if row is None:
                raise RegistryStoreError(
                    "registry generation metadata is missing"
                )
            return int(row[0])

        def active_target(
            connection: sqlite3.Connection,
            model_version_id: str,
            *,
            require_ready: bool,
        ) -> sqlite3.Row:
            row = connection.execute(
                """
                SELECT
                    mv.model_version_id,
                    mv.bundle_id,
                    mv.state,
                    mvl.relative_root,
                    al.current_bundle_id,
                    al.present,
                    cm.manifest_sha256
                FROM model_versions AS mv
                JOIN model_version_locations AS mvl
                  ON mvl.model_version_id=mv.model_version_id
                JOIN artifact_locations AS al
                  ON al.relative_root=mvl.relative_root
                JOIN capability_manifests AS cm
                  ON cm.model_version_id=mv.model_version_id
                WHERE mv.model_version_id=?
                """,
                (model_version_id,),
            ).fetchone()
            if (
                row is None
                or row["present"] != 1
                or row["current_bundle_id"] != row["bundle_id"]
                or (require_ready and row["state"] != ModelState.READY.value)
            ):
                raise AliasTransactionConflict(
                    "alias transaction model target is not active"
                )
            return row

        def prior_event(
            connection: sqlite3.Connection,
            event_identity: str,
        ) -> dict[str, Any] | None:
            rows = connection.execute(
                """
                SELECT detail_json
                FROM registry_events
                WHERE event_type='default_alias_promoted'
                  AND subject_id=?
                ORDER BY generation DESC
                """,
                (alias,),
            ).fetchall()
            for row in rows:
                try:
                    detail = json.loads(str(row["detail_json"]))
                except (json.JSONDecodeError, TypeError):
                    continue
                if (
                    isinstance(detail, dict)
                    and detail.get("alias_event_identity") == event_identity
                ):
                    return detail
            return None

        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            observed_generation = current_generation(connection)
            existing_alias = connection.execute(
                """
                SELECT a.model_version_id,a.alias_kind,ab.relative_root
                FROM aliases AS a
                LEFT JOIN alias_bindings AS ab ON ab.alias=a.alias
                WHERE a.alias=?
                """,
                (alias,),
            ).fetchone()

            # A completed exact request is idempotent even though its expected
            # pre-mutation generation and alias target no longer apply.
            event_rows = connection.execute(
                """
                SELECT generation,event_type,detail_json
                FROM registry_events
                WHERE subject_id=?
                  AND event_type IN (
                      'default_alias_promoted',
                      'default_alias_rolled_back',
                      'default_alias_cleared'
                  )
                ORDER BY generation DESC
                """,
                (alias,),
            ).fetchall()
            for event_row in event_rows:
                try:
                    detail = json.loads(str(event_row["detail_json"]))
                except (json.JSONDecodeError, TypeError):
                    continue
                if (
                    isinstance(detail, dict)
                    and detail.get("alias_event_identity") == alias_event_identity
                    and detail.get("event_basis") == event_basis
                    and (
                        (
                            new_target is None
                            and existing_alias is None
                        )
                        or (
                            existing_alias is not None
                            and existing_alias["model_version_id"]
                            == new_target
                            and existing_alias["relative_root"]
                            == target_relative_root
                        )
                    )
                ):
                    return {
                        "action": action,
                        "alias": alias,
                        "alias_event_identity": alias_event_identity,
                        "changed": False,
                        "new_registry_generation": int(event_row["generation"]),
                        "new_target": new_target,
                        "observed_registry_generation": observed_generation,
                        "previous_target": expected_current_target,
                        "promotion_transaction_id": transaction_id,
                    }

            if observed_generation != expected_registry_generation:
                raise AliasTransactionConflict(
                    "alias transaction registry generation changed"
                )
            observed_target = (
                str(existing_alias["model_version_id"])
                if existing_alias is not None
                else None
            )
            if observed_target != expected_current_target:
                raise AliasTransactionConflict(
                    "alias transaction current target changed"
                )
            if existing_alias is not None and (
                existing_alias["alias_kind"] != "default"
                or existing_alias["relative_root"] is None
            ):
                raise AliasTransactionConflict(
                    "default alias binding is incomplete"
                )

            target = None
            if new_target is not None:
                target = active_target(
                    connection,
                    new_target,
                    require_ready=True,
                )
                if (
                    target["bundle_id"] != target_artifact_version_id
                    or "sha256:" + str(target["manifest_sha256"])
                    != target_capability_manifest_identity
                    or target["relative_root"] != target_relative_root
                ):
                    raise AliasTransactionConflict(
                        "alias transaction target identity changed"
                    )

            if expected_current_target is not None:
                current = active_target(
                    connection,
                    expected_current_target,
                    require_ready=(action == "promote"),
                )
                if existing_alias["relative_root"] != current["relative_root"]:
                    raise AliasTransactionConflict(
                        "alias transaction current binding changed"
                    )

            if action == "rollback":
                promotion_event = prior_event(
                    connection,
                    str(promotion_alias_event_identity),
                )
                if (
                    promotion_event is None
                    or promotion_event.get("promotion_transaction_id")
                    != transaction_id
                    or promotion_event.get("previous_target") != new_target
                    or promotion_event.get("new_target")
                    != expected_current_target
                ):
                    raise AliasTransactionConflict(
                        "rollback promotion event is not authentic"
                    )

            now = utc_now()
            if existing_alias is None:
                if action != "promote" or expected_current_target is not None:
                    raise AliasTransactionConflict(
                        "only first-model promotion may create default"
                    )
                connection.execute(
                    """
                    INSERT INTO aliases(
                        alias,model_version_id,alias_kind,created_utc,updated_utc
                    ) VALUES (?,?,?,?,?)
                    """,
                    (alias, new_target, "default", now, now),
                )
                connection.execute(
                    """
                    INSERT INTO alias_bindings(
                        alias,relative_root,promotion_policy,
                        created_utc,updated_utc
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        alias,
                        target_relative_root,
                        "on_ready_same_location",
                        now,
                        now,
                    ),
                )
            elif new_target is None:
                deleted = connection.execute(
                    """
                    DELETE FROM aliases
                    WHERE alias=? AND model_version_id=?
                    """,
                    (alias, expected_current_target),
                ).rowcount
                if deleted != 1:
                    raise AliasTransactionConflict(
                        "alias transaction compare-and-swap failed"
                    )
            else:
                alias_changes = connection.execute(
                    """
                    UPDATE aliases
                    SET model_version_id=?,updated_utc=?
                    WHERE alias=? AND model_version_id=?
                    """,
                    (
                        new_target,
                        now,
                        alias,
                        expected_current_target,
                    ),
                ).rowcount
                binding_changes = connection.execute(
                    """
                    UPDATE alias_bindings
                    SET relative_root=?,updated_utc=?
                    WHERE alias=? AND relative_root=?
                    """,
                    (
                        target_relative_root,
                        now,
                        alias,
                        existing_alias["relative_root"],
                    ),
                ).rowcount
                if alias_changes != 1 or binding_changes != 1:
                    raise AliasTransactionConflict(
                        "alias transaction compare-and-swap failed"
                    )

            generation = self._next_generation(connection)
            event_detail = {
                "alias": alias,
                "alias_event_identity": alias_event_identity,
                "event_basis": event_basis,
                "new_registry_generation": generation,
                "new_target": new_target,
                "previous_target": expected_current_target,
                "promotion_alias_event_identity": (
                    promotion_alias_event_identity
                ),
                "promotion_transaction_id": transaction_id,
            }
            self._insert_event(
                connection,
                generation,
                {
                    "promote": "default_alias_promoted",
                    "rollback": "default_alias_rolled_back",
                    "clear": "default_alias_cleared",
                }[action],
                alias,
                event_detail,
                now,
            )
            return {
                "action": action,
                "alias": alias,
                "alias_event_identity": alias_event_identity,
                "changed": True,
                "new_registry_generation": generation,
                "new_target": new_target,
                "observed_registry_generation": generation,
                "previous_target": expected_current_target,
                "promotion_transaction_id": transaction_id,
            }

        return await self._write(transaction)

    async def record_runtime_capability(
        self,
        model_version_id: str,
        capability_name: str,
        request_id: str,
        service_transaction_id: str,
        router_transaction_id: str,
        observed_protocol_surfaces: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Persist one validated live capability proof without request content."""

        if capability_name not in {
            "tool_calling",
            "structured_output",
            "streaming",
        }:
            raise RegistryStoreError("runtime capability name is invalid")
        if not request_id or len(request_id) > 128:
            raise RegistryStoreError("runtime capability request ID is invalid")
        if not service_transaction_id or len(service_transaction_id) > 128:
            raise RegistryStoreError(
                "runtime capability service transaction ID is invalid"
            )
        if not router_transaction_id or len(router_transaction_id) > 128:
            raise RegistryStoreError(
                "runtime capability router transaction ID is invalid"
            )
        if (
            not isinstance(observed_protocol_surfaces, tuple)
            or not all(
                isinstance(surface, str)
                and surface in STREAMING_PROTOCOL_SURFACES
                for surface in observed_protocol_surfaces
            )
        ):
            raise RegistryStoreError(
                "runtime capability protocol surfaces are invalid"
            )
        protocol_surfaces = tuple(sorted(set(observed_protocol_surfaces)))
        if capability_name == "streaming" and not protocol_surfaces:
            raise RegistryStoreError(
                "streaming capability requires an observed protocol surface"
            )
        if capability_name != "streaming" and protocol_surfaces:
            raise RegistryStoreError(
                "protocol surfaces are only valid for streaming capability"
            )

        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            row = connection.execute(
                """
                SELECT mv.state,cm.manifest_json,cm.props_payload_sha256
                FROM model_versions AS mv
                JOIN capability_manifests AS cm
                    ON cm.model_version_id=mv.model_version_id
                WHERE mv.model_version_id=?
                """,
                (model_version_id,),
            ).fetchone()
            if row is None or row["state"] != ModelState.READY.value:
                raise RegistryStoreError(
                    "runtime capability model is not ready"
                )
            try:
                manifest = json.loads(str(row["manifest_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RegistryStoreError(
                    "runtime capability manifest is invalid"
                ) from exc
            if not isinstance(manifest, dict):
                raise RegistryStoreError(
                    "runtime capability manifest is invalid"
                )
            tests = manifest.get("runtime_generation_tests")
            if not isinstance(tests, dict):
                raise RegistryStoreError(
                    "runtime capability test map is invalid"
                )
            evidence = manifest.get("runtime_capability_evidence")
            if evidence is None:
                evidence = {}
                manifest["runtime_capability_evidence"] = evidence
            if not isinstance(evidence, dict):
                raise RegistryStoreError(
                    "runtime capability evidence map is invalid"
                )
            existing = evidence.get(capability_name)
            if (
                capability_name != "streaming"
                and
                tests.get(capability_name) == "AVAILABLE"
                and isinstance(existing, dict)
                and existing.get("state") == "AVAILABLE"
            ):
                generation = int(
                    connection.execute(
                        "SELECT value FROM registry_metadata "
                        "WHERE key='registry_generation'"
                    ).fetchone()[0]
                )
                return {
                    "changed": False,
                    "generation": generation,
                    "capability": capability_name,
                }

            now = utc_now()
            tests[capability_name] = "AVAILABLE"
            if capability_name == "streaming":
                prior_surfaces: list[str] = []
                observations: dict[str, Any] = {}
                if isinstance(existing, dict):
                    raw_surfaces = existing.get("observed_protocol_surfaces", [])
                    raw_observations = existing.get("observations", {})
                    if (
                        not isinstance(raw_surfaces, list)
                        or not all(
                            isinstance(surface, str)
                            and surface in STREAMING_PROTOCOL_SURFACES
                            for surface in raw_surfaces
                        )
                        or not isinstance(raw_observations, dict)
                    ):
                        raise RegistryStoreError(
                            "streaming capability evidence is invalid"
                        )
                    prior_surfaces = raw_surfaces
                    observations = dict(raw_observations)
                    if existing.get("state") == "AVAILABLE":
                        if set(observations) != set(prior_surfaces):
                            raise RegistryStoreError(
                                "streaming capability evidence is invalid"
                            )
                        for surface in prior_surfaces:
                            observation = observations.get(surface)
                            if (
                                not isinstance(observation, dict)
                                or any(
                                    not isinstance(observation.get(field), str)
                                    or not observation[field]
                                    or len(observation[field]) > 128
                                    for field in (
                                        "request_id",
                                        "service_transaction_id",
                                        "router_transaction_id",
                                    )
                                )
                                or not isinstance(
                                    observation.get("observed_utc"), str
                                )
                                or not observation["observed_utc"]
                            ):
                                raise RegistryStoreError(
                                    "streaming capability evidence is invalid"
                                )
                merged_surfaces = sorted(
                    set(prior_surfaces).union(protocol_surfaces)
                )
                same_evidence = (
                    tests.get(capability_name) == "AVAILABLE"
                    and isinstance(existing, dict)
                    and existing.get("state") == "AVAILABLE"
                    and merged_surfaces == sorted(set(prior_surfaces))
                    and all(
                        isinstance(observations.get(surface), dict)
                        and observations[surface].get("request_id") == request_id
                        and observations[surface].get(
                            "service_transaction_id"
                        )
                        == service_transaction_id
                        and observations[surface].get(
                            "router_transaction_id"
                        )
                        == router_transaction_id
                        for surface in protocol_surfaces
                    )
                )
                if same_evidence:
                    generation = int(
                        connection.execute(
                            "SELECT value FROM registry_metadata "
                            "WHERE key='registry_generation'"
                        ).fetchone()[0]
                    )
                    return {
                        "changed": False,
                        "generation": generation,
                        "capability": capability_name,
                        "observed_protocol_surfaces": merged_surfaces,
                    }
                for surface in protocol_surfaces:
                    observations[surface] = {
                        "request_id": request_id,
                        "service_transaction_id": service_transaction_id,
                        "router_transaction_id": router_transaction_id,
                        "observed_utc": now,
                    }
                evidence[capability_name] = {
                    "state": "AVAILABLE",
                    "source": "validated_live_inference",
                    "model_version_id": model_version_id,
                    "request_id": request_id,
                    "service_transaction_id": service_transaction_id,
                    "router_transaction_id": router_transaction_id,
                    "observed_protocol_surfaces": merged_surfaces,
                    "observations": observations,
                    "observed_utc": now,
                }
            else:
                evidence[capability_name] = {
                    "state": "AVAILABLE",
                    "source": "validated_live_inference",
                    "request_id": request_id,
                    "service_transaction_id": service_transaction_id,
                    "router_transaction_id": router_transaction_id,
                    "observed_utc": now,
                }
            layers = manifest.get("evidence_layers")
            if not isinstance(layers, dict):
                raise RegistryStoreError(
                    "runtime capability evidence layers are invalid"
                )
            runtime_layer = layers.get("runtime_generation")
            if runtime_layer == "NOT_TESTED":
                runtime_layer = {}
                layers["runtime_generation"] = runtime_layer
            if not isinstance(runtime_layer, dict):
                raise RegistryStoreError(
                    "runtime generation evidence is invalid"
                )
            runtime_layer[capability_name] = "AVAILABLE"
            manifest_json = canonical_json(manifest)
            manifest_sha256 = hashlib.sha256(
                manifest_json.encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                UPDATE capability_manifests
                SET manifest_json=?,manifest_sha256=?,observed_utc=?
                WHERE model_version_id=?
                """,
                (manifest_json, manifest_sha256, now, model_version_id),
            )
            generation = self._next_generation(connection)
            self._insert_event(
                connection,
                generation,
                "runtime_capability_proven",
                model_version_id,
                {
                    "capability": capability_name,
                    "source": "validated_live_inference",
                    "request_id": request_id,
                    "service_transaction_id": service_transaction_id,
                    "router_transaction_id": router_transaction_id,
                    "observed_protocol_surfaces": list(protocol_surfaces),
                },
                now,
            )
            observed_surfaces = (
                merged_surfaces
                if capability_name == "streaming"
                else list(protocol_surfaces)
            )
            return {
                "changed": True,
                "generation": generation,
                "capability": capability_name,
                "manifest_sha256": manifest_sha256,
                "observed_protocol_surfaces": observed_surfaces,
            }

        return await self._write(transaction)

    async def record_rejection(
        self,
        relative_path: str,
        reason_code: str,
        detail: dict[str, Any],
    ) -> dict[str, Any]:
        detail_json = canonical_json(detail)

        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            now = utc_now()
            existing = connection.execute(
                """
                SELECT reason_code,detail_json FROM artifact_rejections
                WHERE relative_path=?
                """,
                (relative_path,),
            ).fetchone()
            if (
                existing
                and existing["reason_code"] == reason_code
                and existing["detail_json"] == detail_json
            ):
                connection.execute(
                    "UPDATE artifact_rejections SET last_seen_utc=? "
                    "WHERE relative_path=?",
                    (now, relative_path),
                )
                generation = int(
                    connection.execute(
                        "SELECT value FROM registry_metadata "
                        "WHERE key='registry_generation'"
                    ).fetchone()[0]
                )
                return {"changed": False, "generation": generation}
            connection.execute(
                """
                INSERT INTO artifact_rejections(
                    relative_path,reason_code,detail_json,first_seen_utc,last_seen_utc
                ) VALUES (?,?,?,?,?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    reason_code=excluded.reason_code,
                    detail_json=excluded.detail_json,
                    last_seen_utc=excluded.last_seen_utc
                """,
                (relative_path, reason_code, detail_json, now, now),
            )
            generation = self._next_generation(connection)
            self._insert_event(
                connection,
                generation,
                "artifact_rejected",
                relative_path,
                {"reason_code": reason_code, "detail": detail},
                now,
            )
            return {"changed": True, "generation": generation}

        return await self._write(transaction)

    async def mark_missing(self, seen_relative_roots: set[str]) -> list[dict[str, Any]]:
        present = await self._read(
            lambda connection: [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT relative_root,current_bundle_id FROM artifact_locations
                    WHERE present=1
                    """
                )
            ]
        )
        results = []
        for location in present:
            relative_root = str(location["relative_root"])
            if relative_root in seen_relative_roots:
                continue

            def transaction(
                connection: sqlite3.Connection,
                relative_root: str = relative_root,
                bundle_id: str = str(location["current_bundle_id"]),
            ) -> dict[str, Any]:
                now = utc_now()
                connection.execute(
                    "UPDATE artifact_locations SET present=0,last_seen_utc=? "
                    "WHERE relative_root=?",
                    (now, relative_root),
                )
                removed_count = connection.execute(
                    """
                    UPDATE model_versions
                    SET state='REMOVED',updated_utc=?
                    WHERE model_version_id IN (
                        SELECT model_version_id
                        FROM model_version_locations
                        WHERE relative_root=?
                    )
                      AND state IN (
                        'REGISTERED','PROBING','READY','UNAVAILABLE','REPLACED'
                      )
                    """,
                    (now, relative_root),
                ).rowcount
                generation = self._next_generation(connection)
                self._insert_event(
                    connection,
                    generation,
                    "artifact_location_removed",
                    relative_root,
                    {
                        "bundle_id": bundle_id,
                        "removed_model_count": removed_count,
                    },
                    now,
                )
                return {
                    "relative_root": relative_root,
                    "generation": generation,
                    "removed_model_count": removed_count,
                }

            results.append(await self._write(transaction))
        return results

    async def mark_location_invalid(
        self, relative_root: str, reason_code: str
    ) -> dict[str, Any]:
        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            location = connection.execute(
                """
                SELECT current_bundle_id,present FROM artifact_locations
                WHERE relative_root=?
                """,
                (relative_root,),
            ).fetchone()
            if location is None or location["present"] != 1:
                generation = int(
                    connection.execute(
                        "SELECT value FROM registry_metadata "
                        "WHERE key='registry_generation'"
                    ).fetchone()[0]
                )
                return {"changed": False, "generation": generation}
            now = utc_now()
            connection.execute(
                "UPDATE artifact_locations SET present=0,last_seen_utc=? "
                "WHERE relative_root=?",
                (now, relative_root),
            )
            unavailable_count = connection.execute(
                """
                UPDATE model_versions
                SET state='UNAVAILABLE',updated_utc=?
                WHERE model_version_id IN (
                    SELECT model_version_id
                    FROM model_version_locations
                    WHERE relative_root=?
                )
                  AND state IN ('PROBING','READY')
                """,
                (now, relative_root),
            ).rowcount
            generation = self._next_generation(connection)
            self._insert_event(
                connection,
                generation,
                "artifact_location_invalid",
                relative_root,
                {
                    "bundle_id": location["current_bundle_id"],
                    "reason_code": reason_code,
                    "unavailable_model_count": unavailable_count,
                },
                now,
            )
            return {
                "changed": True,
                "generation": generation,
                "unavailable_model_count": unavailable_count,
            }

        return await self._write(transaction)

    async def location_hash_cache(self, relative_root: str) -> HashCache:
        def read(connection: sqlite3.Connection) -> HashCache:
            location = connection.execute(
                """
                SELECT current_bundle_id,physical_manifest_json,present
                FROM artifact_locations WHERE relative_root=?
                """,
                (relative_root,),
            ).fetchone()
            if location is None or location["present"] != 1:
                return {}
            try:
                physical = {
                    item["relative_path"]: PhysicalIdentity(
                        device=int(item["device"]),
                        inode=int(item["inode"]),
                        mode=int(item["mode"]),
                        size=int(item["size"]),
                        mtime_ns=int(item["mtime_ns"]),
                    )
                    for item in json.loads(location["physical_manifest_json"])
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return {}
            hashes = {
                row["relative_path"]: row["file_sha256"]
                for row in connection.execute(
                    """
                    SELECT relative_path,file_sha256 FROM artifact_files
                    WHERE bundle_id=?
                    """,
                    (location["current_bundle_id"],),
                )
            }
            if set(physical) != set(hashes):
                return {}
            return {
                relative: (identity, hashes[relative])
                for relative, identity in physical.items()
            }

        return await self._read(read)

    async def location_record(
        self, relative_root: str
    ) -> dict[str, Any] | None:
        return await self._read(
            lambda connection: (
                dict(row)
                if (
                    row := connection.execute(
                        """
                        SELECT
                            relative_root,current_bundle_id,present,
                            physical_manifest_json
                        FROM artifact_locations
                        WHERE relative_root=?
                        """,
                        (relative_root,),
                    ).fetchone()
                )
                is not None
                else None
            )
        )

    async def present_location_roots(self) -> set[str]:
        return await self._read(
            lambda connection: {
                str(row["relative_root"])
                for row in connection.execute(
                    """
                    SELECT relative_root
                    FROM artifact_locations
                    WHERE present=1
                    """
                )
            }
        )

    async def model_version_for_location_bundle(
        self,
        relative_root: str,
        bundle_id: str,
    ) -> str | None:
        return await self._read(
            lambda connection: (
                str(row["model_version_id"])
                if (
                    row := connection.execute(
                        """
                        SELECT mv.model_version_id
                        FROM model_versions AS mv
                        JOIN model_version_locations AS mvl
                          ON mvl.model_version_id=mv.model_version_id
                        WHERE mvl.relative_root=?
                          AND mv.bundle_id=?
                        """,
                        (relative_root, bundle_id),
                    ).fetchone()
                )
                is not None
                else None
            )
        )

    async def models_needing_capability(self) -> list[dict[str, Any]]:
        return await self._read(
            lambda connection: [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT mv.model_version_id,mv.bundle_id,mv.router_model_id,
                           mv.router_metadata_json,mv.state,mvl.relative_root
                    FROM model_versions AS mv
                    JOIN model_version_locations AS mvl
                        ON mvl.model_version_id=mv.model_version_id
                    JOIN artifact_locations AS al
                        ON al.relative_root=mvl.relative_root
                    LEFT JOIN capability_manifests AS cm
                        ON cm.model_version_id=mv.model_version_id
                    WHERE mv.state IN ('REGISTERED','UNAVAILABLE')
                      AND cm.model_version_id IS NULL
                      AND al.present=1
                      AND al.current_bundle_id=mv.bundle_id
                    ORDER BY mv.created_utc,mv.model_version_id
                    """
                )
            ]
        )

    async def set_last_reconcile(self, observed_utc: str) -> None:
        def transaction(connection: sqlite3.Connection) -> None:
            connection.execute(
                "UPDATE registry_metadata SET value=? WHERE key='last_reconcile_utc'",
                (observed_utc,),
            )

        await self._write(transaction)

    async def summary(
        self, default_alias: str = "default"
    ) -> dict[str, Any]:
        def read(connection: sqlite3.Connection) -> dict[str, Any]:
            metadata = dict(
                connection.execute(
                    "SELECT key,value FROM registry_metadata"
                ).fetchall()
            )
            alias_row = connection.execute(
                "SELECT model_version_id FROM aliases WHERE alias=?",
                (default_alias,),
            ).fetchone()
            default_alias_model_id = (
                str(alias_row["model_version_id"])
                if alias_row is not None
                else None
            )
            default_alias_ready = False
            if default_alias_model_id is not None:
                default_alias_ready = bool(
                    connection.execute(
                        """
                        SELECT 1
                        FROM model_versions AS mv
                        JOIN model_version_locations AS mvl
                          ON mvl.model_version_id=mv.model_version_id
                        JOIN artifact_locations AS al
                          ON al.relative_root=mvl.relative_root
                        JOIN capability_manifests AS cm
                          ON cm.model_version_id=mv.model_version_id
                        WHERE mv.model_version_id=?
                          AND mv.state='READY'
                          AND al.present=1
                          AND al.current_bundle_id=mv.bundle_id
                        LIMIT 1
                        """,
                        (default_alias_model_id,),
                    ).fetchone()
                )
            return {
                "registered_model_count": connection.execute(
                    """
                    SELECT count(*) FROM model_versions
                    WHERE state NOT IN ('REPLACED','REMOVED')
                    """
                ).fetchone()[0],
                "ready_model_count": connection.execute(
                    """
                    SELECT count(*)
                    FROM model_versions AS mv
                    JOIN model_version_locations AS mvl
                      ON mvl.model_version_id=mv.model_version_id
                    JOIN artifact_locations AS al
                      ON al.relative_root=mvl.relative_root
                    JOIN capability_manifests AS cm
                      ON cm.model_version_id=mv.model_version_id
                    WHERE mv.state='READY'
                      AND al.present=1
                      AND al.current_bundle_id=mv.bundle_id
                    """
                ).fetchone()[0],
                "candidate_model_count": connection.execute(
                    """
                    SELECT count(*)
                    FROM model_versions AS mv
                    JOIN model_version_locations AS mvl
                      ON mvl.model_version_id=mv.model_version_id
                    JOIN artifact_locations AS al
                      ON al.relative_root=mvl.relative_root
                    WHERE mv.state IN ('REGISTERED','PROBING')
                      AND al.present=1
                      AND al.current_bundle_id=mv.bundle_id
                    """
                ).fetchone()[0],
                "rejected_artifact_count": connection.execute(
                    "SELECT count(*) FROM artifact_rejections"
                ).fetchone()[0],
                "registry_generation": int(metadata["registry_generation"]),
                "last_reconcile_utc": metadata["last_reconcile_utc"] or None,
                "default_alias_model_id": default_alias_model_id,
                "default_alias_ready": default_alias_ready,
            }

        return await self._read(read)

    @staticmethod
    def _public_model_row(
        connection: sqlite3.Connection,
        model_version_id: str,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT
                mv.model_version_id,
                mv.bundle_id,
                mv.router_model_id,
                mv.state,
                mv.created_utc,
                cm.manifest_json AS capability_manifest_json,
                cm.manifest_sha256 AS capability_manifest_sha256
            FROM model_versions AS mv
            LEFT JOIN capability_manifests AS cm
                ON cm.model_version_id=mv.model_version_id
            WHERE mv.model_version_id=?
            """,
            (model_version_id,),
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["aliases"] = [
            str(alias["alias"])
            for alias in connection.execute(
                """
                SELECT alias FROM aliases
                WHERE model_version_id=?
                ORDER BY alias
                """,
                (model_version_id,),
            )
        ]
        value["artifact_present"] = bool(
            connection.execute(
                """
                SELECT 1
                FROM model_version_locations AS mvl
                JOIN artifact_locations AS al
                  ON al.relative_root=mvl.relative_root
                WHERE mvl.model_version_id=?
                  AND al.current_bundle_id=?
                  AND al.present=1
                LIMIT 1
                """,
                (model_version_id, value["bundle_id"]),
            ).fetchone()
        )
        return value

    async def public_model_rows(self) -> dict[str, Any]:
        """Return one atomic internal snapshot for the public READY catalogue."""

        def read(connection: sqlite3.Connection) -> dict[str, Any]:
            generation_row = connection.execute(
                """
                SELECT value FROM registry_metadata
                WHERE key='registry_generation'
                """
            ).fetchone()
            if generation_row is None:
                raise RegistryStoreError("registry generation metadata is missing")
            identifiers = [
                str(row["model_version_id"])
                for row in connection.execute(
                    """
                    SELECT mv.model_version_id
                    FROM model_versions AS mv
                    JOIN capability_manifests AS cm
                        ON cm.model_version_id=mv.model_version_id
                    WHERE mv.state='READY'
                      AND EXISTS (
                        SELECT 1
                        FROM model_version_locations AS mvl
                        JOIN artifact_locations AS al
                          ON al.relative_root=mvl.relative_root
                        WHERE mvl.model_version_id=mv.model_version_id
                          AND al.current_bundle_id=mv.bundle_id
                          AND al.present=1
                      )
                    ORDER BY mv.model_version_id
                    """
                )
            ]
            rows = [
                self._public_model_row(connection, identifier)
                for identifier in identifiers
            ]
            if any(row is None for row in rows):
                raise RegistryStoreError("public catalogue snapshot was inconsistent")
            return {
                "registry_generation": int(generation_row[0]),
                "models": rows,
            }

        return await self._read(read)

    async def resolve_public_model(self, reference: str) -> dict[str, Any]:
        """Resolve one immutable public ID or active alias in one read transaction."""

        def read(connection: sqlite3.Connection) -> dict[str, Any]:
            generation_row = connection.execute(
                """
                SELECT value FROM registry_metadata
                WHERE key='registry_generation'
                """
            ).fetchone()
            if generation_row is None:
                raise RegistryStoreError("registry generation metadata is missing")
            model_version_id = reference
            direct = connection.execute(
                """
                SELECT model_version_id FROM model_versions
                WHERE model_version_id=?
                """,
                (reference,),
            ).fetchone()
            resolution_kind = "immutable_id"
            if direct is None:
                alias = connection.execute(
                    """
                    SELECT model_version_id FROM aliases
                    WHERE alias=?
                    """,
                    (reference,),
                ).fetchone()
                if alias is None:
                    return {
                        "registry_generation": int(generation_row[0]),
                        "resolution": "not_found",
                        "model": None,
                    }
                model_version_id = str(alias["model_version_id"])
                resolution_kind = "alias"
            row = self._public_model_row(connection, model_version_id)
            if row is None:
                return {
                    "registry_generation": int(generation_row[0]),
                    "resolution": "not_found",
                    "model": None,
                }
            state = str(row["state"])
            if state == ModelState.UNAVAILABLE.value:
                resolution = "unavailable"
            elif (
                state != ModelState.READY.value
                or not row["artifact_present"]
                or row["capability_manifest_json"] is None
                or row["capability_manifest_sha256"] is None
            ):
                resolution = "not_found"
            else:
                resolution = "ready"
            return {
                "registry_generation": int(generation_row[0]),
                "resolution": resolution,
                "resolution_kind": resolution_kind,
                "model": row,
            }

        return await self._read(read)

    async def public_model_snapshot_matches(
        self,
        reference: str,
        expected: dict[str, Any],
    ) -> bool:
        """Fail closed when a request's public-to-private mapping has changed."""

        current = await self.resolve_public_model(reference)
        row = current.get("model")
        if current.get("resolution") != "ready" or not isinstance(row, dict):
            return False
        keys = (
            "model_version_id",
            "bundle_id",
            "router_model_id",
            "state",
            "capability_manifest_sha256",
            "created_utc",
            "aliases",
            "artifact_present",
        )
        return all(row.get(key) == expected.get(key) for key in keys)

    async def snapshot(self) -> dict[str, Any]:
        tables = (
            "registry_metadata",
            "artifact_bundles",
            "artifact_files",
            "artifact_locations",
            "model_versions",
            "model_version_locations",
            "aliases",
            "alias_bindings",
            "capability_manifests",
            "artifact_rejections",
            "registry_events",
        )

        def read(connection: sqlite3.Connection) -> dict[str, Any]:
            result = {}
            for table in tables:
                result[table] = [
                    dict(row)
                    for row in connection.execute(
                        f"SELECT * FROM {table} ORDER BY rowid"
                    )
                ]
            return result

        return await self._read(read)

    async def integrity(self) -> dict[str, Any]:
        def read(connection: sqlite3.Connection) -> dict[str, Any]:
            return {
                "integrity_check": connection.execute(
                    "PRAGMA integrity_check"
                ).fetchone()[0],
                "foreign_keys": connection.execute(
                    "PRAGMA foreign_keys"
                ).fetchone()[0],
                "journal_mode": connection.execute(
                    "PRAGMA journal_mode"
                ).fetchone()[0],
                "database_mode": oct(
                    stat.S_IMODE(self.database_path.stat().st_mode)
                ),
            }

        return await self._read(read)

    def _checkpoint_sync(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            result = tuple(connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
            return {
                "wal_checkpoint_truncate": list(result),
                "database_bytes": self.database_path.stat().st_size,
                "wal_present": os.path.lexists(f"{self.database_path}-wal"),
                "shm_present": os.path.lexists(f"{self.database_path}-shm"),
            }
        finally:
            connection.close()

    async def checkpoint_and_close(self) -> dict[str, Any]:
        self._require_initialized()
        async with self._writer_lock:
            result = await asyncio.to_thread(self._checkpoint_sync)
            self._initialized = False
            return result
