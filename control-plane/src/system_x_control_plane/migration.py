from __future__ import annotations
import sqlite3
def migrate(db:sqlite3.Connection,target:int=1)->None:
    db.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"); db.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(1,'v5')"); db.commit()
