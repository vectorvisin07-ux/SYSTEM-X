from __future__ import annotations
import json,os,sqlite3,threading
from pathlib import Path
from typing import Any
from .serialization import canonical_hash
from .errors import IdempotencyConflict
SCHEMA='''CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS operations(operation_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL, operation_type TEXT NOT NULL, state TEXT NOT NULL, result_json TEXT, generation_before INTEGER NOT NULL, generation_after INTEGER NOT NULL, UNIQUE(actor_id,idempotency_key)); CREATE TABLE IF NOT EXISTS idempotency_keys(actor_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL, PRIMARY KEY(actor_id,idempotency_key)); CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, sequence INTEGER NOT NULL, schema_version TEXT NOT NULL, event_type TEXT NOT NULL, generation INTEGER NOT NULL, reason_code TEXT NOT NULL, payload_json TEXT NOT NULL, event_hash TEXT NOT NULL, UNIQUE(operation_id,sequence)); CREATE TABLE IF NOT EXISTS resource_generations(resource_identity TEXT PRIMARY KEY, generation INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS reconciliation_budgets(resource_identity TEXT PRIMARY KEY, attempts INTEGER NOT NULL, window_started TEXT NOT NULL);'''
class ControlStore:
    def __init__(self,path:Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); os.chmod(self.path.parent,0o700); self._lock=threading.RLock(); self._db=sqlite3.connect(self.path,timeout=5,isolation_level=None,check_same_thread=False); self._db.row_factory=sqlite3.Row; self._db.execute("PRAGMA foreign_keys=ON"); self._db.execute("PRAGMA journal_mode=WAL"); self._db.execute("PRAGMA synchronous=FULL"); self._db.executescript(SCHEMA); os.chmod(self.path,0o600)
    def close(self): self._db.close()
    def integrity(self)->bool:return self._db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
    def put_operation(self,*,operation_id:str,actor_id:str,idempotency_key:str,request_hash:str,operation_type:str,generation:int)->dict[str,Any]:
        with self._lock:
            row=self._db.execute("SELECT * FROM operations WHERE actor_id=? AND idempotency_key=?",(actor_id,idempotency_key)).fetchone()
            if row:
                if row["request_hash"]!=request_hash: raise IdempotencyConflict("IDEMPOTENCY_CONFLICT")
                return dict(row)|{"reused_existing_result":True}
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?)",(operation_id,actor_id,idempotency_key,request_hash,operation_type,"REQUESTED",None,generation,generation))
                self._db.execute("INSERT INTO idempotency_keys VALUES(?,?,?,?,datetime('now'))",(actor_id,idempotency_key,request_hash,operation_id))
                self._db.execute("COMMIT")
            except Exception:self._db.execute("ROLLBACK"); raise
            return {"operation_id":operation_id,"reused_existing_result":False}
    def append_event(self,operation_id:str,event_type:str,reason_code:str,generation:int,payload:dict[str,Any])->str:
        with self._lock:
            seq=self._db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM events WHERE operation_id=?",(operation_id,)).fetchone()[0]; body={"operation_id":operation_id,"sequence":seq,"event_type":event_type,"reason_code":reason_code,"generation":generation,"payload":payload}; h=canonical_hash(body); eid=canonical_hash({"event":h}); self._db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)",(eid,operation_id,seq,"system-x.control-event.v1",event_type,generation,reason_code,json.dumps(payload,sort_keys=True,separators=(",",":")),h)); return h
    def update_operation(self,operation_id:str,state:str,result:dict[str,Any]|None=None,generation_after:int|None=None):self._db.execute("UPDATE operations SET state=?,result_json=?,generation_after=COALESCE(?,generation_after) WHERE operation_id=?",(state,json.dumps(result,sort_keys=True,separators=(",",":")) if result else None,generation_after,operation_id))
    def events(self,operation_id:str):return [dict(x) for x in self._db.execute("SELECT * FROM events WHERE operation_id=? ORDER BY sequence",(operation_id,))]
