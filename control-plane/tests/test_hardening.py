from __future__ import annotations
import concurrent.futures, json, socket
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from system_x_control_plane.budgets import RepairBudget
from system_x_control_plane.diagnostics import deep_bundle
from system_x_control_plane.ipc import ControlSocket
from system_x_control_plane.migration import migrate
from system_x_control_plane.replay import replay
from system_x_control_plane.serialization import canonical_bytes

def test_migration_is_idempotent(tmp_path):
    import sqlite3
    db=sqlite3.connect(tmp_path/"m.sqlite3"); migrate(db); migrate(db); assert db.execute("select count(*) from schema_migrations").fetchone()[0]==1
def test_budget_is_bounded():
    b=RepairBudget(maximum_attempts=2); assert b.allows(0) and b.allows(1) and not b.allows(2)
def test_replay_is_byte_deterministic():
    value={"z":1,"a":[2,3]}; assert replay(value)==replay(json.loads(canonical_bytes(value)))
def test_diagnostics_redact_private_fields():
    out=deep_bundle({"api_key":"secret","model_path":"/private/model","safe":"ok"}); raw=json.dumps(out); assert '"api_key": "secret"' not in raw and "/private/model" not in raw and out["raw_secret_exposure_count"]==0
def test_owner_only_socket(tmp_path):
    s=ControlSocket(tmp_path/"control.sock"); listener=s.bind(); c=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); c.connect(str(s.path)); assert s.validate_peer(c); c.close(); listener.close()
def test_idempotency_race(tmp_path):
    from system_x_control_plane.journal import ControlStore
    s=ControlStore(tmp_path/"race.sqlite3")
    def put(i): return s.put_operation(operation_id=f"operation-{i:016d}",actor_id="a",idempotency_key="same",request_hash="a"*64,operation_type="RepairService",generation=0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(put,range(8)))
    assert sum(not x.get("reused_existing_result") for x in results)==1
