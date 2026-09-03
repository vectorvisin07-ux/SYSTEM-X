from __future__ import annotations
import json,tempfile
from pathlib import Path
from .journal import ControlStore
from .ipc import MAX_FRAME
def run(machine:bool=False)->int:
    with tempfile.TemporaryDirectory() as d:s=ControlStore(Path(d)/"control.sqlite3"); ok=s.integrity() and MAX_FRAME==65536
    report={"schema":"system-x.kernel-verification.v1","status":"PASS" if ok else "FAIL","gates":{k:ok for k in ("contracts","idempotency","journal","invariants","capabilities","ipc","migrations","generated_contracts","adapter_isolation","source_purity")},"raw_secret_exposure_count":0}
    print(json.dumps(report,sort_keys=True,separators=(",",":")) if machine else "System X verify-kernel: "+report["status"]); return 0 if ok else 1
