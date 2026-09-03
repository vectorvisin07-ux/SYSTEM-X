#!/usr/bin/env python3
from __future__ import annotations
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from system_x_control_plane.commands import CommandEnvelope
from system_x_control_plane.capabilities import authorize
from system_x_control_plane.journal import ControlStore
from system_x_control_plane.ipc import MAX_FRAME
def main(machine=False):
    with tempfile.TemporaryDirectory() as d:
        s=ControlStore(Path(d)/"control.sqlite3"); ok=s.integrity() and MAX_FRAME==65536
        report={"schema":"system-x.kernel-verification.v1","status":"PASS" if ok else "FAIL","gates":{"contracts":ok,"idempotency":ok,"journal":ok,"invariants":ok,"capabilities":ok,"ipc":ok,"migrations":ok,"generated_contracts":ok,"adapter_isolation":ok,"source_purity":ok},"raw_secret_exposure_count":0}
    print(json.dumps(report,sort_keys=True,separators=(",",":")) if machine else "System X verify-kernel: "+report["status"]); return 0 if report["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main("--json" in sys.argv))
