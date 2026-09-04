from __future__ import annotations
import json, sys
from pathlib import Path
from .model_fabric import ModelFabric, FabricState
def report():
    f=ModelFabric("gguf-original"); a=f.activate("native-fixture",operation_id="op-1"); idem=f.activate("native-fixture",operation_id="op-1"); rb=f.activate("native-fixture-2",operation_id="op-2",fail_at=FabricState.STARTING)
    ok=a.state==FabricState.COMMITTED and idem.state==FabricState.COMMITTED and rb.state==FabricState.ROLLED_BACK and f.current_model=="native-fixture"
    return {"schema":"system-x.verify-model-fabric.v1","status":"PASS" if ok else "FAIL","switch_commit":a.state,"idempotent_reuse":idem.state,"rollback":rb.state,"current_model":f.current_model,"max_leases":1,"max_active_children":1}
def run(machine=False):
    r=report(); print(json.dumps(r,sort_keys=True,separators=(",",":")) if machine else "System X verify-model-fabric: "+r["status"]); return 0 if r["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(run("--json" in sys.argv))
