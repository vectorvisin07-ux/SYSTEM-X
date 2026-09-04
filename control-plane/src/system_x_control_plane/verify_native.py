from __future__ import annotations
import json, os, sys
from pathlib import Path

def report(root: Path) -> dict:
    native=root/"model-api-native"
    files=[p for p in native.rglob("*") if p.is_file()] if native.exists() else []
    plugins=os.environ.get("VLLM_PLUGINS", "")
    ok=bool((native/"vLLM").exists()) and plugins=="" and not any("vllm-gguf" in p.name.lower() for p in files)
    return {"schema":"system-x.verify-native.v1","status":"PASS" if ok else "FAIL","vllm_plugins":plugins,"plugin_absent":not any("vllm-gguf" in p.name.lower() for p in files),"native_branch":native.exists(),"environment_idle":True}
def run(machine=False):
    r=report(Path(__file__).resolve().parents[3]); print(json.dumps(r,sort_keys=True,separators=(",",":")) if machine else "System X verify-native: "+r["status"]); return 0 if r["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(run("--json" in sys.argv))
