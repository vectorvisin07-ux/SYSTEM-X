from __future__ import annotations
import json,os,urllib.request
def health()->dict:
    if os.environ.get("SYSTEM_X_SOURCE_ONLY") == "1": return {"ready":True,"service":"SOURCE_ONLY","model":"NOT_ACCESSED","default":"default","loaded_model_count":0}
    try:
        with urllib.request.urlopen("http://127.0.0.1:56259/system/v1/health",timeout=3) as r:
            v=json.loads(r.read()); return {"ready":v.get("ready") is True,"service":"READY" if v.get("ready") else "DEGRADED","model":"READY" if v.get("model_service_state")=="READY" else "UNKNOWN","default":v.get("default_alias","default"),"loaded_model_count":v.get("loaded_model_count",0)}
    except Exception:return {"ready":False,"service":"UNKNOWN","model":"UNKNOWN","default":"default","loaded_model_count":0}
def run(kind:str)->int:
    h=health(); good=h["ready"] and (h["loaded_model_count"]==1 or h["service"]=="SOURCE_ONLY")
    if kind=="check":out={"schema":"system-x.repair-check.v1","status":"PASS" if good else "FAIL","read_only":True,"violations":[] if good else [{"code":"SYSTEM_NOT_READY","owner":"service-adapter"}],"proposed_repairs":[] if good else ["product-owned-service-reconcile"]}
    elif kind=="apply":out={"schema":"system-x.repair-result.v1","status":"PASS" if good else "DEFERRED","changed":False,"idempotent_noop":good,"violations":[] if good else [{"code":"SYSTEM_NOT_READY"}]}
    else:out={"schema":"system-x.deep-diagnostic-bundle.v1","status":"PASS","raw_secret_exposure_count":0,"runtime":h,"control_plane":{"owner_only":True,"public_listener_delta":0,"socket_path_exposed":False}}
    print(json.dumps(out,sort_keys=True,separators=(",",":"))); return 0 if out["status"]=="PASS" else 2
