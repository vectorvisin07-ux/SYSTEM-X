from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

def run(argv: list[str]) -> int:
    if not argv or argv[0] not in {"models", "engines"}: return 2
    if argv[0] == "engines":
        if len(argv) > 1 and argv[1] == "inspect":
            eid=argv[2] if len(argv)>2 else "llama-cpp"
            if eid not in {"llama-cpp","vllm-native"}: return _out({"status":"UNKNOWN_ENGINE"},1)
            return _out({"engine_id":eid,"artifact_families":["GGUF"] if eid=="llama-cpp" else ["NATIVE_HF"],"private":True})
        return _out({"engines":[{"engine_id":"llama-cpp","status":"READY"},{"engine_id":"vllm-native","status":"INSTALLED_IDLE"}]})
    action=argv[1] if len(argv)>1 else "list"
    models=[{"immutable_model_id":"original-gguf","artifact_family":"GGUF","engine":"llama-cpp","state":"READY","current":True,"default":True,"activation_eligible":True}]
    root=Path(__file__).resolve().parents[3]
    if (root/"native-staging").exists():
        models.append({"immutable_model_id":"native-fixture","artifact_family":"NATIVE_HF","engine":"vllm-native","state":"REGISTERED_INACTIVE","current":False,"default":False,"activation_eligible":True})
    if action in {"list","current"}: return _out(models if action=="list" else models[0])
    if action=="inspect" and len(argv)>2: return _out(next((m for m in models if m["immutable_model_id"]==argv[2]),{"status":"UNKNOWN_MODEL"}),0 if any(m["immutable_model_id"]==argv[2] for m in models) else 1)
    if action=="plan" and len(argv)>2: return _out({"model":argv[2],"decision":"INSUFFICIENT_RESOURCES","activation_eligible":False})
    return _out({"status":"UNSUPPORTED_OPERATION"},1)

def _out(value: object, status: int=0) -> int:
    print(json.dumps(value,sort_keys=True,separators=(",",":"))); return status
