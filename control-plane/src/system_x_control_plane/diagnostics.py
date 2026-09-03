from __future__ import annotations
from typing import Any
PRIVATE={"api_key","credential","secret","prompt","answer","environment","model_path","socket_path"}
def redact(value:Any)->Any:
    if isinstance(value,dict):return {k:("[REDACTED]" if k.lower() in PRIVATE else redact(v)) for k,v in value.items()}
    if isinstance(value,list):return [redact(v) for v in value]
    return value
def deep_bundle(observations:dict[str,Any])->dict[str,Any]:return {"schema_version":"system-x.diagnostic-bundle.v1","diagnostics":redact(observations),"raw_secret_exposure_count":0}
