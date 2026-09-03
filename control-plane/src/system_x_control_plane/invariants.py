from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True,slots=True)
class Observation:
    schema_version:str="system-x.control-observation.v1"; release_identity:str=""; service_state:str=""; model_state:str=""; connection_state:str=""; model_child_count:int=0; duplicate_owners:int=0
@dataclass(frozen=True,slots=True)
class Violation:
    code:str; owner:str; message:str; repair:str
def evaluate(o:Observation)->list[Violation]:
    out=[]
    if o.duplicate_owners:out.append(Violation("DUPLICATE_OWNER","adapter","more than one physical owner","repair-owned-duplicates"))
    if o.model_child_count>1:out.append(Violation("MODEL_OWNER_COUNT","model-adapter","multiple model children","reconcile-model"))
    return out
