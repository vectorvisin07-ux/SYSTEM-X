from __future__ import annotations
import uuid
from pathlib import Path
from .commands import CommandEnvelope
from .contracts import ResultEnvelope,utc_now
from .capabilities import authorize
from .journal import ControlStore
from .states import OperationState
class Kernel:
    def __init__(self,root:Path):self.root=Path(root); self.store=ControlStore(self.root/"control-plane.sqlite3")
    def execute(self,command:dict,required_capability:str)->dict:
        envelope=CommandEnvelope.parse(command); authorize(envelope.value["capability_set"],required_capability,actor=envelope.actor); now=utc_now(); existing=self.store.put_operation(operation_id=command["operation_id"],actor_id=envelope.actor,idempotency_key=envelope.key,request_hash=command["request_hash"],operation_type=command["operation_type"],generation=command["expected_generation"])
        if existing.get("reused_existing_result") and existing.get("result_json"): return __import__("json").loads(existing["result_json"])
        op=command["operation_id"]; self.store.append_event(op,"OperationRequested","OK",command["expected_generation"],{}); self.store.append_event(op,"AuthorizationAccepted","OK",command["expected_generation"],{}); self.store.update_operation(op,"COMPLETED",{"accepted":True}); result=ResultEnvelope("system-x.control-result.v1",op,envelope.key,command["operation_type"],OperationState.COMPLETED,"OK","accepted","accepted",now,utc_now(),command["expected_generation"],command["expected_generation"],{"accepted":True},"none",False).as_dict(); self.store.update_operation(op,"COMPLETED",result); self.store.append_event(op,"OperationCompleted","OK",command["expected_generation"],{}); return result
