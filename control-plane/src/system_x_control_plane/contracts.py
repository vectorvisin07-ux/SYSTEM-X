from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from .states import OperationState
def utc_now() -> str: return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    schema_version:str; operation_id:str; idempotency_key:str; operation_type:str; state:OperationState; reason_code:str; public_message:str; operator_message:str; started_at:str; completed_at:str|None; resource_generation_before:int; resource_generation_after:int; result_projection:dict[str,Any]; retry_classification:str; reused_existing_result:bool
    def as_dict(self)->dict[str,Any]: return {"schema_version":self.schema_version,"operation_id":self.operation_id,"idempotency_key":self.idempotency_key,"operation_type":self.operation_type,"state":self.state.value,"reason_code":self.reason_code,"public_message":self.public_message,"operator_message":self.operator_message,"started_at":self.started_at,"completed_at":self.completed_at,"resource_generation_before":self.resource_generation_before,"resource_generation_after":self.resource_generation_after,"result_projection":self.result_projection,"retry_classification":self.retry_classification,"reused_existing_result":self.reused_existing_result}
