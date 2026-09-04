"""Transactional, engine-neutral model fabric with one exclusive lease."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
import hashlib, json

class FabricState(StrEnum):
    IDLE="IDLE"; PREPARING="PREPARING"; DRAINING="DRAINING"; STOPPING="STOPPING"; STARTING="STARTING"; PROBING="PROBING"; COMMITTED="COMMITTED"; ROLLED_BACK="ROLLED_BACK"

@dataclass(frozen=True)
class SwitchResult:
    operation_id: str
    state: FabricState
    current_model: str
    previous_model: str | None
    reason: str | None = None

class ModelFabric:
    def __init__(self, current_model: str):
        self._lock=Lock(); self.current_model=current_model; self.state=FabricState.IDLE; self.operation_hashes={}; self.lease_owner=None

    def activate(self, model_id: str, *, operation_id: str, request_hash: str | None = None, generation: int = 0, fail_at: str | None = None) -> SwitchResult:
        digest=request_hash or hashlib.sha256(json.dumps([model_id,generation],separators=(",",":")).encode()).hexdigest()
        with self._lock:
            if operation_id in self.operation_hashes:
                if self.operation_hashes[operation_id] != digest: raise ValueError("idempotency conflict")
                return SwitchResult(operation_id, FabricState.COMMITTED, self.current_model, None)
            if self.state != FabricState.IDLE: raise RuntimeError("MODEL_SWITCH_IN_PROGRESS")
            old=self.current_model; self.operation_hashes[operation_id]=digest; self.state=FabricState.PREPARING; self.lease_owner=operation_id
            try:
                for state in (FabricState.DRAINING,FabricState.STOPPING,FabricState.STARTING,FabricState.PROBING):
                    self.state=state
                    if fail_at == state: raise RuntimeError("target activation failed")
                self.current_model=model_id; self.state=FabricState.COMMITTED; self.lease_owner=None
                result=SwitchResult(operation_id,self.state,model_id,old); self.state=FabricState.IDLE
                return result
            except Exception as exc:
                self.current_model=old; self.state=FabricState.ROLLED_BACK; self.lease_owner=None
                result=SwitchResult(operation_id,self.state,old,old,str(exc)); self.state=FabricState.IDLE; return result
