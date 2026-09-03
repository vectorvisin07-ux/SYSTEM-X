from __future__ import annotations
from enum import StrEnum
from .errors import StateTransitionError
class OperationState(StrEnum):
    REQUESTED="REQUESTED"; AUTHORIZED="AUTHORIZED"; PRECONDITIONS_VERIFIED="PRECONDITIONS_VERIFIED"; RUNNING="RUNNING"; PHYSICAL_MUTATION_OBSERVED="PHYSICAL_MUTATION_OBSERVED"; COMMITTING="COMMITTING"; COMPLETED="COMPLETED"; FAILED_CLEAN="FAILED_CLEAN"; FAILED_DIRTY="FAILED_DIRTY"; CANCELLING="CANCELLING"; CANCELLED="CANCELLED"; RECOVERING="RECOVERING"; ROLLING_BACK="ROLLING_BACK"; ROLLED_BACK="ROLLED_BACK"
LEGAL = {OperationState.REQUESTED:{OperationState.AUTHORIZED,OperationState.FAILED_CLEAN},OperationState.AUTHORIZED:{OperationState.PRECONDITIONS_VERIFIED,OperationState.FAILED_CLEAN},OperationState.PRECONDITIONS_VERIFIED:{OperationState.RUNNING,OperationState.FAILED_CLEAN},OperationState.RUNNING:{OperationState.PHYSICAL_MUTATION_OBSERVED,OperationState.CANCELLING,OperationState.FAILED_DIRTY,OperationState.FAILED_CLEAN},OperationState.PHYSICAL_MUTATION_OBSERVED:{OperationState.COMMITTING,OperationState.ROLLING_BACK,OperationState.FAILED_DIRTY},OperationState.COMMITTING:{OperationState.COMPLETED,OperationState.FAILED_DIRTY},OperationState.CANCELLING:{OperationState.CANCELLED,OperationState.COMPLETED,OperationState.FAILED_DIRTY},OperationState.ROLLING_BACK:{OperationState.ROLLED_BACK,OperationState.FAILED_DIRTY}}
def transition(before: OperationState, after: OperationState) -> OperationState:
    if after not in LEGAL.get(before, set()): raise StateTransitionError("ILLEGAL_STATE_TRANSITION", f"{before}->{after}")
    return after
