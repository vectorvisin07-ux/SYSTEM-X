from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

class Plan(StrEnum): FIT="FIT"; FIT_WITH_REDUCED_CONTEXT="FIT_WITH_REDUCED_CONTEXT"; CPU_OFFLOAD_ELIGIBLE="CPU_OFFLOAD_ELIGIBLE"; INSUFFICIENT_RESOURCES="INSUFFICIENT_RESOURCES"; UNSUPPORTED_TOPOLOGY="UNSUPPORTED_TOPOLOGY"
@dataclass(frozen=True)
class ResourceDecision:
    decision: Plan; reason: str; estimated_bytes: int; available_bytes: int
def plan(*, artifact_bytes: int, available_bytes: int, overhead_bytes: int = 0, context_limit: int = 2048) -> ResourceDecision:
    need=artifact_bytes+overhead_bytes+context_limit*1024
    if available_bytes >= need: return ResourceDecision(Plan.FIT,"within observed budget",need,available_bytes)
    if available_bytes >= artifact_bytes+overhead_bytes: return ResourceDecision(Plan.FIT_WITH_REDUCED_CONTEXT,"context reduction required",need,available_bytes)
    return ResourceDecision(Plan.INSUFFICIENT_RESOURCES,"observed memory is insufficient",need,available_bytes)
