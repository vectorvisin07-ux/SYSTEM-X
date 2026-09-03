from dataclasses import dataclass
@dataclass(frozen=True,slots=True)
class RepairBudget:
    maximum_attempts:int=3; window_seconds:float=30.0; backoff_seconds:float=1.0; circuit_reset_seconds:float=60.0
    def allows(self,attempts:int)->bool:return 0<=attempts<self.maximum_attempts
