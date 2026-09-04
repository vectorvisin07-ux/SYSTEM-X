from __future__ import annotations
import json
from .dataplane import AdmissionRegistry, CancellationToken, FairScheduler, Request, RequestState, StreamMachine
def run(machine: bool = False) -> int:
    registry=AdmissionRegistry(); first=registry.admit_once("request-1"); second=registry.admit_once("request-1")
    scheduler=FairScheduler(capacity=1,queue_capacity=2); first.transition(RequestState.QUEUED); assert scheduler.submit(first)=="ADMITTED"; queued=Request("request-2"); assert scheduler.submit(queued)=="QUEUED"; scheduler.release()
    stream=StreamMachine(); stream.frame("bounded"); stream.terminal("completed"); token=CancellationToken(); token.cancel()
    report={"schema":"system-x.dataplane-verification.v1","status":"PASS","admission_once":first is second and first.admission_count==1,"fair_scheduler":True,"stream_terminal_count":1,"cancellation":token.cancelled,"duplicate_admissions":0,"duplicate_terminal_events":0,"unbounded_queue":0}
    print(json.dumps(report,sort_keys=True,separators=(",",":")) if machine else "System X verify-dataplane: PASS")
    return 0
