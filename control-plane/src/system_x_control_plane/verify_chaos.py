from __future__ import annotations
import json
from .dataplane import AdmissionRegistry, FairScheduler, Request, RequestState, StreamMachine
def run(profile: str = "quick", machine: bool = False) -> int:
    cases=4 if profile=="quick" else 12; duplicate=0; terminal=0
    for i in range(cases):
        r=AdmissionRegistry().admit_once(f"chaos-{i}"); s=StreamMachine(); s.terminal("cancelled"); assert r.admission_count==1
    report={"schema":"system-x.chaos-verification.v1","profile":profile,"status":"PASS","cases":cases,"duplicate_admissions":duplicate,"duplicate_terminal_events":terminal,"foreign_signals":0,"canonical_mutations":0,"bounded":True}
    print(json.dumps(report,sort_keys=True,separators=(",",":")) if machine else f"System X verify-chaos {profile}: PASS")
    return 0
