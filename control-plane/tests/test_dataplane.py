from system_x_control_plane.dataplane import *
def test_admission_and_terminal_invariants():
    a=AdmissionRegistry(); r=a.admit_once("x"); assert a.admit_once("x") is r and r.admission_count==1
    s=StreamMachine(); s.terminal("cancelled")
    try:s.terminal("completed"); assert False
    except ValueError:pass
def test_scheduler_is_bounded_and_fifo():
    q=FairScheduler(1,1); a=Request("a"); a.transition(RequestState.QUEUED); assert q.submit(a)=="ADMITTED"; b=Request("b"); assert q.submit(b)=="QUEUED"; c=Request("c"); assert q.submit(c)=="BUSY"
