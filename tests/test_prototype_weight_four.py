import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools.resume_prototype_weights_four import EXPECTED,remaining_jobs,process_status,has_finished,stop_frozen_parent


def test_exact_remainder_never_repeats_completed_or_inflight():
    completed=[(v,f'S{i}') for v in ['weight_020','weight_040'] for i in [1,2]]
    legacy=[('weight_020','S3')]
    result=remaining_jobs(completed,legacy,legacy)
    assert len(result)==7 and not set(result)&(set(completed)|set(legacy))
    assert set(result)|set(completed)|set(legacy)==EXPECTED


def test_unknown_partial_requires_inspection_not_fresh_training():
    with pytest.raises(ValueError):remaining_jobs([],[],[('weight_020','S3')])
    with pytest.raises(ValueError):remaining_jobs([('weight_005','S1')],[],[])
    with pytest.raises(ValueError):remaining_jobs([('weight_020','S1')],[('weight_020','S1')],[])


def test_reused_pid_is_not_signalled():
    current=process_status(os.getpid())
    with pytest.raises(ValueError):has_finished(os.getpid(),'wrong-start-time')
    assert not has_finished(os.getpid(),current['start_ticks'])


def test_no_inflight_is_supported():
    assert has_finished(None,None)
    assert len(remaining_jobs([],[],[]))==12


@pytest.mark.skipif(os.name=='nt',reason='真实Linux进程组/冻结语义')
def test_freezing_supervisor_does_not_interrupt_child_and_drain_is_safe(tmp_path):
    program='''import subprocess,sys,signal,time,os
from pathlib import Path
folder=Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-c',"import time;from pathlib import Path;time.sleep(1.);Path("+repr(str(folder/'child_complete'))+").write_text('ok')"])
(folder/'child_pid').write_text(str(child.pid))
def stop(s,f):raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
try:
 while True:time.sleep(.1)
finally:
 if child.poll() is None:child.terminate()
 child.wait()
'''
    parent=subprocess.Popen([sys.executable,'-B','-c',program,str(tmp_path)],start_new_session=True)
    record=None
    try:
        until=time.monotonic()+10
        while not (tmp_path/'child_pid').exists():
            assert time.monotonic()<until;time.sleep(.02)
        child=int((tmp_path/'child_pid').read_text());cstatus=process_status(child)
        os.kill(parent.pid,signal.SIGSTOP)
        while process_status(parent.pid)['state']!='T':time.sleep(.01)
        record=dict(parent=parent.pid,parent_start_ticks=process_status(parent.pid)['start_ticks'],
            children=[dict(pid=child,start_ticks=cstatus['start_ticks'])])
        with pytest.raises(ValueError):stop_frozen_parent(record)
        while not has_finished(child,cstatus['start_ticks']):
            assert time.monotonic()<until;time.sleep(.02)
        assert (tmp_path/'child_complete').read_text()=='ok'
        stop_frozen_parent(record);parent.wait(timeout=5)
        assert parent.returncode==143
    finally:
        if parent.poll() is None:
            parent.terminate();os.kill(parent.pid,signal.SIGCONT);parent.wait(timeout=5)
