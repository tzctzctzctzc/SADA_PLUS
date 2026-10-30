import json
from pathlib import Path
import subprocess
import sys
import time
import os
import signal
import pytest
from tools.takeover_background_oracle import memory_requirement,process_identity,validate_snapshot


def test_resource_threshold_uses_actual_peak():
    assert memory_requirement([dict(peak_reserved_mib=4430),dict(peak_reserved_mib=4092)])==8192
    assert memory_requirement([dict(peak_reserved_mib=7000)])==10072
    with pytest.raises(ValueError):memory_requirement([dict(peak_reserved_mib=float('nan'))])


def test_rejects_duplicate_or_changed_task_snapshot(tmp_path):
    with pytest.raises(ValueError):validate_snapshot(dict(pid=1,completed=[{}],active={}),tmp_path,1)
    with pytest.raises(ValueError):validate_snapshot(dict(pid=1,completed=[],active=dict.fromkeys(['S1','S2','S3']),pending=['S4']),tmp_path,1)


@pytest.mark.skipif(sys.platform!='linux',reason='实际Linux进程信号核验')
def test_cpu_parent_replacement_does_not_stop_own_child(tmp_path):
    pidfile=tmp_path/'child.pid'
    program="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
    parent=subprocess.Popen([sys.executable,'-c',program,str(pidfile)],start_new_session=True)
    child=None
    try:
        deadline=time.monotonic()+5
        while not pidfile.exists():
            assert time.monotonic()<deadline;time.sleep(.02)
        child=int(pidfile.read_text());before=process_identity(child)
        assert before['parent']==parent.pid
        os.kill(parent.pid,signal.SIGSTOP);assert process_identity(child)['state']!='T'
        parent.kill();parent.wait(timeout=5)
        after=process_identity(child);assert after['start']==before['start'] and after['state'] not in ('T','Z')
    finally:
        if parent.poll() is None:parent.kill();parent.wait()
        if child is not None:
            try:os.killpg(child,signal.SIGTERM)
            except ProcessLookupError:pass
