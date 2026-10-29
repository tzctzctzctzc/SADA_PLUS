import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from tools.run_prototype_methods_four import (JOBS,pending_jobs,process_state,wait_natural_children,
                                             stop_parent_after_children,exact_spawn)


def test_exact_missing_set_excludes_completed_and_all_live_previous_jobs():
    complete={('temporal_pool','S1'),('multi_prototype','S1')}
    live={('temporal_pool','S2'),('multi_prototype','S2')}
    pending=pending_jobs(complete,live)
    assert len(pending)==8 and set(pending)|complete|live==JOBS
    assert not set(pending)&(complete|live)
    with pytest.raises(ValueError):pending_jobs(complete,complete)
    with pytest.raises(ValueError):pending_jobs({('other','S1')},set())


def test_spawn_keeps_original_training_code_config_and_numeric_environment(tmp_path,monkeypatch):
    entry=tmp_path/'entry';entry.mkdir();training=tmp_path/'original-code'
    cfg={'seed':123,'baseline':{'tmpdir':str(tmp_path/'tmp')}}
    config=entry/'config.json';config.write_text(json.dumps(cfg));before=config.read_bytes();seen={}
    def fake(command,**kwargs):seen.update(command=command,**kwargs);return object()
    monkeypatch.setattr(subprocess,'Popen',fake)
    lease=dict(uuid='physical-gpu-2',lease='/receipt',lease_sha256='sha')
    exact_spawn('train',entry,lease,Path('/data'),training)
    assert seen['command'][2]==str(training/'tools/run_sada_baseline.py')
    assert seen['command'][3:5]==['--child','train'] and seen['cwd']==training
    assert seen['env']['CUDA_VISIBLE_DEVICES']=='physical-gpu-2'
    assert seen['env']['PYTHONPATH']==str(training)
    assert seen['env']['OMP_NUM_THREADS']==seen['env']['MKL_NUM_THREADS']=='8'
    assert config.read_bytes()==before and seen['start_new_session'] is True
    with pytest.raises(FileExistsError):exact_spawn('train',entry,lease,Path('/data'),training)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux /proc/flock handoff')
def test_supervisor_freeze_leaves_children_running_and_prevents_next_dispatch(tmp_path):
    ready=tmp_path/'ready.json';finished=tmp_path/'child-finished';dispatched=tmp_path/'next-dispatch'
    program="""import subprocess,sys,json,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c',"import time;from pathlib import Path;time.sleep(1.5);Path("+repr(sys.argv[2])+").write_text('done')"],start_new_session=True)
Path(sys.argv[1]).write_text(json.dumps({'pid':child.pid}))
while True:
 if child.poll() is not None:Path(sys.argv[3]).write_text('unexpected');time.sleep(30)
 time.sleep(.05)
"""
    external=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    parent=subprocess.Popen([sys.executable,'-c',program,str(ready),str(finished),str(dispatched)],start_new_session=True)
    try:
        deadline=time.time()+10
        while not ready.exists() and time.time()<deadline:time.sleep(.02)
        assert ready.exists();child=json.loads(ready.read_text())['pid']
        os.kill(parent.pid,signal.SIGSTOP)
        statuses=wait_natural_children({'old':{'pid':child}},tmp_path/'status.json')
        assert statuses['old']['exit_status']==0 and finished.exists() and not dispatched.exists()
        stop_parent_after_children(parent.pid);parent.wait(timeout=5)
        assert not dispatched.exists() and external.poll() is None
    finally:
        if parent.poll() is None:
            os.kill(parent.pid,signal.SIGTERM);os.kill(parent.pid,signal.SIGCONT);parent.wait(timeout=5)
        external.terminate();external.wait(timeout=5)
