"""真实CPU父子进程交接：仅暂停派发器，已运行子阶段自然完成。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pytest

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))


@pytest.mark.skipif(os.name!='posix',reason='requires Linux /proc and SIGSTOP')
def test_preserve_existing_process_and_only_update_unstarted_config(tmp_path):
    from tools.baseline_protocol import save,read,sha
    from tools.native_queue_handoff import adopt,retire_previous,process_state
    old_code=tmp_path/'old_code';root=tmp_path/'run';root.mkdir()
    for rel in ['tools/baseline_engine.py','libs/models/SADA/meta_archs.py','libs/models/SADA/prototype_capacity.py',
                'libs/trainer/da_trainer.py','libs/trainer/base_trainer_da.py','libs/datasets/epic_kitchens.py']:
        p=old_code/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(CODE/rel,p)
    identity={'commit':'fast','core_files':{'model':'same'}}
    tasks=['S1_sada','S1_prototype'];save(root/'plan.json',dict(tasks=tasks))
    for task in tasks:
        (root/task).mkdir();save(root/task/'config.json',{'model':{'loss':1},'baseline':{'code':{'commit':'old','core_files':{'model':'same'}}}})
    save(tmp_path/'proof.json',dict(passed=True,fast_code_sha256=sha(CODE/'tools/native_ap_fast.py')))
    script=old_code/'tools/driver.py'
    script.write_text("""import json,os,pathlib,signal,subprocess,sys,time
r=pathlib.Path(sys.argv[-1]);entry=r/'S1_sada'
worker="import pathlib,time,json;time.sleep(2);pathlib.Path("+repr(str(entry/'train_complete.json'))+").write_text(json.dumps({'passed':True}))"
p=subprocess.Popen([sys.executable,'-c',worker,'--entry',str(entry),'--stage','train'])
def stop(sig,frame):
 (r/'stop.json').write_text(json.dumps({'signal':sig,'automatic_restart_authorized':False}));raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
(r/'state.json').write_text(json.dumps({'owner_pid':os.getpid(),'active':{'S1_sada':{'pid':p.pid,'stage':'train','gpu':0}},'pending':['S1_prototype'],'waiting_evaluate':[],'completed':[],'failed':{}}))
while True:time.sleep(.1)
""")
    parent=subprocess.Popen([sys.executable,str(script),'--root',str(root)])
    try:
        for _ in range(100):
            if (root/'state.json').exists():break
            time.sleep(.02)
        pending,waiting,active,done,failed,handoff=adopt(root,identity,tmp_path/'proof.json',CODE)
        assert pending==['S1_prototype'] and not done and not failed and not waiting
        existing=active['S1_sada']['p'];assert existing.pid==handoff['previous_state']['active']['S1_sada']['pid']
        assert process_state(parent.pid) in ('T','t')
        assert process_state(existing.pid) not in ('T','t')
        assert read(root/'S1_sada/config.json')['baseline']['code']['commit']=='old'
        assert read(root/'S1_prototype/config.json')['baseline']['code']==identity
        for _ in range(150):
            if existing.poll() is not None:break
            time.sleep(.03)
        assert existing.poll()==0 and read(root/'S1_sada/train_complete.json')['passed']
        retire_previous(root,handoff)
        assert not (root/'stop.json').exists() and (root/'dispatcher_retired_for_efficiency.json').exists()
        parent.wait(timeout=5)
    finally:
        if parent.poll() is None:
            import signal
            os.kill(parent.pid,signal.SIGCONT);parent.terminate();parent.wait(timeout=5)
