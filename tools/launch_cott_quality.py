"""独立子进程隔离prepare/evaluate(CPU)与VLM(GPU)，完整结果自动评估。"""
import argparse
import fcntl
import os
import subprocess
import sys
import traceback
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.cott_protocol import save,sha,read
from tools.run_cott_quality import UUIDS

def launch(root):
    root=Path(root)
    # This is a single-experiment duplicate-controller lock, never a shared GPU lock.
    with root.with_name(root.name+'_controller.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=os.environ.copy();base.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
        stages=[]
        if not root.exists():stages.append(('prepare','prepare_cott_quality.py',False))
        else:assert (root/'plan.json').exists(), 'partial preparation needs explicit diagnosis'
        if not (root/'inference_complete.json').exists():stages.append(('generate','run_cott_quality.py',True))
        if not (root/'complete.json').exists():stages.append(('evaluate','evaluate_cott_quality.py',False))
        for phase,script,cuda in stages:
            env=base.copy();env['CUDA_VISIBLE_DEVICES']=','.join(UUIDS) if cuda else ''
            log=root.with_name(root.name+'_'+phase+'.log')
            with log.open('a') as f:
                command=[sys.executable,'-u','-B',str(CODE/'tools'/script),'--root',str(root)]
                child=subprocess.Popen(command,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
                receipt=dict(controller=os.getpid(),child=child.pid,phase=phase,command=command,
                    cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],log=str(log))
                save(root.with_name(root.name+'_supervisor.json'),receipt)
                print('PHASE',phase,child.pid,flush=True)
                rc=child.wait()
            receipt['exit_code']=rc;save(root.with_name(root.name+'_supervisor.json'),receipt)
            if rc:raise RuntimeError(f'{phase} exited {rc}; preserve outputs and inspect {log}')
        assert (root/'complete.json').exists()
        print('PILOT_COMPLETE',str(root),sha(root/'complete.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    try:launch(a.root)
    except BaseException as exc:
        save(Path(a.root).with_name(Path(a.root).name+'_supervisor_error.json'),
            dict(error=str(exc),traceback=traceback.format_exc()))
        raise
