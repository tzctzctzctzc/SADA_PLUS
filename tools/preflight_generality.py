"""两种完整检测器×三种条件的真实GPU预检；不启动正式训练。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,code_identity
from tools.baseline_gpu import slot


def run(root):
    identity=code_identity();reports={}
    for detector in ('actionformer','tridet'):
        for condition in ('so','sada','sada_prototype'):
            key=detector+'/'+condition;entry=root/'preflight'/detector/condition/'S1'
            save(entry/'config.json',read(root/detector/condition/'S1/config.json'))
            while True:
                admitted=False
                for gpu in range(4):
                    ctx=slot(gpu,'/vepfs/vlms/SADA_PLUS',root,16384,identity['commit'])
                    try:lease=ctx.__enter__()
                    except BlockingIOError:continue
                    admitted=True
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],PYTHONPATH=str(CODE),
                        PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='123',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=lease['lease'],
                        SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                    try:
                        with (entry/'probe.log').open('x') as output:
                            p=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/run_generality.py'),
                                'probe','--entry',str(entry)],cwd=CODE,env=env,stdout=output,stderr=subprocess.STDOUT)
                        save(root/'preflight_state.json',dict(active=key,pid=p.pid,gpu=gpu,completed=list(reports)))
                        code=p.wait()
                        if code:
                            save(root/'preflight_batch.json',dict(passed=False,failed=key,exitcode=code,
                                log=str(entry/'probe.log'),completed=reports,formal_training_runs=0))
                            raise SystemExit(1)
                        reports[key]=read(entry/'probe.json')
                    finally:ctx.__exit__(None,None,None)
                    break
                if admitted:break
                save(root/'preflight_state.json',dict(active=None,waiting_for_gpu=True,completed=list(reports)))
                time.sleep(10)
    save(root/'preflight_batch.json',dict(passed=True,completed=reports,formal_training_runs=0,
        original_prototype=True,old_hybrid_not_tested=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(Path(a.root))
