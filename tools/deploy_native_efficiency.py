"""完整真实CPU等价/吞吐门禁通过后，交接本任务尚未开始的训练。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity


def deploy(root,output):
    identity=code_identity();output.mkdir(parents=True,exist_ok=True)
    source=Path('/vepfs/vlms/head_replacements12_764a29a_1104/S1_interaction')
    proof=output/'real_prediction_benchmark.json'
    if not proof.exists():
        with (output/'benchmark.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-B',str(CODE/'tools/benchmark_native_ap_fast.py'),str(source),str(proof),
                '--slot-root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True)
    benchmark=read(proof);assert benchmark['passed'] and benchmark['fast_code_sha256']==sha(CODE/'tools/native_ap_fast.py')
    save(output/'acceptance.json',dict(passed=True,code=identity,benchmark_sha256=sha(proof),
        model_forward_losses_optimizers_dataset_and_train_control_unchanged=True,
        scope='CPU evaluation optimization only; all AP TP FP FN and CPU/Python/NumPy RNG exact on full saved source+target predictions',
        speedup={k:v['speedup'] for k,v in benchmark['results'].items()},GPU_training_speedup_claimed=False))
    if (root/'final.json').exists() or (root/'efficiency_handoff.json').exists():
        save(output/'deployment.json',dict(status='already_complete_or_handed_off_no_duplicate_dispatch',root=str(root)))
        return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (output/'dispatcher.log').open('x') as log:
        driver=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/run_release_data_reproduction.py'),
            '--root',str(root),'--adoption-proof',str(proof)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        if (root/'efficiency_handoff.json').exists() or driver.poll() is not None:break
        time.sleep(.2)
    if not (root/'efficiency_handoff.json').exists():
        raise RuntimeError('handoff did not complete; see dispatcher.log; old dispatcher is restored on pre-handoff failure')
    save(output/'deployment.json',dict(status='validated_optimization_dispatched_for_unstarted_tasks',
        root=str(root),driver_pid=driver.pid,commit=identity['commit'],
        handoff_sha256=sha(root/'efficiency_handoff.json'),existing_training_restarted=False,
        new_task_GPU_acceptance='each new task still requires real forward/gradient/update/full fixed-batch state replay'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    try:deploy(a.root,a.output)
    except BaseException as error:
        a.output.mkdir(parents=True,exist_ok=True)
        save(a.output/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise
