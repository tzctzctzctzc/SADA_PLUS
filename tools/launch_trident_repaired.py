"""只在独立原语和真实模型重放诊断通过后，补全未正式训练的六项Trident。"""
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


def launch(root,output):
    identity=code_identity();output.mkdir(exist_ok=True)
    diag=Path('/vepfs/vlms/SADA_PLUS_worktrees/trident-replay-diagnostic-638149a-1104')
    if not diag.exists():
        subprocess.run(['git','-C','/vepfs/vlms/SADA_PLUS','worktree','add','--detach',str(diag),
            '638149a8e81451515d080baec6710e6b4ba88df3'],check=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(diag),
        PYTHONHASHSEED='123',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    proofs={}
    for name,tool,filename in [('window','diagnose_trident_replay.py','kernel_diagnosis.json'),
                               ('model','diagnose_trident_model.py','model_replay_diagnosis.json')]:
        stage=Path('/vepfs/vlms/trident_replay_'+name+'_638149a_1104')
        if not stage.exists():
            with (output/(name+'_diagnostic_driver.log')).open('x') as log:
                subprocess.run([sys.executable,'-u','-B',str(diag/'tools'/tool),'--root',str(stage)],
                    cwd=diag,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (stage/filename).exists(),'已有诊断根未通过，保留现场，不盲重启：'+str(stage)
        proofs[name]=read(stage/filename);assert proofs[name]['passed']
    assert all(x['native_unique_gradients']>1 for x in proofs['window']['results'].values())
    assert proofs['model']['evidence']['native']['full_state_unique']>1
    assert proofs['model']['evidence']['fixed']['full_state_unique']==1
    assert proofs['model']['evidence']['fixed']['loss_all_gradient_unique']==1
    old=Path('/vepfs/vlms/trident_complete6_8fdd2ae_1104')
    assert not read(old/'state.json')['active']
    assert all(not (old/f'S{s}_trident'/'formal_initialization.json').exists() and
               not (old/f'S{s}_trident'/'metrics.jsonl').exists() for s in range(1,7))
    save(output/'repair_acceptance.json',dict(passed=True,proofs=proofs,code=identity,
        old_root=str(old),old_formal_updates=0,native_window_gradient_nondeterminism_observed=True,
        same_forward_loss=True,new_full_state_replay_bit_exact=True,
        no_gate_tolerance_relaxed=True,protocol='same as previous interaction branch, not native-training reproduction'))
    if root.exists() or Path(str(root)+'_launch.json').exists():
        save(output/'launch_status.json',dict(already_present=str(root),no_duplicate_submission=True));return
    env.update(PYTHONPATH=str(CODE))
    with Path(str(root)+'.log').open('x') as log:
        controller=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/run_head_replacements.py'),'--root',str(root)],
            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(root=str(root),code=str(CODE),commit=identity['commit'],pid=controller.pid,
        repair_acceptance=str(output/'repair_acceptance.json'),repair_acceptance_sha256=sha(output/'repair_acceptance.json'),
        tasks=[f'S{s}_trident' for s in range(1,7)],fresh_random_initialization=True,trained_weights_reused=False,time_unix=time.time())
    save(Path(str(root)+'_launch.json'),receipt)
    for _ in range(100):
        if root.exists() or controller.poll() is not None:break
        time.sleep(.1)
    if root.exists() and controller.poll() is None:
        with (root/'independent_audit.log').open('x') as log:
            auditor=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/audit_head_replacements.py'),str(root)],
                cwd=CODE,env=dict(env,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'independent_audit_launch.json',dict(pid=auditor.pid,code=str(CODE),commit=identity['commit'],
            scope='six repaired Trident plus six immutable completed interaction reports'))
    save(output/'launch_status.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    try:launch(a.root,a.output)
    except BaseException as error:
        a.output.mkdir(exist_ok=True)
        save(a.output/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise
