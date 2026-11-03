"""仅补已经正常结束训练、因输出目录缺失失败的评估，绝不调用train。"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, termination
from tools.baseline_gpu import snapshot


def run(root, task):
    entry = root/task
    state = read(root/'state.json')
    assert not (root/'stop.json').exists()
    assert task not in state['active'] and state['failed'][task]['stage'] == 'evaluate'
    assert read(entry/'train_complete.json')['passed']
    assert not (entry/'evaluate_complete.json').exists() and not (entry/'complete.json').exists()
    assert 'Parent directory' in (entry/'evaluate.log').read_text()
    assert 'does not exist' in (entry/'evaluate.log').read_text()
    cfg = read(entry/'config.json')
    source_code = Path(read(root/'_launch.json')['code'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_code, text=True).strip() == cfg['baseline']['code']['commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source_code, text=True).strip()
    selected = termination(read(entry/'termination.json')['history'])
    immutable = {name:sha(entry/name) for name in ('best.pt','latest.pt','metrics.jsonl','termination.json','train_complete.json','config.json')}
    for name in ('S1_ae','S1_vae','S2_ae','S2_vae'):
        (root/name/'evaluation').mkdir(exist_ok=True)
    receipt = entry/'evaluate_recovery_launch.json'
    if receipt.exists():
        raise RuntimeError('recovery already launched; inspect old owner instead of duplicate')
    child = None
    def stop(sig, frame):
        save(entry/'evaluate_recovery_stop.json',dict(signal=sig,automatic_restart_authorized=False))
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)
    while True:
        if (root/'stop.json').exists():
            return
        state = read(root/'state.json')
        occupied = {value['gpu'] for value in state['active'].values()}
        choices = [snapshot(gpu) for gpu in range(4) if gpu not in occupied]
        choices = [value for value in choices if value['free_mib'] >= 16384]
        if choices:
            observed = choices[0]
            break
        time.sleep(10)
    lockpath = root/('recovery_private_gpu_'+str(observed['index'])+'.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        leasepath = root/'recovery_leases'/(task+'_'+str(time.time_ns())+'.json')
        save(leasepath,dict(owner_pid=os.getpid(),lock=str(lockpath),observed=observed,
             minimum_free_mib=16384,commit=cfg['baseline']['code']['commit']))
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=observed['uuid'],PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=str(source_code),PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=str(leasepath),
            SADA_BASELINE_LEASE_SHA256=sha(leasepath),TMPDIR=cfg['baseline']['tmpdir'],
            TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'])
        with (entry/'evaluate_recovery.log').open('x') as output:
            child = subprocess.Popen([sys.executable,'-u','-B',str(source_code/'tools/run_domain_factor_vae.py'),
                '--entry',str(entry),'--stage','evaluate'],cwd=source_code,env=env,stdout=output,
                stderr=subprocess.STDOUT,start_new_session=True)
        save(receipt,dict(owner_pid=os.getpid(),child_pid=child.pid,task=task,
            original_training_owner=state['owner_pid'],lease=str(leasepath),lease_sha256=sha(leasepath),
            source_code=str(source_code),source_commit=cfg['baseline']['code']['commit'],
            repair_code=str(CODE),before=immutable,selected=selected,
            reason='create missing evaluation directories; rerun evaluation only',training_repeated=False))
        while child.poll() is None:
            if (root/'stop.json').exists():
                stop(signal.SIGTERM,None)
            time.sleep(5)
        after = {name:sha(entry/name) for name in immutable}
        assert immutable == after, 'evaluation altered saved training artifacts'
        if child.returncode:
            save(entry/'evaluate_recovery_error.json',dict(exit_code=child.returncode,training_artifacts_unchanged=True))
            raise RuntimeError('supplemental evaluation failed; preserve logs')
        files = ['config.json','origin_binding.json','probe.json','formal_initialization.json',
                 'train_complete.json','evaluate_complete.json','best.pt','latest.pt','metrics.jsonl',
                 'termination.json','evaluation/source_val.json','evaluation/target_val.json',
                 'evaluation/source_predictions.pt','evaluation/target_predictions.pt']
        save(entry/'evaluate_recovery_complete.json',dict(passed=True,receipt_sha256=sha(receipt),
            unchanged_training=after,original_producer_failure_retained=True))
        save(entry/'complete.json',dict(passed=True,artifacts={name:sha(entry/name) for name in files},
             recovered_evaluation=True,recovery_sha256=sha(entry/'evaluate_recovery_complete.json')))
        print('EVALUATION_RECOVERED',task,read(entry/'evaluation/target_val.json'),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--task',default='S1_ae')
    args=parser.parse_args()
    run(args.root,args.task)
