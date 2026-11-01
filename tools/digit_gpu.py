"""用户授权共享四卡：不获取公共排他锁，只记录显存准入与首次CUDA凭证。"""
import os
from pathlib import Path
import subprocess
import time
from tools.digit_protocol import read,save,sha

def snapshot(index):
    line=subprocess.check_output(['nvidia-smi','-i',str(index),
        '--query-gpu=uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,total,used,util=[v.strip() for v in line]
    return dict(index=int(index),uuid=uuid,total_mib=int(total),used_mib=int(used),
                free_mib=int(total)-int(used),utilization=int(util))

def admit(index,root,minimum_free_mib,commit):
    observed=snapshot(index)
    if observed['free_mib']<minimum_free_mib:raise BlockingIOError('显存余量不足')
    path=Path(root)/'gpu_admissions'/('%d_%d.json'%(os.getpid(),time.time_ns()))
    save(path,dict(owner_pid=os.getpid(),observed=observed,minimum_free_mib=minimum_free_mib,
        commit=commit,policy='shared_without_public_exclusive_lock'))
    return dict(index=index,uuid=observed['uuid'],lease=str(path),lease_sha256=sha(path))

def first_cuda_guard():
    path=Path(os.environ['SADA_BASELINE_LEASE'])
    assert sha(path)==os.environ['SADA_BASELINE_LEASE_SHA256']
    lease=read(path)
    assert lease['policy']=='shared_without_public_exclusive_lock'
    assert os.environ['CUDA_VISIBLE_DEVICES']==lease['observed']['uuid']
    while True:
        assert os.getppid()==lease['owner_pid']
        now=snapshot(lease['observed']['index'])
        assert now['uuid']==lease['observed']['uuid']
        if now['free_mib']>=lease['minimum_free_mib']:break
        print('WAITING_BEFORE_CUDA',now['free_mib'],flush=True);time.sleep(10)
    proof=path.with_name(path.stem+'_guard_%d.json'%os.getpid())
    save(proof,dict(lease=str(path),lease_sha256=sha(path),observed=now,
        child_pid=os.getpid(),owner_pid=lease['owner_pid']))
    print('FIRST_CUDA_GUARD_PASS',now['index'],now['free_mib'],flush=True)
    return str(proof)
