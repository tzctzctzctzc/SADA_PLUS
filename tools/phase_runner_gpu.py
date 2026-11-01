"""用户允许四卡共享；本矩阵每卡一个自有worker，不使用公共互斥锁。"""
from contextlib import contextmanager
import os,time
from pathlib import Path
from tools.baseline_gpu import snapshot
from tools.baseline_protocol import save,sha,read

@contextmanager
def slot(index,data,root,minimum,commit):
 s=snapshot(index)
 if s['free_mib']<minimum:raise BlockingIOError('free VRAM')
 p=Path(root)/'gpu_admissions'/('%d_%d.json'%(os.getpid(),time.time_ns()))
 save(p,dict(owner_pid=os.getpid(),observed=s,minimum_free_mib=minimum,commit=commit,shared_no_public_lock=True))
 yield dict(index=index,uuid=s['uuid'],lease=str(p),lease_sha256=sha(p))

def acquire(indices,data,root,minimum,commit):
 # 自有live worker的卡由父调度器显式排除；外部任务不读取命令/环境、不干预。
 for index in indices:
  c=slot(index,data,root,minimum,commit)
  try:card=c.__enter__();return c,card
  except BlockingIOError:continue
 return None

def first_cuda_guard():
 p=Path(os.environ['SADA_BASELINE_LEASE']);assert sha(p)==os.environ['SADA_BASELINE_LEASE_SHA256']
 lease=read(p);assert lease['owner_pid']==os.getppid()
 now=snapshot(lease['observed']['index'])
 assert now['uuid']==os.environ['CUDA_VISIBLE_DEVICES']==lease['observed']['uuid']
 assert now['free_mib']>=lease['minimum_free_mib']
 proof=p.with_name(p.stem+'_guard_%d.json'%os.getpid())
 save(proof,dict(lease=str(p),lease_sha256=sha(p),observed=now,child_pid=os.getpid(),owner_pid=os.getppid()))
 return str(proof)
