"""只与用户明确指定、身份核实的研究监督共享；每卡最多一个额外任务。"""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import json
import threading
import time

from tools.baseline_protocol import read,save,sha
from tools.baseline_gpu import snapshot


def process_status(pid):
    p=Path('/proc/%d/stat'%int(pid))
    if not p.exists():return None
    a=p.read_text().rsplit(')',1)[1].split()
    return dict(state=a[0],ppid=int(a[1]),start_ticks=a[19])


def public_owner(path):
    result=subprocess.check_output(['lslocks','--json','--output','PID,PATH'],text=True)
    owners=set()
    for row in json.loads(result).get('locks',[]):
        if row.get('path')==str(path):owners.add(int(row['pid']));continue
        if row.get('path') not in (None,'','/...'):continue
        # overlay等文件系统上lslocks可能只显示/…；按真实FD身份识别，不猜路径。
        directory=Path('/proc/%d/fd'%row['pid'])
        try:descriptors=list(directory.iterdir())
        except (FileNotFoundError,PermissionError):continue
        for descriptor in descriptors:
            try:
                if not os.path.samefile(descriptor,path):continue
                lines=(directory.parent/'fdinfo'/descriptor.name).read_text().splitlines()
                for line in lines:
                    if line.startswith('lock:') and ' FLOCK ' in line and ' WRITE ' in line:
                        parts=line.split();owners.add(int(parts[parts.index('WRITE')+1]))
            except (FileNotFoundError,PermissionError):continue
    if len(owners)>1:raise ValueError('公共锁存在多个拥有者')
    return next(iter(owners),None)


def load_policy(path,digest=None):
    if digest is not None and sha(path)!=digest:raise ValueError('共享授权工件SHA改变')
    p=read(path)
    if p['schema']!='verified-research-peer-share-v1' or not p['user_authorized']:
        raise ValueError('缺少明确研究任务共享授权')
    return p


def verify_peer(index,policy):
    status=process_status(policy['peer_pid'])
    if not status or status['state']=='Z' or status['start_ticks']!=policy['peer_start_ticks']:
        raise BlockingIOError('指定共享监督身份已结束或改变')
    root=Path(policy['peer_root'])
    if sha(root/'plan.json')!=policy['peer_plan_sha256']:raise ValueError('原研究实验计划改变')
    plan=read(root/'plan.json');state=read(root/'state.json')
    if plan['code']['commit']!=policy['peer_commit'] or state['pid']!=policy['peer_pid']:
        raise ValueError('共享监督代码/实验身份不符')
    entries=[x for x in state.get('active',{}).values() if x['gpu']==index]
    if len(entries)!=1:raise BlockingIOError('等待原监督发布对应卡的阶段身份')
    child=entries[0];child_status=process_status(child['pid'])
    if child_status and child_status['state']!='Z' and child_status['ppid']!=policy['peer_pid']:
        raise ValueError('原研究子进程父子关系不符')
    return dict(peer_pid=policy['peer_pid'],peer_start_ticks=policy['peer_start_ticks'],peer_commit=policy['peer_commit'],
                peer_plan_sha256=policy['peer_plan_sha256'],peer_child_pid=child['pid'],peer_stage=child['stage'])


@contextmanager
def slot(index,data,root,minimum,commit,policy_path):
    import fcntl
    policy_path=Path(policy_path);policy=load_policy(policy_path)
    if str(index) not in policy['gpu_uuids']:raise BlockingIOError('该卡不在共享授权中')
    # 不删除、释放或修改原监督的锁；额外槽使用共同的独立锁，防止重复新增。
    extra=(Path(data)/('.rq2_gpu%d.research_extra.lock'%index)).open('a')
    primary=(Path(data)/('.rq2_gpu%d.lock'%index)).open('a')
    done=threading.Event();thread=None
    try:
        fcntl.flock(extra,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB);mode='primary_owned';peer=None
        except BlockingIOError:
            if public_owner(primary.name)!=policy['peer_pid']:raise BlockingIOError('不是已核实的共享监督')
            peer=verify_peer(index,policy);mode='verified_peer'
        observed=snapshot(index)
        if observed['uuid']!=policy['gpu_uuids'][str(index)]:raise ValueError('共享GPU UUID改变')
        if observed['free_mib']<minimum:raise BlockingIOError('共享显存余量不足')
        lease=Path(root)/'gpu_admissions'/('%d_%d.json'%(os.getpid(),time.time_ns()))
        shared=dict(policy=str(policy_path),policy_sha256=sha(policy_path),primary_lock=str(Path(primary.name).absolute()),
                    admission_mode=mode,peer_at_admission=peer)
        save(lease,dict(owner_pid=os.getpid(),lock=str(Path(extra.name).absolute()),observed=observed,
                        minimum_free_mib=minimum,commit=commit,research_share=shared))
        if mode=='verified_peer':
            # 原阶段释放公共锁后自动接管；不向原监督或其他GPU进程发送信号。
            def promote():
                while not done.wait(.2):
                    try:fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    save(lease.with_name(lease.stem+'_promotion.json'),dict(owner_pid=os.getpid(),primary_lock=shared['primary_lock']))
                    break
            thread=threading.Thread(target=promote,daemon=True);thread.start()
        yield dict(index=index,uuid=observed['uuid'],lease=str(lease),lease_sha256=sha(lease))
    finally:
        done.set()
        if thread is not None:thread.join(timeout=2)
        primary.close();extra.close()


def first_cuda_membership(lease):
    shared=lease['research_share'];policy=load_policy(shared['policy'],shared['policy_sha256'])
    index=lease['observed']['index']
    if policy['gpu_uuids'].get(str(index))!=lease['observed']['uuid']:raise ValueError('未授权的共享GPU')
    if Path(lease['lock']).name!='.rq2_gpu%d.research_extra.lock'%index:raise ValueError('共享额外槽锁不符')
    while True:
        if os.getppid()!=lease['owner_pid']:raise ValueError('共享阶段父进程改变')
        owner=public_owner(shared['primary_lock'])
        if owner==lease['owner_pid']:
            return dict(mode='primary_owned',owner_pid=owner,policy_sha256=shared['policy_sha256'])
        if owner==policy['peer_pid']:
            try:peer=verify_peer(index,policy)
            except BlockingIOError:pass
            else:return dict(mode='verified_peer',policy_sha256=shared['policy_sha256'],**peer)
        print('WAITING_FOR_VERIFIED_GPU_COHORT',index,flush=True);time.sleep(2)


def audit_guard(lease,guard):
    shared=lease['research_share'];p=load_policy(shared['policy'],shared['policy_sha256']);e=guard['research_share']
    index=lease['observed']['index']
    if p['gpu_uuids'].get(str(index))!=guard['observed']['uuid'] or e['policy_sha256']!=shared['policy_sha256']:
        raise ValueError('共享首CUDA的GPU/授权SHA不符')
    if e['mode']=='verified_peer':
        for key in ('peer_pid','peer_start_ticks','peer_commit','peer_plan_sha256'):
            if e[key]!=p[key]:raise ValueError('共享首CUDA的研究监督身份不符')
    elif e['mode']!='primary_owned' or e['owner_pid']!=lease['owner_pid']:
        raise ValueError('共享公共锁接管证明不符')
