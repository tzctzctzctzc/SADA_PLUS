import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools import research_gpu_share as sharing
from tools.baseline_protocol import save,sha,read


def policy(tmp_path):
    root=tmp_path/'peer';save(root/'plan.json',dict(code={'commit':'peer-code'}))
    save(root/'state.json',dict(pid=101,active={'S6':dict(gpu=1,pid=102,stage='train')}))
    path=tmp_path/'policy.json'
    value=dict(schema='verified-research-peer-share-v1',user_authorized=True,peer_pid=101,peer_start_ticks='1000',
        peer_root=str(root),peer_commit='peer-code',peer_plan_sha256=sha(root/'plan.json'),gpu_uuids={'1':'uuid1'})
    save(path,value);return path,value


@pytest.mark.parametrize('change',['pid_reused','plan_changed','wrong_parent','wrong_commit'])
def test_peer_identity_is_more_than_same_user(tmp_path,monkeypatch,change):
    path,p=policy(tmp_path)
    status={101:dict(state='S',ppid=1,start_ticks='1000'),102:dict(state='R',ppid=101,start_ticks='1100')}
    monkeypatch.setattr(sharing,'process_status',lambda pid:status.get(pid))
    assert sharing.verify_peer(1,p)['peer_child_pid']==102
    if change=='pid_reused':status[101]['start_ticks']='other'
    elif change=='plan_changed':save(Path(p['peer_root'])/'plan.json',{'other':'plan'})
    elif change=='wrong_parent':status[102]['ppid']=999
    else:p['peer_commit']='other'
    with pytest.raises((ValueError,BlockingIOError)):sharing.verify_peer(1,p)


def test_audit_keeps_shared_authorization_and_uuid_bound(tmp_path):
    path,p=policy(tmp_path);digest=sha(path)
    lease=dict(owner_pid=201,observed=dict(index=1,uuid='uuid1'),research_share=dict(policy=str(path),policy_sha256=digest))
    evidence=dict(mode='verified_peer',policy_sha256=digest,**{k:p[k] for k in ('peer_pid','peer_start_ticks','peer_commit','peer_plan_sha256')})
    guard=dict(observed={'uuid':'uuid1'},research_share=evidence)
    sharing.audit_guard(lease,guard)
    guard['research_share']['peer_pid']=999
    with pytest.raises(ValueError):sharing.audit_guard(lease,guard)
    guard['research_share']=dict(mode='primary_owned',owner_pid=201,policy_sha256=digest)
    sharing.audit_guard(lease,guard)
    guard['observed']['uuid']='other'
    with pytest.raises(ValueError):sharing.audit_guard(lease,guard)
    save(path,dict(p,user_authorized=False))
    with pytest.raises(ValueError):sharing.load_policy(path,digest)


@pytest.mark.skipif(os.name=='nt',reason='真实Linux flock/进程组')
def test_extra_slot_coexists_without_stopping_peer_and_promotes_public_lock(tmp_path,monkeypatch):
    import fcntl
    from tools.baseline_gpu import slot as ordinary_slot
    data=tmp_path/'data';data.mkdir();primary=data/'.rq2_gpu1.lock';ready=tmp_path/'ready'
    program="import fcntl,time,sys;from pathlib import Path;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);Path(sys.argv[2]).write_text('ready');time.sleep(60)"
    peer=subprocess.Popen([sys.executable,'-B','-c',program,str(primary),str(ready)],start_new_session=True)
    try:
        until=time.monotonic()+5
        while not ready.exists():assert time.monotonic()<until;time.sleep(.02)
        path,p=policy(tmp_path);p.update(peer_pid=peer.pid,peer_start_ticks=sharing.process_status(peer.pid)['start_ticks']);save(path,p)
        monkeypatch.setattr(sharing,'verify_peer',lambda index,p:{'peer_pid':peer.pid})
        monkeypatch.setattr(sharing,'snapshot',lambda index:dict(index=index,uuid='uuid1',free_mib=20000,external_pids=[peer.pid]))
        with pytest.raises(BlockingIOError):
            with ordinary_slot(1,data,tmp_path/'ordinary',9000,'code'):pass
        with sharing.slot(1,data,tmp_path/'shared',12000,'code',path) as card:
            assert peer.poll() is None and read(card['lease'])['research_share']['admission_mode']=='verified_peer'
            with pytest.raises(BlockingIOError):
                with sharing.slot(1,data,tmp_path/'duplicate',12000,'code',path):pass
            peer.terminate();peer.wait(timeout=5)
            until=time.monotonic()+5
            while sharing.public_owner(primary)!=os.getpid():assert time.monotonic()<until;time.sleep(.05)
            with primary.open('a') as other:
                with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with primary.open('a') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
        if peer.poll() is None:peer.terminate();peer.wait(timeout=5)


@pytest.mark.skipif(os.name=='nt',reason='真实Linux flock')
@pytest.mark.parametrize('case',['unknown_owner','low_memory','wrong_uuid'])
def test_slot_refuses_unknown_owner_or_insufficient_resource(tmp_path,monkeypatch,case):
    import fcntl
    data=tmp_path/'data';data.mkdir();held=(data/'.rq2_gpu1.lock').open('a');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path,p=policy(tmp_path)
    monkeypatch.setattr(sharing,'public_owner',lambda path:999 if case=='unknown_owner' else p['peer_pid'])
    monkeypatch.setattr(sharing,'verify_peer',lambda *args:dict(peer_pid=p['peer_pid']))
    monkeypatch.setattr(sharing,'snapshot',lambda index:dict(index=index,uuid='other' if case=='wrong_uuid' else 'uuid1',free_mib=8000 if case=='low_memory' else 20000))
    try:
        with pytest.raises((BlockingIOError,ValueError)):
            with sharing.slot(1,data,tmp_path/'out',12000,'code',path):pass
    finally:held.close()
