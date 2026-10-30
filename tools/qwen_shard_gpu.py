"""四卡承载同一模型；必须同时取得全部物理卡锁，首CUDA前逐卡复核。"""
from contextlib import ExitStack
import os
from pathlib import Path
import time

from tools.baseline_protocol import read,save,sha
from tools.baseline_gpu import slot,snapshot

MIN_FREE_MIB=12288
WEIGHT_BUDGET_MIB=8192


def acquire_four(indices,data,root,commit,slot_fn=slot):
    if len(indices)!=4 or set(indices)!={0,1,2,3}:
        raise ValueError('四卡分片必须精确使用物理0/1/2/3')
    held=ExitStack();cards=[]
    try:
        for index in sorted(indices):
            cards.append(held.enter_context(slot_fn(index,data,root,MIN_FREE_MIB,commit)))
    except BlockingIOError:
        held.close();return None
    except BaseException:
        held.close();raise
    return held,cards


def verify_cards(cards,parent,visible,observe,sig_sha=sha):
    if len(cards)!=4 or {c['index'] for c in cards}!={0,1,2,3}:
        raise ValueError('四卡租约不完整')
    if visible!=','.join(c['uuid'] for c in cards):
        raise ValueError('可见GPU与四卡租约顺序不符')
    snapshots=[]
    for card in cards:
        lease=read(card['lease'])
        if (sig_sha(card['lease'])!=card['lease_sha256'] or lease['owner_pid']!=parent
                or lease['observed']['uuid']!=card['uuid'] or lease['minimum_free_mib']!=MIN_FREE_MIB):
            raise ValueError('租约SHA/直接父进程/卡身份改变')
        now=observe(card['index'])
        if now['uuid']!=card['uuid']:
            raise ValueError('首CUDA GPU UUID改变')
        snapshots.append(now)
    return snapshots,all(x['free_mib']>=MIN_FREE_MIB for x in snapshots)


def first_cuda_guard_four(root):
    import fcntl
    path=Path(os.environ['QWEN_SHARD_LEASES'])
    if sha(path)!=os.environ['QWEN_SHARD_LEASES_SHA256']:
        raise ValueError('四卡租约总清单SHA改变')
    cards=read(path)
    while True:
        observed,enough=verify_cards(cards,os.getppid(),os.environ.get('CUDA_VISIBLE_DEVICES'),snapshot)
        for card in cards:
            lease=read(card['lease'])
            with Path(lease['lock']).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise ValueError('父进程未持有四张物理卡锁')
        if enough:break
        print('WAIT_BEFORE_FOUR_CUDA',[(x['index'],x['free_mib']) for x in observed],flush=True);time.sleep(10)
    proof=Path(root)/'first_cuda_four.json'
    save(proof,dict(status='passed',cards=cards,observed=observed,child_pid=os.getpid(),owner_pid=os.getppid(),
        leases_sha256=sha(path),model_weight_budget_per_gpu_mib=WEIGHT_BUDGET_MIB))
    print('FIRST_CUDA_FOUR_PASS',[(x['index'],x['free_mib']) for x in observed],flush=True)
    return str(proof)
