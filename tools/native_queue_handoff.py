"""只交接未开始的任务；已经运行的阶段保留原进程、配置和全部状态。"""
import json
import os
from pathlib import Path
import signal
import time
from tools.baseline_protocol import read,save,sha


def process_state(pid):
    p=Path('/proc')/str(pid)
    try:return (p/'stat').read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError:return None


class ExistingProcess:
    def __init__(self,pid,entry,stage):self.pid,self.entry,self.stage,self.returncode=pid,entry,stage,None
    def poll(self):
        if process_state(self.pid) not in (None,'Z','X'):return None
        self.returncode=0 if (self.entry/(self.stage+'_complete.json')).exists() else 1
        return self.returncode


class RetainedParentLock:
    def close(self):pass


def adopt(root,identity,proof,code):
    proof=Path(proof);assert read(proof)['passed']
    assert read(proof)['fast_code_sha256']==sha(code/'tools/native_ap_fast.py')
    assert not (root/'stop.json').exists() and not (root/'efficiency_handoff.json').exists()
    state=read(root/'state.json');old_owner=state['owner_pid'];proc=Path('/proc')/str(old_owner)
    args=(proc/'cmdline').read_bytes().split(b'\0');assert str(root).encode() in args and b'--root' in args
    old_code=Path(args[args.index(b'--root')-1].decode()).parents[1]
    # 模型/损失/优化器/训练控制逐字节保留；优化只在新进程替换CPU评测函数。
    for rel in ['tools/baseline_engine.py','libs/models/SADA/meta_archs.py','libs/models/SADA/prototype_capacity.py',
                'libs/trainer/da_trainer.py','libs/trainer/base_trainer_da.py','libs/datasets/epic_kitchens.py']:
        assert sha(old_code/rel)==sha(code/rel),rel
    os.kill(old_owner,signal.SIGSTOP);time.sleep(.2)
    try:
        state=read(root/'state.json');actual=set()
        for path in Path('/proc').glob('[0-9]*'):
            try:
                fields=(path/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[1])!=old_owner or fields[0]=='Z':continue
                cmd=(path/'cmdline').read_bytes().split(b'\0')
                if b'--entry' in cmd:actual.add(int(path.name))
            except (FileNotFoundError,ProcessLookupError):pass
        expected={a['pid'] for a in state['active'].values() if process_state(a['pid']) not in (None,'Z','X')}
        assert actual==expected,'调度器正处于派发边界；先恢复原队列，不猜测worker归属'
        tasks=read(root/'plan.json')['tasks'];active={};waiting=[];done=list(state['completed']);failed=dict(state['failed'])
        old_entries=set(state['active'])|set(state.get('waiting_evaluate',[]))|set(done)|set(failed)
        pending=[t for t in tasks if t not in old_entries]
        versions={}
        for task in tasks:
            entry=root/task;cfg=read(entry/'config.json')
            versions[task]=str(old_code if task in old_entries else code)
            if task not in old_entries:
                assert not (entry/'train.log').exists() and not (entry/'formal_initialization.json').exists()
                save(entry/'config_before_efficiency.json',cfg)
                assert cfg['baseline']['code']['core_files']==identity['core_files']
                cfg['baseline']['code']=identity;save(entry/'config.json',cfg)
        for task,a in state['active'].items():
            active[task]=dict(p=ExistingProcess(a['pid'],root/task,a['stage']),gpu=a['gpu'],stage=a['stage'],
                lease=None,lock=RetainedParentLock(),inherited=True)
        for task in state.get('waiting_evaluate',[]):
            # 原阶段已完成；仅补缺失评估，不重跑训练。
            record=read(root/task/'train_complete.json');guard=read(record['guard'])
            waiting.append((task,dict(gpu=guard['observed']['index'],lease=None,lock=RetainedParentLock(),inherited=True)))
        value=dict(previous_owner=old_owner,previous_code=str(old_code),previous_state=state,
            previous_controller_start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19],
            new_owner=os.getpid(),new_code=str(code),new_commit=identity['commit'],versions=versions,
            inherited_tasks=sorted(old_entries),optimization_proof=str(proof),optimization_proof_sha256=sha(proof),
            original_plan_sha256=sha(root/'plan.json'),reason='equivalent CPU evaluation optimization for unstarted tasks only',
            no_training_worker_stopped=True,no_trained_prefix_repeated=True,time_unix=time.time())
        save(root/'efficiency_handoff.json',value)
        audit=root/'independent_audit_launch.json'
        if audit.exists():
            pid=read(audit)['pid'];p=Path('/proc')/str(pid)
            if p.exists():
                cmd=(p/'cmdline').read_bytes().split(b'\0')
                if str(root).encode() in cmd and any(b'audit_release_reproduction.py' in x for x in cmd):
                    os.kill(pid,signal.SIGTERM)
        return pending,waiting,active,done,failed,value
    except BaseException:
        # 派发边界检查失败时不让临时暂停阻塞原训练队列。
        for p in root.glob('S*/config_before_efficiency.json'):
            save(p.parent/'config.json',read(p))
        if (root/'efficiency_handoff.json').exists():
            (root/'efficiency_handoff.json').rename(root/'efficiency_handoff_aborted.json')
        os.kill(old_owner,signal.SIGCONT)
        raise


def retire_previous(root,handoff):
    pid=handoff['previous_owner'];p=Path('/proc')/str(pid)
    if p.exists():
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        assert fields[19]==handoff['previous_controller_start_ticks']
        assert fields[0] in ('T','t')
        # 所有原worker已自然结束，才让原调度器执行自身退出清理。
        assert all(process_state(a['pid']) in (None,'Z','X') for a in handoff['previous_state']['active'].values())
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        for _ in range(100):
            if process_state(pid) in (None,'Z','X'):break
            time.sleep(.1)
        assert process_state(pid) in (None,'Z','X')
    if (root/'stop.json').exists():
        stop=read(root/'stop.json');assert stop['signal']==signal.SIGTERM
        (root/'stop.json').rename(root/'dispatcher_retired_for_efficiency.json')
    save(root/'previous_dispatcher_retired.json',dict(pid=pid,time_unix=time.time(),scientific_failure=False,
        all_original_workers_finished_naturally=True))
