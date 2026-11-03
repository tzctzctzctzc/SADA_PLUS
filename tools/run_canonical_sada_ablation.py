"""原训练器的SADA对抗开关对照：原初始化、原Prototype、原数据流及选模。"""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination

ORIGIN=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
DATA=Path('/vepfs/vlms/SADA_PLUS')
TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in ('on','off'))


def paired_config(original,mode,identity,tmpdir):
    assert mode in ('on','off') and original['baseline']['regime']=='uda'
    cfg=deepcopy(original)
    cfg['baseline']['code']=identity;cfg['baseline']['tmpdir']=str(tmpdir)
    if mode=='off':cfg['model']['loss_weights']['weight_adv']=0.
    return cfg


def assert_pair(on,off):
    expected=deepcopy(on);expected['model']['loss_weights']['weight_adv']=0.
    assert expected==off,'除SADA对抗权重之外改变了配置'


def worker(entry,stage):
    from tools import baseline_engine as base
    from tools.baseline_gpu import first_cuda_guard
    cfg=read(entry/'config.json');binding=read(entry/'origin_binding.json');shift,mode=entry.name.split('_')
    assert cfg['baseline']['code']==code_identity()
    for path,digest in binding['inputs'].items():assert sha(path)==digest
    os.chdir(DATA);guard=first_cuda_guard();started=time.monotonic()
    if stage=='train':
        # probe结束后原train会重新播种/重建模型、EMA、三优化器及loader；不转移probe参数。
        base.probe(cfg,entry/'probe.json');report=read(entry/'probe.json')
        assert report['initial_state_sha']==binding['initial_state_sha']
        assert report['rng_roundtrip']['status']=='passed'
        active=report['phases']['5'];estimate=active['seconds']*binding['old_stop_epoch']+180
        save(entry/'throughput.json',dict(full_epoch_seconds=active['seconds'],historical_stop_epoch=binding['old_stop_epoch'],
            projected_seconds=estimate,max50_training_seconds=active['seconds']*50,admission_budget_seconds=1800,
            selection_rule='original source-validation patience15/max50; historical epoch only estimates runtime'))
        if estimate>1800:
            save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,estimate_seconds=estimate));return
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        base.train(cfg,entry)
        end=read(entry/'termination.json');assert end['initial_state_sha']==binding['initial_state_sha']
        termination(end['history'])
    else:
        base.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(stage+'_complete.json'),dict(passed=True,guard=str(guard),guard_sha256=sha(guard),
        config_sha256=sha(entry/'config.json'),seconds=time.monotonic()-started,stage=stage,
        baseline_engine_unchanged=True,from_original_initialization=True,target_GT_in_training=False))


def run(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    identity=code_identity();root.mkdir(parents=True,exist_ok=False);active={};failed={};paused=[];completed=[]
    # 原训练核心和Prototype逐字节不变；不采用额外微调包装器。
    paths=['tools/baseline_engine.py','libs/trainer/da_trainer.py','libs/models/SADA/meta_archs.py','libs/models/SADA/prototype_capacity.py']
    core={}
    import hashlib
    for path in paths:
        old=subprocess.check_output(['git','show','5f9a157e17e4a605ff4a6ed8dac4bd725b44d7b4:'+path],cwd=CODE)
        assert hashlib.sha256(old).hexdigest()==sha(CODE/path),path
        core[path]=sha(CODE/path)
    save(root/'plan.json',dict(code=identity,core_unchanged_from_original=core,tasks=TASKS,
        intervention='only original SADA adversarial weight; Prototype source and target objectives retained in both',
        initialization='original seed123 random initialization; never load trained best.pt',
        policy='original source validation every5, patience15, max50, original EMA best',
        target_data='same unlabeled targetTRAIN loader and budget in both arms',
        caveat='S1/S2 historically exposed development; on-arm reproducibility must be checked before causal attribution'))
    for shift in ('S1','S2'):
        original=read(ORIGIN/shift/'config.json');term=read(ORIGIN/shift/'termination.json')
        assert original['baseline']['code']['core_files']==identity['core_files']
        inputs={p:sha(p) for p in (original['dataset']['json_file'],original['baseline']['target_manifest'],original['prototype_capacity']['prepared_cache'])}
        assert inputs[original['dataset']['json_file']]==original['baseline']['source_manifest_sha256']
        assert inputs[original['baseline']['target_manifest']]==original['baseline']['target_manifest_sha256']
        assert inputs[original['prototype_capacity']['prepared_cache']]==original['prototype_capacity']['prepared_cache_sha256']
        configs={mode:paired_config(original,mode,identity,root/'tmp') for mode in ('on','off')};assert_pair(configs['on'],configs['off'])
        for mode,cfg in configs.items():
            entry=root/(shift+'_'+mode);entry.mkdir();save(entry/'config.json',cfg)
            save(entry/'origin_binding.json',dict(origin=str(ORIGIN/shift),config_sha256=sha(ORIGIN/shift/'config.json'),
                initial_state_sha=term['initial_state_sha'],old_stop_epoch=term['decision']['epoch'],inputs=inputs,
                old_target_map=read(ORIGIN/shift/'evaluation/target_val.json')['map_0.1_to_0.5']))
    (root/'tmp').mkdir()
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for a in active.values():
            if a['p'].poll() is None:os.killpg(a['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    pending=list(TASKS);waiting=[]
    def start(task,stage,g,lease,lock):
        e=root/task;cfg=read(e/'config.json')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['observed']['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
            PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            SADA_BASELINE_LEASE=lease['path'],SADA_BASELINE_LEASE_SHA256=sha(lease['path']),TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'])
        with (e/(stage+'.log')).open('x') as out:p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e),'--stage',stage],cwd=CODE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        active[task]=dict(p=p,gpu=g,stage=stage,lease=lease,lock=lock)
    try:
        while pending or waiting or active:
            for task,a in list(active.items()):
                if a['p'].poll() is None:continue
                del active[task]
                if a['p'].returncode:
                    failed[task]=dict(stage=a['stage'],exit_code=a['p'].returncode);a['lock'].close();continue
                if (root/task/'budget_pause.json').exists():paused.append(task);a['lock'].close();continue
                if a['stage']=='train':waiting.append((task,a));continue
                files=['config.json','origin_binding.json','probe.json','throughput.json','train_complete.json','evaluate_complete.json','runtime_config.json','seed_proof.json','metrics.jsonl','termination.json','best.pt','latest.pt','evaluation/source_val.json','evaluation/target_val.json']
                save(root/task/'complete.json',dict(passed=True,artifacts={f:sha(root/task/f) for f in files}));completed.append(task);a['lock'].close()
            while waiting and sum(a['stage']=='evaluate' for a in active.values())<2:
                task,a=waiting.pop(0);start(task,'evaluate',a['gpu'],a['lease'],a['lock'])
            occupied=[a['gpu'] for a in active.values()]+[a['gpu'] for _,a in waiting]
            for g in range(4):
                if not pending or g in occupied:continue
                view=snapshot(g)
                if view['free_mib']<16384:continue
                # 自有私有lease锁，绝不获取共享公共GPU锁，也不阻塞外部任务。
                lockpath=root/('private_gpu_'+str(g)+'.lock');lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                path=root/'leases'/(str(g)+'_'+str(time.time_ns())+'.json')
                lease=dict(owner_pid=os.getpid(),lock=str(lockpath),observed=view,minimum_free_mib=16384,commit=identity['commit']);save(path,lease);lease['path']=str(path)
                start(pending.pop(0),'train',g,lease,lock)
            save(root/'state.json',dict(owner_pid=os.getpid(),pending=pending,waiting_evaluate=[t for t,_ in waiting],
                active={t:dict(pid=a['p'].pid,gpu=a['gpu'],stage=a['stage']) for t,a in active.items()},completed=completed,failed=failed,paused=paused))
            time.sleep(5)
        results={}
        for task in completed:
            results[task]=dict(target=read(root/task/'evaluation/target_val.json'),source=read(root/task/'evaluation/source_val.json'),termination=read(root/task/'termination.json'),complete_sha256=sha(root/task/'complete.json'))
        save(root/'final.json',dict(passed=len(completed)==4,completed=completed,failed=failed,paused=paused,results=results))
    finally:
        for a in active.values():
            if a['p'].poll() is None:os.killpg(a['p'].pid,signal.SIGTERM)
        for a in list(active.values())+[a for _,a in waiting]:a['lock'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--stage',choices=['train','evaluate']);a=p.parse_args()
    worker(a.entry,a.stage) if a.entry else run(a.root)
