"""六向一次完整融合结构训练；最强版冻结推理设置作为固定对照。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination
from tools.unified_prototype_classifier import SPEC
REFERENCE=Path('/vepfs/vlms/trident_repaired6_820b9e3_1104')
DATA=Path('/vepfs/vlms/SADA_PLUS');SHIFTS=tuple('S'+str(i) for i in range(1,7))


def configuration(root,shift,identity):
    cfg=read(REFERENCE/(shift+'_trident')/'config.json');cfg['unified_prototype']=SPEC
    cfg['baseline'].update(code=identity,tmpdir=str(root/'tmp'));return cfg


def prepare(root,auth):
    assert auth['full_training_runs']==6 and auth['seed']==123 and auth['shifts']==list(SHIFTS)
    assert read(REFERENCE/'independent_final_audit.json')['passed']
    identity=code_identity();root.mkdir(exist_ok=False);(root/'tmp').mkdir()
    save(root/'authorization.json',auth);save(root/'plan.json',dict(code=identity,spec=SPEC,full_training_runs=6,seed=123,
        reference=str(REFERENCE),strong_reference_mean_percent=56.22733524999254,native_reference_mean_percent=55.73843961801179,
        checkpoint_selection='source_validation_cls_plus_reg_every5_patience15_max50_bestEMA',new_target_GT=False,new_VLM_calls=0,
        efficiency='single shared dense embedding; no extra proposal Transformer; cached masked features reused by readout',
        known_original_cold_Adam_limitation='historical failed record preserved; not repeated as a new check'))
    for shift in SHIFTS:
        e=root/shift;e.mkdir();cfg=configuration(root,shift,identity);old=REFERENCE/(shift+'_trident')
        assert read(old/'complete.json')['passed'];compare=deepcopy(cfg);compare.pop('unified_prototype')
        original=read(old/'config.json');compare['baseline'].update(code=original['baseline']['code'],tmpdir=original['baseline']['tmpdir']);assert compare==original
        paths=[Path(cfg['dataset']['json_file']),Path(cfg['baseline']['target_manifest']),Path(cfg['prototype_capacity']['prepared_cache'])]
        inputs={str(p):sha(p) for p in paths}
        assert inputs[cfg['dataset']['json_file']]==cfg['baseline']['source_manifest_sha256']
        assert inputs[cfg['baseline']['target_manifest']]==cfg['baseline']['target_manifest_sha256']
        assert inputs[cfg['prototype_capacity']['prepared_cache']]==cfg['prototype_capacity']['prepared_cache_sha256']
        source=read(paths[0])['database'];target=read(paths[1])['database'];assert not set(source)&set(target) and all('annotations' not in v for v in target.values())
        save(e/'config.json',cfg);save(e/'input_binding.json',dict(inputs=inputs,reference=str(old),reference_complete_sha256=sha(old/'complete.json'),historical_initial_sha=read(old/'termination.json')['initial_state_sha']))
    save(root/'state.json',dict(stage='prepared',active={},pending=list(SHIFTS),completed=[],failed={}))


class FixedBatch:
    def __init__(self,batch):self.batch=batch
    def __len__(self):return 1
    def __iter__(self):yield self.batch


def probe(root,shift):
    import torch
    from tools import baseline_engine as base
    from libs.models.SADA.prototype_capacity import fully_visible_cache
    e=root/'probes'/shift;cfg=read(root/shift/'config.json');ctx=base.build(cfg)
    binding=read(root/shift/'input_binding.json');assert ctx.model.unified_base_initial_sha==binding['historical_initial_sha']
    assert ctx.model.cls_head is ctx.model.cls_head_target is ctx.model.cls_head_pl
    assert not any('rq3_capacity.branch' in n for n,_ in ctx.model.named_parameters())
    target=None;desired=set()
    for batch in ctx.train_loader_tgt:
        kept,_,_=fully_visible_cache(ctx.model.rq3_capacity.cache,batch,ctx.model.rq3_capacity.confidence_threshold)
        if any(kept):
            target=batch;desired={max(range(len(p['class_probabilities'])),key=lambda k:p['class_probabilities'][k]) for rows in kept for p in rows};break
    assert target is not None
    source=None
    for batch in ctx.train_loader_src:
        labels={int(x) for v in batch for x in v['labels']}
        if labels&desired:source=batch;break
    assert source is not None,'真实源batch未覆盖该目标候选类别'
    ctx.train_loader_src=FixedBatch(source);ctx.train_loader_tgt=FixedBatch(target)
    summaries=[]
    def step(epoch):
        base.epoch_setup(ctx,epoch);torch.cuda.synchronize();start=time.monotonic();ctx._train_epoch(epoch);torch.cuda.synchronize()
        metrics=ctx.train_metrics.result();grads={n:p.grad for n,p in ctx.model.named_parameters() if p.grad is not None}
        assert float(metrics['t_rq3_source_loss'])==0 and float(metrics['t_rq3_target_gradient_norm'])>0
        assert ctx.model.cls_head.projection.weight.grad.abs().sum()>0 and ctx.model.cls_head.class_bias.grad.abs().sum()>0
        return dict(full_state_sha=base.tensor_digest(base.capture(ctx,cfg,epoch,[])),gradient_sha=base.tensor_digest(grads),
            loss=float(metrics['t_final_loss']),seconds=time.monotonic()-start,auxiliary={k[2:]:float(v) for k,v in metrics.items() if k.startswith('t_rq3_')})
    summaries.append(step(1));summaries.append(step(5))
    snapshot=e/'full_state.pt';base.checkpoint(snapshot,base.capture(ctx,cfg,5,[]))
    expected=step(5);base.restore(ctx,torch.load(snapshot,map_location='cpu'));actual=step(5)
    assert all(expected[k]==actual[k] for k in ['full_state_sha','gradient_sha','loss']),'融合版完整磁盘恢复下一更新未逐位一致'
    save(e/'result.json',dict(passed=True,shift=shift,base_initial_sha=ctx.model.unified_base_initial_sha,initial_state_sha=ctx.initial_state_sha,
        full_disk_next_update_exact=True,source_background_and_projection_gradients=True,source_auxiliary_CE=False,
        parameter_count=sum(p.numel() for p in ctx.model.parameters()),target_GT=False,steps=summaries,next_update=actual,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,probe_weights_not_transferred_to_training=True))


def worker(root,shift,stage):
    from tools.baseline_gpu import first_cuda_guard
    guard=first_cuda_guard();os.chdir(DATA)
    e=root/'probes'/shift if stage=='probe' else root/shift;cfg=read(root/shift/'config.json');assert cfg==configuration(root,shift,code_identity())
    for path,digest in read(root/shift/'input_binding.json')['inputs'].items():assert sha(path)==digest
    from tools import baseline_engine as base
    if stage=='probe':probe(root,shift)
    elif stage=='train':
        build=base.build
        def initialized(config):
            ctx=build(config);proof=read(root/'probes'/shift/'result.json')
            assert ctx.initial_state_sha==proof['initial_state_sha'] and ctx.model.unified_base_initial_sha==proof['base_initial_sha']
            save(e/'formal_initialization.json',dict(passed=True,initial_state_sha=ctx.initial_state_sha,base_initial_sha=ctx.model.unified_base_initial_sha,
                loaded_trained_checkpoint=False,source_auxiliary_CE=False,shared_classifier=True))
            return ctx
        base.build=initialized
        try:base.train(cfg,e)
        finally:base.build=build
    elif stage=='evaluate':
        from tools.native_ap_fast import install
        benchmark=read(Path('/vepfs/vlms/native_efficiency_caaa3e0_1104/real_prediction_benchmark.json'))
        assert benchmark['passed'] and benchmark['fast_code_sha256']==sha(CODE/'tools/native_ap_fast.py')
        install();base.evaluate(cfg,e,cfg['baseline']['annotation_for_offline_evaluation'])
    else:
        from tools.unified_readout_eval import evaluate_shift
        args=SimpleNamespace(reference_root=root,output_root=root/'fixed_readout',cached_root=None,device='cuda:0',shared=False,limit_videos=None)
        evaluate_shift(args,shift)
    save(e/(stage+'_gpu.json'),dict(guard=guard,guard_sha256=sha(guard),config_sha256=sha(root/shift/'config.json')))


def audit(root,shift):
    import numpy as np
    import torch
    import pandas as pd
    from tools.audit_head_replacements import metrics
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    e=root/shift;cfg=read(e/'config.json');assert cfg==configuration(root,shift,cfg['baseline']['code'])
    binding=read(e/'input_binding.json')
    for path,digest in binding['inputs'].items():assert sha(path)==digest
    assert sha(Path(binding['reference'])/'complete.json')==binding['reference_complete_sha256']
    term=read(e/'termination.json');decision=termination(term['history']);rows=[json.loads(line) for line in (e/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,decision['epoch']+1))
    assert [[r['epoch'],r['source_validation_loss']] for r in rows if 'source_validation_loss' in r]==term['history']
    assert all(r['target_gt_cls_loss']==r['target_gt_reg_loss']==r['target_gt_downstream_loss']==0 for r in rows)
    assert all(r['auxiliary']['rq3_source_loss']==0 for r in rows)
    assert any(r['auxiliary']['rq3_target_gradient_norm']>0 for r in rows)
    initial=read(e/'formal_initialization.json');assert initial['passed'] and not initial['loaded_trained_checkpoint']
    assert read(e/'runtime_config.json')==cfg and read(e/'seed_proof.json')['actual_cuda_seed']==123
    for file,epoch in [('best.pt',decision['best_epoch']),('latest.pt',decision['epoch'])]:
        saved=torch.load(e/file,map_location='cpu');assert saved['config']==cfg and saved['epoch']==epoch
        assert len(saved['optimizers'])==len(saved['schedulers'])==3
        assert set(saved['rng'])=={'python','numpy','cpu','cuda','loaders'} and len(saved['rng']['cuda'])==1
        assert 'cls_head.source_prototypes' in saved['state_dict'] and 'cls_head.source_prototypes' in saved['ema_state_dict']
        assert not any('rq3_capacity.branch' in k or 'cls_head.cls_head.conv' in k for k in saved['state_dict'])
        del saved
    results={}
    for domain,splits in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
        report=read(e/'evaluation'/(domain+'_val.json'));assert report['checkpoint_sha256']==sha(e/'best.pt') and report['selected_epoch']==decision['best_epoch']
        gt=load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],split=splits,action_subsampling='10_majority_classes');gt=gt[gt['t-start']!=gt['t-end']].reset_index(drop=True)
        pred=pd.DataFrame(torch.load(e/'evaluation'/(domain+'_predictions.pt'),map_location='cpu'));independent=metrics(gt,pred)
        assert np.max(np.abs(np.asarray(independent['map_by_tiou'])-np.asarray(report['map_by_tiou'])))<1e-7
        results[domain]=report
    for stage in ['train','evaluate','readout']:
        proof=read(e/(stage+'_gpu.json'));assert sha(proof['guard'])==proof['guard_sha256'];g=read(proof['guard']);assert sha(g['lease'])==g['lease_sha256']
        assert read(g['lease'])['commit']==cfg['baseline']['code']['commit']
    final=read(root/'fixed_readout'/shift/'result.json');assert final['full_dataset'] and final['checkpoint_sha256']==sha(e/'best.pt')
    files=['config.json','input_binding.json','formal_initialization.json','metrics.jsonl','termination.json','best.pt','latest.pt','seed_proof.json',
        'runtime_config.json','train_gpu.json','evaluate_gpu.json','readout_gpu.json','evaluation/manifest.json','evaluation/source_val.json',
        'evaluation/target_val.json','evaluation/source_predictions.pt','evaluation/target_predictions.pt']
    save(e/'complete.json',dict(passed=True,artifacts={f:sha(e/f) for f in files}))
    save(root/'job_audits'/(shift+'.json'),dict(passed=True,shift=shift,results=results,fixed_readout=final,decision=decision,complete_sha256=sha(e/'complete.json')))


def summarize(root):
    reports={s:read(root/'job_audits'/(s+'.json')) for s in SHIFTS};assert all(r['passed'] for r in reports.values())
    native=sum(r['results']['target']['map_0.1_to_0.5'] for r in reports.values())/6*100
    final=sum(r['fixed_readout']['map_percent'] for r in reports.values())/6
    save(root/'summary.json',dict(passed=True,all_six_complete=True,seed=123,native_mean_percent=native,fixed_readout_mean_percent=final,
        native_delta_vs_trident_pp=native-55.73843961801179,final_delta_vs_strongest_pp=final-56.22733524999254,
        directions={s:dict(native=r['results']['target']['map_0.1_to_0.5']*100,final=r['fixed_readout']['map_percent']) for s,r in reports.items()},
        single_seed=True,target_validation_development=True,extra_training_budget=0,
        readout_interpretation='Region classifier logits come from the SAME unified prototype classifier, not an independent CE head.'))


def run(root):
    from tools.baseline_gpu import slot,snapshot
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'state.json')['stage']=='prepared'
    identity=code_identity();mutex=threading.Lock();stop=threading.Event();evaluation_slots=threading.Semaphore(2);cpu_slots=threading.Semaphore(2)
    shared=dict(stage='gpu_acceptance',active={},pending=list(SHIFTS),completed=[],failed={},probes_complete=[])
    def update():save(root/'state.json',dict(shared,checked_at=time.time(),owner_pid=os.getpid()))
    def task(shift,probing=False):
        context=None;e=root/'probes'/shift if probing else root/shift
        try:
            while context is None:
                if stop.is_set():return
                for gpu in sorted([snapshot(i) for i in range(4)],key=lambda x:x['utilization']):
                    candidate=slot(gpu['index'],DATA,root,32768,identity['commit'])
                    try:lease=candidate.__enter__();context=candidate;break
                    except BlockingIOError:continue
                if context is None:time.sleep(15)
            e.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
                PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='2',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(root/'tmp'),TMP=str(root/'tmp'),TEMP=str(root/'tmp'))
            if not probing:
                with (e/'train_claim.json').open('x') as claim:json.dump(dict(owner=os.getpid(),claimed=time.time()),claim)
                with mutex:shared['pending'].remove(shift);update()
            for stage in (['probe'] if probing else ['train','evaluate','readout']):
                if stage=='evaluate':evaluation_slots.acquire()
                try:
                    with (e/(stage+'.log')).open('x') as log:
                        child=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--root',str(root),'--shift',shift,'--stage',stage],cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        receipt=dict(pid=child.pid,start_ticks=(Path('/proc')/str(child.pid)/'stat').read_text().split(') ',1)[1].split()[19],stage=stage,gpu=lease['index'],started=time.time())
                        save(e/(stage+'_launch.json'),receipt)
                        with mutex:shared['active'][shift]=receipt;update()
                        assert child.wait()==0,shift+'/'+stage+' failed'
                finally:
                    if stage=='evaluate':evaluation_slots.release()
            context.__exit__(None,None,None);context=None
            if probing:
                assert read(e/'result.json')['passed']
                with mutex:shared['probes_complete'].append(shift);shared['active'].pop(shift);update()
            else:
                with mutex:shared['active'][shift]['stage']='cpu_audit';update()
                with cpu_slots:audit(root,shift)
                with mutex:shared['completed'].append(shift);shared['active'].pop(shift);update()
        except Exception as error:
            stop.set();save(e/'error.json',dict(error=str(error),type=type(error).__name__))
            with mutex:shared['failed'][shift]=str(error);shared['active'].pop(shift,None);update()
        finally:
            if context:context.__exit__(None,None,None)
    update()
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda s:task(s,True),SHIFTS))
    if not stop.is_set():
        assert set(shared['probes_complete'])==set(SHIFTS)
        save(root/'acceptance.json',dict(passed=True,reports={s:sha(root/'probes'/s/'result.json') for s in SHIFTS}))
        shared['stage']='training';update()
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(task,SHIFTS))
    if not shared['failed'] and len(shared['completed'])==6:summarize(root);shared['stage']='complete'
    else:shared['stage']='incomplete_requires_review'
    update();lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',type=Path);p.add_argument('--shift',choices=SHIFTS);p.add_argument('--stage',choices=['probe','train','evaluate','readout']);a=p.parse_args()
    try:
        if a.prepare:prepare(a.root,read(a.prepare))
        elif a.shift:worker(a.root,a.shift,a.stage)
        else:run(a.root)
    except Exception as error:
        e=a.root/'probes'/a.shift if a.stage=='probe' else a.root/(a.shift or '')
        save(e/'controller_error.json',dict(error=str(error),type=type(error).__name__));raise
