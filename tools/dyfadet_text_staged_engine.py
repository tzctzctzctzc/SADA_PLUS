"""分阶段训练完整文本检测器；源选模走实际推理、前景融合和NMS。"""
from copy import deepcopy
from pathlib import Path
import importlib.util, itertools, json, math, os, subprocess, sys, time
import numpy as np
import torch
from tools import dyfadet_text_engine as eng
from tools import dyfadet_engine as base
from tools.dyfadet_text_staged_protocol import SPEC,SCHEMA,CODE,read,save,sha,decision,stage_config

def bootstrap(ctx, state):
    """完整源学生复制给新学生和EMA；不继承随机分支EMA或源优化器动量。"""
    assert state['mode']=='source_pretrain' and state['epoch']>0
    assert not ctx['opt'].state and ctx['step']==0
    ctx['model'].load_state_dict(state['model'],strict=True)
    ctx['ema'].module.load_state_dict(state['model'],strict=True)
    ctx['model'].loss_normalizer=ctx['ema'].module.loss_normalizer=state['normalizers'][0]
    a=base.digest(ctx['model'].state_dict());b=base.digest(ctx['ema'].module.state_dict())
    assert a==b==base.digest(state['model'])
    return dict(student_sha256=a,teacher_sha256=b,source_epoch=state['epoch'],
        teacher_equals_trained_student=True,target_updates_before_handoff=0,optimizer_reset=True)

def build(cfg, mode, entry=None, device='cuda:0', workers=None):
    c=stage_config(cfg,mode);ctx=eng.build(c,mode,device,workers)
    if mode!='source_pretrain':
        p=Path(entry)/'source_pretrain/best.pt';s=torch.load(p,map_location='cpu')
        assert s['schema']==SCHEMA and s['mode']=='source_pretrain'
        ctx['handoff']=dict(bootstrap(ctx,s),source_checkpoint=str(p),source_checkpoint_sha256=sha(p));del s
    return ctx

def capture(ctx,epoch,history,rows):
    s=eng.capture(ctx,epoch,history,rows);s['schema']=SCHEMA
    if 'handoff' in ctx:s['handoff']=ctx['handoff']
    return s

def restore(ctx,s):
    assert s['schema']==SCHEMA
    if 'handoff' in ctx:assert s['handoff']==ctx['handoff']
    eng.restore(ctx,s)

@torch.no_grad()
def predict(model,cfg,manifest,path):
    model.eval();ds=base.BenchmarkDataset(cfg,manifest,unlabeled=True)
    cols={k:[] for k in ['video-id','t-start','t-end','label','score']}
    for v in ds:
        p=eng.predict_unlabeled(model,[v])[0];cols['video-id'].extend([v['video_id']]*len(p['scores']))
        for k,x in [('t-start',p['segments'][:,0]),('t-end',p['segments'][:,1]),('label',p['labels']),('score',p['scores'])]:cols[k].extend(x.cpu().tolist())
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,**{k:np.asarray(v,dtype=str if k=='video-id' else np.int64 if k=='label' else np.float32) for k,v in cols.items()})
    model.last_semantic=None
    return ds.ids

def source_ap(model,cfg,d,epoch):
    p=d/'selection'/('epoch_%03d.npz'%epoch);report=p.with_suffix('.json')
    manifest=cfg['benchmark']['inference']['source_val']
    predict(model,cfg,manifest,p)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(CODE))
    subprocess.run([sys.executable,'-u','-B',str(CODE/'tools/evaluate_text_source_selection.py'),
        '--annotation',cfg['benchmark']['source_val'],'--prediction',str(p),'--manifest',manifest,'--output',str(report)],
        env=env,cwd=CODE,check=True)
    result=read(report);assert not result['target_GT_used']
    return result['map_0.1_to_0.5']

def train_mode(cfg,entry,mode,guard,matched_stop=None):
    d=entry/mode;d.mkdir(exist_ok=True)
    if (d/'termination.json').exists():return read(d/'termination.json')['decision']['epoch']
    ctx=build(cfg,mode,entry);rows=[];history=[];start=1
    init=dict(student_sha256=base.digest(ctx['model'].state_dict()),teacher_sha256=base.digest(ctx['ema'].module.state_dict()),
        reference_best_sha256=cfg['benchmark']['reference_best_sha256'],semantic_parameter_count=sum(p.numel() for p in ctx['model'].semantic.parameters()))
    init.update(ctx.get('handoff',{}));save(d/'initialization.json',init)
    if (d/'latest.pt').exists():
        s=torch.load(d/'latest.pt',map_location='cpu');restore(ctx,s)
        history=s['history'];rows=s['metrics'];start=s['epoch']+1
        assert base.digest(capture(ctx,s['epoch'],history,rows))==base.digest(s)
        lines=(d/'events.jsonl').read_text().splitlines();prefix=lines[:ctx['step']]
        assert len(prefix)==ctx['step'] and (json.loads(prefix[-1])['sha256'] if prefix else '')==ctx['ledger_digest']
        if len(lines)>len(prefix):
            save(d/('uncommitted_suffix_%d.json'%time.time_ns()),dict(lines=lines[len(prefix):]))
            (d/'events.jsonl').write_text(''.join(x+'\n' for x in prefix))
        save(d/('resume_%d.json'%time.time_ns()),dict(start_epoch=start,checkpoint_sha256=sha(d/'latest.pt'),guard=guard,guard_sha256=sha(guard)));del s
    else:
        value=source_ap(ctx['model'] if mode=='source_pretrain' else ctx['ema'].module,ctx['cfg'],d,0);history=[[0,value]]
        if mode!='source_pretrain':
            source_t=read(entry/'source_pretrain/termination.json')
            assert abs(value-source_t['decision']['best_map'])<1e-7,'handoff changed source detection AP'
        s=capture(ctx,0,history,rows);base.checkpoint(d/'latest.pt',s)
        if mode!='source_pretrain':base.checkpoint(d/'best.pt',s)
        del s;(d/'events.jsonl').touch();save(d/'metrics.json',rows)
    maximum=50 if mode=='source_pretrain' else 30
    for epoch in range(start,maximum+1):
        if mode!='source_control' and decision(history,mode)['reason']!='running':break
        if matched_stop is not None and epoch>matched_stop:break
        started=time.monotonic();totals={};classes=[0]*10;norms={};zero=0
        sources=itertools.cycle(ctx['train']);targets=itertools.cycle(ctx['target']) if mode=='uda' else None
        for n in range(cfg['benchmark']['steps_per_epoch']):
            src=next(sources);tgt=next(targets) if targets is not None else None
            losses,g=eng.update(ctx,src,tgt,measure=not norms)
            if g:norms=g
            for k,v in losses.items():
                if k!='target_classes':totals[k]=totals.get(k,0.)+v
            classes=[a+b for a,b in zip(classes,losses['target_classes'])]
            zero+=int(tgt is not None and losses['target_positive_points']==0)
            eng.record(ctx,d/'events.jsonl',src,tgt,losses)
        del sources,targets
        row={k:v/(n+1) for k,v in totals.items()};row.update(epoch=epoch,steps=n+1,target_classes=classes,
            target_gradient_norms=norms,target_zero_positive_batches=zero,target_training_batches=n+1 if mode=='uda' else 0,
            lr=ctx['opt'].param_groups[0]['lr'],semantic_lr=ctx['opt'].param_groups[2]['lr'])
        improved=False
        if epoch%5==0:
            previous=decision(history,mode)['best_map']
            value=source_ap(ctx['model'] if mode=='source_pretrain' else ctx['ema'].module,ctx['cfg'],d,epoch)
            history.append([epoch,value]);row['source_validation_map']=value
            improved=previous is None or value>previous
        torch.cuda.synchronize();row['seconds']=time.monotonic()-started;rows.append(row)
        s=capture(ctx,epoch,history,rows);base.checkpoint(d/'latest.pt',s)
        if improved:base.checkpoint(d/'best.pt',s)
        del s;save(d/'metrics.json',rows);print('EPOCH',mode,json.dumps(row),flush=True)
    selected=decision(history,mode)
    if mode=='source_control':selected=dict(selected,reason='matched_UDA_source_update_budget');assert selected['epoch']==matched_stop
    else:assert selected['reason']!='running'
    save(d/'termination.json',dict(decision=selected,history=history,ledger_digest=ctx['ledger_digest'],guard=guard,guard_sha256=sha(guard),
        source_AP_below_original_SO=selected['best_map']<read(Path(cfg['benchmark']['reference'])/'evaluation/source_val.json')['map_0.1_to_0.5']))
    del ctx;torch.cuda.empty_cache();return selected['epoch']

def infer(cfg,entry,mode):
    d=entry/mode;s=torch.load(d/'best.pt',map_location='cpu')
    model=eng.wrap(base.make_meta_arch(cfg['model_name'],**deepcopy(cfg['model'])).to('cuda:0'),cfg,'cuda:0')
    model.load_state_dict(s['model' if mode=='source_pretrain' else 'ema'],strict=True);epoch=s['epoch'];del s
    for role,path in cfg['benchmark']['inference'].items():
        proof=d/(role+'_inference.json');p=d/(role+'.npz')
        if proof.exists() and sha(p)==read(proof)['prediction_sha256']:continue
        videos=predict(model,cfg,path,p)
        save(proof,dict(videos=videos,count=len(videos),selected_epoch=epoch,manifest_sha256=sha(path),prediction_sha256=sha(p),target_GT_used=False))
    del model;torch.cuda.empty_cache()

def train_and_infer(cfg,entry,guard):
    train_mode(cfg,entry,'source_pretrain',guard)
    stop=train_mode(cfg,entry,'uda',guard)
    train_mode(cfg,entry,'source_control',guard,stop)
    for mode in SPEC['stages']:infer(cfg,entry,mode)
    save(entry/'gpu_complete.json',dict(status='passed',guard=guard,guard_sha256=sha(guard),target_GT_used=False))

def evaluate(entry):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=read(entry/'config.json');offline=read(entry/'offline_evaluation.json');assert sha(offline['annotation'])==offline['annotation_sha256']
    path=CODE/'libs/evaluation/evaluator_anet.py';s=importlib.util.spec_from_file_location('staged_final_evaluator',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    for mode in SPEC['stages']:
        d=entry/mode
        for role,splits in offline['splits'].items():
            with np.load(d/(role+'.npz')) as f:columns={k:f[k] for k in f.files}
            result=m.ANETdetection(offline['annotation'],splits,num_workers=2,action_subsampling='10_majority_classes').evaluate(columns,verbose=True)
            proof=read(d/(role+'_inference.json'))
            save(d/'evaluation'/(role+'.json'),dict(map_0.1_to_0.5=float(result[1]),map_by_tiou=[float(x) for x in result[0]],
                selected_epoch=proof['selected_epoch'],videos=proof['count'],prediction_sha256=sha(d/(role+'.npz')),evaluator_sha256=sha(path),annotation_sha256=offline['annotation_sha256']))
    artifacts={str(p):sha(p) for p in entry.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.pt','.npz'] and p.name!='complete.json'}
    save(entry/'complete.json',dict(schema=SCHEMA,shift=cfg['benchmark']['shift'],code=cfg['benchmark']['code'],artifacts=artifacts))

def probe(cfg,entry,guard):
    eng.probe(cfg,entry,guard)
    ctx=build(cfg,'source_pretrain');src=next(iter(ctx['train']));before=base.digest(ctx['model'].state_dict())
    for _ in range(2):eng.update(ctx,src)
    trained=base.digest(ctx['model'].state_dict());assert trained!=before
    s=deepcopy(capture(ctx,1,[],[]));s['model']={k:v.cpu() for k,v in s['model'].items()}
    del ctx;torch.cuda.empty_cache()
    ctx=eng.build(stage_config(cfg,'uda'),'uda');handoff=bootstrap(ctx,s);del s
    v=next(iter(ctx['val']))[0];un={k:x for k,x in v.items() if k not in ['segments','labels']};model=ctx['ema'].module;model.eval()
    with torch.no_grad():
        out=base.dense(model,[un]);hard,soft,_=model.last_semantic
        assert any(float(h.abs().sum())>0 for h in hard)
        # 前景概率必须进入最终分数，而不只训练辅助loss。
        raw=torch.cat(out[2],1).sigmoid();fg=torch.cat(hard,1).sigmoid()
        assert not torch.equal(raw,raw*fg)
    probe_cfg=deepcopy(cfg);db=read(cfg['benchmark']['source_val']);db['database']={v['video_id']:db['database'][v['video_id']]}
    gt=entry/'probe_source_gt.json';manifest=entry/'probe_source_unlabeled.json';save(gt,db)
    save(manifest,dict(database={k:{a:b for a,b in x.items() if a!='annotations'} for k,x in db['database'].items()}))
    probe_cfg['benchmark']['source_val']=str(gt);probe_cfg['benchmark']['inference']['source_val']=str(manifest)
    value=source_ap(model,probe_cfg,entry/'staged_probe',0);assert math.isfinite(value)
    handoff.update(source_parameters_updated=True,foreground_affects_final_scores=True,source_AP_pipeline_passed=True,
        source_AP=value,scope='two_source_steps_wiring_probe_not_mature_model_or_method_result')
    report=read(entry/'probe.json');report['staged_handoff']=handoff;report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
    save(entry/'probe.json',report)
