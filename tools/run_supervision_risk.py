"""小型监督风险分支：六方向源域可学性和GPU接线通过后四卡冷启动。"""
import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,decision,termination,validate_cuda_rng
from tools.supervision_risk import MODES,SPEC,SCHEMA,install,enable_controls,attach
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
SHIFTS=tuple('S%d'%i for i in range(1,7))


def build(cfg):
    from tools import baseline_engine as base
    enable_controls();ctx=base.build(cfg)
    ref=Path(cfg['supervision_risk']['reference'])
    assert ctx.initial_state_sha==read(ref/'termination.json')['initial_state_sha']
    assert ctx.initial_detector_sha==read(ref/'detector_initialization.json')['sha256']
    install(ctx,cfg)
    from tools.cold_metric_readback import configure
    configure(ctx)
    return ctx


def epoch_metrics(ctx,epoch,elapsed):
    m=ctx.train_metrics.result()
    return dict(epoch=epoch,seconds=elapsed,train_loss=float(m['t_final_loss']),
        source_cls_loss=float(m['t_cls_loss']),source_reg_loss=float(m['t_reg_loss']),
        risk={k[7:]:float(v) for k,v in m.items() if k.startswith('t_risk_')},
        Prototype={k[6:]:float(v) for k,v in m.items() if k.startswith('t_rq3_')})


def probe(cfg,entry,guard):
    import torch
    from tools import baseline_engine as base
    ctx=build(cfg);ctx.model.train()
    src=base.move_to_device(next(iter(ctx.train_loader_src)),ctx.model.device)
    tgt=base.move_to_device(next(iter(ctx.train_loader_tgt)),ctx.model.device)
    before=deepcopy(base.capture(ctx,cfg,0,[]))
    kwargs=dict(video_list_src=src,video_list_tgt=tgt,mode='train',curr_epoch=1,
                cls_static_anchors=None,mean_class_conf=None)
    expected={k:v.detach().clone() for k,v in ctx.risk_original_forward(**kwargs)[0].items()}
    base.restore(ctx,before);ctx.risk_enabled=False
    actual=ctx.model(**kwargs)[0]
    assert all(torch.equal(v,actual[k].detach()) for k,v in expected.items())
    base.restore(ctx,before);ctx.risk_enabled=True
    clean=[dict(src[0],segments=src[0]['segments'].new_empty((0,2)),labels=src[0]['labels'].new_empty(0))]
    ctx.model.eval()
    test_kwargs=dict(video_list_src=clean,video_list_tgt=clean,mode='test',domain='source',curr_epoch=1)
    with torch.no_grad():
        expected_prediction=base.tensor_digest(ctx.risk_original_forward(**test_kwargs))
        assert base.tensor_digest(ctx.model(**test_kwargs))==expected_prediction
    del src,tgt,expected,actual,before
    phases={}
    for epoch,steps,mature in [(1,2,False),(15,6,True)]:
        if mature:
            saved=torch.load(REFERENCE/cfg['baseline']['shift']/'best.pt',map_location='cpu')
            for model in [ctx.model,ctx.model_ema.module]:
                missing=model.load_state_dict(saved['ema_state_dict'],strict=False)
                assert not missing.unexpected_keys and all(k.startswith('supervision_risk.') for k in missing.missing_keys)
                risk=torch.load(Path(cfg['supervision_risk']['feasibility_entry'])/'best_risk.pt',map_location='cpu')
                model.supervision_risk.load_state_dict(risk['state_dict'],strict=True)
            del saved,risk
        base.epoch_setup(ctx,epoch);loaders=ctx.train_loader_src,ctx.train_loader_tgt
        ctx.train_loader_src,ctx.train_loader_tgt=[base.StepLoader(x,steps) for x in loaders]
        modules=[('SGP',ctx.model.backbone_source),('cls',ctx.model.cls_head),('offset',ctx.model.reg_head.offset_head),
                 ('risk',ctx.model.supervision_risk)]
        old={name:base.tensor_digest(module.state_dict()) for name,module in modules}
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        ctx._train_epoch(epoch);torch.cuda.synchronize()
        result=epoch_metrics(ctx,epoch,time.monotonic()-started)
        result['updated']={name:old[name]!=base.tensor_digest(module.state_dict()) for name,module in modules}
        result['peak_mib']=torch.cuda.max_memory_reserved()/2**20
        assert all(result['updated'].values())
        if mature:
            assert result['risk'].get('target_reg_gradient',0)==0 and result['risk']['target_regression']==0
        phases[str(epoch)]=result;ctx.train_loader_src,ctx.train_loader_tgt=loaders
        print('RISK_GPU_PROBE',json.dumps(result),flush=True)
    for opt in [ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment]:
        ids={id(p) for group in opt.param_groups for p in group['params']}
        assert all(id(p) in ids for p in ctx.model.supervision_risk.parameters())
    saved=deepcopy(base.capture(ctx,cfg,15,[]));stream=io.BytesIO();torch.save(saved,stream);stream.seek(0)
    restored=torch.load(stream,map_location='cpu');stream.close();base.restore(ctx,restored)
    assert base.tensor_digest(base.capture(ctx,cfg,15,[]))==base.tensor_digest(saved)
    def draws():
        import random,numpy as np
        return dict(python=random.random(),numpy=np.random.rand(3),cpu=torch.rand(3),cuda=torch.rand(3,device='cuda:0'),
                    loaders={k:torch.randperm(8,generator=g) for k,g in ctx.generators.items()})
    base.restore(ctx,restored);expected=base.tensor_digest(draws())
    base.restore(ctx,restored);assert expected==base.tensor_digest(draws())
    risk=phases['15']['risk']
    ready=(risk['positive']>0 and risk['negative']>0 and risk.get('target_SGP_gradient',0)>0 and risk.get('target_cls_gradient',0)>0)
    save(entry/'result.json',dict(status='passed' if ready else 'target_mechanism_unready',schema=SCHEMA,
        shift=cfg['baseline']['shift'],code=code_identity(),config_sha256=sha(entry/'config.json'),
        cold_initial_state_sha=ctx.initial_state_sha,cold_initial_detector_sha=ctx.initial_detector_sha,
        disabled_all_losses_exact=True,disabled_complete_forward_NMS_voting_exact=True,
        phases=phases,full_model_EMA_three_optimizer_scheduler_controls_RNG_restore=True,
        mature_and_source_fit_head_only_for_probe_not_formal=True,formal_initialization='fresh_cold',
        peak_mib=max(x['peak_mib'] for x in phases.values()),guard=guard,guard_sha256=sha(guard),
        scientific_effectiveness_not_claimed=True))


def train(cfg,entry):
    import torch
    from tools import baseline_engine as base
    ctx=build(cfg);ctx.risk_ledger=str(entry/'risk_events.jsonl')
    save(entry/'runtime_config.json',ctx.cfg);save(entry/'seed_proof.json',ctx.cuda_seed_proof)
    save(entry/'cold_initialization.json',dict(initial_state_sha=ctx.initial_state_sha,
        initial_detector_sha=ctx.initial_detector_sha,mature_checkpoint_loaded=False,
        original_parameter_set_unchanged=True,new_trainable_parameters=sum(p.numel() for p in ctx.model.supervision_risk.parameters())))
    history=[];rows=[];best=float('inf')
    for epoch in range(1,51):
        base.epoch_setup(ctx,epoch);before=ctx.checked_steps;start=time.monotonic()
        ctx._train_epoch(epoch);torch.cuda.synchronize()
        assert ctx.checked_steps-before==cfg['baseline']['epoch_budget']['iterations_per_epoch']
        row=epoch_metrics(ctx,epoch,time.monotonic()-start)
        if epoch%5==0:
            ctx.valid_metrics_src.reset();value=float(ctx._valid_epoch(epoch,ctx.val_loader_src,domain='src'))
            assert math.isfinite(value);history.append([epoch,value]);row['source_validation_loss']=value
        rows.append(row);base.checkpoint(entry/'latest.pt',base.capture(ctx,cfg,epoch,history))
        if epoch%5==0:
            os.link(entry/'latest.pt',entry/('epoch%03d.pt'%epoch))
            if value<best:
                best=value;tmp=entry/'best.pt.tmp';os.link(entry/'latest.pt',tmp);tmp.replace(entry/'best.pt')
        save(entry/'metrics.json',rows);print('RISK_METRICS',json.dumps(row,allow_nan=False),flush=True)
        if epoch%5==0 and decision(history)['reason']!='running':
            save(entry/'termination.json',dict(history=history,decision=termination(history),
                initialization='original_cold',initial_state_sha=ctx.initial_state_sha));break


def inference(cfg,entry,epochs,domains,manifest_name,diagnostics=False):
    import torch,numpy as np
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    from tools.cold_oracle_fastpath import install as fast
    fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    def construct(weights):
        m=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda();fast(m)
        attach(m);state=None;m.load_state_dict(weights,strict=True);m.eval()
        return m,state
    reports=[]
    for epoch in epochs:
        path=entry/('epoch%03d.pt'%epoch);saved=torch.load(path,map_location='cpu')
        model,state=construct(saved['ema_state_dict'])
        for k,v in saved['ema_controls'].items():setattr(model,k,deepcopy(v))
        if diagnostics:
            from tools.supervision_risk_signals import diagnostic_arrays
            diag=[];d=entry/('signals/e%03d'%epoch);d.mkdir(parents=True,exist_ok=True)
        del saved
        for domain in domains:
            manifest=entry/(domain+'_val_unlabeled.json')
            assert all('annotations' not in x for x in read(manifest)['database'].values())
            ds=base.dataset(cfg,manifest,cfg['src_val_split'] if domain=='source' else cfg['tgt_val_split'],False,True)
            columns={k:[] for k in ['video-id','t-start','t-end','label','score']}
            with torch.no_grad():
                for i in range(len(ds)):
                    videos=base.move_to_device([ds[i]],'cuda:0');v=videos[0]
                    assert not v['segments'].numel() and not v['labels'].numel()
                    model.eval();pred=model(video_list_src=videos,video_list_tgt=videos,mode='test',domain=domain,curr_epoch=epoch)[0 if domain=='source' else 1][0]
                    columns['video-id'].extend([pred['video_id']]*len(pred['scores']))
                    for k,z in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:columns[k].extend(z.cpu().tolist())
                    if diagnostics:
                        arrays,row=diagnostic_arrays(model,v)
                        output=d/(v['metadata']['video_id']+'.npz');np.savez_compressed(output,**arrays)
                        row.update(arrays=str(output),arrays_sha256=sha(output));diag.append(row)
                    print('RISK_INFERENCE',epoch,domain,i+1,len(ds),flush=True)
            prediction=entry/('predictions/%s_e%03d.npz'%(domain,epoch));prediction.parent.mkdir(exist_ok=True)
            tmp=prediction.with_name(prediction.name+'.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,**{k:np.asarray(v) for k,v in columns.items()})
            tmp.replace(prediction)
            reports.append(dict(epoch=epoch,domain=domain,prediction=str(prediction),prediction_sha256=sha(prediction),checkpoint=str(path),checkpoint_sha256=sha(path)))
        if diagnostics:
            save(d/'diagnostics.json',dict(videos=diag,GT_loaded=False,scope='fixed_source_crop_only'))
        del model
    save(entry/manifest_name,dict(reports=reports,GT_loaded=False,code=code_identity()))


def evaluate_cpu(cfg,entry,manifest_name,output,diagnostics=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from tools.run_prototype_cold_oracle import evaluate_cpu as evaluate
    evaluate(cfg,entry,Path(manifest_name),Path(output))
    if diagnostics:
        from tools.supervision_risk_signals import evaluate_diagnostics
        epoch=read(entry/manifest_name)['reports'][0]['epoch'];evaluate_diagnostics(cfg,entry,epoch)


def audit(entry):
    import torch
    cfg=read(entry/'config.json');complete=read(entry/'complete.json')
    assert complete['schema']==SCHEMA and complete['code']==code_identity()
    assert cfg['supervision_risk']['spec']==SPEC and not cfg['supervision_risk']['target_GT_used']
    original=read(Path(cfg['supervision_risk']['reference'])/'config.json');paired=deepcopy(cfg);paired.pop('supervision_risk')
    for k in ['code','tmpdir','schema']:paired['baseline'][k]=original['baseline'][k]
    assert paired==original
    for p,digest in complete['artifacts'].items():assert sha(p)==digest,p
    term=read(entry/'termination.json');selected=termination(term['history']);assert selected==term['decision']
    latest=torch.load(entry/'latest.pt',map_location='cpu');best=torch.load(entry/'best.pt',map_location='cpu')
    assert latest['epoch']==selected['epoch'] and best['epoch']==selected['best_epoch']
    assert latest['history']==term['history'] and len(latest['optimizers'])==len(latest['schedulers'])==3
    validate_cuda_rng(latest['rng']['cuda'])
    initial=read(entry/'cold_initialization.json')
    assert initial['initial_state_sha']==read(Path(cfg['supervision_risk']['reference'])/'termination.json')['initial_state_sha']
    assert not initial['mature_checkpoint_loaded']
    target_db=read(cfg['baseline']['target_manifest'])['database'];src_db=read(cfg['dataset']['json_file'])['database']
    source_train={k for k,v in src_db.items() if v['subset'] in cfg['src_train_split']}
    source_val={k for k,v in src_db.items() if v['subset'] in cfg['src_val_split']}
    assert not source_train&source_val and all('annotations' not in v for v in target_db.values())
    digest='0'*64;steps=0
    for line in (entry/'risk_events.jsonl').open():
        event=json.loads(line);given=event.pop('digest');steps+=1
        assert event['step']==steps and all(v in source_train for v in event['source_videos'])
        assert all(x['video'] in target_db for x in event['observations'])
        assert event['stats']['ramp']==max(0.,min(1.,(event['epoch']-10)/5))
        if event['epoch']<=10:assert not event['observations']
        assert event['stats']['target_regression']==event['stats'].get('target_reg_gradient',0)==0
        digest=__import__('hashlib').sha256((digest+json.dumps(event,sort_keys=True)).encode()).hexdigest();assert digest==given
    control=latest['model_controls']['risk_state']
    assert steps==control['step']==selected['epoch']*cfg['baseline']['epoch_budget']['iterations_per_epoch'] and digest==control['digest']
    for domain in ['source','target']:assert all('annotations' not in v for v in read(entry/(domain+'_val_unlabeled.json'))['database'].values())
    for stage in ['train','inference']:
        receipt=read(entry/(stage+'_gpu.json'));guard=read(receipt['guard']);lease=read(guard['lease'])
        assert sha(receipt['guard'])==receipt['guard_sha256'] and sha(guard['lease'])==guard['lease_sha256']
        assert guard['owner_pid']==lease['owner_pid'] and guard['observed']['uuid']==lease['observed']['uuid']
        assert guard['observed']['free_mib']>=lease['minimum_free_mib']>=12288
    results=read(entry/'evaluation/final.json')['results']
    primary=next(x for x in results if x['epoch']==selected['best_epoch'] and x['domain']=='target')
    ap_epoch=source_ap_selection(entry,term['history'])['epoch']
    ap_result=next(x for x in results if x['epoch']==ap_epoch and x['domain']=='target')
    return dict(mode=cfg['supervision_risk']['mode'],shift=cfg['baseline']['shift'],map=primary['map'],
                best_epoch=selected['best_epoch'],source_AP_selected_map=ap_result['map'],source_AP_selected_epoch=ap_epoch,
                stop_epoch=selected['epoch'],complete_sha256=sha(entry/'complete.json'))


def source_ap_selection(entry,history):
    candidates=[]
    for epoch,_ in history:
        path=entry/('signals/e%03d/evaluation.json'%epoch)
        row=read(path)['results'][0]
        assert row['domain']=='source' and row['epoch']==epoch
        candidates.append(dict(epoch=epoch,map=row['map'],report=str(path),sha256=sha(path)))
    best=min(candidates,key=lambda x:(-x['map'],x['epoch']))
    return dict(epoch=best['epoch'],candidates=candidates,rule='max_source_AP_earliest_tie')


def worker(args):
    cfg=read(args.entry/'config.json');assert cfg['baseline']['code']==code_identity()
    for path,digest in [(cfg['dataset']['json_file'],cfg['baseline']['source_manifest_sha256']),
            (cfg['baseline']['target_manifest'],cfg['baseline']['target_manifest_sha256']),
            (cfg['prototype_capacity']['prepared_cache'],cfg['prototype_capacity']['prepared_cache_sha256'])]:
        assert sha(path)==digest
    if args.stage in ('evaluate_cpu','signal_cpu'):
        evaluate_cpu(cfg,args.entry,'inference_manifest.json' if args.stage=='evaluate_cpu' else 'signals/e%03d/manifest.json'%args.epoch,
            'evaluation/final.json' if args.stage=='evaluate_cpu' else 'signals/e%03d/evaluation.json'%args.epoch,args.stage=='signal_cpu');return
    from tools.phase_runner_gpu import first_cuda_guard
    guard=first_cuda_guard();os.chdir(args.data_root)
    if args.stage=='source_fit':
        from tools.supervision_risk_feasibility import run
        run(cfg,args.entry)
    elif args.stage=='probe':probe(cfg,args.entry,guard)
    elif args.stage=='train':train(cfg,args.entry)
    elif args.stage=='signal':inference(cfg,args.entry,[args.epoch],['source'],'signals/e%03d/manifest.json'%args.epoch,True)
    else:
        terminal=read(args.entry/'termination.json');d=terminal['decision']
        ap=source_ap_selection(args.entry,terminal['history']);save(args.entry/'source_AP_selection.json',ap)
        inference(cfg,args.entry,sorted({d['epoch'],d['best_epoch'],ap['epoch']}),['target'],'inference_manifest.json')
    name='signal_%03d_gpu.json'%args.epoch if args.stage=='signal' else args.stage+'_gpu.json'
    save(args.entry/name,dict(guard=guard,guard_sha256=sha(guard),code=code_identity(),
        target_GT_loaded=False,source_GT_used=args.stage in ('source_fit','probe','train')))


def main(args):
    from tools.run_phase_boundary import reuse_reference_snapshot
    from tools.phase_runner_gpu import acquire
    from tools.baseline_gpu import snapshot
    root=args.output;root.mkdir(exist_ok=False);identity=code_identity()
    references=reuse_reference_snapshot(root)
    tmp=Path('/vepfs/vlms/tmp')/('supervision_risk_'+identity['commit'][:7]);tmp.mkdir(exist_ok=False)
    configs={}
    for shift in SHIFTS:
        original=read(REFERENCE/shift/'config.json')
        source_db=read(original['dataset']['json_file'])['database']
        target_db=read(original['baseline']['target_manifest'])['database']
        train_ids={k for k,v in source_db.items() if v['subset'] in original['src_train_split']}
        val_ids={k for k,v in source_db.items() if v['subset'] in original['src_val_split']}
        assert train_ids and val_ids and not train_ids&val_ids
        assert not (train_ids|val_ids)&set(target_db)
        assert all('annotations' not in v for v in target_db.values())
        for mode in MODES:
            cfg=deepcopy(original);cfg['baseline'].update(code=identity,tmpdir=str(tmp),schema=SCHEMA)
            cfg['supervision_risk']=dict(mode=mode,spec=SPEC,reference=str(REFERENCE/shift),target_GT_used=False,
                feasibility_entry=str(root/'feasibility'/mode/shift))
            configs[mode,shift]=cfg
            for folder in ['feasibility','probes','formal']:
                e=root/folder/mode/shift;e.mkdir(parents=True);save(e/'config.json',cfg)
                if folder=='formal':
                    for domain in ['source','target']:
                        save(e/(domain+'_val_unlabeled.json'),read(root/'reference_snapshot/formal'/shift/(domain+'_val_unlabeled.json')),sort_keys=False)
    save(root/'plan.json',dict(schema=SCHEMA,spec=SPEC,modes=list(MODES),shifts=list(SHIFTS),count=6,code=identity,A_not_retrained=True))
    save(root/'inputs.json',dict(references=references,config_sha={m+'/'+s:sha(root/'formal'/m/s/'config.json') for m,s in configs}))
    pending=[(m,s,'source_fit',None) for s in SHIFTS for m in MODES];active={};reports={};done=[];signals=set();source_reports={};source_reviewed=False;probes_reviewed=False
    formal=False;minimum=12288
    def spawn(m,s,stage,epoch,card):
        e=root/('feasibility' if stage=='source_fit' else 'probes' if stage=='probe' else 'formal')/m/s
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',
            PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
        if card:env.update(CUDA_VISIBLE_DEVICES=card['uuid'],SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        command=[sys.executable,'-u','-B',__file__,'--output',str(root),'--entry',str(e),'--stage',stage]
        if epoch is not None:command += ['--epoch',str(epoch)]
        name=stage+('_%03d'%epoch if epoch is not None else '')+'.log'
        with (e/name).open('x') as log:p=subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return dict(process=p,mode=m,shift=s,stage=stage,epoch=epoch,gpu=card['index'] if card else None)
    try:
        while pending or active:
            for key,t in list(active.items()):
                rc=t['process'].poll()
                if rc is None:continue
                if t.get('context'):t['context'].__exit__(None,None,None)
                del active[key];assert rc==0,(key,rc)
                m,s,stage,epoch=[t[k] for k in ['mode','shift','stage','epoch']];e=root/'formal'/m/s
                if stage=='source_fit':
                    from tools.supervision_risk_feasibility import audit as audit_source
                    audit_source(configs[m,s],root/'feasibility'/m/s)
                    path=root/'feasibility'/m/s/'result.json';source_reports[m+'/'+s]=dict(path=str(path),sha256=sha(path),report=read(path))
                elif stage=='probe':
                    path=root/'probes'/m/s/'result.json';r=read(path)
                    reports[m+'/'+s]=dict(path=str(path),sha256=sha(path),report=r)
                elif stage=='train':pending.append((m,s,'inference',None))
                elif stage=='inference':pending.append((m,s,'evaluate_cpu',None))
                elif stage=='signal':pending.append((m,s,'signal_cpu',epoch))
            if len(source_reports)==6 and not source_reviewed:
                source_reviewed=True
                ready=all(x['report']['status']=='ready' for x in source_reports.values())
                save(root/'source_mechanism_review.json',dict(ready=ready,reports=source_reports))
                if not ready:
                    assert not pending and not active
                    save(root/'state.json',dict(pid=os.getpid(),phase='source_mechanism_unready',complete_count=0,pending=[],active={}))
                    return
                pending.extend((m,s,'probe',None) for s in SHIFTS for m in MODES)
            if len(reports)==6 and not probes_reviewed:
                probes_reviewed=True
                ready=all(x['report']['status']=='passed' for x in reports.values())
                if not ready:
                    save(root/'target_mechanism_review.json',dict(ready=False,reports=reports))
                    save(root/'state.json',dict(pid=os.getpid(),phase='target_mechanism_unready',complete_count=0,pending=[],active={}))
                    return
            if len(reports)==6 and not formal:
                minimum=max(12288,math.ceil(max(x['report']['peak_mib'] for x in reports.values())+4096))
                save(root/'acceptance.json',dict(status='passed',reports=reports,minimum_free_mib=minimum))
                pending.extend((m,s,'train',None) for s in SHIFTS for m in MODES);formal=True
            if formal:
                for m,s in configs:
                    e=root/'formal'/m/s
                    for p in sorted(e.glob('epoch*.pt')):
                        epoch=int(p.stem[5:]);token=(m,s,epoch)
                        if token not in signals:signals.add(token);pending.insert(0,(m,s,'signal',epoch))
                    if (e/'evaluation/final.json').exists() and not (e/'complete.json').exists():
                        busy=any(t['mode']==m and t['shift']==s for t in active.values()) or any(x[0]==m and x[1]==s for x in pending)
                        if not busy:
                            paths=[p for p in e.rglob('*') if p.is_file() and p.name!='complete.json']
                            save(e/'complete.json',dict(schema=SCHEMA,code=identity,artifacts={str(p):sha(p) for p in paths}))
                            done.append(audit(e));save(root/'progress.json',dict(count=len(done),results=done))
            priority={'signal':0,'signal_cpu':0,'inference':1,'evaluate_cpu':1,'source_fit':2,'probe':2,'train':3}
            for m,s,stage,epoch in sorted(list(pending),key=lambda x:priority[x[2]]):
                card=context=None
                if stage=='inference':
                    e=root/'formal'/m/s
                    if not all((e/('signals/e%03d/evaluation.json'%ep)).exists() for ep,_ in read(e/'termination.json')['history']):continue
                if stage in ('signal_cpu','evaluate_cpu'):
                    if sum(t['stage'] in ('signal_cpu','evaluate_cpu') for t in active.values())>=2:continue
                else:
                    is_signal=stage=='signal'
                    if is_signal and any(t['stage']=='signal' for t in active.values()):continue
                    if not is_signal and sum(t['gpu'] is not None and t['stage']!='signal' for t in active.values())>=4:continue
                    counts={g:sum(t['gpu']==g for t in active.values()) for g in range(4)}
                    cards=[g for g in sorted([2,0,3,1],key=counts.get) if counts[g]<(2 if is_signal else 1)]
                    cards=[g for g in cards if not(stage in ('source_fit','probe','train') and any(t['gpu']==g and t['mode']==m and t['stage'] in ('source_fit','probe','train') for t in active.values()))]
                    cards=[g for g in cards if snapshot(g)['free_mib']>=minimum*(1+counts[g])]
                    found=acquire(cards,args.data_root,root,minimum,identity['commit'])
                    if not found:continue
                    context,card=found
                task=spawn(m,s,stage,epoch,card)
                if context:task['context']=context
                active['%s/%s/%s/%s'%(m,s,stage,epoch)]=task;pending.remove((m,s,stage,epoch))
            save(root/'state.json',dict(pid=os.getpid(),phase='formal' if formal else 'GPU_acceptance' if source_reviewed else 'source_feasibility',formal_started=formal,
                complete_count=len(done),pending=pending,active={k:dict(pid=t['process'].pid,**{x:t[x] for x in ['mode','shift','stage','epoch','gpu']}) for k,t in active.items()}))
            time.sleep(3)
        assert len(done)==6 and len({(x['mode'],x['shift']) for x in done})==6
        save(root/'final.json',dict(count=6,results=done,
            means={m:sum(x['map'] for x in done if x['mode']==m)/6 for m in MODES},
            source_AP_selected_means={m:sum(x['source_AP_selected_map'] for x in done if x['mode']==m)/6 for m in MODES},schema=SCHEMA))
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
            if t.get('context'):t['context'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--entry',type=Path)
    p.add_argument('--stage',choices=['source_fit','probe','train','inference','evaluate_cpu','signal','signal_cpu']);p.add_argument('--epoch',type=int)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.stage:worker(args)
    else:main(args)
