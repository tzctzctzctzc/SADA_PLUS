"""固定教师缓存→真实验收→三组六方向定位训练；目标验证只离线评估。"""
import argparse
from copy import deepcopy
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,assert_no_target_gt
from tools.run_sada_baseline import acquire,prepare_file_limit
from tools.localization_uda import SPEC
from tools.run_tta import load_model,evaluate_predictions
from tools.run_masked_uda import validation,prediction_item

SHIFTS=tuple('S%d'%i for i in range(1,7));ACTIVE={}


def cache_teacher(job):
    import torch
    from tools.baseline_engine import dataset,tensor_digest
    from tools.tta_core import dense,unlabeled,crop,predict_full
    from tools.boundary_diagnostic import ranges,candidates,attach_stability
    from tools.localization_uda import select_pseudo
    cfg=read(job['config']);model=load_model(cfg,job['checkpoint']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    initial=tensor_digest(model.state_dict());ds=dataset(cfg,job['target_manifest'],cfg['tgt_train_split'],False,True)
    videos={};counts=[];exact=False
    with torch.no_grad():
        for i in range(len(ds)):
            v=unlabeled(ds[i]);vid=v['metadata']['video_id'];windows=ranges(v['feats'].shape[-1]);outputs=[]
            for begin,end in windows:
                raw=dense(model,crop(v,begin,end))
                outputs.append({k:[x.detach().cpu() for x in raw[k]] for k in ('points','offsets','logits','masks')})
            if i==0:
                ours=predict_full(model,v);original=model([v],[v],mode='test')[0][0]
                exact=all(torch.equal(ours[k],original[k]) for k in ('segments','scores','labels'));assert exact
            rows=attach_stability(candidates(outputs[0],v['feats'].shape[-1]),outputs,windows)
            selected={k:select_pseudo(rows,k) for k in ('confidence','stability')}
            a={b:sum(x['selector_bin']==b for x in selected['confidence']) for b in range(16)}
            b={b:sum(x['selector_bin']==b for x in selected['stability']) for b in range(16)}
            assert a==b
            videos[vid]=dict(candidates=rows,**selected)
            counts.append(dict(video_id=vid,views=len(windows),eligible=sum('relative_std' in x for x in rows),selected=len(selected['stability'])))
            if i%10==0:print('CACHE',i+1,len(ds),json.dumps(counts[-1]),flush=True)
    assert tensor_digest(model.state_dict())==initial
    output=Path(job['output']);save(output/'proposals.json',dict(specification=SPEC,teacher_checkpoint_sha256=job['checkpoint_sha256'],
        teacher_initial_state_sha256=initial,target_GT_used=False,videos=videos))
    save(output/'coverage.json',counts)
    return dict(status='passed',videos=len(ds),selected_videos=sum(x['selected']>0 for x in counts),
        selected_candidates=sum(x['selected'] for x in counts),ordinary_prediction_exact=exact,model_unchanged=True,
        initial_state_sha256=initial,target_GT_used=False,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)


def training(job,probe=False):
    import numpy as np
    import torch
    from tools.baseline_engine import dataset,loader,tensor_digest,checkpoint,controls
    from tools.tta_core import dense,unlabeled,predict_full
    from tools.ttt_recovery import source_detection_loss
    from tools.localization_uda import trainable_parameters,frozen_tensors,delta,apply_delta,update_ema,pseudo_regression,choose
    cfg=read(job['config']);out=Path(job['output']);variant=job['variant'];cache=read(job['cache'])
    assert cache['specification']==SPEC and cache['teacher_checkpoint_sha256']==job['checkpoint_sha256'] and not cache['target_GT_used']
    model=load_model(cfg,job['checkpoint']);named=trainable_parameters(model)
    assert model.backbone_source is model.backbone_target and model.neck_source is model.neck_target and model.cls_head is model.cls_head_target
    initial=tensor_digest(model.state_dict());assert initial==cache['teacher_initial_state_sha256']
    frozen=tensor_digest(frozen_tensors(model,named));initial_delta=delta(named)
    ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    ema_named={n:p for n,p in ema.named_parameters() if n in named}
    opt=torch.optim.AdamW(list(named.values()),lr=SPEC['learning_rate'],weight_decay=SPEC['weight_decay'])
    generators={k:torch.Generator().manual_seed(123+i*1000003) for i,k in enumerate(('source','target'))}
    src=dataset(cfg,job['source_manifest'],cfg['src_train_split'],True)
    tgt=dataset(cfg,job['target_manifest'],cfg['tgt_train_split'],True,True)
    source=loader(src,True,generators['source'],2,0);target=loader(tgt,True,generators['target'],2,0)
    budget=cfg['baseline']['epoch_budget']['iterations_per_epoch'];history=[];epochs=[];target_gradient_seen=False
    assert budget>=max(len(source),len(target))
    probe_video=None;probe_logits=None;probe_prediction=None
    if probe:
        full=dataset(cfg,job['target_manifest'],cfg['tgt_train_split'],False,True)
        wanted=next(v for v,x in cache['videos'].items() if x['stability'])
        probe_video=next(unlabeled(full[i]) for i in range(len(full)) if full[i]['metadata']['video_id']==wanted)
        with torch.no_grad():
            probe_logits=[x.cpu().clone() for x in dense(model,probe_video)['logits']];probe_prediction=predict_full(model,probe_video)
    else:
        score=validation(ema,cfg);history=[dict(epoch=0,map=score)]
        save(out/'source_history.json',history)
        checkpoint(out/'best.pt',dict(config=cfg,regression_delta=delta(ema_named),epoch=0,parent_checkpoint_sha256=job['checkpoint_sha256'],
            specification=SPEC,variant=variant,initial_state_sha256=initial))
        print('SOURCE_VALIDATION',0,score,flush=True)
    last_epoch=0
    for epoch in range(1,(1 if probe else SPEC['max_epochs'])+1):
        source_iter=itertools.cycle(source);target_iter=itertools.cycle(target);reports=[]
        for step in range(3 if probe else budget):
            global_step=(epoch-1)*budget+step;random.seed(123+global_step);np.random.seed(123+global_step)
            sources=next(source_iter);targets=next(target_iter)
            if probe:targets=[probe_video]
            assert_no_target_gt(targets);opt.zero_grad(set_to_none=True)
            supervised=torch.stack([source_detection_loss(model,v) for v in sources]).mean()
            terms=[];points=0
            if variant!='source_only':
                for v in targets:
                    loss,count=pseudo_regression(model,unlabeled(v),cache['videos'][v['metadata']['video_id']][variant])
                    terms.append(loss);points+=count
            pseudo=torch.stack(terms).mean() if terms else next(iter(named.values())).sum()*0
            target_norm=0.
            if probe and variant!='source_only' and points:
                grads=torch.autograd.grad(pseudo,list(named.values()),retain_graph=True,allow_unused=True)
                values=[g.norm() for g in grads if g is not None];target_norm=float(torch.stack(values).norm()) if values else 0.
                target_gradient_seen|=target_norm>1e-8
            loss=supervised+SPEC['pseudo_regression_weight']*pseudo
            if not torch.isfinite(loss):raise ValueError('定位训练loss非有限')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(named.values()),SPEC['clip_grad'])
            if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('定位回归梯度无效')
            opt.step();update_ema(ema,named)
            report=dict(source=float(supervised.detach()),pseudo=float(pseudo.detach()),gradient_norm=float(norm),
                pseudo_points=points,target_gradient_norm=target_norm,target_GT_used=False,target_annotations=0)
            reports.append(report)
            if step%20==0:print('TRAIN',variant,epoch,step+1,budget,json.dumps(report),flush=True)
        last_epoch=epoch
        aggregate=dict(epoch=epoch,steps=len(reports),source=sum(x['source'] for x in reports)/len(reports),
            pseudo=sum(x['pseudo'] for x in reports)/len(reports),pseudo_points=sum(x['pseudo_points'] for x in reports),
            target_GT_used=False,target_annotations=0)
        epochs.append(aggregate);save(out/'epochs.json',epochs)
        if probe:break
        if epoch%SPEC['validate_every']==0:
            score=validation(ema,cfg);previous_best=choose(history)[0]['map'];history.append(dict(epoch=epoch,map=score));save(out/'source_history.json',history)
            if score>previous_best:
                checkpoint(out/'best.pt',dict(config=cfg,regression_delta=delta(ema_named),epoch=epoch,parent_checkpoint_sha256=job['checkpoint_sha256'],
                    specification=SPEC,variant=variant,initial_state_sha256=initial))
            print('SOURCE_VALIDATION',epoch,score,flush=True)
        checkpoint(out/'latest.pt',dict(config=cfg,regression_delta=delta(named),ema_regression_delta=delta(ema_named),
            optimizer=opt.state_dict(),epoch=epoch,history=history,epochs=epochs,specification=SPEC,variant=variant,
            parent_checkpoint_sha256=job['checkpoint_sha256'],model_controls=controls(model),ema_controls=controls(ema),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0),
                loaders={k:g.get_state() for k,g in generators.items()})))
        if history[-1]['epoch']==epoch and choose(history)[1]:break
    assert tensor_digest(frozen_tensors(model,named))==frozen
    result=dict(status='passed',variant=variant,initial_state_sha256=initial,frozen_parameters_unchanged=True,
        trainable_names=list(named),trainable_count=sum(p.numel() for p in named.values()),target_GT_used=False,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    if probe:
        with torch.no_grad():after=[x.cpu() for x in dense(model,probe_video)['logits']]
        exact=all(torch.equal(a,b) for a,b in zip(probe_logits,after));assert exact
        final_delta=delta(named);assert tensor_digest(final_delta)!=tensor_digest(initial_delta)
        apply_delta(named,initial_delta);restored=predict_full(model,probe_video)
        roundtrip=all(torch.equal(probe_prediction[k],restored[k]) for k in ('segments','scores','labels'));assert roundtrip
        apply_delta(named,final_delta)
        if variant!='source_only' and not target_gradient_seen:raise ValueError('定位伪监督未产生真实回归梯度')
        result.update(classifier_logits_exact=exact,restore_roundtrip_exact=roundtrip,real_target_gradient_seen=target_gradient_seen,regression_updated=True)
    else:
        chosen,stop=choose(history);assert stop
        term=dict(status='passed',variant=variant,best_epoch=chosen['epoch'],best_source_map=chosen['map'],stop_epoch=last_epoch,
            initial_state_sha256=initial,frozen_parameters_sha256=frozen,trainable_names=list(named),epochs=epochs,source_history=history,
            target_GT_used=False,frozen_parameters_unchanged=True,specification=SPEC)
        save(out/'termination.json',term);result.update(best_epoch=chosen['epoch'],best_source_map=chosen['map'],stop_epoch=last_epoch)
    return result


def evaluate(job):
    import torch
    from tools.baseline_engine import dataset
    from tools.tta_core import predict_full,unlabeled
    from tools.localization_uda import trainable_parameters,apply_delta
    cfg=read(job['config']);out=Path(job['output'])
    assert sha(out/'train_complete.json')==job['train_complete_sha256']
    assert sha(out/'best.pt')==read(out/'train_complete.json')['artifacts']['best.pt']
    saved=torch.load(out/'best.pt',map_location='cpu')
    assert saved['config']==cfg and saved['parent_checkpoint_sha256']==job['checkpoint_sha256'] and saved['specification']==SPEC
    assert saved['variant']==job['variant'] and saved['epoch']==read(out/'termination.json')['best_epoch']
    model=load_model(cfg,job['checkpoint']);apply_delta(trainable_parameters(model),saved['regression_delta'])
    ds=dataset(cfg,job['evaluation_metadata'],cfg['tgt_val_split'],False,True);predictions=[]
    for i in range(len(ds)):
        predictions.append(prediction_item(predict_full(model,unlabeled(ds[i]))));print('EVAL',i+1,len(ds),flush=True)
    save(out/'predictions.json',predictions)
    result=evaluate_predictions(predictions,job['evaluation_annotation'],cfg['tgt_val_split'])
    result.update(status='passed',target_GT_during_training=False,videos=len(ds));return result


def worker(args):
    from tools.baseline_gpu import first_cuda_guard
    job=read(args.job);out=Path(job['output']);assert job['code']==code_identity() and job['specification']==SPEC
    for field in ('config','checkpoint','source_manifest','target_manifest','cache','evaluation_metadata','evaluation_annotation'):
        if field in job and sha(job[field])!=job[field+'_sha256']:raise ValueError('定位输入SHA改变:'+field)
    assert all('annotations' not in x for x in read(job['target_manifest'])['database'].values())
    guard=first_cuda_guard();os.chdir(args.data_root)
    result=cache_teacher(job) if job['kind']=='cache' else training(job,job['kind']=='probe') if job['kind'] in ('train','probe') else evaluate(job)
    names=('proposals.json','coverage.json','epochs.json','source_history.json','best.pt','latest.pt','termination.json','predictions.json')
    name='train_complete.json' if job['kind']=='train' else 'complete.json'
    save(out/name,dict(status='passed',job_sha256=sha(args.job),code=job['code'],guard=guard,guard_sha256=sha(guard),
        result=result,artifacts={n:sha(out/n) for n in names if (out/n).exists()}))


def audit(path):
    from tools.localization_uda import choose
    job=read(path);d=Path(job['output']);cp=d/('train_complete.json' if job['kind']=='train' else 'complete.json');a=read(cp)
    assert a['status']=='passed' and a['job_sha256']==sha(path) and a['code']==job['code']
    for field in ('config','checkpoint','source_manifest','target_manifest','cache','evaluation_metadata','evaluation_annotation'):
        if field in job:assert sha(job[field])==job[field+'_sha256']
    for name,h in a['artifacts'].items():assert sha(d/name)==h
    g=read(a['guard']);l=read(g['lease'])
    assert sha(a['guard'])==a['guard_sha256'] and sha(g['lease'])==g['lease_sha256'] and l['commit']==job['code']['commit']
    assert g['observed']['uuid']==l['observed']['uuid'] and g['observed']['free_mib']>=12288 and g['owner_pid']==l['owner_pid']
    if job['kind'] in ('train','evaluate'):
        t=read(d/'termination.json');best,stop=choose(t['source_history'])
        assert stop and best['epoch']==t['best_epoch'] and t['epochs'][-1]['epoch']==t['stop_epoch']
        assert [x['epoch'] for x in t['epochs']]==list(range(1,t['stop_epoch']+1))
        assert not any(choose(t['source_history'][:i])[1] for i in range(1,len(t['source_history'])))
        assert not any(x['target_GT_used'] or x['target_annotations'] for x in t['epochs']) and t['frozen_parameters_unchanged']
        if job['kind']=='evaluate':
            assert sha(d/'train_complete.json')==job['train_complete_sha256']
            assert a['artifacts']['best.pt']==read(d/'train_complete.json')['artifacts']['best.pt']
    return dict(a['result'],kind=job['kind'],shift=job['shift'],variant=job.get('variant'),job=str(path),complete_sha256=sha(cp))


def run_jobs(root,jobs,data,identity,phase):
    pending=list(jobs);done=[]
    if pending and read(pending[0])['kind']=='train':pending.sort(key=lambda p:read(read(p)['config'])['baseline']['epoch_budget']['iterations_per_epoch'],reverse=True)
    while pending or ACTIVE:
        for key,t in list(ACTIVE.items()):
            if t['process'].poll() is None:continue
            t['context'].__exit__(None,None,None);del ACTIVE[key]
            if t['process'].returncode:raise RuntimeError('定位任务失败:'+key)
            done.append(audit(t['job']))
        for path in list(pending):
            found=acquire([0,1],data,root,12288,identity['commit'])
            if not found:break
            context,slot=found;j=read(path);out=Path(j['output'])
            env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=slot['uuid'],
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',SADA_BASELINE_LEASE=slot['lease'],SADA_BASELINE_LEASE_SHA256=slot['lease_sha256'],TMPDIR=str(root/'tmp'))
            with (out/(j['kind']+'.log')).open('x') as stream:
                p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--job',str(path),'--data-root',str(data)],cwd=CODE,
                    env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            key=str(out.relative_to(root));ACTIVE[key]=dict(process=p,context=context,job=path,gpu=slot['index']);pending.remove(path)
        save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=len(done),total=len(jobs),
            active={k:dict(pid=t['process'].pid,gpu=t['gpu']) for k,t in ACTIVE.items()}))
        if pending or ACTIVE:time.sleep(3)
    return done


def make_job(root,relative,kind,s,args,identity,variant=None):
    base=args.prototype/'formal'/s/'prototype';cfg=read(base/'config.json');out=root/relative;out.mkdir(parents=True,exist_ok=False)
    j=dict(kind=kind,shift=s,variant=variant,output=str(out),code=identity,specification=SPEC,config=str(base/'config.json'),
        checkpoint=str(base/'best.pt'),source_manifest=cfg['dataset']['json_file'],target_manifest=cfg['baseline']['target_manifest'])
    if kind in ('train','probe'):j['cache']=str(root/'cache'/s/'proposals.json')
    for field in ('config','checkpoint','source_manifest','target_manifest','cache'):
        if field in j:j[field+'_sha256']=sha(j[field])
    path=out/(kind+'_job.json');save(path,j);return path


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir();identity=code_identity()
    save(root/'resources.json',prepare_file_limit());previous=read(args.diagnostic/'monitor/final_independent_audit.json')
    assert previous['status']=='passed' and previous['count']==6 and previous['decision']['mechanism_gate_passed']
    save(root/'plan.json',dict(code=identity,specification=SPEC,formal_count=18,source_diagnostic_final_sha256=sha(args.diagnostic/'monitor/final_independent_audit.json'),
        diagnostic_duration_control_sha256=sha(args.diagnostic/'monitor/duration_confound_audit.json'),prototype_final_sha256=sha(args.prototype/'final_audit.json')))
    try:
        caches=run_jobs(root,[make_job(root,'cache/'+s,'cache',s,args,identity) for s in SHIFTS],args.data_root,identity,'fixed_teacher_cache')
        assert {x['shift'] for x in caches}==set(SHIFTS) and len(caches)==6
        save(root/'cache/final_audit.json',dict(status='passed',count=6,results=caches))
        probes=run_jobs(root,[make_job(root,'probe/'+v,'probe','S1',args,identity,v) for v in ('source_only','stability')],args.data_root,identity,'real_gpu_acceptance')
        assert len({x['initial_state_sha256'] for x in probes})==1
        assert all(x['classifier_logits_exact'] and x['frozen_parameters_unchanged'] and x['restore_roundtrip_exact'] and x['regression_updated'] for x in probes)
        assert next(x for x in probes if x['variant']=='stability')['real_target_gradient_seen']
        save(root/'acceptance.json',dict(status='passed',code=identity,results=probes))
        groups={}
        for v in SPEC['variants']:
            jobs=[make_job(root,'formal/'+v+'/'+s,'train',s,args,identity,v) for s in SHIFTS]
            trained=run_jobs(root,jobs,args.data_root,identity,'training_'+v);evaluations=[]
            for p in jobs:
                j=read(p);out=Path(j['output']);cfg=read(j['config'])
                meta=root/'inputs'/j['shift']/'evaluation_unlabeled.json'
                previous_meta=args.evaluation_inputs/j['shift']/'evaluation_unlabeled.json'
                payload=read(previous_meta);assert all('annotations' not in row for row in payload['database'].values());save(meta,payload)
                j.update(kind='evaluate',evaluation_metadata=str(meta),evaluation_metadata_sha256=sha(meta),
                    evaluation_annotation=cfg['baseline']['annotation_for_offline_evaluation'],evaluation_annotation_sha256=sha(cfg['baseline']['annotation_for_offline_evaluation']),
                    train_complete_sha256=sha(out/'train_complete.json'),reused_evaluation_metadata=str(previous_meta),reused_evaluation_metadata_sha256=sha(previous_meta))
                q=out/'evaluate_job.json';save(q,j);evaluations.append(q)
            result=run_jobs(root,evaluations,args.data_root,identity,'evaluation_'+v);groups[v]=result
            save(root/'formal'/v/'final_audit.json',dict(status='passed',count=6,training=trained,results=result,mean_map=sum(x['map'] for x in result)/6))
        for s in SHIFTS:
            terms=[read(root/'formal'/v/s/'termination.json') for v in SPEC['variants']]
            assert len({x['initial_state_sha256'] for x in terms})==1
            assert max(x['source_history'][0]['map'] for x in terms)-min(x['source_history'][0]['map'] for x in terms)<1e-10
        save(root/'final_audit.json',dict(status='passed',count=18,results=groups));save(root/'state.json',dict(phase='complete',pid=os.getpid(),active={},count=18))
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)));save(root/'state.json',dict(phase='failed',pid=os.getpid(),active={}))
        raise
    finally:
        for t in ACTIVE.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in ACTIVE.values():
            try:t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
            t['context'].__exit__(None,None,None)


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);raise SystemExit(128+signum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prototype',type=Path);p.add_argument('--diagnostic',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--job',type=Path)
    p.add_argument('--evaluation-inputs',type=Path,default=Path('/root/autodl-tmp/vlm/prototype_masked_uda18_cd9402f_1101/inputs'))
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.job:worker(args)
    else:main(args)
