"""完整检测微调：原focal配对对照 versus AP阈值质量分类，均保留源检测/目标SADA。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_native_joint import SOURCE,DENSE
MODES=('focal','iacs');TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
READOUTS=('native','scale_score','scale_boundary')
SPEC=dict(schema='full-detector-AP-IACS-SADA-v1',shifts=['S1','S2'],modes=MODES,epochs=20,checkpoints=[1,5,10,20],
    source='ALL originalsourceTRAIN51/124 videos, all dense detection anchors; sourceval excluded. Previoushead-heldout now entersFIT and is NOT independent evaluation.',
    task='originalsource classification and DIoU regression; IACS changes originalclassification targets to fraction of fixed tIoUs[.1,.2,.3,.4,.5] passed by assigned predictedsourcebox; targetdetached',
    quality_loss='authors originalVarifocalLoss alpha.75 gamma2 iou_weightedTrue; applied to actual10class logits, noauxBGhead or multiplicationofuntrainedquality',
    adaptation='same originalSADA conditionalGRL/discriminator/domain/class/level weights inboth groups fromE6; sourceGT assignments, frozen originalPrototype semantic teacher targetpseudo groups, never targetGT',
    semantic_teacher='originalbinaryclass semantics/threshold preserved while student scores change toquality; eachtarget fullFPN logits cached once, teacher never updated',
    trained='shared originalpretrainedSGP/neck AND originalclassification/regression heads jointly learn from E1; body dropout remains eval like priornativeprecheck; detectorfiles/base immutable',
    prototype='originalPrototype learned baseweights retained; prototype auxiliary not rerun in this score-calibration phase, identical forpairedcontrol',
    main_lr=1e-5,weight_decay=.05,clip=1.,ema=.999,seed=311104,order_seed=123,target_seed=311108,
    original_branch='originalconfigured discriminatorlr/architecture/dropout/beta/domainweights/weightclip; fixedadv/downstreamratio; no parametersearch',
    fresh='newmodel/AdamW/EMA/order/RNG from originalPrototype bestEMA, no oldbackgroundexperimentcheckpoint/optimizer/history',
    inference='recompute all scores/boundaries/candidates with originalNMS config, no deletequota; fixedEMA20 only; native + preregistered3scale composition, all reported',
    readouts=READOUTS,scales=[.75,1.,1.25],primary_tracks=['iacs/native','iacs/scale_boundary'],
    target_train_GT_read=False,target_val_used_for_selection=False,budget_seconds=1800,
    acceptance='fixedtwo-shift targetmeanAP@.1:.5 gain>=1pp, bothpositive; then methodfrozen nextdirections; allpairedcontrols/negativecurves retained',
    caveat='historically exposedS1/S2 development; not independent six-direction finalmean; metricqualitytarget is not exactAP gradient or oraclematching')


def prepare(root):
    import torch
    from tools import baseline_engine as base
    folder=root/'inputs';folder.mkdir();reports={}
    for s in ('S1','S2'):
        cfg=read(SOURCE/s/'config.json');manifest=SOURCE/s/'source_fit_unlabeled.json';truth=read(SOURCE/s/'source_truth.json');ds=base.dataset(cfg,manifest,cfg['src_train_split'],False,True);rows=[]
        dpath=DENSE/'inputs'/s/'data.pt';assert sha(dpath)==read(DENSE/'prepare_complete.json')['results'][s]['data_sha256'];prior=torch.load(dpath,map_location='cpu');allowed={'video','segments','scores','labels','hard','grid_boxes','video_metadata','raw_path','raw_size','raw_mtime_ns','raw_sha256','downsample'}
        for video in ds:
            assert not video['segments'].numel() and not video['labels'].numel();vid=video['metadata']['video_id'];gt=truth[vid];labels=torch.tensor(gt['labels'],dtype=torch.long);assert not labels.numel() or (labels.min()>=0 and labels.max()<10)
            rows.append(dict(video=vid,raw=video['feats'].T.contiguous(),metadata=video['metadata'],gt_segments=torch.tensor(gt['segments'],dtype=torch.float32).reshape(-1,2),gt_labels=labels))
        rows.sort(key=lambda x:x['video']);ids={r['video'] for r in rows};assert ids==set(truth)==set(read(manifest)['database'])
        assert ids=={r['video'] for role in ('fit','heldout') for r in prior['roles'][role]}
        assert all(not ids&{r['video'] for r in prior['roles'][role]} for role in ('validation','target_train','target_val'))
        target=prior['roles']['target_train'];evaluation={role:prior['roles'][role] for role in ('validation','target_val')};assert all(set(r)<=allowed for r in target+[r for v in evaluation.values() for r in v]);base.checkpoint(folder/(s+'_source.pt'),rows);base.checkpoint(folder/(s+'_unlabeled.pt'),dict(target_train=target,evaluation=evaluation))
        reports[s]=dict(source_sha256=sha(folder/(s+'_source.pt')),unlabeled_sha256=sha(folder/(s+'_unlabeled.pt')),source_videos=[r['video'] for r in rows],target_videos=[r['video'] for r in target],source_truth_sha256=sha(SOURCE/s/'source_truth.json'),source_config_sha256=sha(SOURCE/s/'config.json'),prior_audited_input_sha256=sha(dpath),source_labels_only=True)
    save(root/'prepare.json',dict(passed=True,shifts=reports,target_train_GT_read=False))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.detector_iacs_math import QualityDetector,ema_step,restore_optimizer
    from tools.native_sada_math import conditional_loss
    from tools.run_foreground_calibration_oracle import load_detector,REFERENCE
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_dense_revisit import seed,raw_load
    from tools.run_uda_background import TargetStream
    from tools.detector_scale_math import rescale_raw,fuse_heads
    guard=first_cuda_guard(entry);s,mode=entry.name.split('_');quality=mode=='iacs';began=time.monotonic();inputdir=entry.parent/'inputs';prepared=read(entry.parent/'prepare.json')['shifts'][s]
    assert sha(inputdir/(s+'_source.pt'))==prepared['source_sha256'] and sha(inputdir/(s+'_unlabeled.pt'))==prepared['unlabeled_sha256'];fit=torch.load(inputdir/(s+'_source.pt'),map_location='cpu');data=torch.load(inputdir/(s+'_unlabeled.pt'),map_location='cpu');target=data['target_train'];evaluation=data['evaluation'];cfg=read(SOURCE/s/'config.json')
    raw=[r['raw'].pin_memory() for r in fit];gt=[(r['gt_segments'].cuda(),r['gt_labels'].cuda()) for r in fit];targetraw={r['video']:raw_load(r).pin_memory() for r in target}
    def build():
        d=load_detector(SOURCE/s);d.requires_grad_(False);seed();m=QualityDetector(d).cuda();m.eval();m.body_mode(True);m.cls_head.requires_grad_(True);m.reg_head.requires_grad_(True);m.disc_dom_conditional_adv.requires_grad_(True);m.learnable_embeddings.requires_grad_(True)
        params=list(m.backbone.parameters())+list(m.neck.parameters())+list(m.cls_head.parameters())+list(m.reg_head.parameters());o=torch.optim.AdamW(params,lr=1e-5,weight_decay=.05);disc=torch.optim.AdamW(m.da_parameters(),lr=cfg['opt']['disc_learning_rate'],weight_decay=.05)
        return m,o,disc
    m,o,d=build();ema,_,_=build();ema.load_state_dict(m.state_dict());ema.requires_grad_(False);ema.eval();teacher,_,_=build();teacher.requires_grad_(False);teacher.eval();teacher_digest=base.tensor_digest(teacher.state_dict());cache={}
    def controls(m):return dict(loss_normalizer=m.loss_normalizer,loss_normalizer_pseudo_labels=m.loss_normalizer_pseudo_labels)
    def capture(m,e,o,d,order,stream,history,epoch,step):
        return dict(model=deepcopy(m.state_dict()),ema=deepcopy(e.state_dict()),optimizer=deepcopy(o.state_dict()),disc_optimizer=deepcopy(d.state_dict()),controls=controls(m),order_rng=order.get_state(),target_order=stream.state(),semantic_cache=deepcopy(cache),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),history=deepcopy(history),epoch=epoch,step=step,spec=SPEC,source_input_sha256=prepared['source_sha256'],unlabeled_input_sha256=prepared['unlabeled_sha256'],original_checkpoint_sha256=sha(REFERENCE/s/'best.pt'))
    def restore(st,m,e,o,d,order,stream):
        m.load_state_dict(st['model']);e.load_state_dict(st['ema']);restore_optimizer(o,st['optimizer']);restore_optimizer(d,st['disc_optimizer']);m.loss_normalizer=st['controls']['loss_normalizer'];m.loss_normalizer_pseudo_labels=st['controls']['loss_normalizer_pseudo_labels'];order.set_state(st['order_rng']);stream.restore(st['target_order']);cache.clear();cache.update(deepcopy(st['semantic_cache']));random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng']);torch.set_rng_state(st['cpu_rng']);torch.cuda.set_rng_state(st['cuda_rng'],0)
    def update(m,e,o,d,i,epoch,ti=None):
        m.body_mode(True);m.cls_head.train();m.reg_head.train();m.disc_dom_conditional_adv.train(epoch>5);o.zero_grad(set_to_none=True);d.zero_grad(set_to_none=True);r=fit[i];f,v,c,z,a=m.forward_heads(raw[i].to('cuda',non_blocking=True),r['metadata']);loss,details,sl=m.source_loss(f,v,c,z,a,*gt[i],quality);adv=loss.new_zeros(())
        if epoch>5:
            tr=target[ti];x=targetraw[tr['video']].to('cuda',non_blocking=True);tf,tm=m.features(x,tr['video_metadata'])
            if tr['video'] not in cache:
                with torch.no_grad():
                    original,masks=teacher.features(x,tr['video_metadata']);logits=teacher.pseudo_logits(original,masks);assert all(torch.equal(a,b) for a,b in zip(masks,tm));cache[tr['video']]=[a.cpu() for a in logits]
            tl=[a.cuda() for a in cache[tr['video']]];adv,_=conditional_loss(m,f,v,tf,tm,sl,tl,True);loss=loss+cfg['model']['loss_weights']['weight_adv']/cfg['model']['loss_weights']['weight_downstream']*adv
        assert torch.isfinite(loss);loss.backward()
        if epoch>5 and cfg['model']['weight_clip_value']>0:
            for p in m.disc_dom_conditional_adv.parameters():p.data.clamp_(-cfg['model']['weight_clip_value'],cfg['model']['weight_clip_value'])
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
        if epoch>5:d.step()
        o.step();ema_step(e,m,.999)
        return dict(loss=float(loss.detach()),cls=float(details['cls'].detach()),reg=float(details['reg'].detach()),adv=float(adv.detach()),grad=float(norm),positive_anchors=details['positive_anchors'],quality_target_sum=float(details['quality_target_sum']))
    seed();order=torch.Generator().manual_seed(123);stream=TargetStream(len(target));probe=next(i for i,r in enumerate(fit) if len(r['gt_labels']));baseline=evaluation['validation'][0]
    with torch.no_grad():
        old=m.decode(raw_load(baseline).cuda(),baseline['video_metadata'])
        for k in ('segments','scores','labels'):assert torch.equal(old[k].cpu(),baseline[k])
    before={name:base.tensor_digest(getattr(m,name).state_dict()) for name in ('backbone','cls_head','reg_head')};ti=int(stream.draw(1)[0]);update(m,ema,o,d,probe,6,ti)
    assert all(base.tensor_digest(getattr(m,name).state_dict())!=value for name,value in before.items());base.checkpoint(entry/'kernel.pt',capture(m,ema,o,d,order,stream,[],0,1));st=torch.load(entry/'kernel.pt',map_location='cpu')
    restore(st,m,ema,o,d,order,stream);a=(torch.randperm(len(fit),generator=order),stream.draw(1),torch.rand(3),torch.rand(3,device='cuda'));restore(st,m,ema,o,d,order,stream);b=(torch.randperm(len(fit),generator=order),stream.draw(1),torch.rand(3),torch.rand(3,device='cuda'));assert all(torch.equal(x,y) for x,y in zip(a,b))
    checkpoint_digest=base.tensor_digest(st);restore(st,m,ema,o,d,order,stream);ti=int(stream.draw(1)[0]);a=update(m,ema,o,d,probe,6,ti);md=base.tensor_digest(m.state_dict());ed=base.tensor_digest(ema.state_dict());od=base.tensor_digest(o.state_dict());dd=base.tensor_digest(d.state_dict());cd=controls(m);assert checkpoint_digest==base.tensor_digest(st)
    other,oo,odisc=build();otherema,_,_=build();otherema.requires_grad_(False);restore(st,other,otherema,oo,odisc,order,stream);ti=int(stream.draw(1)[0]);b=update(other,otherema,oo,odisc,probe,6,ti)
    comparisons=dict(loss_terms=a==b,model=md==base.tensor_digest(other.state_dict()),ema=ed==base.tensor_digest(otherema.state_dict()),optimizer=od==base.tensor_digest(oo.state_dict()),disc_optimizer=dd==base.tensor_digest(odisc.state_dict()),normalizers=cd==controls(other),saved_state_immutable=checkpoint_digest==base.tensor_digest(st));save(entry/'kernel_reconstruction_comparison.json',comparisons);assert all(comparisons.values()),comparisons
    save(entry/'kernel_complete.json',dict(passed=True,original_pretrained_decode_arrays_bitwise_equal=True,original_focal_loss_exact_when_disabled=True,authors_VFL_used_directly=True,real_backbone_classifier_regressor_updates=True,full_model_EMA_dualopts_controls_RNG_targetcache_nextstep_exact=True,kernel_sha256=sha(entry/'kernel.pt'),guard_sha256=sha(guard)))
    del m,o,d,ema,other,oo,odisc,otherema,st;torch.cuda.empty_cache();cache.clear();m,o,d=build();ema,_,_=build();ema.load_state_dict(m.state_dict());ema.requires_grad_(False);ema.eval();initial=base.tensor_digest(m.state_dict());seed();order=torch.Generator().manual_seed(123);stream=TargetStream(len(target));history=[];step=0;formal=time.monotonic()
    for epoch in range(1,21):
        tick=time.monotonic();records=[]
        for i in torch.randperm(len(fit),generator=order).tolist():records.append(update(m,ema,o,d,i,epoch,int(stream.draw(1)[0]) if epoch>5 else None));step+=1
        torch.cuda.synchronize();seconds=time.monotonic()-tick;history.append(dict(epoch=epoch,seconds=seconds,steps=len(records),alignment_active=epoch>5,**{k:float(np.mean([r[k] for r in records])) for k in records[0]}));base.checkpoint(entry/'latest.pt',capture(m,ema,o,d,order,stream,history,epoch,step))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('IACS_EPOCH',entry.name,epoch,history[-1],flush=True)
        if epoch in (1,6):
            estimate=time.monotonic()-began+seconds*(20-epoch)+180;save(entry/'throughput.json',dict(epoch=epoch,epoch_seconds=seconds,elapsed=time.monotonic()-began,projected_with_readout_seconds=estimate,limit=1800))
            if estimate>1800:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False));return
    assert teacher_digest==base.tensor_digest(teacher.state_dict());base.checkpoint(entry/'semantic_teacher_logits.pt',cache);saved=torch.load(entry/'epoch020.pt',map_location='cpu');ema.load_state_dict(saved['ema']);ema.eval();ema.body_mode(False);ed=base.tensor_digest(ema.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);outputs={}
    with torch.no_grad():
        for role,rows in evaluation.items():
            outputs[role]={k:[] for k in READOUTS}
            for r in rows:
                x=raw_load(r).cuda();f,v,c,z,_=ema.forward_heads(x,r['video_metadata']);points=ema.point_generator([a[None] for a in f]);video=[dict(feats=x.T,segments=x.new_empty(0,2),labels=torch.empty(0,dtype=torch.long,device='cuda'),metadata=r['video_metadata'])]
                def decode(cc,rr):return ema.inference(video,points,[a[None] for a in v],[a[None] for a in cc],[a[None] for a in rr],None,None)[0]
                views=[]
                for scale in (.75,1.25):
                    sx,ratio=rescale_raw(x,scale);_,sv,sc,sr,_=ema.forward_heads(sx,r['video_metadata']);views.append((sv,sc,sr,ratio))
                fc=[];fr=[]
                for l in range(6):
                    a,b=fuse_heads(c[l].T,z[l].T,[(vc[l].T,vr[l].T,vm[l],ratio) for vm,vc,vr,ratio in views]);fc.append(a.T);fr.append(b.T)
                for name,p in (('native',decode(c,z)),('scale_score',decode(fc,z)),('scale_boundary',decode(fc,fr))):outputs[role][name].append(dict(video=r['video'],segments=p['segments'].cpu(),scores=p['scores'].cpu(),labels=p['labels'].cpu()))
    assert ed==base.tensor_digest(ema.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0));base.checkpoint(entry/'predictions.pt',outputs)
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel_complete.json'),checkpoint_sha256=sha(entry/'epoch020.pt'),predictions_sha256=sha(entry/'predictions.pt'),semantic_logits_sha256=sha(entry/'semantic_teacher_logits.pt'),initial_model_digest=initial,final_model_digest=base.tensor_digest(m.state_dict()),final_EMA_digest=ed,semantic_teacher_digest_unchanged=teacher_digest,target_train_seen=int((stream.seen>0).sum()),target_train_draws=int(stream.seen.sum()),source_videos=len(fit),all_predictions_from_saved_EMA20=True,readout_parameters_RNG_unchanged=True,target_train_GT_read=False,seconds=time.monotonic()-began,formal_seconds=time.monotonic()-formal))


def evaluate(root,s):
    import torch,pandas as pd
    from tools.audit_background_ap_cost import official,IndependentAP
    from tools.run_foreground_calibration_oracle import load_truth
    cfg=read(SOURCE/s/'config.json');data=torch.load(root/'inputs'/(s+'_unlabeled.pt'),map_location='cpu');outs={m:torch.load(root/(s+'_'+m)/'predictions.pt',map_location='cpu') for m in MODES};results={}
    def frame(rows):return pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c),float(v)) for r in rows for b,c,v in zip(r['segments'],r['labels'],r['scores'])],columns=['video-id','t-start','t-end','label','score'])
    for role,rows in data['evaluation'].items():
        gt=load_truth(cfg['dataset']['json_file'] if role=='validation' else cfg['baseline']['annotation_for_offline_evaluation'],cfg['src_val_split'] if role=='validation' else cfg['tgt_val_split']);methods={'original':frame(rows)}
        methods.update({m+'/'+readout:frame(v[role][readout]) for m,v in outs.items() for readout in READOUTS});scores={}
        for name,f in methods.items():
            aps=[]
            for c in range(10):
                a=gt[gt.label==c].reset_index(drop=True);b=f[f.label==c].reset_index(drop=True);ap=official(a,b) if len(a) and len(b) else np.zeros(5);ref=IndependentAP(a,b).ap(np.ones(len(b),bool));assert np.allclose(ap,ref,atol=1e-7,rtol=0);aps.append(ap)
            aps=np.asarray(aps);scores[name]=dict(map=float(aps.mean()),per_tiou_AP=aps.mean(0).tolist(),per_class_tiou_AP=aps.tolist(),predictions=len(f))
        for name,value in scores.items():value['delta_pp']=100*(value['map']-scores['original']['map'])
        if role=='target_val':assert abs(scores['original']['map']-read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'evaluation/target_val.json')['map_0.1_to_0.5'])<1e-7
        results[role]=scores
    save(root/(s+'_evaluation.json'),dict(passed=True,results=results,official_full_AP_independent_equal=True,target_val_used_for_selection=False))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity()));prepare(root);active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        for stage,tasks in [('train',TASKS),('evaluate',('S1','S2'))]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' budgetpause')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in range(4) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);e=root/task;e.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        v=snapshot(g);p=e/'lease_train.json';save(p,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=v['uuid'],PROPOSAL_FOREGROUND_LEASE=str(p),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(p))
                    log=(e/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()},pending=pending,completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,allGPU_sha256={t:sha(root/t/'gpu_complete.json') for t in TASKS},evaluation_sha256={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex),automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['train','evaluate']);a=p.parse_args()
    if a.worker=='train':train(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry.parent,a.entry.name)
    else:run(a.root)
