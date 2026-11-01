"""简单EMA自训练：源官方损失 + 目标正点监督，无域判别器或VLM。"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib,io,itertools,json,math,os,random,time,importlib.util
from tools import dyfadet_engine as base
from tools.dyfadet_uda_protocol import SPEC,SCHEMA,CODE,read,save,sha,decision,terminal
import numpy as np
import torch
from libs.modeling.losses import sigmoid_focal_loss,ctr_diou_loss_1d
from libs.utils.nms import batched_nms
from tools.dyfadet_split_policy import temporal_view,inverse_segments,stability_filter,assignment_metadata

MODULES=('backbone','neck','cls_head','reg_head')

class TargetDataset(base.BenchmarkDataset):
    def __init__(self,cfg,path):super().__init__(cfg,path,unlabeled=True)
    def __getitem__(self,index):
        d=super().__getitem__(index);t=d['feats'].shape[-1];cap=min(t,self.cfg['max_seq_len'])
        lo,hi=self.cfg['crop_ratio'];length=random.randint(max(1,int(cap*lo)),max(1,int(cap*hi)))
        start=random.randint(0,t-length)
        d['feats']=d['feats'][:,start:start+length].clone()
        d.update(crop_start_grid=start,crop_length=length)
        assert 'segments' not in d and 'labels' not in d
        return d

def rng_get(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
        cuda=[torch.cuda.get_rng_state(0)] if device.startswith('cuda') else [])

def rng_set(r,device):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
    if device.startswith('cuda'):torch.cuda.set_rng_state(r['cuda'][0].cpu(),0)

@contextmanager
def target_stream(ctx):
    source=rng_get(ctx['device']);rng_set(ctx['target_rng'],ctx['device'])
    try:yield
    finally:ctx['target_rng']=rng_get(ctx['device']);rng_set(source,ctx['device'])

def strong(videos,generator):
    result=[]
    for v in videos:
        d=dict(v);keep=(torch.rand((v['feats'].shape[0],1),generator=generator)>=SPEC['channel_dropout'])
        d['feats']=v['feats']*keep.to(v['feats'].device)/(1-SPEC['channel_dropout']);result.append(d)
    return result

@torch.no_grad()
def raw_pseudo(teacher,videos):
    teacher.eval();result=[]
    # The official evaluation preprocessor accepts one video; preserve it exactly.
    for v in videos:
        points,masks,cls,reg=base.dense(teacher,[v])
        p=teacher.inference_single_video(points,[x[0] for x in masks],[x[0] for x in cls],[x[0] for x in reg])
        seg=p['segments'].detach().cpu().clamp(0,v['feats'].shape[-1]);score=p['scores'].detach().cpu();label=p['labels'].detach().cpu()
        keep=(score>=SPEC['target_threshold'])&((seg[:,1]-seg[:,0])>teacher.test_duration_thresh)
        seg,score,label=batched_nms(seg[keep],score[keep],label[keep],teacher.test_iou_threshold,SPEC['target_threshold'],teacher.test_max_seg_num,
            use_soft_nms=teacher.test_nms_method=='soft',multiclass=teacher.test_multiclass_nms,sigma=teacher.test_nms_sigma,voting_thresh=teacher.test_voting_thresh)
        keep=score>=SPEC['target_threshold'];result.append(dict(segments=seg[keep],scores=score[keep],labels=label[keep]))
    return result

@torch.no_grad()
def pseudo(teacher,videos):
    original=raw_pseudo(teacher,videos)
    for v,p in zip(videos,original):
        views=[]
        for scale in SPEC['regression_temporal_scales']:
            changed,ratio=temporal_view(v,scale)
            prediction=raw_pseudo(teacher,[changed])[0]
            prediction['segments']=inverse_segments(prediction['segments'],ratio,v['feats'].shape[-1])
            views.append(prediction)
        p['regression_keep'],p['stability_iou'],p['stability_endpoint']=stability_filter(
            p,views,SPEC['regression_min_iou'],SPEC['regression_max_endpoint_fraction'])
    return original


def positive_loss(model,outputs,predictions):
    points,masks,cls,reg=outputs
    logits=torch.cat(cls,1);offsets=torch.cat(reg,1)
    valid=torch.cat(masks,1);targets=[];weights=[];reg_segments=[];reg_labels=[]
    for p in predictions:
        seg=p['segments'].to(model.device);lab=p['labels'].to(model.device)
        score=p['scores'].to(model.device)
        target,_,weight=assignment_metadata(model,points,seg,lab,score)
        targets.append(target);weights.append(weight)
        keep=p['regression_keep'].to(model.device)
        reg_segments.append(seg[keep]);reg_labels.append(lab[keep])
    gt=torch.stack(targets);weight=torch.stack(weights)
    pos=(gt.sum(-1)>0)&valid;n=int(pos.sum())
    classification=logits.sum()*0
    if n:
        labels=gt[pos]*(1-model.train_label_smoothing)+model.train_label_smoothing/(model.num_classes+1)
        per_point=sigmoid_focal_loss(logits[pos],labels,reduction='none').sum(-1)
        # Unweighted denominator preserves absolute confidence attenuation.
        classification=(per_point*weight[pos]).sum()/n
    rcls,rgt=model.label_points(points,reg_segments,reg_labels)
    rg=torch.stack(rcls);rpos=(rg.sum(-1)>0)&valid;nr=int(rpos.sum())
    regression=offsets.sum()*0
    if nr:
        vid=torch.where(rg[rpos])[0]
        regression=ctr_diou_loss_1d(offsets[rpos][vid],torch.stack(rgt)[rpos][vid],reduction='sum')/nr
    return classification,regression,n,nr,float(weight[pos].mean()) if n else 0.


def predict_unlabeled(model,videos):
    # Official forward filters `segments` before its eval branch. Reuse the
    # identical dense network and official decoder without requiring fake GT.
    assert all('segments' not in v and 'labels' not in v for v in videos)
    return model.inference(videos,*base.dense(model,videos))

def build(cfg,mode,device='cuda:0',workers=None):
    ctx=base.build(cfg,device,workers);ref=torch.load(Path(cfg['benchmark']['reference'])/'best.pt',map_location='cpu')
    ctx['model'].load_state_dict(ref['ema'],strict=True);ctx['ema'].module.load_state_dict(ref['ema'],strict=True)
    ctx['model'].loss_normalizer=ctx['ema'].module.loss_normalizer=ref['normalizers'][1]
    ctx['initial_sha']=base.digest(ref['ema']);del ref
    for p in ctx['ema'].module.parameters():p.requires_grad_(False)
    ctx['generators']['target']=torch.Generator().manual_seed(3123)
    w=cfg['loader']['num_workers'] if workers is None else workers
    ctx['target']=base.loader(TargetDataset(cfg,cfg['benchmark']['target_train']),ctx['generators']['target'],True,2,w)
    ctx['augmentation']=torch.Generator().manual_seed(4123)
    source=rng_get(device);random.seed(5123);np.random.seed(5123);torch.manual_seed(5123)
    if device.startswith('cuda'):torch.cuda.manual_seed(5123)
    ctx['target_rng']=rng_get(device);rng_set(source,device)
    ctx.update(mode=mode,step=0,ledger_digest='',cfg=cfg)
    return ctx

def capture(ctx,epoch,history,metrics):
    state=base.capture(ctx,ctx['cfg'],epoch,history,metrics)
    state.update(schema=SCHEMA,mode=ctx['mode'],step=ctx['step'],target_rng=deepcopy(ctx['target_rng']),
        augmentation_rng=ctx['augmentation'].get_state(),ledger_digest=ctx['ledger_digest'])
    return state

def restore(ctx,state):
    assert state['config']==ctx['cfg'] and state['mode']==ctx['mode']
    assert state['step']==state['epoch']*ctx['cfg']['benchmark']['steps_per_epoch'] or state.get('probe')
    base.restore(ctx,state);ctx['target_rng']=deepcopy(state['target_rng']);ctx['augmentation'].set_state(state['augmentation_rng'].cpu())
    ctx['step']=state['step'];ctx['ledger_digest']=state['ledger_digest']

def update(ctx,source,target=None,measure=False):
    if target is None:
        # Control and disabled UDA execute the official function, not a reimplementation.
        result,_=base.step(ctx,source);ctx['step']+=1
        result.update(target_cls=0.,target_reg=0.,target_positive_points=0,target_proposals=0,target_classes=[0]*10,ramp=0.,target_regression_points=0,target_regression_proposals=0,target_cls_weight=0.)
        return result,{}
    m=ctx['model'];m.train();ctx['opt'].zero_grad(set_to_none=True)
    source_loss=m(source);assert all(torch.isfinite(x).all() for x in source_loss.values())
    source_loss['final_loss'].backward()
    result={k:float(v.detach()) for k,v in source_loss.items()};norms={}
    result.update(target_cls=0.,target_reg=0.,target_positive_points=0,target_proposals=0,target_classes=[0]*10,ramp=0.,target_regression_points=0,target_regression_proposals=0,target_cls_weight=0.)
    if target is not None:
        assert ctx['mode']=='uda' and all('segments' not in v and 'labels' not in v for v in target)
        with target_stream(ctx):
            labels=pseudo(ctx['ema'].module,target);normalizer=m.loss_normalizer
            c,r,n,nr,confidence=positive_loss(m,base.dense(m,strong(target,ctx['augmentation'])),labels)
            assert m.loss_normalizer==normalizer
            loss=c+SPEC['target_regression_weight']*r;assert torch.isfinite(loss)
            if measure and nr:
                for name in MODULES:
                    params=[p for p in getattr(m,name).parameters() if p.requires_grad]
                    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                    norms[name]=float(torch.stack([g.detach().square().sum() for g in gs if g is not None]).sum().sqrt())
            ramp=min((ctx['step']+1)/(SPEC['target_ramp_epochs']*ctx['cfg']['benchmark']['steps_per_epoch']),1.)
            (ramp*loss).backward()
            classes=torch.bincount(torch.cat([x['labels'] for x in labels]),minlength=10).tolist()
            result.update(target_cls=float(c.detach()),target_reg=float(r.detach()),target_positive_points=n,
                target_proposals=sum(len(x['labels']) for x in labels),target_classes=classes,ramp=ramp)
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
    ctx['opt'].step();ctx['sched'].step();ctx['ema'].update(m);ctx['step']+=1
    assert all(p.grad is None for p in ctx['ema'].module.parameters())
    return result,norms

def record(ctx,path,source,target,result):
    x=dict(step=ctx['step'],previous=ctx['ledger_digest'],source_videos=[v['video_id'] for v in source],
        target_videos=[v['video_id'] for v in target] if target is not None else [],
        target_crops=[dict(video_id=v['video_id'],start=v['crop_start_grid'],length=v['crop_length']) for v in target] if target is not None else [],
        target_positive_points=result['target_positive_points'],target_proposals=result['target_proposals'],target_classes=result['target_classes'],
        target_regression_points=result['target_regression_points'],target_regression_proposals=result['target_regression_proposals'],target_cls_weight=result['target_cls_weight'])
    h=hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest();x['sha256']=h;ctx['ledger_digest']=h
    with Path(path).open('a') as f:f.write(json.dumps(x)+'\n')

def frozen_state(ctx,epoch=0,history=None,metrics=None):
    buf=io.BytesIO();torch.save(capture(ctx,epoch,history or [],metrics or []),buf);buf.seek(0)
    return torch.load(buf,map_location='cpu')

def gradients(ctx):
    return {k:p.grad.detach().cpu().clone() for k,p in ctx['model'].named_parameters() if p.grad is not None}

def numerical_step_check(a,b,ga,gb,learning_rate):
    # Official GPU repeatability was measured before defining this gate. Gradients
    # are checked tightly; tiny sign changes in Adam's first moment can cause up
    # to two learning-rate units of parameter difference near a zero gradient.
    assert set(ga)==set(gb)
    # Per-element relative errors are ill-conditioned for nearly zero gradients.
    # Compare the entire clipped gradient vector instead; keep a fixed 1e-4 cap.
    norm_a=math.sqrt(sum(float(x.double().square().sum()) for x in ga.values()))
    norm_b=math.sqrt(sum(float(x.double().square().sum()) for x in gb.values()))
    difference=math.sqrt(sum(float((ga[k].double()-gb[k].double()).square().sum()) for k in ga))
    relative=difference/max(norm_a,norm_b,1e-12)
    assert relative<=1e-4,('relative_gradient_error',relative)
    bound=2*learning_rate+1e-7
    for name,atol in [('model',bound),('ema',.001*bound+1e-7)]:
        assert set(a[name])==set(b[name])
        for k in a[name]:assert torch.allclose(a[name][k],b[name][k],atol=atol,rtol=1e-6),(name,k)
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    assert set(a['optimizer']['state'])==set(b['optimizer']['state'])
    for index,values in a['optimizer']['state'].items():
        other=b['optimizer']['state'][index];assert values.keys()==other.keys()
        for k,v in values.items():
            if k=='step':assert torch.equal(v,other[k])
            else:assert torch.allclose(v,other[k],atol=1e-6 if k=='exp_avg' else 1e-7,rtol=1e-5),('optimizer',index,k)
    for k in set(a)-{'model','ema','optimizer'}:assert base.digest(a[k])==base.digest(b[k]),k
    return dict(exact=base.digest(a)==base.digest(b),max_parameter_delta=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model']),
        max_gradient_delta=max(float((ga[k]-gb[k]).abs().max()) for k in ga),gradient_relative_l2=relative,gradient_relative_l2_limit=1e-4,parameter_atol=bound)

def probe(cfg,entry,guard):
    ctx=build(cfg,'uda');initial=ctx['initial_sha'];source=next(iter(ctx['train']))
    target_batches=list(itertools.islice(ctx['target'],8));assert target_batches
    before={k:base.digest(getattr(ctx['model'],k).state_dict()) for k in MODULES}
    # At identical state the disabled adaptation path must equal official source training.
    state=frozen_state(ctx);state['probe']=True;lr=ctx['opt'].param_groups[0]['lr']
    a,_=base.step(ctx,source);ctx['step']+=1;official=frozen_state(ctx);ga=gradients(ctx)
    restore(ctx,deepcopy(state));b,_=update(ctx,source)
    assert all(a[k]==b[k] for k in a)
    disabled=numerical_step_check(official,frozen_state(ctx),ga,gradients(ctx),lr)
    restore(ctx,deepcopy(state));del state,official,ga
    all_norms={};loss=None;chosen=None
    for batch in target_batches:
        loss,norms=update(ctx,source,batch,True)
        if norms:all_norms=norms;chosen=batch;break
    assert chosen is not None and all(v>0 and math.isfinite(v) for v in all_norms.values()),('no target support',loss,all_norms)
    # Actual Teacher predictions verify that filtering gates regression separately.
    labels=pseudo(ctx['ema'].module,chosen)
    out=base.dense(ctx['model'],chosen)
    c,r,n,nr,confidence=positive_loss(ctx['model'],out,labels)
    assert n>0 and nr>0
    cls_params=[p for p in ctx['model'].cls_head.parameters() if p.requires_grad];reg_params=[p for p in ctx['model'].reg_head.parameters() if p.requires_grad]
    def norm(loss,params):
        gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))
    routing=dict(classification_to_cls=norm(c,cls_params),classification_to_reg=norm(c,reg_params),
        regression_to_cls=norm(r,cls_params),regression_to_reg=norm(r,reg_params))
    assert routing['classification_to_cls']>0 and routing['regression_to_reg']>0
    assert routing['classification_to_reg']==routing['regression_to_cls']==0
    split_support=dict(classification_proposals=sum(len(p['labels']) for p in labels),
        regression_proposals=sum(int(p['regression_keep'].sum()) for p in labels),
        classification_points=n,regression_points=nr,classification_weight=confidence)
    del out,c,r,labels
    changed={k:before[k]!=base.digest(getattr(ctx['model'],k).state_dict()) for k in MODULES};assert all(changed.values())
    state=frozen_state(ctx);state['probe']=True
    restore(ctx,deepcopy(state));again=capture(ctx,0,[],[]);again['probe']=True
    assert base.digest(again)==base.digest(state);del again
    lr=ctx['opt'].param_groups[0]['lr'];l1,_=update(ctx,source,chosen);after1=frozen_state(ctx);g1=gradients(ctx)
    restore(ctx,deepcopy(state));l2,_=update(ctx,source,chosen);after2=frozen_state(ctx)
    assert all(np.allclose(l1[k],l2[k],atol=1e-6,rtol=1e-5) for k in l1)
    resumed=numerical_step_check(after1,after2,g1,gradients(ctx),lr)
    del state,after1,after2,g1
    m=ctx['ema'].module;v=next(iter(ctx['val']));m.eval()
    with torch.no_grad():
        p1=m(v);p2=m.inference(v,*base.dense(m,v))
        un=[{k:x for k,x in y.items() if k not in ('segments','labels')} for y in v]
        p3=predict_unlabeled(m,un)
    assert base.digest(p1)==base.digest(p2)==base.digest(p3)
    torch.cuda.synchronize()
    save(entry/'probe.json',dict(status='passed',initial_ema_sha256=initial,parameters_updated=changed,
        target_gradient_norms=all_norms,target_gradient_positive=True,source_disabled_path_equal=True,source_disabled_calls_official=True,
        disabled_GPU_repeat=disabled,full_state_restore=True,next_step_exact=resumed['exact'],next_step_numerical=True,next_step_GPU_repeat=resumed,
        teacher_no_grad=True,target_GT_used=False,split_loss_routing=routing,split_support=split_support,
        original_inference_equal=True,loss=loss,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        guard=guard,guard_sha256=sha(guard)))

def train_mode(cfg,entry,mode,guard,matched_stop=None):
    d=entry/mode;d.mkdir(exist_ok=True)
    if (d/'termination.json').exists():return read(d/'termination.json')['decision']['epoch']
    ctx=build(cfg,mode);history=[];rows=[];start=1
    init=dict(student_sha256=base.digest(ctx['model'].state_dict()),teacher_sha256=base.digest(ctx['ema'].module.state_dict()),
        reference_ema_sha256=ctx['initial_sha'],reference_best_sha256=cfg['benchmark']['reference_best_sha256'])
    assert len({init[k] for k in ('student_sha256','teacher_sha256','reference_ema_sha256')})==1
    save(d/'initialization.json',init)
    if (d/'latest.pt').exists():
        saved=torch.load(d/'latest.pt',map_location='cpu');restore(ctx,saved)
        history=saved['history'];rows=saved['metrics'];start=saved['epoch']+1
        assert base.digest(capture(ctx,saved['epoch'],history,rows))==base.digest(saved)
        # Discard only the uncommitted partial-epoch ledger suffix; the saved prefix is immutable.
        lines=(d/'events.jsonl').read_text().splitlines() if (d/'events.jsonl').exists() else []
        assert len(lines)>=ctx['step'];prefix=lines[:ctx['step']]
        assert (json.loads(prefix[-1])['sha256'] if prefix else '')==ctx['ledger_digest']
        if len(lines)>len(prefix):
            save(d/('partial_suffix_%d.json'%time.time_ns()),dict(lines=lines[len(prefix):]))
            (d/'events.jsonl').write_text(''.join(x+'\n' for x in prefix))
        save(d/('resume_%d.json'%time.time_ns()),dict(checkpoint_sha256=sha(d/'latest.pt'),start_epoch=start,guard=guard,guard_sha256=sha(guard)))
        del saved
    else:
        value=base.validation(ctx['ema'].module,ctx['val']);history=[[0,value]]
        original=torch.load(Path(cfg['benchmark']['reference'])/'best.pt',map_location='cpu')
        old=min(original['history'],key=lambda x:x[1])[1];del original
        assert abs(value-old)<1e-7,('source epoch0 mismatch',value,old)
        state=capture(ctx,0,history,rows);base.checkpoint(d/'best.pt',state);base.checkpoint(d/'latest.pt',state);del state
        (d/'events.jsonl').touch();save(d/'metrics.json',rows)
    best=min(x[1] for x in history)
    for epoch in range(start,31):
        if mode=='uda' and decision(history)['reason']!='running':break
        if matched_stop is not None and epoch>matched_stop:break
        started=time.monotonic();totals={};norms={};classes=[0]*10;zeros=0
        source_iter=itertools.cycle(ctx['train']);target_iter=itertools.cycle(ctx['target']) if mode=='uda' else None
        for n in range(cfg['benchmark']['steps_per_epoch']):
            src=next(source_iter);tgt=next(target_iter) if target_iter is not None else None
            losses,g=update(ctx,src,tgt,measure=not norms)
            if g:norms=g
            zeros+=int(tgt is not None and losses['target_positive_points']==0)
            classes=[a+b for a,b in zip(classes,losses['target_classes'])]
            for k,v in losses.items():
                if k!='target_classes':totals[k]=totals.get(k,0.)+v
            record(ctx,d/'events.jsonl',src,tgt,losses)
        del source_iter,target_iter
        row={k:v/(n+1) for k,v in totals.items()};row.update(epoch=epoch,steps=n+1,target_gradient_norms=norms,target_classes=classes,
            target_zero_positive_batches=zeros,target_training_batches=n+1 if mode=='uda' else 0,lr=ctx['opt'].param_groups[0]['lr'])
        improved=False
        if epoch%5==0:
            value=base.validation(ctx['ema'].module,ctx['val']);history.append([epoch,value]);row['source_validation_loss']=value
            improved=value<best
            if improved:best=value
        torch.cuda.synchronize();row['seconds']=time.monotonic()-started;rows.append(row)
        state=capture(ctx,epoch,history,rows);base.checkpoint(d/'latest.pt',state)
        if improved:base.checkpoint(d/'best.pt',state)
        if epoch%5==0:base.checkpoint(d/('epoch_%03d.pt'%epoch),state)
        del state;save(d/'metrics.json',rows);print('EPOCH',mode,json.dumps(row),flush=True)
    selected=terminal(history) if mode=='uda' else dict(decision(history),reason='matched_UDA_source_update_budget')
    if matched_stop is not None:assert selected['epoch']==matched_stop
    save(d/'termination.json',dict(decision=selected,history=history,ledger_digest=ctx['ledger_digest'],guard=guard,guard_sha256=sha(guard)))
    del ctx;torch.cuda.empty_cache();return selected['epoch']

def infer(cfg,entry,mode):
    d=entry/mode;saved=torch.load(d/'best.pt',map_location='cpu')
    m=base.make_meta_arch(cfg['model_name'],**deepcopy(cfg['model'])).to('cuda:0');m.load_state_dict(saved['ema'],strict=True)
    epoch=saved['epoch'];del saved;m.eval()
    for role,path in cfg['benchmark']['inference'].items():
        proof=d/(role+'_inference.json')
        if proof.exists() and sha(d/(role+'.npz'))==read(proof)['prediction_sha256']:continue
        ds=base.BenchmarkDataset(cfg,path,unlabeled=True);cols={k:[] for k in ('video-id','t-start','t-end','label','score')};seen=[]
        with torch.no_grad():
            for i,v in enumerate(ds):
                seen.append(v['video_id']);pred=predict_unlabeled(m,[v])[0];num=len(pred['scores']);cols['video-id'].extend([v['video_id']]*num)
                for k,x in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:cols[k].extend(x.cpu().tolist())
                if i%10==0:print('INFERENCE',mode,role,i,len(ds),flush=True)
        np.savez_compressed(d/(role+'.npz'),**{k:np.asarray(v,dtype=str if k=='video-id' else np.int64 if k=='label' else np.float32) for k,v in cols.items()})
        save(proof,dict(videos=seen,count=len(seen),selected_epoch=epoch,manifest_sha256=sha(path),prediction_sha256=sha(d/(role+'.npz')),target_GT_used=False))
    del m;torch.cuda.empty_cache()

def train_and_infer(cfg,entry,guard):
    stop=train_mode(cfg,entry,'uda',guard)
    train_mode(cfg,entry,'source_control',guard,stop)
    for mode in ('uda','source_control'):infer(cfg,entry,mode)
    save(entry/'gpu_complete.json',dict(status='passed',guard=guard,guard_sha256=sha(guard),config_sha256=sha(entry/'config.json')))

def evaluate(entry):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=read(entry/'config.json');offline=read(entry/'offline_evaluation.json');assert sha(offline['annotation'])==offline['annotation_sha256']
    path=CODE/'libs/evaluation/evaluator_anet.py';spec=importlib.util.spec_from_file_location('uda_evaluator',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for mode in ('uda','source_control'):
        d=entry/mode
        for role,splits in offline['splits'].items():
            with np.load(d/(role+'.npz')) as f:columns={k:f[k] for k in f.files}
            ev=mod.ANETdetection(offline['annotation'],splits,num_workers=4,action_subsampling='10_majority_classes');result=ev.evaluate(columns,verbose=True)
            proof=read(d/(role+'_inference.json'));assert proof['videos']==list(read(cfg['benchmark']['inference'][role])['database'])
            save(d/'evaluation'/(role+'.json'),dict({'map_0.1_to_0.5':float(result[1])},map_by_tiou=[float(x) for x in result[0]],
                selected_epoch=proof['selected_epoch'],videos=proof['count'],prediction_sha256=sha(d/(role+'.npz')),evaluator_sha256=sha(path),annotation_sha256=offline['annotation_sha256']))
    artifacts={str(p):sha(p) for p in entry.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.npz') and p.name!='complete.json'}
    save(entry/'complete.json',dict(schema=SCHEMA,shift=cfg['benchmark']['shift'],artifacts=artifacts,code=cfg['benchmark']['code']))
