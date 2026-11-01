"""保留官方DyFADet数值路径；仅接入现有数据、源选模和完整恢复工件。"""
import sys
from pathlib import Path
from copy import deepcopy
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import random
import time

from tools.dyfadet_protocol import CODE,VENDOR,SPEC,SCHEMA,read,save,sha,decision,terminal
sys.path.insert(0,str(VENDOR))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from libs.modeling import make_meta_arch
from libs.datasets.data_utils import truncate_feats,trivial_batch_collator,worker_init_reset_seed
from libs.utils import make_optimizer,make_scheduler,ModelEma,fix_random_seed

class BenchmarkDataset(Dataset):
    def __init__(self,cfg,path,training=False,unlabeled=False):
        self.cfg=cfg['dataset']; self.db=read(path)['database'];self.ids=list(self.db)
        self.training=training;self.unlabeled=unlabeled
        if unlabeled: assert all('annotations' not in v for v in self.db.values())
        else: assert all(all(int(a['label_id']) in range(10) for a in v['annotations']) for v in self.db.values())
    def __len__(self): return len(self.ids)
    def __getitem__(self,index):
        key=self.ids[index];v=self.db[key];c=self.cfg
        path=Path(c['feat_folder'])/((c.get('file_prefix') or '')+key+c['file_ext'])
        with np.load(path) as f: a=f['feats'].astype(np.float32)
        feats=torch.from_numpy(np.ascontiguousarray(a[::c['downsample_rate']].T))
        fps=c['default_fps'] if c['default_fps'] is not None else v['fps'];stride=c['feat_stride']*c['downsample_rate']
        d=dict(video_id=key,feats=feats,fps=fps,duration=v['duration'],feat_stride=stride,feat_num_frames=c['num_frames'])
        if not self.unlabeled:
            ants=v['annotations'];seg=np.asarray([a['segment'] for a in ants],dtype=np.float32).reshape(-1,2)
            d['segments']=torch.from_numpy((seg*fps-.5*c['num_frames'])/stride)
            d['labels']=torch.tensor([a['label_id'] for a in ants],dtype=torch.long)
            order=torch.argsort(d['segments'][:,0]);d['segments']=d['segments'][order];d['labels']=d['labels'][order]
            if self.training: d=truncate_feats(d,c['max_seq_len'],c['trunc_thresh'],c['crop_ratio'])
        return d

def loader(ds,generator,training,batch,workers):
    return DataLoader(ds,batch_size=batch,num_workers=workers,shuffle=training,drop_last=training,
        generator=generator,collate_fn=trivial_batch_collator,persistent_workers=False,
        worker_init_fn=worker_init_reset_seed if training else None)

def digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        elif torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def checkpoint(path,state):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);tmp.replace(path)

def dense(model,videos):
    x,m=model.preprocessing(videos);features,masks=model.backbone(x,m);fpn,masks=model.neck(features,masks)
    points=model.point_generator(fpn)
    cls=[x.permute(0,2,1) for x in model.cls_head(fpn,masks)]
    reg=[x.permute(0,2,1) for x in model.reg_head(fpn,masks)]
    return points,[x.squeeze(1) for x in masks],cls,reg

def validation(model,dl):
    model.eval();values=[];original=model.loss_normalizer
    with torch.no_grad():
        for videos in dl:
            points,masks,cls,reg=dense(model,videos)
            gt_cls,gt_reg=model.label_points(points,[v['segments'].to(model.device) for v in videos],[v['labels'].to(model.device) for v in videos])
            n=int(((torch.stack(gt_cls).sum(-1)>0)&torch.cat(masks,1)).sum())
            # 与旧benchmark一样：验证按本视频正点数归一化，不让训练EMA分母决定选模。
            model.loss_normalizer=float(max(n,1))
            loss=model.losses(masks,cls,reg,gt_cls,gt_reg)
            values.append(float(loss['cls_loss']+loss['reg_loss']))
    model.loss_normalizer=original
    assert values and all(math.isfinite(v) for v in values)
    return sum(values)/len(values)

def build(cfg,device='cuda:0',workers=None):
    fix_random_seed(123,include_cuda=device.startswith('cuda'))
    if device.startswith('cuda'): torch.cuda.manual_seed(123)
    model=make_meta_arch(cfg['model_name'],**deepcopy(cfg['model'])).to(device)
    ema=ModelEma(model,decay=.999)
    opt=make_optimizer(model,cfg['opt']);sched=make_scheduler(opt,cfg['opt'],cfg['benchmark']['steps_per_epoch'])
    gs={k:torch.Generator().manual_seed(s) for k,s in [('source',123),('validation',2000129)]}
    w=cfg['loader']['num_workers'] if workers is None else workers
    dl=loader(BenchmarkDataset(cfg,cfg['benchmark']['source_train'],True),gs['source'],True,2,w)
    vl=loader(BenchmarkDataset(cfg,cfg['benchmark']['source_val']),gs['validation'],False,1,w)
    return dict(model=model,ema=ema,opt=opt,sched=sched,generators=gs,train=dl,val=vl,device=device)

def capture(ctx,cfg,epoch,history,metrics):
    return dict(schema=SCHEMA,config=cfg,epoch=epoch,history=deepcopy(history),metrics=deepcopy(metrics),
        model=ctx['model'].state_dict(),ema=ctx['ema'].module.state_dict(),optimizer=ctx['opt'].state_dict(),scheduler=ctx['sched'].state_dict(),
        normalizers=[ctx['model'].loss_normalizer,ctx['ema'].module.loss_normalizer],
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                 cuda=[torch.cuda.get_rng_state(0)] if ctx['device'].startswith('cuda') else [],
                 generators={k:v.get_state() for k,v in ctx['generators'].items()}))

def restore(ctx,state):
    ctx['model'].load_state_dict(state['model'],strict=True);ctx['ema'].module.load_state_dict(state['ema'],strict=True)
    ctx['opt'].load_state_dict(state['optimizer']);ctx['sched'].load_state_dict(state['scheduler'])
    ctx['model'].loss_normalizer,ctx['ema'].module.loss_normalizer=state['normalizers']
    r=state['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
    if ctx['device'].startswith('cuda'):
        assert len(r['cuda'])==1 and r['cuda'][0].numel()>0
        torch.cuda.set_rng_state(r['cuda'][0].cpu(),0)
    for k,g in ctx['generators'].items():g.set_state(r['generators'][k].cpu())

def step(ctx,batch,measure=False):
    m=ctx['model'];m.train();ctx['opt'].zero_grad(set_to_none=True)
    loss=m(batch)
    assert all(torch.isfinite(v).all() for v in loss.values())
    loss['final_loss'].backward();norms={}
    if measure:
        for name in ('backbone','neck','cls_head','reg_head'):
            gs=[p.grad.detach().norm() for p in getattr(m,name).parameters() if p.grad is not None]
            norms[name]=float(torch.linalg.vector_norm(torch.stack(gs))) if gs else 0.
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    assert torch.isfinite(norm)
    ctx['opt'].step();ctx['sched'].step();ctx['ema'].update(m)
    return {k:float(v.detach()) for k,v in loss.items()},norms

def probe(cfg,entry,guard):
    ctx=build(cfg);m=ctx['model'];initial=digest(m.state_dict())
    before={k:digest(getattr(m,k).state_dict()) for k in ('backbone','neck','cls_head','reg_head')}
    batches=list(itertools.islice(ctx['train'],3));norms={}
    for batch in batches:loss,norms=step(ctx,batch,True)
    updated={k:before[k]!=digest(getattr(m,k).state_dict()) for k in before}
    assert all(updated.values()) and all(v>0 and math.isfinite(v) for v in norms.values()),(updated,norms)
    # PyTorch 1.13在UUID隔离下可能把device_count报告成0，先落CPU再严格恢复到当前逻辑卡。
    buf=io.BytesIO();torch.save(capture(ctx,cfg,0,[],[]),buf);buf.seek(0);state=torch.load(buf,map_location='cpu')
    saved_digest=digest(state);restore(ctx,deepcopy(state));assert digest(capture(ctx,cfg,0,[],[]))==saved_digest
    loss1,_=step(ctx,batches[-1]);after1=digest(capture(ctx,cfg,0,[],[]))
    restore(ctx,deepcopy(state));loss2,_=step(ctx,batches[-1]);after2=digest(capture(ctx,cfg,0,[],[]))
    assert loss1==loss2 and after1==after2,'resume next-step mismatch'
    m.eval();source=next(iter(ctx['val']))
    with torch.no_grad():
        p1=m(source);p2=m.inference(source,*dense(m,source))
    assert digest(p1)==digest(p2),'official forward mismatch'
    unlabeled=[{k:v for k,v in item.items() if k not in ('segments','labels')} for item in source]
    with torch.no_grad(): p3=m.inference(unlabeled,*dense(m,unlabeled))
    assert digest(p2)==digest(p3),'inference depends on GT'
    normalizers=(m.loss_normalizer,ctx['ema'].module.loss_normalizer)
    value=validation(ctx['ema'].module,ctx['val'])
    assert normalizers==(m.loss_normalizer,ctx['ema'].module.loss_normalizer)
    torch.cuda.synchronize()
    save(entry/'probe.json',dict(status='passed',initial_model_sha256=initial,gradient_norms=norms,parameters_updated=updated,
        source_loss=loss,source_validation_loss=value,full_state_restore=True,next_step_exact=True,
        official_inference_equal=True,unlabeled_inference_equal=True,target_training_batches=0,
        parameter_count=sum(p.numel() for p in m.parameters()),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        guard=guard,guard_sha256=sha(guard),actual_cuda_seed=torch.cuda.default_generators[0].initial_seed()))

def train_and_infer(cfg,entry,guard):
    ctx=build(cfg);init=digest(ctx['model'].state_dict());probe_report=read(entry.parents[1]/'probes'/entry.name/'probe.json')
    assert init==probe_report['initial_model_sha256'],'formal cold init differs from probe'
    save(entry/'initialization.json',dict(model_sha256=init,ema_sha256=digest(ctx['ema'].module.state_dict()),pretrained_checkpoint_loaded=False,
        parameter_counts={k:sum(p.numel() for p in getattr(ctx['model'],k).parameters()) for k in ('backbone','neck','cls_head','reg_head')}))
    history=[];rows=[];best=float('inf');start=1
    if (entry/'latest.pt').exists():
        saved=torch.load(entry/'latest.pt',map_location='cpu');assert saved['config']==cfg
        restore(ctx,saved);history=saved['history'];rows=saved['metrics'];start=saved['epoch']+1
        best=min([x[1] for x in history],default=float('inf'))
        assert digest(capture(ctx,cfg,saved['epoch'],history,rows))==digest(saved)
        save(entry/'resume_proof.json',dict(status='passed',checkpoint_sha256=sha(entry/'latest.pt'),start_epoch=start,guard=guard,guard_sha256=sha(guard)))
        del saved
    save(entry/'train_gpu.json',dict(guard=guard,guard_sha256=sha(guard),config_sha256=sha(entry/'config.json'),start_epoch=start))
    state=None
    for epoch in range(start,51):
        if history and decision(history)['reason']!='running':break
        started=time.monotonic();totals={};norms={};iterator=itertools.cycle(ctx['train'])
        for n in range(cfg['benchmark']['steps_per_epoch']):
            losses,g=step(ctx,next(iterator),measure=n==0)
            if n==0:norms=g
            for k,v in losses.items():totals[k]=totals.get(k,0.)+v
        del iterator
        row=dict(epoch=epoch,steps=n+1,train_loss=totals['final_loss']/(n+1),
            cls_loss=totals['cls_loss']/(n+1),reg_loss=totals['reg_loss']/(n+1),
            gradient_norms=norms,lr=ctx['opt'].param_groups[0]['lr'],target_training_batches=0)
        improved=False
        if epoch%5==0:
            value=validation(ctx['ema'].module,ctx['val']);history.append([epoch,value]);row['source_validation_loss']=value
            improved=value<best
            if improved:best=value
        torch.cuda.synchronize();row['seconds']=time.monotonic()-started;rows.append(row)
        state=capture(ctx,cfg,epoch,history,rows)
        checkpoint(entry/'latest.pt',state)
        if improved:checkpoint(entry/'best.pt',state)
        save(entry/'metrics.json',rows)
        print('EPOCH',json.dumps(row),flush=True)
        if history and decision(history)['reason']!='running':break
    d=terminal(history);save(entry/'termination.json',dict(history=history,decision=d,initial_state_sha256=init))
    # Free optimizer/student before long-video inference; weights remain in atomic checkpoints.
    del state,ctx
    torch.cuda.empty_cache()
    saved=torch.load(entry/'best.pt',map_location='cpu');m=make_meta_arch(cfg['model_name'],**deepcopy(cfg['model'])).to('cuda:0')
    m.load_state_dict(saved['ema'],strict=True);selected_epoch=saved['epoch'];del saved;m.eval()
    for role,path in cfg['benchmark']['inference'].items():
        ds=BenchmarkDataset(cfg,path,unlabeled=True);cols={k:[] for k in ('video-id','t-start','t-end','label','score')};seen=[]
        with torch.no_grad():
            for index,v in enumerate(ds):
                seen.append(v['video_id']);pred=m.inference([v],*dense(m,[v]))[0]
                n=len(pred['scores']);cols['video-id'].extend([v['video_id']]*n)
                for key,values in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:cols[key].extend(values.cpu().tolist())
                if index%10==0:print('INFERENCE',role,index,len(ds),flush=True)
        np.savez_compressed(entry/(role+'.npz'),**{k:np.asarray(v,dtype=str if k=='video-id' else np.int64 if k=='label' else np.float32) for k,v in cols.items()})
        save(entry/(role+'_inference.json'),dict(videos=seen,count=len(seen),prediction_count=len(cols['score']),selected_epoch=selected_epoch,
            manifest_sha256=sha(path),prediction_sha256=sha(entry/(role+'.npz')),target_GT_used=False))
    save(entry/'gpu_complete.json',dict(status='passed',guard=guard,guard_sha256=sha(guard),selected_epoch=selected_epoch,config_sha256=sha(entry/'config.json'),target_training_batches=0))

def evaluate(entry):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU evaluator requires hidden CUDA'
    cfg=read(entry/'config.json');offline=read(entry/'offline_evaluation.json')
    assert sha(offline['annotation'])==offline['annotation_sha256']
    module_path=CODE/'libs/evaluation/evaluator_anet.py'
    spec=importlib.util.spec_from_file_location('benchmark_evaluator',module_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for role,splits in offline['splits'].items():
        with np.load(entry/(role+'.npz')) as f: columns={k:f[k] for k in f.files}
        ev=mod.ANETdetection(offline['annotation'],splits,num_workers=4,action_subsampling='10_majority_classes')
        result=ev.evaluate(columns,verbose=True);proof=read(entry/(role+'_inference.json'))
        assert proof['videos']==list(read(cfg['benchmark']['inference'][role])['database'])
        save(entry/'evaluation'/(role+'.json'),dict({'map_0.1_to_0.5':float(result[1])},map_by_tiou=[float(x) for x in result[0]],
            selected_epoch=proof['selected_epoch'],videos=proof['count'],prediction_sha256=sha(entry/(role+'.npz')),
            evaluator_sha256=sha(module_path),annotation_sha256=offline['annotation_sha256']))
    artifacts={str(p):sha(p) for p in entry.rglob('*') if p.is_file() and p.suffix in ('.json','.pt','.npz') and p.name!='complete.json'}
    save(entry/'complete.json',dict(schema=SCHEMA,shift=cfg['benchmark']['shift'],artifacts=artifacts,code=cfg['benchmark']['code']))
