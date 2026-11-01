"""官方DiGIT完整数值路径，接入固定六方向source-only协议。"""
import sys,os,json,math,random,time,itertools,io,importlib.util
from pathlib import Path
from copy import deepcopy
from argparse import Namespace
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from tools.digit_protocol import CODE,VENDOR,SPEC,SCHEMA,read,save,sha,decision,terminal
sys.path.insert(0,str(VENDOR))
from models.digit.model import build as official_build
from util.misc import collate_fn
from util.get_param_dicts import get_param_dict
# Common benchmark crop is unchanged, including retained empty negative videos.
_sp=importlib.util.spec_from_file_location('digit_benchmark_crop',CODE/'vendor/dyfadet/libs/datasets/data_utils.py')
_crop=importlib.util.module_from_spec(_sp);sys.modules[_sp.name]=_crop;_sp.loader.exec_module(_crop)
truncate_feats,trivial_batch_collator,worker_init_reset_seed=(_crop.truncate_feats,_crop.trivial_batch_collator,_crop.worker_init_reset_seed)
_sp=importlib.util.spec_from_file_location('digit_benchmark_nms',CODE/'vendor/dyfadet/libs/utils/nms.py')
_nms=importlib.util.module_from_spec(_sp);_sp.loader.exec_module(_nms)

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

def digit_item(item):
    fps=float(item['fps']);stride=int(item['feat_stride']);frames=int(item['feat_num_frames'])
    info=dict(video_name=item['video_id'],video_duration=torch.tensor(item['duration'],dtype=torch.float32),
        feature_duration=torch.tensor(item['feats'].shape[1]*stride/fps),fps=torch.tensor(fps),
        base_frames=torch.tensor(frames),stride=torch.tensor(stride),offset=torch.tensor((frames-stride)*.5/fps))
    if 'segments' in item:
        # Benchmark grid center = (i*stride+frames/2)/fps. DiGIT internal center
        # = (i+.5)*stride/fps, and PostProcess adds the remaining offset back.
        info.update(segments=(item['segments']+.5)*stride/fps,labels=item['labels'])
    return item['feats'],info

class DiGITDataset(BenchmarkDataset):
    def __getitem__(self,index):return digit_item(super().__getitem__(index))

def digit_loader(ds,generator,training,batch,workers):
    return DataLoader(ds,batch_size=batch,num_workers=workers,shuffle=training,drop_last=training,
        generator=generator,collate_fn=collate_fn,persistent_workers=False,
        worker_init_fn=worker_init_reset_seed if training else None)

def device_batch(batch,device):
    x,info=batch
    return x.to(device),[{k:v.to(device) if torch.is_tensor(v) else v for k,v in row.items()} for row in info]

def build(cfg,device='cuda:0',workers=None):
    random.seed(123);np.random.seed(123);torch.manual_seed(123)
    if device.startswith('cuda'):torch.cuda.manual_seed(123)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    args=Namespace(**cfg['model']);m,criterion,post=official_build(args)
    m=m.to(device);criterion=criterion.to(device)
    ema=torch.optim.swa_utils.AveragedModel(m,multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999))
    opt=torch.optim.AdamW(get_param_dict(args,m),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=args.lr_drop)
    gs={k:torch.Generator().manual_seed(s) for k,s in [('source',123),('validation',2000129)]}
    w=cfg['loader']['num_workers'] if workers is None else workers
    dl=digit_loader(DiGITDataset(cfg,cfg['benchmark']['source_train'],True),gs['source'],True,2,w)
    vl=digit_loader(DiGITDataset(cfg,cfg['benchmark']['source_val']),gs['validation'],False,1,w)
    return dict(model=m,criterion=criterion,post=post,ema=ema,opt=opt,sched=sched,generators=gs,train=dl,val=vl,device=device)

def capture(ctx,cfg,epoch,history,metrics):
    return dict(schema=SCHEMA,config=cfg,epoch=epoch,history=deepcopy(history),metrics=deepcopy(metrics),
        model=ctx['model'].state_dict(),ema=ctx['ema'].module.state_dict(),ema_updates=ctx['ema'].n_averaged.clone(),
        criterion=ctx['criterion'].state_dict(),optimizer=ctx['opt'].state_dict(),scheduler=ctx['sched'].state_dict(),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
            cuda=[torch.cuda.get_rng_state(0)] if ctx['device'].startswith('cuda') else [],
            generators={k:g.get_state() for k,g in ctx['generators'].items()}))

def restore(ctx,s):
    ctx['model'].load_state_dict(s['model'],strict=True);ctx['ema'].module.load_state_dict(s['ema'],strict=True)
    ctx['ema'].n_averaged.copy_(s['ema_updates']);ctx['criterion'].load_state_dict(s['criterion'],strict=True)
    ctx['opt'].load_state_dict(s['optimizer']);ctx['sched'].load_state_dict(s['scheduler'])
    r=s['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
    if ctx['device'].startswith('cuda'):torch.cuda.set_rng_state(r['cuda'][0].cpu(),0)
    for k,g in ctx['generators'].items():g.set_state(r['generators'][k].cpu())

# enc_proj is the author's parameter-free resizer, not a trainable FPN.
GROUPS={'input_projection':'input_proj.', 'MDGE':'transformer.encoder.', 'memory_projection':'transformer.memory_proj.',
    'CAID_central':'.cross_attn.', 'CAID_adjacent':'.cross_attn_se.', 'classification':'class_embed.', 'regression':'segment_embed.'}

def group_parameters(model):
    return {key:{n:p for n,p in model.named_parameters() if fragment in n} for key,fragment in GROUPS.items()}

def step(ctx,batch,measure=False):
    m=ctx['model'];m.train();ctx['criterion'].train();ctx['opt'].zero_grad(set_to_none=True)
    x,info=device_batch(batch,ctx['device']);out=m(x,info);parts=ctx['criterion'](out,info);weights=ctx['criterion'].weight_dict
    loss=sum(parts[k]*weights[k] for k in weights)
    assert torch.isfinite(loss) and all(torch.isfinite(parts[k]).all() for k in weights)
    loss.backward();norms={}
    if measure:
        for name,params in group_parameters(m).items():
            gs=[p.grad.detach().norm() for p in params.values() if p.grad is not None]
            norms[name]=float(torch.linalg.vector_norm(torch.stack(gs))) if gs else 0.
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),.1);assert torch.isfinite(norm)
    ctx['opt'].step();ctx['ema'].update_parameters(m)
    return dict(final_loss=float(loss.detach()),cls_loss=float(parts['loss_ce'].detach()),reg_loss=float(parts['loss_iou'].detach())),norms

def validation(ctx):
    m=ctx['ema'].module;m.eval();ctx['criterion'].eval();values=[]
    with torch.no_grad():
        for batch in ctx['val']:
            x,info=device_batch(batch,ctx['device']);parts=ctx['criterion'](m(x,info),info)
            weights=ctx['criterion'].weight_dict
            values.append(float(parts['loss_ce']*weights['loss_ce']+parts['loss_iou']*weights['loss_iou']))
    assert values and all(math.isfinite(v) for v in values)
    return sum(values)/len(values)

def predict(ctx,batch,cfg):
    x,info=device_batch(batch,ctx['device']);out=ctx['model'](x,info)
    assert len(info)==1 and not out['mask'].any(), 'single video must have no padded queries'
    args=[torch.stack([v[k] for v in info]) for k in ('video_duration','feature_duration','stride','offset')]
    # Native DiGIT emits one class per query. Its seconds threshold is converted
    # from the frozen benchmark's feature-grid duration threshold.
    seconds=cfg['test_cfg']['duration_thresh']*float(info[0]['stride']/info[0]['fps'])
    raw=ctx['post'](out,*args,duration_thresh=seconds)[0]
    tc=cfg['test_cfg'];segs=raw['segments'].cpu();scores=raw['scores'].cpu();labels=raw['labels'].cpu()
    keep=scores>tc['pre_nms_thresh'];segs,scores,labels=segs[keep],scores[keep],labels[keep]
    order=scores.argsort(descending=True)[:tc['pre_nms_topk']];segs,scores,labels=segs[order],scores[order],labels[order]
    segs,scores,labels=_nms.batched_nms(segs,scores,labels,tc['iou_threshold'],tc['min_score'],tc['max_seg_num'],
        use_soft_nms=True,multiclass=tc['multiclass_nms'],sigma=tc['nms_sigma'],voting_thresh=tc['voting_thresh'])
    return dict(segments=segs,scores=scores,labels=labels),out

def cuda_operator_check():
    from models.digit.ops.temporal_deform_attn import MSDeformAttnFunction,deform_attn_core_pytorch
    reports=[]
    # Independent CPU reference checks both interpolation and its three gradients.
    for dtype,tol in [(torch.float64,1e-8),(torch.float32,2e-5)]:
        gen=torch.Generator().manual_seed(321)
        value=torch.randn(2,24,4,8,generator=gen,dtype=dtype)
        loc=torch.rand(2,13,4,2,4,1,generator=gen,dtype=dtype)*1.2-.1
        weight=torch.rand(2,13,4,2,4,generator=gen,dtype=dtype)
        lens=torch.tensor([16,8]);starts=torch.tensor([0,16])
        cpu=[v.clone().requires_grad_() for v in (value,loc,weight)]
        gpu=[v.cuda().requires_grad_() for v in (value,loc,weight)]
        ref=deform_attn_core_pytorch(cpu[0],lens,cpu[1],cpu[2])
        got=MSDeformAttnFunction.apply(gpu[0],lens.cuda(),starts.cuda(),gpu[1],gpu[2],64)
        upstream=torch.randn(ref.shape,generator=gen,dtype=dtype)
        a=torch.autograd.grad(ref,cpu,upstream);b=torch.autograd.grad(got,gpu,upstream.cuda())
        diffs=[]
        for expected,actual in zip([ref]+list(a),[got]+list(b)):
            actual=actual.cpu();torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
            diffs.append(float((actual-expected).abs().max()))
        reports.append(dict(dtype=str(dtype),tolerance=tol,max_absolute_errors=diffs))
    return reports

def probe(cfg,entry,guard):
    operator=cuda_operator_check()
    ctx=build(cfg);m=ctx['model'];initial=digest(m.state_dict());before={k:digest(p) for k,p in group_parameters(m).items()}
    batches=list(itertools.islice(ctx['train'],3))
    for batch in batches:loss,norms=step(ctx,batch,True)
    updated={k:before[k]!=digest(p) for k,p in group_parameters(m).items()}
    assert all(updated.values()) and all(math.isfinite(v) and v>0 for v in norms.values()),(updated,norms)
    buf=io.BytesIO();torch.save(capture(ctx,cfg,0,[],[]),buf);buf.seek(0);state=torch.load(buf,map_location='cpu',weights_only=False)
    saved_digest=digest(state);restore(ctx,deepcopy(state));assert digest(capture(ctx,cfg,0,[],[]))==saved_digest
    l1,_=step(ctx,batches[-1]);d1=digest(capture(ctx,cfg,0,[],[]))
    restore(ctx,deepcopy(state));l2,_=step(ctx,batches[-1]);d2=digest(capture(ctx,cfg,0,[],[]))
    assert l1==l2 and d1==d2,'resume next-step mismatch'
    m.eval();source=next(iter(ctx['val']));x,info=device_batch(source,ctx['device'])
    clean=[{k:v for k,v in row.items() if k not in ('segments','labels')} for row in info]
    with torch.no_grad():
        a=m(x,info);b=m(x,clean);assert digest(a)==digest(b),'forward depends on GT'
        pred,_=predict(ctx,source,cfg);assert all(torch.isfinite(v).all() for v in pred.values())
    value=validation(ctx);torch.cuda.synchronize()
    save(entry/'probe.json',dict(status='passed',initial_model_sha256=initial,gradient_norms=norms,parameters_updated=updated,
        source_loss=loss,source_validation_loss=value,full_state_restore=True,next_step_exact=True,official_model_and_postprocessor=True,
        cuda_operator_reference=operator,
        unlabeled_inference_equal=True,target_training_batches=0,parameter_count=sum(p.numel() for p in m.parameters()),
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,guard=guard,guard_sha256=sha(guard),
        actual_cuda_seed=torch.cuda.default_generators[0].initial_seed(),torch_version=torch.__version__,cuda_version=torch.version.cuda))

def train_and_infer(cfg,entry,guard):
    ctx=build(cfg);init=digest(ctx['model'].state_dict());probe_report=read(entry.parents[1]/'probes'/entry.name/'probe.json')
    assert init==probe_report['initial_model_sha256']
    save(entry/'initialization.json',dict(model_sha256=init,ema_sha256=digest(ctx['ema'].module.state_dict()),pretrained_checkpoint_loaded=False,
        parameter_counts={k:sum(p.numel() for p in params.values()) for k,params in group_parameters(ctx['model']).items()}))
    history=[];rows=[];best=float('inf');start=1
    if (entry/'latest.pt').exists():
        saved=torch.load(entry/'latest.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
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
            losses,g=step(ctx,next(iterator),n==0)
            if n==0:norms=g
            for k,v in losses.items():totals[k]=totals.get(k,0.)+v
        del iterator
        ctx['sched'].step();row=dict(epoch=epoch,steps=n+1,train_loss=totals['final_loss']/(n+1),cls_loss=totals['cls_loss']/(n+1),
            reg_loss=totals['reg_loss']/(n+1),gradient_norms=norms,lr=ctx['opt'].param_groups[0]['lr'],target_training_batches=0)
        improved=False
        if epoch%5==0:
            value=validation(ctx);history.append([epoch,value]);row['source_validation_loss']=value;improved=value<best
            if improved:best=value
        torch.cuda.synchronize();row['seconds']=time.monotonic()-started;rows.append(row);state=capture(ctx,cfg,epoch,history,rows)
        checkpoint(entry/'latest.pt',state)
        if improved:checkpoint(entry/'best.pt',state)
        save(entry/'metrics.json',rows);print('EPOCH',json.dumps(row),flush=True)
        if history and decision(history)['reason']!='running':break
    d=terminal(history);save(entry/'termination.json',dict(history=history,decision=d,initial_state_sha256=init))
    del state,ctx;torch.cuda.empty_cache()
    saved=torch.load(entry/'best.pt',map_location='cpu',weights_only=False)
    m,_,post=official_build(Namespace(**cfg['model']));m=m.to('cuda:0');m.load_state_dict(saved['ema'],strict=True);selected=saved['epoch'];del saved;m.eval()
    ctx=dict(model=m,post=post,device='cuda:0')
    for role,path in cfg['benchmark']['inference'].items():
        ds=DiGITDataset(cfg,path,unlabeled=True);cols={k:[] for k in ('video-id','t-start','t-end','label','score')};seen=[]
        with torch.no_grad():
            for index,v in enumerate(ds):
                assert 'segments' not in v[1] and 'labels' not in v[1];seen.append(v[1]['video_name']);pred,_=predict(ctx,collate_fn([v]),cfg)
                n=len(pred['scores']);cols['video-id'].extend([v[1]['video_name']]*n)
                for key,values in [('t-start',pred['segments'][:,0]),('t-end',pred['segments'][:,1]),('label',pred['labels']),('score',pred['scores'])]:cols[key].extend(values.tolist())
                if index%10==0:print('INFERENCE',role,index,len(ds),flush=True)
        np.savez_compressed(entry/(role+'.npz'),**{k:np.asarray(v,dtype=str if k=='video-id' else np.int64 if k=='label' else np.float32) for k,v in cols.items()})
        save(entry/(role+'_inference.json'),dict(videos=seen,count=len(seen),prediction_count=len(cols['score']),selected_epoch=selected,
            manifest_sha256=sha(path),prediction_sha256=sha(entry/(role+'.npz')),target_GT_used=False))
    save(entry/'gpu_complete.json',dict(status='passed',guard=guard,guard_sha256=sha(guard),selected_epoch=selected,config_sha256=sha(entry/'config.json'),target_training_batches=0))

def evaluate(entry):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=read(entry/'config.json');offline=read(entry/'offline_evaluation.json');assert sha(offline['annotation'])==offline['annotation_sha256']
    module_path=CODE/'libs/evaluation/evaluator_anet.py';sp=importlib.util.spec_from_file_location('benchmark_evaluator',module_path);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
    for role,splits in offline['splits'].items():
        with np.load(entry/(role+'.npz')) as f:columns={k:f[k] for k in f.files}
        ev=mod.ANETdetection(offline['annotation'],splits,num_workers=4,action_subsampling='10_majority_classes')
        result=ev.evaluate(columns,verbose=True);proof=read(entry/(role+'_inference.json'))
        assert proof['videos']==list(read(cfg['benchmark']['inference'][role])['database'])
        save(entry/'evaluation'/(role+'.json'),dict({'map_0.1_to_0.5':float(result[1])},map_by_tiou=[float(x) for x in result[0]],
            selected_epoch=proof['selected_epoch'],videos=proof['count'],prediction_sha256=sha(entry/(role+'.npz')),
            evaluator_sha256=sha(module_path),annotation_sha256=offline['annotation_sha256']))
    artifacts={str(p):sha(p) for p in entry.rglob('*') if p.is_file() and p.suffix in ('.json','.pt','.npz') and p.name!='complete.json'}
    save(entry/'complete.json',dict(schema=SCHEMA,shift=cfg['benchmark']['shift'],artifacts=artifacts,code=cfg['benchmark']['code']))

