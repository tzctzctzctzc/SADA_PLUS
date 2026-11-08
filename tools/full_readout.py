"""冻结的动作内部参照读出，与原 SADA 六向共用数学实现。"""
import numpy as np
import torch
import pandas as pd
from torch.nn import functional as F
from tools.full_tf_alignment import path_features
from tools import baseline_engine as base
from libs.evaluation import ANETdetection
from libs.evaluation.evaluator_anet import load_gt_seg_from_json
SEM_DIM=23
STEP=.10


def apply_readout(segments, scores, labels, video_limit, shift_config):
    boxes = np.asarray(segments)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError('候选坐标、分数、类别数目不一致')
    refined = boxes.astype(float, copy=True)
    ranked = scores.astype(float, copy=True)
    lengths = refined[:, 1] - refined[:, 0]
    positive = lengths[lengths > 0]
    if not positive.size:
        return refined, ranked
    relative = np.clip(lengths / np.median(positive), .25, 4.)
    for class_id in np.unique(labels):
        selected = labels == class_id
        params = shift_config['classes'][str(int(class_id))]
        power, width = params['power'], params['width']
        if power:
            ranked[selected] *= relative[selected] ** power
        if width != 1.:
            center = refined[selected].mean(axis=1)
            half = lengths[selected] * width / 2
            refined[selected] = np.column_stack([center - half, center + half]).clip(0, video_limit)
    return refined, ranked


def span_limit(meta):
    return meta['duration'] if meta['use_temporal_grid_feats'] else meta['video_length']


def default_prediction(pred, meta, params):
    b, s = apply_readout(pred['segments'].cpu().numpy(), pred['scores'].cpu().numpy(),
                         pred['labels'].cpu().numpy(), span_limit(meta), params)
    return b, s, pred['labels'].cpu().numpy()


def edge_features(model, meta, features, boxes, scores, labels):
    maps, masks, logits = features
    device = maps[0].device
    box = torch.as_tensor(boxes,dtype=torch.float32,device=device)
    cls = torch.as_tensor(labels,dtype=torch.long,device=device)
    sc = torch.as_tensor(scores,dtype=torch.float32,device=device)
    if meta['use_temporal_grid_feats']:
        box = (box * meta['fps'] - .5 * meta['feat_num_frames']) / meta['feat_stride']
    length = (box[:,1]-box[:,0]).clamp_min(0)
    levels = torch.full_like(cls,len(maps)-1)
    for lev,(lo,hi) in reversed(list(enumerate(model.rq3_capacity.regression_ranges))):
        half = length.clamp_min(1)/2
        levels[(lo<=half)&(half<=hi)] = lev
    result = np.zeros((len(box),2,SEM_DIM+12),dtype=np.float32)
    for lev,(feat,mask,raw) in enumerate(zip(maps,masks,logits)):
        ids = torch.where(levels==lev)[0]
        if not len(ids):continue
        valid = int(mask[0].sum())
        if valid<1:continue
        proj = model.adapt_route.projection @ feat[0,:,:valid]
        sem = raw[0,:,:valid].sigmoid()
        cls_ids = cls[ids]
        for side,sign in enumerate((-1,1)):
            # 第一半在框内，由远及近；第二半在框外，由近及远。
            u=(torch.arange(16,device=device)+.5)/8-1
            pos=(box[ids,side,None]+sign*length[ids,None]*.2*u)/model.fpn_strides[lev]
            pos=pos.clamp(0,valid-1)
            left=pos.long(); right=(left+1).clamp_max(valid-1); w=pos-left
            q=sem[cls_ids[:,None],left]*(1-w)+sem[cls_ids[:,None],right]*w
            z=(proj[:,left]*(1-w)+proj[:,right]*w).permute(1,0,2)
            qi,qo=q[:,:8],q[:,8:]
            zi,zo=z[:,:,:8],z[:,:,8:]
            # 两侧分别计算均值、轨迹与频带能量；保留内外差异。
            window=torch.hann_window(8,periodic=False,device=device)
            def band(signal,prob):
                weighted=signal*prob[:,None,:]*window
                return torch.fft.rfft(weighted,dim=-1,norm='ortho').abs().square().mean(1).log1p()
            pi,po=band(zi,qi),band(zo,qo)
            mi=zi.mean(2); mo=zo.mean(2)
            tf=torch.cat((po-pi,(po-pi).abs(),F.cosine_similarity(mi,mo)[:,None],
                          (zi.flip(2)-zo).square().mean((1,2)).sqrt()[:,None]),1)
            semantic=torch.cat((F.one_hot(cls_ids,10),sc[ids,None],length[ids,None].log1p(),
                levels[ids,None],torch.full_like(sc[ids,None],sign),q.reshape(-1,8,2).mean(2),
                (qo.mean(1)-qi.mean(1))[:,None]),1)
            result[ids.cpu().numpy(),side]=torch.cat((semantic,tf),1).cpu().numpy()
    assert np.isfinite(result).all()
    return result


def overlap(boxes, truth):
    inter=np.maximum(0,np.minimum(boxes[...,1],truth[...,1])-np.maximum(boxes[...,0],truth[...,0]))
    return inter/np.maximum(1e-9,np.diff(boxes,axis=-1)[...,0]+np.diff(truth,axis=-1)[...,0]-inter)


def edge_labels(boxes,labels,gt,limit):
    targets=np.zeros((len(boxes),2,3))
    for c in np.unique(labels):
        ids=np.flatnonzero(labels==c)
        truth=gt[gt.label==c][['t-start','t-end']].to_numpy()
        if not len(truth):continue
        base=boxes[ids]
        matrix=overlap(base[:,None,:],truth[None,:,:])
        match=truth[matrix.argmax(1)]
        initial=matrix.max(1)
        delta=np.diff(base,axis=1)[:,0]*STEP
        for side,sign in enumerate((-1,1)):
            for k,direction in enumerate((-1,1)):
                trial=base.copy()
                trial[:,side]=(base[:,side]+sign*direction*delta).clip(0,limit)
                targets[ids,side,k]=np.where(initial>0,overlap(trial,match)-initial,0)
            outer=(base[:,side]+sign*delta).clip(0,limit)
            lo=np.minimum(outer,base[:,side]);hi=np.maximum(outer,base[:,side])
            inside=np.maximum(0,np.minimum(hi,match[:,1])-np.maximum(lo,match[:,0]))
            targets[ids,side,2]=np.where(initial>0,inside/np.maximum(hi-lo,1e-9),0)
    return targets


def correct(boxes,x,head,limit):
    n=len(head['mean'])
    y=((x[:,:,:n]-head['mean'])/head['scale'])@head['coef']+head['bias']
    # 仅在外侧主要属于当前实例时允许向外延伸。
    gain=y[:,:,:2].copy()
    gain[:,:,1]=np.where(y[:,:,2]>.5,gain[:,:,1],-np.inf)
    action=np.where(gain.max(2)>0,np.where(gain.argmax(2)==0,-1,1),0)
    return (boxes+action*np.array([-1,1])*STEP*np.diff(boxes,axis=1)).clip(0,limit)


class RidgeStats:
    """流式拟合标准化岭回归；不保存视频或大型 FPN 缓存。"""
    def __init__(self,n):
        self.n=n
        self.num=0
        self.sum=np.zeros(n)
        self.xy=np.zeros((n,3))
        self.xx=np.zeros((n,n))
        self.ys=np.zeros(3)

    def add(self,x,y):
        x=x.reshape(-1,x.shape[-1])[:,:self.n].astype(float)
        y=y.reshape(-1,3)
        self.num+=len(y);self.sum+=x.sum(0);self.ys+=y.sum(0)
        self.xx+=x.T@x;self.xy+=x.T@y

    def fit(self):
        assert self.num>0
        mean=self.sum/self.num; bias=self.ys/self.num
        cov=self.xx/self.num-np.outer(mean,mean)
        scale=np.sqrt(np.maximum(cov.diagonal(),1e-6))
        coef=np.linalg.solve(cov/np.outer(scale,scale)+.1*np.eye(self.n),
                             (self.xy/self.num-np.outer(mean,bias))/scale[:,None])
        return dict(mean=mean,scale=scale,coef=coef,bias=bias)


def load_data(cfg):
    src=base.dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False,unlabeled=True)
    tgt=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False,unlabeled=True)
    ids=lambda ds:{v['id'] for v in ds.data_list}
    assert ids(src).isdisjoint(ids(tgt)) and len(src) and len(tgt)
    g=load_gt_seg_from_json(cfg['dataset']['json_file'],cfg['src_train_split'],action_subsampling='10_majority_classes')
    g=g[g['t-start']<g['t-end']]
    assert set(g['video-id'])<=ids(src)
    return src,tgt,g


def score(cfg,cols):
    evaluator=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],
                             num_workers=1,action_subsampling='10_majority_classes')
    by,mean,*_=evaluator.evaluate({k:np.asarray(v) for k,v in cols.items()},verbose=False)
    return dict(mAP=100*float(mean),by_tiou=(100*np.asarray(by)).tolist())


def features(model, meta, maps, b, scores, classes):
    old = edge_features(model, meta, maps, b, scores, classes)
    boxes = torch.as_tensor(b, dtype=torch.float32)
    if meta['use_temporal_grid_feats']:
        boxes = (boxes * meta['fps'] - .5 * meta['feat_num_frames']) / meta['feat_stride']
    width = (boxes[:, 1] - boxes[:, 0]).clamp_min(0)
    levels = old[:, 0, 12].astype(int)
    extra = torch.zeros(len(b), 2, 2)
    phase = (torch.arange(32)+.5)/32
    # 中部及两端的内外条带都取同样的相对宽度，按正向时间采样。
    starts = torch.tensor([.5-STEP/2, -STEP, 0., 1., 1.-STEP])
    fractions = starts[:, None] + STEP*phase
    for lvl in np.unique(levels):
        ids = np.flatnonzero(levels == lvl)
        valid = int(maps[1][lvl][0].sum())
        if valid < 2:
            continue
        z = model.adapt_route.projection @ maps[0][lvl][0, :, :valid]
        q = maps[2][lvl][0, :, :valid].sigmoid()
        eye = torch.eye(len(z))
        for offset in range(0, len(ids), 128):
            ix = ids[offset:offset+128]
            pos = (boxes[ix, 0, None, None] + width[ix,None,None]*fractions) / model.fpn_strides[lvl]
            # 忽略完全位于视频外的条带，其余条带在可用网格内取样。
            pos = pos.clamp(0,valid-1)
            usable = pos[:,:,0] < pos[:,:,-1]
            lo = pos.long()
            hi = (lo+1).clamp_max(valid-1)
            w = pos-lo
            cls = torch.as_tensor(classes[ix],dtype=torch.long)[:,None,None]
            probability = torch.lerp(q[cls,lo],q[cls,hi],w).reshape(-1,1,32)
            trajectory = torch.lerp(z[:,lo],z[:,hi],w).permute(1,2,0,3).reshape(-1,len(z),32)
            desc,_ = path_features(trajectory,probability,eye,model.adapt_route.spec['bins'])
            sims = []
            for part in desc:
                v = part.reshape(len(ix),5,-1)  # path_features 已逐描述归一化
                sims.append((v[:,:1]*v[:,1:]).sum(-1))
            similarity = torch.stack(sims).mean(0) * (usable[:,:1] & usable[:,1:])
            extra[ix] = similarity.reshape(-1,2,2)
    return np.concatenate((old,extra.numpy()),axis=-1)

def batch_prediction(model, video, params):
    from tools.generalization_models import native_videos
    meta=video['metadata']
    assert meta['use_temporal_grid_feats']
    model.native.test_nms_sigma=params['sigma']
    model._captured={}
    try:
        pred=model.native(native_videos([video]))[0]
        maps,masks=model._captured['pyramid']
        logits=model._captured['logits']
    finally:
        model._captured=None
    b,s,c=default_prediction(pred,meta,params)
    x=features(model,meta,(maps,masks,logits),b,s,c)
    assert np.isfinite(x).all()
    return b,s,c,x


def parameters(cfg):
    from pathlib import Path
    import json
    r=json.loads(Path(__file__).with_name('full_readout_recipe.json').read_text())[cfg['baseline']['shift']]
    return dict(sigma=r['sigma'],classes={str(i):dict(width=r['width'],power=r['power']) for i in range(10)})


def probe_wiring(cfg,entry,engine):
    from libs.trainer.da_trainer import move_to_device
    from tools.baseline_protocol import save
    ctx=engine.build(cfg)
    m=ctx.model
    m.train()
    src=move_to_device(next(iter(ctx.train_loader_src)),m.device)
    tgt=move_to_device(next(iter(ctx.train_loader_tgt)),m.device)
    values=[]
    hook=m.adapt_route.register_forward_hook(lambda mod,args,out:values.append((args,out)))
    epoch=max(5,cfg['train_cfg']['warmup_adv'])
    engine.epoch_setup(ctx,epoch)
    losses=m(src,tgt,mode='train',curr_epoch=epoch)[0]
    hook.remove()
    args,out=values[0]
    s,t=tuple(args[3]),tuple(args[4])
    g=torch.autograd.grad(out[0],s+t,retain_graph=True,allow_unused=True)
    norms=[sum(float(x.square().sum()) for x in group if x is not None)**.5 for group in (g[:len(s)],g[len(s):])]
    assert norms[0]==0 and norms[1]>0 and out[1]['aligned_groups']>0
    (losses['downstream_loss']+losses['loss_conditional_adv_rev']).backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in m.rq3_capacity.parameters())
    # 与正式读出相同的 CPU 路径做一次完整检测和特征计算。
    m.eval().cpu()
    with torch.inference_mode():
        video=ctx.val_loader_src.dataset[0]
        b,_,_,x=batch_prediction(m,video,parameters(cfg))
        assert len(b)>0 and x.shape==(len(b),2,37)
    save(entry/'probe.json',dict(passed=True,source_target_alignment_grad=norms,readout_dim=37,
                                 target_train_gt=False,probe_weights_reused=False))


@torch.inference_mode()
def run(cfg, entry):
    from tools.baseline_protocol import read,save,sha
    from tools.generalization_models import DetectorWithAdaptation
    torch.set_num_threads(2)
    saved=torch.load(entry/'best.pt',map_location='cpu')
    model=DetectorWithAdaptation(cfg,'cpu').eval()
    model.load_state_dict(saved['ema_state_dict'],strict=True)
    del saved
    src,tgt,truth=load_data(cfg)
    params=parameters(cfg)
    fit=RidgeStats(37)
    for i in range(len(src)):
        v=src[i]; b,s,c,x=batch_prediction(model,v,params)
        gt=truth[truth['video-id']==v['metadata']['video_id']]
        fit.add(x,edge_labels(b,c,gt,span_limit(v['metadata'])))
    head=fit.fit()
    np.savez(entry/'full_readout.npz',**head)
    rows=[]
    for i in range(len(tgt)):
        v=tgt[i];assert not v['labels'].numel() and not v['segments'].numel()
        b,s,c,x=batch_prediction(model,v,params)
        b=correct(b,x,head,span_limit(v['metadata']))
        rows.append(pd.DataFrame({'video-id':v['metadata']['video_id'],'t-start':b[:,0],'t-end':b[:,1],'score':s,'label':c}))
    cols=pd.concat(rows,ignore_index=True).to_dict('list')
    torch.save(cols,entry/'full_predictions.pt')
    result=score(cfg,cols)
    assert np.isfinite(result['mAP'])
    save(entry/'full_result.json',dict(**result,params=params,source_train_endpoints=fit.num,
                                      checkpoint_sha256=read(entry/'audit.json')['checkpoint_sha256'],
                                      readout_sha256=sha(entry/'full_readout.npz'),target_dev_recipe_frozen=True))
