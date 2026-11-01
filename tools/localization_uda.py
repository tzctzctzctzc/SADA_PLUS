"""定位专用UDA：固定Prototype表征/分类，只更新共享回归头。"""
import numpy as np
import torch

SPEC=dict(schema='prototype-localization-only-uda-v1',variants=['source_only','confidence','stability'],
    seed=123,batch=2,learning_rate=1e-5,weight_decay=.01,clip_grad=1.,ema_momentum=.99,
    max_epochs=10,validate_every=2,patience_epochs=6,include_original_epoch_zero=True,
    selection='full-source-validation-EMA-map; earliest strict maximum',
    optimizer='fresh-AdamW-regression-parameters-only',pseudo_regression_weight=1.,
    pseudo_teacher='fixed-original-Prototype-source-selected-EMA',pseudo_boundary='original-full-view',
    selection_bins='predicted-duration-quartile x classification-confidence-quartile',
    selection_fraction=.5,pseudo_weight='teacher-probability-squared-normalized',
    target_loss='official-1D-DIoU-at-original-level-nearest-physical-point',
    source_loss='official-source-val-positive-normalization-GT-focal-plus-DIoU; classifier frozen',
    frozen='SGP/FPN/classifier/Prototype/adversarial/actionness parameters',target_GT=False,
    training='persistent-target-train-only; no-test-time-updates',
    checkpoint='regression-delta-plus-SHA-bound-original-full-EMA')


def select_pseudo(rows,variant):
    if variant=='source_only':return []
    if variant not in ('confidence','stability'):raise ValueError('未知定位变体')
    eligible=[x for x in rows if 'relative_std' in x]
    ordered=sorted(eligible,key=lambda x:(x['segment'][1]-x['segment'][0],x['level'],x['point']))
    chosen=[]
    for duration_bin,indices in enumerate(np.array_split(np.arange(len(ordered)),4)):
        group=sorted([ordered[int(i)] for i in indices],key=lambda x:(x['score'],x['level'],x['point']))
        for confidence_bin,subset in enumerate(np.array_split(np.arange(len(group)),4)):
            items=[group[int(i)] for i in subset];count=len(items)//2
            key=(lambda x:(-x['score'],x['level'],x['point'])) if variant=='confidence' else (lambda x:(x['relative_std'],-x['score'],x['level'],x['point']))
            chosen.extend(dict(x,selector_bin=duration_bin*4+confidence_bin) for x in sorted(items,key=key)[:count])
    return chosen


def trainable_parameters(model):
    if model.reg_head_target is not model.reg_head or model.reg_head_pl is not model.reg_head:
        raise ValueError('定位实验依赖原共享reg_head身份')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    allowed={id(p) for n,p in model.reg_head.named_parameters() if not n.startswith('actionness_head.')}
    named={n:p for n,p in model.named_parameters() if id(p) in allowed}
    for p in named.values():p.requires_grad_(True)
    if not named:raise ValueError('没有回归头可训练参数')
    return named


def frozen_tensors(model,named):
    ids={id(p) for p in named.values()}
    return {**{n:p for n,p in model.named_parameters() if id(p) not in ids},
            **{'buffer:'+n:p for n,p in model.named_buffers()}}


def delta(named):
    return {n:p.detach().cpu().clone() for n,p in named.items()}


def apply_delta(named,saved):
    if set(named)!=set(saved):raise ValueError('回归delta参数名单不符')
    with torch.no_grad():
        for n,p in named.items():p.copy_(saved[n])


@torch.no_grad()
def update_ema(ema,student):
    target=dict(ema.named_parameters())
    for n,p in student.items():target[n].lerp_(p,1-SPEC['ema_momentum'])


def regression_targets(output,video,rows):
    """将完整视频的教师边界移到无GT随机crop；真实边界不参与点选择。"""
    begin=video['metadata'].get('crop_start',0);end=begin+video['feats'].shape[-1]
    pred=[];targets=[];weights=[]
    for row in rows:
        left,right=row['segment'];level=row['level']
        if not (begin<=left<right<=end):continue
        points=output['points'][level];index=int((points[:,0]+begin-row['point']).abs().argmin())
        if not bool(output['masks'][level][0,index]):continue
        point=points[index,0]+begin;stride=points[index,3]
        if not left<float(point)<right:continue
        pred.append(output['offsets'][level][0,index]);targets.append(torch.stack(((point-left)/stride,(right-point)/stride)))
        weights.append(row['score']**2)
    if not pred:return None,None,None
    return torch.stack(pred),torch.stack(targets).detach(),torch.tensor(weights,device=pred[0].device)


def pseudo_regression(model,video,rows):
    from tools.tta_core import dense,assert_unlabeled
    from libs.models.SADA.losses_aux import ctr_diou_loss_1d
    assert_unlabeled(video);out=dense(model,video)
    pred,target,weight=regression_targets(out,video,rows)
    if pred is None:return out['offsets'][0].sum()*0,0
    loss=ctr_diou_loss_1d(pred,target,reduction='none')
    return (loss*weight).sum()/weight.sum().clamp_min(1e-8),len(pred)


def choose(history):
    expected=list(range(0,history[-1]['epoch']+1,2)) if history else []
    if not history or [x['epoch'] for x in history]!=expected:raise ValueError('源验证历史必须包含epoch0并连续')
    if any(not np.isfinite(x['map']) or not 0<=x['map']<=1 for x in history):raise ValueError('源mAP非法')
    best=max(history,key=lambda x:x['map'])
    stop=history[-1]['epoch']-best['epoch']>=SPEC['patience_epochs'] or history[-1]['epoch']>=SPEC['max_epochs']
    return best,bool(stop)
