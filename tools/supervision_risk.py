"""小型监督风险头：只控制额外目标分类，原检测器与推理保持原样。"""
from copy import deepcopy
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA='sada-prototype-supervision-risk-v1'
MODES=('risk',)
SPEC=dict(schema=SCHEMA,seed=123,batch=2,initialization='original_cold_with_fresh_risk_head',
    head='Conv1d(FPN+10_logits,64,3)+ReLU+Conv1d(64,2,1); detached_inputs',
    source_risk_weight=.1,target_classification_weight=.1,negative_weight=.25,
    candidate_floor=.1,class_reliability=.8,background_reliability=.95,
    warmup_epochs=10,ramp_epochs=5,max_epochs=50,validation_every=5,patience=15,
    source_detection='unchanged',target_regression=False,inference='original_unchanged',
    target_view='original_student_training_FPN_vs_EMA_eval_same_features_no_extra_strong_view',
    risk_labels='source_GT_original_multilabel_assignment; background_outside_all_benchmark_GT',
    class_loss='accepted_argmax_positive_only_other_classes_ignored',
    conflict='class_and_background_both_accepted_then_ignore',
    risk_loss='unweighted_BCE_video_mean; class_candidates_only; background_all_valid',
    source_feasibility=dict(epochs=10,lr=3e-4,weight_decay=.05,detector='frozen_mature_A_EMA',
        crop='one_stable_crop_per_source_video_max_seq_len; full_membership',
        selection='source_val_risk_BCE_earliest_best',head_not_transferred_to_formal=True),
    readiness='source_val_equal_count_both_ranks_not_worse_one_positive; accepted_accuracy_class>=.8_BG>=.95; both_coverage>=1percent',
    selection='original_source_cls_plus_reg_EMA; secondary_source_AP_max_earliest_tie',
    target_GT=False,efficiency='reuse_original_source_target_FPN_and_student_logits; one_teacher_cls_path; no_extra_reg_forward')


class SupervisionRisk(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.temporal=nn.Conv1d(channels+10,64,3,padding=1)
        self.output=nn.Conv1d(64,2,1)
    def forward(self,features,logits,masks):
        result=[]
        for x,p,m in zip(features,logits,masks):
            mask=m.to(x.dtype)
            # valid mask先于卷积，避免padding logits污染邻点；不修改原FPN。
            z=torch.cat([x.detach()*mask,p.detach()*mask],1)
            y=self.output(F.relu(self.temporal(z)))*mask
            result.append(y.permute(0,2,1))
        return tuple(result)


def attach(model):
    if hasattr(model,'supervision_risk'):return
    channels=model.cls_head.cls_head.conv.in_channels
    # 不改变原检测器/loader随机流，新增头使用独立、固定初始化。
    device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(198731)
        model.add_module('supervision_risk',SupervisionRisk(channels).to(device))
    model.risk_state=dict(step=0,digest='0'*64,positive_normalizer=250.,negative_normalizer=250.)


def enable_controls():
    from tools import baseline_engine as base
    if getattr(base,'supervision_risk_controls_enabled',False):return
    native=base.controls
    def controls(model):
        result=native(model)
        if hasattr(model,'risk_state'):result['risk_state']=deepcopy(model.risk_state)
        return result
    base.controls=controls;base.supervision_risk_controls_enabled=True


def risk_labels(classes,coordinates,segments,valid):
    """categories可多热；安全背景比原检测负点严格，区间端点也排除。"""
    if len(segments):
        occupied=((coordinates[:,None]>=segments[None,:,0]) &
                  (coordinates[:,None]<=segments[None,:,1])).any(-1)
    else:occupied=torch.zeros_like(valid)
    background=valid & ~occupied & ~(classes.sum(-1)>0)
    return background


def source_loss(model,pred,videos):
    classes,_=model.label_points(pred['points'],[v['segments'] for v in videos],[v['labels'] for v in videos])
    coordinates=torch.cat(pred['points'])[:,0]
    losses=[];stats=dict(source_candidates=0,source_correct=0,source_safe_background=0)
    for i,(video,truth) in enumerate(zip(videos,classes)):
        p=pred['cls'][i].detach().sigmoid();score,label=p.max(-1)
        valid=pred['valid'][i];candidate=valid & (score>=SPEC['candidate_floor'])
        correct=truth.gather(1,label[:,None])[:,0]>0
        # 原训练crop会丢掉覆盖不足的截断实例；安全背景必须查询完整源GT。
        record=model.risk_source_truth[video['metadata']['video_id']]
        segments=coordinates.new_tensor(record['segments']).reshape(-1,2)-float(video['metadata'].get('crop_start',0))
        bg=risk_labels(truth,coordinates,segments,valid)
        r=pred['risk'][i]
        cls=F.binary_cross_entropy_with_logits(r[candidate,0],correct[candidate].float()) if candidate.any() else r[:,0].sum()*0
        background=F.binary_cross_entropy_with_logits(r[valid,1],bg[valid].float())
        losses.append(cls+background)
        stats['source_candidates']+=int(candidate.sum());stats['source_correct']+=int((correct&candidate).sum())
        stats['source_safe_background']+=int(bg.sum())
    return torch.stack(losses).mean(),stats


def source_truth(cfg):
    from tools.baseline_protocol import read
    records={}
    # 只装入src_train，不把src_val或任何target标注交给训练模块。
    for video,row in read(cfg['dataset']['json_file'])['database'].items():
        if row['subset'] not in cfg['src_train_split']:continue
        stride=cfg['dataset']['feat_stride']*cfg['dataset'].get('downsample_rate',1)
        frames=cfg['dataset']['num_frames'];fps=cfg['dataset'].get('default_fps') or row['fps']
        segments=[[(float(t)*fps-.5*frames)/stride for t in a['segment']]
                  for a in row.get('annotations',[]) if 0<=int(a['label_id'])<10]
        records[video]=dict(segments=segments)
    return records


def select(probability,reliability,valid):
    score,label=probability.max(-1)
    positive=valid & (score>=SPEC['candidate_floor']) & (reliability[:,0]>=SPEC['class_reliability'])
    negative=valid & (score<SPEC['candidate_floor']) & (reliability[:,1]>=SPEC['background_reliability'])
    # 独立预测器即使发生互相矛盾，也不能同时给正/负监督。
    conflict=valid & (score>=SPEC['candidate_floor']) & (reliability[:,0]>=SPEC['class_reliability']) & (reliability[:,1]>=SPEC['background_reliability'])
    positive &= ~conflict
    return dict(positive=positive,negative=negative,ignored=valid&~(positive|negative),conflict=conflict,label=label)


def positive_negative_loss(logits,label,positive,negative,positive_norm,negative_norm):
    """只正监督被接受类别；重叠动作其它类别不会被伪造为负。"""
    from libs.models.SADA.losses_aux import sigmoid_focal_loss
    chosen=logits.gather(1,label[:,None])[:,0]
    a=sigmoid_focal_loss(chosen,torch.ones_like(chosen),alpha=.25,gamma=2.,reduction='none')
    b=sigmoid_focal_loss(logits,torch.zeros_like(logits),alpha=.25,gamma=2.,reduction='none')
    return (a*positive).sum()/positive_norm+SPEC['negative_weight']*(b*negative[:,None]).sum()/negative_norm


def capture_path(model):
    state=dict(active=False,calls=[])
    def hook(module,args,output):
        if state['active']:state['calls'].append((args,output))
    model.cls_head.register_forward_hook(hook)
    return state


def from_call(model,call,include_risk=True):
    args,out=call;features,masks=args
    cls=out[0];r=model.supervision_risk(features,cls,masks) if include_risk else None
    return dict(features=features,raw_cls=cls,masks=masks,points=model.point_generator(features),
        cls=torch.cat([x.permute(0,2,1) for x in cls],1),risk=torch.cat(r,1) if r is not None else None,
        valid=torch.cat([m[:,0].bool() for m in masks],1))


def teacher_predictions(model,videos):
    model.eval()
    x,mask=model.preprocessing(videos,mode='train')
    x=model.dropout(x)
    features,masks,_=model.backbone_source(x,mask);features,masks=model.neck_source(features,masks)
    cls=model.cls_head(features,masks)
    return from_call(model,((features,masks),cls))


def install(ctx,cfg):
    enable_controls()
    from tools.cold_oracle_fastpath import install as fast
    for m in [ctx.model,ctx.model_ema.module]:attach(m);fast(m)
    ctx.model.risk_source_truth=source_truth(cfg)
    ctx.model_ema.module.supervision_risk.load_state_dict(ctx.model.supervision_risk.state_dict())
    # 只加入已有source/alignment两组，scheduler组数和原参数顺序保持。
    for opt in [ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment]:
        for name,p in ctx.model.supervision_risk.named_parameters():opt.param_groups[1 if name.endswith('bias') else 0]['params'].append(p)
    state=capture_path(ctx.model);ctx.risk_enabled=True;ctx.risk_ledger=None;ctx.risk_gradient_epoch=None
    native=ctx.model.forward;ctx.risk_original_forward=native
    def forward(*args,**kwargs):
        if kwargs.get('mode')!='train' or not ctx.risk_enabled:return native(*args,**kwargs)
        source_videos=kwargs['video_list_src'];target_videos=kwargs['video_list_tgt']
        assert all(not v['segments'].numel() and not v['labels'].numel() for v in target_videos)
        state['calls'].clear();state['active']=True
        try:output=native(*args,**kwargs)
        finally:state['active']=False
        assert len(state['calls'])==3
        source=from_call(ctx.model,state['calls'][0]);student_call=state['calls'][1]
        source_risk,stats=source_loss(ctx.model,source,source_videos)
        losses=output[0];epoch=kwargs['curr_epoch'];control=ctx.model.risk_state
        ramp=max(0.,min(1.,(epoch-SPEC['warmup_epochs'])/SPEC['ramp_epochs']))
        extra=losses['cls_loss']*0;obs=[]
        stats.update(positive=0,negative=0,ignored=0,conflict=0,valid=0,ramp=ramp,target_regression=0.)
        if ramp:
            with torch.no_grad():teacher=teacher_predictions(ctx.model_ema.module,target_videos)
            # 原target原样本/同FPN坐标，不增加strong crop或重复student forward。
            student=from_call(ctx.model,student_call,include_risk=False);prepared=[]
            for i,v in enumerate(target_videos):
                valid=teacher['valid'][i]&student['valid'][i]
                item=select(teacher['cls'][i].sigmoid(),teacher['risk'][i].sigmoid(),valid)
                prepared.append(item)
                row={k:int(item[k].sum()) for k in ['positive','negative','ignored','conflict']}
                row.update(video=v['metadata']['video_id'],crop_start=v['metadata'].get('crop_start',0),valid=int(valid.sum()))
                obs.append(row)
                for key in ['positive','negative','ignored','conflict','valid']:stats[key]+=row[key]
            control['positive_normalizer']=.9*control['positive_normalizer']+.1*max(stats['positive'],1.)
            control['negative_normalizer']=.9*control['negative_normalizer']+.1*max(stats['negative'],1.)
            extra=sum(positive_negative_loss(student['cls'][i],x['label'],x['positive'],x['negative'],
                control['positive_normalizer'],control['negative_normalizer']) for i,x in enumerate(prepared))
            if stats['positive']+stats['negative'] and ctx.risk_gradient_epoch!=epoch:
                for key,params in [('target_SGP_gradient',ctx.model.backbone_source.parameters()),
                        ('target_cls_gradient',ctx.model.cls_head.parameters()),('target_reg_gradient',ctx.model.reg_head.parameters())]:
                    grads=torch.autograd.grad(extra,tuple(params),retain_graph=True,allow_unused=True)
                    stats[key]=sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5
                ctx.risk_gradient_epoch=epoch
        losses['downstream_loss']=losses['downstream_loss']+SPEC['source_risk_weight']*source_risk+SPEC['target_classification_weight']*ramp*extra
        stats.update(source_risk_loss=float(source_risk.detach()),target_cls_loss=float(extra.detach()))
        control['step']+=1
        event=dict(step=control['step'],epoch=epoch,source_videos=[v['metadata']['video_id'] for v in source_videos],observations=obs,stats=stats)
        control['digest']=hashlib.sha256((control['digest']+json.dumps(event,sort_keys=True)).encode()).hexdigest()
        if ctx.risk_ledger:
            with open(ctx.risk_ledger,'a') as f:f.write(json.dumps(dict(event,digest=control['digest']))+'\n')
        for key,value in stats.items():losses['risk_'+key]=losses['cls_loss'].new_tensor(value).detach()
        state['calls'].clear()
        return output
    ctx.model.forward=forward
