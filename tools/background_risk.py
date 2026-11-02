"""单输出背景小头，只给原target分类路径补充风险加权负监督。"""
from copy import deepcopy
import hashlib,json
import torch
from torch import nn
from torch.nn import functional as F
from tools.supervision_risk import (enable_controls,capture_path,from_call,teacher_predictions,
    source_truth,risk_labels)

MODES=('background',)
SCHEMA='sada-prototype-background-risk-v1'
SPEC=dict(schema=SCHEMA,initialization='original_cold_fresh_background_head',seed=123,batch=2,
    head='detached_FPN_and_10_logits;Conv3_522_to64_ReLU_Conv1_64to1',
    source_detection='unchanged',original_Prototype_adversarial='unchanged',inference='unchanged',
    target_positive_classification=False,target_regression=False,candidate_max_class_score=.1,
    weighting='sigmoid(a*background_logit+b)^2; a>0; source_video_calibration_only',
    source_aux_weight=.1,target_weight=.1,negative_alpha=.25,warmup_epochs=10,ramp_epochs=5,
    source_aux_role='fit_videos_only; original_source_detector_still_uses_all_source_train',
    source_split='stable_hash_20percent_calibration_rest_fit',
    normalization='EMA_count_of_all_candidate_negative_points; no_normalization_by_predicted_weight_sum',
    calibration='CPU_monotone_logistic_video_mean_NLL_on_source_calib_low_class_points; repeat_every5_on_current_EMA',
    feasibility=dict(cache_reuse='old_source_FPN_cache_readonly_and_SHA_bound',epochs=10,lr=3e-4,
        head_selection='fixed_epoch10_not_source_val',head_not_transferred_to_formal=True),
    source_readiness='six_shift_mean_physical_contamination_gain_vs_uniform_and_equal_weight_confidence>0; every_shift_mass>=.1_and_ESS_fraction>=.25',
    max_epochs=50,validation_every=5,patience=15,
    checkpoint_selection='original_source_cls_reg_EMA_and_all_saved_source_AP_max_earliest_tie',
    target_GT=False,efficiency='reuse_existing_source_target_FPN; teacher_cls_only; reuse_frozen_cache_only_in_feasibility')


class BackgroundRisk(nn.Module):
    def __init__(self,channels):
        super().__init__();self.temporal=nn.Conv1d(channels+10,64,3,padding=1);self.output=nn.Conv1d(64,1,1)
    def forward(self,features,logits,masks):
        return tuple((self.output(F.relu(self.temporal(torch.cat([x.detach()*m,p.detach()*m],1))))*m).permute(0,2,1)
            for x,p,m in zip(features,logits,masks))


def video_split(ids):
    ordered=sorted(ids,key=lambda x:hashlib.sha256(('background-risk-v1/'+x).encode()).hexdigest())
    n=max(1,len(ordered)//5)
    assert len(ordered)>n
    return dict(fit=ordered[n:],calibration=ordered[:n])


def attach(model):
    if hasattr(model,'supervision_risk'):return
    device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(198731)
        model.add_module('supervision_risk',BackgroundRisk(model.cls_head.cls_head.conv.in_channels).to(device))
    model.risk_state=dict(step=0,digest='0'*64,negative_normalizer=250.,
        calibration=dict(slope=1.,bias=0.,epoch=0,scope='source_only'))


def calibrated(logits,calibration):
    assert calibration['slope']>0
    return (logits*calibration['slope']+calibration['bias']).sigmoid()


def negative_loss(logits,weights,candidate,normalizer):
    from libs.models.SADA.losses_aux import sigmoid_focal_loss
    loss=sigmoid_focal_loss(logits,torch.zeros_like(logits),alpha=.25,gamma=2.,reduction='none')
    return (loss*weights.detach()[:,None]*candidate[:,None]).sum()/normalizer


def fit_calibration(rows,epoch=0):
    """rows仅含源calibration视频，每视频等权；不使用source_val或target。"""
    usable=[r for r in rows if len(r['logits'])]
    assert usable
    x=torch.cat([r['logits'].detach().cpu().double() for r in usable])
    y=torch.cat([r['background'].detach().cpu().double() for r in usable])
    w=torch.cat([torch.full_like(r['logits'].cpu().double(),1/(len(r['logits'])*len(usable))) for r in usable])
    log_slope=torch.tensor(0.,dtype=torch.double,requires_grad=True);bias=torch.tensor(0.,dtype=torch.double,requires_grad=True)
    opt=torch.optim.LBFGS([log_slope,bias],lr=.5,max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();slope=log_slope.clamp(-4,4).exp()
        loss=(F.binary_cross_entropy_with_logits(x*slope+bias,y,reduction='none')*w).sum()+1e-4*(log_slope.square()+bias.square())
        loss.backward();return loss
    before=float((F.binary_cross_entropy_with_logits(x,y,reduction='none')*w).sum())
    opt.step(closure)
    slope=float(log_slope.detach().clamp(-4,4).exp());b=float(bias.detach())
    after=float((F.binary_cross_entropy_with_logits(x*slope+b,y,reduction='none')*w).sum())
    assert torch.isfinite(torch.tensor([slope,b,after])).all()
    return dict(slope=slope,bias=b,epoch=epoch,scope='source_calibration_only',videos=[r['video'] for r in usable],
        before_NLL=before,after_NLL=after,point_count=len(x))


def install(ctx,cfg):
    from tools.cold_oracle_fastpath import install as fast
    enable_controls()
    for m in [ctx.model,ctx.model_ema.module]:attach(m);fast(m)
    truth=source_truth(cfg);ctx.model.risk_source_truth=truth
    ctx.background_roles=video_split(truth);fit_ids=set(ctx.background_roles['fit'])
    ctx.model_ema.module.supervision_risk.load_state_dict(ctx.model.supervision_risk.state_dict())
    for opt in [ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment]:
        for name,p in ctx.model.supervision_risk.named_parameters():opt.param_groups[1 if name.endswith('bias') else 0]['params'].append(p)
    state=capture_path(ctx.model);ctx.risk_enabled=True;ctx.risk_ledger=None;ctx.risk_gradient_epoch=None
    native=ctx.model.forward;ctx.risk_original_forward=native
    def forward(*args,**kwargs):
        if kwargs.get('mode')!='train' or not ctx.risk_enabled:return native(*args,**kwargs)
        source_videos=kwargs['video_list_src'];target_videos=kwargs['video_list_tgt']
        assert all(not x['segments'].numel() and not x['labels'].numel() for x in target_videos)
        state['calls'].clear();state['active']=True
        try:output=native(*args,**kwargs)
        finally:state['active']=False
        assert len(state['calls'])==3
        source=from_call(ctx.model,state['calls'][0]);student_call=state['calls'][1]
        coordinates=torch.cat(source['points'])[:,0];source_losses=[]
        nfit=0;ncal=0
        for i,v in enumerate(source_videos):
            video=v['metadata']['video_id']
            if video not in fit_ids:ncal+=1;continue
            nfit+=1;segments=coordinates.new_tensor(truth[video]['segments']).reshape(-1,2)-float(v['metadata'].get('crop_start',0))
            # 背景监督不依赖被裁剪丢弃的原训练segments，完整源GT覆盖才有效。
            valid=source['valid'][i];background=risk_labels(torch.zeros(len(valid),10,device=coordinates.device),coordinates,segments,valid)
            candidate=valid&(source['cls'][i].detach().sigmoid().max(-1).values<SPEC['candidate_max_class_score'])
            raw=source['risk'][i,:,0]
            source_losses.append(F.binary_cross_entropy_with_logits(raw[candidate],background[candidate].float()) if candidate.any() else raw.sum()*0)
        source_risk=torch.stack(source_losses).mean() if source_losses else source['risk'].sum()*0
        losses=output[0];epoch=kwargs['curr_epoch'];control=ctx.model.risk_state
        ramp=max(0.,min(1.,(epoch-SPEC['warmup_epochs'])/SPEC['ramp_epochs']))
        extra=losses['cls_loss']*0;observations=[]
        stats=dict(source_fit_videos=nfit,source_calibration_excluded=ncal,ramp=ramp,
            negative=0,valid=0,weight_sum=0.,target_regression=0.,positive=0)
        if ramp:
            with torch.no_grad():teacher=teacher_predictions(ctx.model_ema.module,target_videos)
            student=from_call(ctx.model,student_call,include_risk=False);prepared=[]
            for i,v in enumerate(target_videos):
                valid=teacher['valid'][i]&student['valid'][i]
                candidate=valid&(teacher['cls'][i].sigmoid().max(-1).values<SPEC['candidate_max_class_score'])
                weights=calibrated(teacher['risk'][i,:,0],ctx.model_ema.module.risk_state['calibration']).square().detach()
                row=dict(video=v['metadata']['video_id'],crop_start=v['metadata'].get('crop_start',0),valid=int(valid.sum()),
                    negative=int(candidate.sum()),weight_sum=float(weights[candidate].sum()),
                    calibration_epoch=ctx.model_ema.module.risk_state['calibration']['epoch'])
                observations.append(row);prepared.append((candidate,weights))
                for k in ['valid','negative','weight_sum']:stats[k]+=row[k]
            control['negative_normalizer']=.9*control['negative_normalizer']+.1*max(stats['negative'],1.)
            extra=sum(negative_loss(student['cls'][i],w,c,control['negative_normalizer']) for i,(c,w) in enumerate(prepared))
            if stats['weight_sum']>0 and ctx.risk_gradient_epoch!=epoch:
                for key,params in [('target_SGP_gradient',ctx.model.backbone_source.parameters()),('target_cls_gradient',ctx.model.cls_head.parameters()),
                        ('target_reg_gradient',ctx.model.reg_head.parameters())]:
                    g=torch.autograd.grad(extra,tuple(params),retain_graph=True,allow_unused=True)
                    stats[key]=sum(float(x.detach().square().sum()) for x in g if x is not None)**.5
                ctx.risk_gradient_epoch=epoch
        losses['downstream_loss']=losses['downstream_loss']+SPEC['source_aux_weight']*source_risk+SPEC['target_weight']*ramp*SPEC['negative_alpha']*extra
        stats.update(source_risk_loss=float(source_risk.detach()),target_cls_loss=float(extra.detach()))
        control['step']+=1
        event=dict(step=control['step'],epoch=epoch,source_videos=[v['metadata']['video_id'] for v in source_videos],observations=observations,stats=stats)
        control['digest']=hashlib.sha256((control['digest']+json.dumps(event,sort_keys=True)).encode()).hexdigest()
        if ctx.risk_ledger:
            with open(ctx.risk_ledger,'a') as f:f.write(json.dumps(dict(event,digest=control['digest']))+'\n')
        for k,v in stats.items():losses['risk_'+k]=losses['cls_loss'].new_tensor(v).detach()
        state['calls'].clear();return output
    ctx.model.forward=forward
