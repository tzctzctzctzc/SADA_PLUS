"""单输出背景小头，只给原target分类路径补充风险加权负监督。"""
from copy import deepcopy
import hashlib,json
import torch
from torch import nn
from torch.nn import functional as F
from tools.supervision_risk import (enable_controls,capture_path,from_call,teacher_predictions,
    source_truth,risk_labels)

MODES=('negative',)
SCHEMA='sada-prototype-oracle-negative-risk-v3'
SPEC=dict(schema=SCHEMA,initialization='original_cold_fresh_background_head',seed=123,batch=2,
    head='detached_FPN_10_logits_and_normalized_FPN_level;Conv3_523to64_ReLU_Conv1_64to1',
    source_detection='unchanged',original_Prototype_adversarial='unchanged',inference='unchanged',
    target_positive_classification=False,target_regression=False,candidate_max_class_score=.1,
    weighting='sigmoid(a*background_logit+b)^2; a>0; source_video_calibration_only',
    source_aux_weight=.1,target_weight=.1,negative_alpha=.25,warmup_epochs=10,ramp_epochs=5,
    source_aux_role='fit_videos_only; original_source_detector_still_uses_all_source_train',
    supervision='original_Oracle_center_range_shortest_multilabel_GT_assignment; label1_iff_all10_detection_labels_zero',
    fit_loss='equal_mean_BCE_of_valid_negative_and_GT_positive_low_class_groups_per_video; source_calibration_restores_population_prior',
    source_error_curriculum='risk_head_only: same_level_same_video_GT_positive_points_receive_detached_real_low_class_GT_negative_logit_vectors; FPN_and_detector_outputs_unchanged',
    source_curriculum_weight=.5,
    source_split='stable_hash_20percent_calibration_rest_fit',
    normalization='EMA_count_of_all_candidate_negative_points; no_normalization_by_predicted_weight_sum',
    calibration='CPU_monotone_logistic_video_mean_NLL_on_source_calib_low_class_points; repeat_every5_on_current_EMA',
    feasibility=dict(cache_reuse='old_source_FPN_cache_readonly_and_SHA_bound',epochs=10,lr=3e-4,
        head_selection='fixed_epoch10_not_source_val',head_not_transferred_to_formal=True),
    source_readiness='six_shift_mean_detector_positive_contamination_gain_vs_uniform_and_equal_weight_confidence>0; every_shift_mass>=.1_and_ESS_fraction>=.25',
    max_epochs=50,validation_every=5,patience=15,
    checkpoint_selection='original_source_cls_reg_EMA_and_all_saved_source_AP_max_earliest_tie',
    target_GT=False,efficiency='reuse_existing_source_target_FPN; teacher_cls_only; reuse_frozen_cache_only_in_feasibility')


class BackgroundRisk(nn.Module):
    def __init__(self,channels):
        super().__init__();self.temporal=nn.Conv1d(channels+11,64,3,padding=1);self.output=nn.Conv1d(64,1,1)
    def forward(self,features,logits,masks):
        return tuple((self.output(F.relu(self.temporal(torch.cat([x.detach()*m,p.detach()*m,
            torch.ones_like(p[:,:1])*float(level)/max(1,len(features)-1)*m],1))))*m).permute(0,2,1)
            for level,(x,p,m) in enumerate(zip(features,logits,masks)))


def balanced_negative_BCE(raw,negative,candidate):
    values=[]
    for group in [candidate&negative,candidate&~negative]:
        if group.any():values.append(F.binary_cross_entropy_with_logits(raw[group],negative[group].float()))
    return torch.stack(values).mean() if values else raw.sum()*0


@torch.no_grad()
def counterfactual_logits(logits,masks,negative,eligible,step):
    """仅源风险头的漏检课程；真实低分向量同视频/同层搬到GT正点，不修改检测器。"""
    result=[];changed=[];offset=0
    for level,(p,m) in enumerate(zip(logits,masks)):
        t=p.shape[-1];copy=p.detach().clone();mask=torch.zeros_like(m[:,0],dtype=torch.bool)
        for i in range(len(p)):
            valid=m[i,0].bool();is_negative=negative[i,offset:offset+t]
            recipients=(valid&~is_negative&eligible[i]).nonzero().flatten()
            donors=(valid&is_negative&(p[i].sigmoid().max(0).values<SPEC['candidate_max_class_score'])).nonzero().flatten()
            if not len(recipients) or not len(donors):continue
            pick=donors[(torch.arange(len(recipients),device=p.device)+step*37+level*13)%len(donors)]
            copy[i,:,recipients]=p[i,:,pick].detach();mask[i,recipients]=True
        result.append(copy);changed.append(mask);offset+=t
    return tuple(result),torch.cat(changed,1)


def oracle_source_truth(cfg):
    from tools.baseline_protocol import read
    records=source_truth(cfg)
    for video,row in read(cfg['dataset']['json_file'])['database'].items():
        if video in records:records[video]['labels']=[int(a['label_id']) for a in row.get('annotations',[]) if 0<=int(a['label_id'])<10]
    return records


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
    truth=oracle_source_truth(cfg);ctx.model.risk_source_truth=truth
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
        coordinates=torch.cat(source['points'])[:,0];source_losses=[];negative_truth=[]
        nfit=0;ncal=0
        for i,v in enumerate(source_videos):
            video=v['metadata']['video_id']
            segments=coordinates.new_tensor(truth[video]['segments']).reshape(-1,2)-float(v['metadata'].get('crop_start',0))
            # 与冷启动classification Oracle相同：全部相交源GT、精确crop、原正负指派。
            length=v['feats'].shape[-1];labels=torch.tensor(truth[video]['labels'],device=coordinates.device,dtype=torch.long)
            keep=(segments[:,1]>0)&(segments[:,0]<length)
            classes,_=ctx.model.label_points(source['points'],[segments[keep].clamp(0,float(length))],[labels[keep]])
            valid=source['valid'][i];background=~classes[0].any(-1)
            negative_truth.append(background)
            if video not in fit_ids:ncal+=1;continue
            nfit+=1
            candidate=valid&(source['cls'][i].detach().sigmoid().max(-1).values<SPEC['candidate_max_class_score'])
            raw=source['risk'][i,:,0]
            source_losses.append(balanced_negative_BCE(raw,background,candidate))
        source_risk=torch.stack(source_losses).mean() if source_losses else source['risk'].sum()*0
        negative_truth=torch.stack(negative_truth)
        eligible=torch.tensor([v['metadata']['video_id'] in fit_ids for v in source_videos],device=coordinates.device)
        synthetic,changed=counterfactual_logits(source['raw_cls'],source['masks'],negative_truth,eligible,ctx.model.risk_state['step'])
        synthetic_raw=torch.cat(ctx.model.supervision_risk(source['features'],synthetic,source['masks']),1)[:,:,0]
        synth_losses=[]
        for i in range(len(source_videos)):
            if not eligible[i]:continue
            candidate=source['valid'][i]&(torch.cat([p[i] for p in synthetic],1).sigmoid().max(0).values<SPEC['candidate_max_class_score'])
            synth_losses.append(balanced_negative_BCE(synthetic_raw[i],negative_truth[i],candidate))
        if synth_losses:source_risk=(1-SPEC['source_curriculum_weight'])*source_risk+SPEC['source_curriculum_weight']*torch.stack(synth_losses).mean()
        losses=output[0];epoch=kwargs['curr_epoch'];control=ctx.model.risk_state
        ramp=max(0.,min(1.,(epoch-SPEC['warmup_epochs'])/SPEC['ramp_epochs']))
        extra=losses['cls_loss']*0;observations=[]
        stats=dict(source_fit_videos=nfit,source_calibration_excluded=ncal,source_simulated_misses=int(changed.sum()),ramp=ramp,
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
