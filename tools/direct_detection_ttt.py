"""直接检测一致性TTA：原EMA起点、全部共享SGP/FPN、一阶逐视频更新后精确重置。

借鉴IoU-filter的检测可靠性原则，不是Faster-RCNN官方复现。时间视图匹配类别/边界，
保留完整视图的原FPN物理点和边界；不把跨视图均值当真值，不推导目标背景标签。
"""
from copy import deepcopy
import hashlib
import math
import torch
from torch.nn import functional as F
from tools.tta_core import dense, predict_full, tiou, match
from tools.meta_ttt import clean_video, check_clean, Encoder

SPEC = dict(schema='direct-detection-consistency-episodic-tta-v1', seed=123,
    initialization='original-A100-source-selected-Prototype-EMA', source_parameter_training=False,
    update='all-shared-SGP-and-all-shared-FPN', optimizer='SGD-no-momentum-no-weight-decay',
    learning_rate=1e-4, steps_candidates=[3,5], gradient_clip=1.,
    teacher='fixed-original-detector-reset-each-video', pseudo_threshold=.5, max_pseudo=64,
    match_tiou=.5, nms_tiou=.5, minimum_support=2, maximum_relative_boundary_std=.1,
    temporal_alignment='maximum-FPN-stride-not-SGP-padding-divisor',
    short_video_views='full-plus-two-complementary-10percent-block-masks',
    student_mask_fraction=.2, block_tokens=16, classification_weight=1., regression_weight=1.,
    preservation_weight=.1, preservation_points_per_level=128,
    pseudo_classification='positive-class-soft-BCE-teacher-full-probability',
    pseudo_regression='official-1D-DIoU-at-original-teacher-FPN-physical-point',
    no_negative_pseudo_labels=True, source_selection='six-full-source-val-mean-map-tie-fewer-steps',
    final_prediction='single-full-view-original-postprocessing-no-fusion',
    target_protocol='offline-whole-video-adapt-predict-finally-reset',
    source_calibration_steps_include_baseline=True, target_GT_used=False,
    reference='https://github.com/XiaoqianRuan1/IoU-filter/blob/main/maskrcnn_benchmark/engine/pseudo_tta.py')


def seed(video_id, step):
    return int(hashlib.sha256(('123:%s:%d'%(video_id,step)).encode()).hexdigest()[:8],16)


def marks(length, fraction, value):
    gen=torch.Generator().manual_seed(value)
    result=torch.zeros(length,dtype=torch.bool)
    wanted=min(length-1,max(1,round(length*fraction))) if length>1 else 0
    for start in (torch.randperm(math.ceil(length/SPEC['block_tokens']),generator=gen)*SPEC['block_tokens']).tolist():
        remaining=wanted-int(result.sum())
        if remaining<=0:break
        result[start:start+min(SPEC['block_tokens'],remaining)]=True
    return result


def views(length, alignment):
    cut=max(alignment,int(.1*length)//alignment*alignment)
    if length-cut>=alignment:
        return [(0,length,None),(0,length-cut,None),(cut,length,None)]
    a=marks(length,.1,123);b=a.roll(max(1,length//2))
    if length>1 and torch.equal(a,b):b=~a
    return [(0,length,None),(0,length,a),(0,length,b)]


def context(video, begin, end, masked=None):
    check_clean(video)
    result=clean_video(video);result['feats']=video['feats'][:,begin:end]
    if masked is not None:
        result['feats']=result['feats'].masked_fill(masked.to(result['feats'].device)[None,:],0.)
    result['metadata']['video_length']=end-begin
    # 所有匹配/监督均在特征坐标内；最终秒坐标只由未裁剪原metadata统一转换。
    return result


def configure(model):
    model.eval().requires_grad_(False)
    if model.backbone_source is not model.backbone_target or model.neck_source is not model.neck_target:
        raise ValueError('要求原SGP/FPN共享别名')
    encoder=Encoder(model).eval().requires_grad_(True)
    return dict(encoder.named_parameters())


def reset(named, initial):
    with torch.no_grad():
        for n,p in named.items():p.copy_(initial[n])


@torch.no_grad()
def candidates(packet, begin, end):
    rows=[]
    for level,(logits,offsets,points,valid) in enumerate(zip(packet['logits'],packet['offsets'],packet['points'],packet['masks'])):
        probability=logits[0,:,:10].sigmoid()
        point_ids,class_ids=((probability>=SPEC['pseudo_threshold'])&valid[0,:,None]).nonzero(as_tuple=True)
        if not len(point_ids):continue
        scores=probability[point_ids,class_ids]
        order=scores.argsort(descending=True)[:SPEC['max_pseudo']*4]
        point_ids,class_ids,scores=point_ids[order],class_ids[order],scores[order]
        p=points[point_ids];o=offsets[0,point_ids]
        bounds=torch.stack((p[:,0]-o[:,0]*p[:,3]+begin,p[:,0]+o[:,1]*p[:,3]+begin),-1)
        keep=(bounds[:,0]>=begin)&(bounds[:,1]<=end)&(bounds[:,1]>bounds[:,0])
        records=torch.cat((bounds,p[:,0:1]+begin,scores[:,None],class_ids[:,None],point_ids[:,None]),1)[keep].cpu().tolist()
        rows.extend(dict(segment=torch.tensor(x[:2]),anchor=x[2],score=x[3],label=int(x[4]),index=int(x[5]),level=level) for x in records)
    result=[]
    for row in sorted(rows,key=lambda r:-r['score']):
        if any(row['label']==r['label'] and float(tiou(row['segment'],r['segment']))>SPEC['nms_tiou'] for r in result):continue
        result.append(row)
        if len(result)==SPEC['max_pseudo']:break
    return result


def reliable(observations):
    base=observations[0];groups=[[r] for r in base]
    for other in observations[1:]:
        for i,j in match(base,other).items():groups[i].append(other[j])
    result=[]
    for original,group in zip(base,groups):
        if len(group)<SPEC['minimum_support']:continue
        bounds=torch.stack([r['segment'] for r in group])
        duration=(original['segment'][1]-original['segment'][0]).clamp(min=1.)
        std=float(bounds.std(0,unbiased=False).mean()/duration)
        if std>SPEC['maximum_relative_boundary_std']:continue
        # 完整视图的原边界/类别/物理点不被求均值替换。
        result.append(dict(original,support=len(group),relative_std=std,weight=original['score']**2/(1+std)))
    return result


@torch.no_grad()
def teacher_rows(teacher,video):
    alignment=max(teacher.fpn_strides);length=video['feats'].shape[-1]
    ranges=views(length,alignment)
    observations=[candidates(dense(teacher,context(video,a,b,m)),a,b) for a,b,m in ranges]
    rows=reliable(observations)
    return rows,dict(views=len(ranges),view_ranges=[[a,b] for a,b,_ in ranges],
        masked_views=sum(m is not None for _,_,m in ranges),candidates=len(observations[0]),reliable=len(rows),
        input_tokens=length,alignment=alignment)


def window(row,length,maximum,alignment):
    if length<=maximum:return 0,length
    begin=int(row['anchor']-maximum/2)
    begin=max(0,min(begin,length-maximum))//alignment*alignment
    return begin,min(length,begin+maximum)


def preservation(packet,reference):
    losses=[]
    for student,teacher,mask in zip(packet['logits'],reference['logits'],reference['masks']):
        ids=mask[0].nonzero().flatten()
        if not len(ids):continue
        p=teacher[0,ids,:10].sigmoid().detach()
        chosen=p.max(-1).values.argsort(descending=True)[:SPEC['preservation_points_per_level']]
        # 连续原概率约束；没有将低分位置/漏检硬编码为目标背景标签。
        losses.append(F.mse_loss(student[0,ids[chosen],:10].sigmoid(),p[chosen]))
    return torch.stack(losses).mean() if losses else packet['logits'][0].sum()*0


def objective(packet,reference,rows,begin,end):
    from libs.models.SADA.losses_aux import ctr_diou_loss_1d
    cls=[];reg=[];weights=[];reg_weights=[]
    for row in rows:
        level=row['level'];points=packet['points'][level]
        local_anchor=row['anchor']-begin
        idx=int((points[:,0]-local_anchor).abs().argmin())
        p=points[idx]
        if abs(float(p[0])-local_anchor)>1e-3 or not bool(packet['masks'][level][0,idx]):continue
        target=row['segment'].to(p.device)-begin
        if not (target[0]<p[0]<target[1]):continue
        logit=packet['logits'][level][0,idx,row['label']]
        cls.append(F.binary_cross_entropy_with_logits(logit,logit.new_tensor(row['score'])))
        weights.append(row['weight'])
        # 只对完整可见边界回归，长动作被crop截断时仍可作类别监督。
        if float(target[0])>=0 and float(target[1])<=end-begin:
            offsets=torch.stack((p[0]-target[0],target[1]-p[0]))/p[3]
            reg.append(ctr_diou_loss_1d(packet['offsets'][level][0,idx:idx+1],offsets[None,:],reduction='mean'))
            reg_weights.append(row['weight'])
    zero=packet['logits'][0].sum()*0
    def weighted(values,w):
        if not values:return zero
        weight=zero.new_tensor(w)
        return (torch.stack(values)*weight).sum()/weight.sum().clamp(min=1e-8)
    classification=weighted(cls,weights);regression=weighted(reg,reg_weights)
    anchor=preservation(packet,reference)
    loss=classification+regression+SPEC['preservation_weight']*anchor
    return loss,dict(classification=float(classification.detach()),regression=float(regression.detach()),
        preservation=float(anchor.detach()),positive_points=len(cls),regression_points=len(reg)),dict(
        classification=classification,regression=regression)


def adapt(model,teacher,named,initial,video,steps,capture=None,mechanism=False):
    """capture用于单次轨迹复用源3/5步校准；finally无条件重置，不跨视频传播状态。"""
    check_clean(video);reset(named,initial)
    outputs={};trace=[];coverage={};parameter_delta=0.;sgp_gradient=0.;classification_gradient=0.;regression_gradient=0.
    snapshots=set(capture if capture is not None else [steps])
    try:
        if 0 in snapshots:outputs[0]=predict_full(model,video)
        if not steps:return outputs,dict(updated=False,trace=[],target_GT_used=False)
        rows,coverage=teacher_rows(teacher,video)
        rows=sorted(rows,key=lambda r:r['anchor']);references={}
        optimizer=torch.optim.SGD(list(named.values()),lr=SPEC['learning_rate'])
        for step in range(1,steps+1):
            if rows:
                # 与总步数无关：源校准5步轨迹的前3步必须等于正式3步轨迹。
                chosen=rows[min(len(rows)-1,int(((step-1)*.61803398875%1)*len(rows)))]
                begin,end=window(chosen,video['feats'].shape[-1],model.max_seq_len,max(model.fpn_strides))
                key=(begin,end)
                if key not in references:
                    with torch.no_grad():
                        ref=dense(teacher,context(video,begin,end))
                        references[key]=dict(logits=[v.detach() for v in ref['logits']],masks=ref['masks'])
                        del ref
                masked=marks(end-begin,SPEC['student_mask_fraction'],seed(video['metadata']['video_id'],step))
                packet=dense(model,context(video,begin,end,masked))
                loss,metrics,components=objective(packet,references[key],rows,begin,end)
                if not torch.isfinite(loss):raise ValueError('检测一致性loss非有限')
                optimizer.zero_grad(set_to_none=True)
                if mechanism and metrics['positive_points']:
                    # 分别证明类别/边界损失确实到达SGP；不以总梯度替代机制证据。
                    for keyloss,field in [('classification','cls'),('regression','reg')]:
                        gradients=torch.autograd.grad(components[keyloss],list(model.backbone_target.parameters()),retain_graph=True,allow_unused=True)
                        norm=sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5
                        if field=='cls':classification_gradient=max(classification_gradient,norm)
                        else:regression_gradient=max(regression_gradient,norm)
                if metrics['positive_points']:
                    loss.backward()
                    squares=[p.grad.square().sum() for p in model.backbone_target.parameters() if p.grad is not None]
                    sgp_gradient=max(sgp_gradient,float(torch.stack(squares).sum().sqrt()) if squares else 0.)
                    norm=torch.nn.utils.clip_grad_norm_(list(named.values()),SPEC['gradient_clip'])
                    if not torch.isfinite(norm):raise ValueError('SGP梯度非有限')
                    if float(norm)>0:optimizer.step()
                else:norm=loss.new_zeros(())
                trace.append(dict(step=step,updated=float(norm)>0,gradient_norm=float(norm),window=[begin,end],**metrics))
                del packet,loss,components
            else:trace.append(dict(step=step,updated=False,reason='no_reliable_pseudo'))
            if step in snapshots:outputs[step]=predict_full(model,video)
        parameter_delta=sum(float((p.detach()-initial[n]).square().sum()) for n,p in named.items())**.5
        return outputs,dict(coverage=coverage,trace=trace,updated=parameter_delta>0,parameter_delta_l2=parameter_delta,
            sgp_gradient=sgp_gradient,classification_path_to_sgp=classification_gradient,
            regression_path_to_sgp=regression_gradient,mechanism_path_measured=mechanism,target_GT_used=False)
    finally:
        reset(named,initial)
