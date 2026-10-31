"""离线、逐视频重置的检测TTA。模型输入只包含特征和无标签元数据。"""
from copy import deepcopy
import hashlib
import math
import torch
from torch.nn import functional as F

VARIANTS = ('multiview', 'self_train', 'prototype', 'uncertainty')
SPEC = dict(schema='prototype-episodic-tta-v1', variants=VARIANTS,
    protocol='offline-whole-video-adapt-then-predict-reset-each-video',
    learning_rates=[1e-5, 1e-4], steps=[1, 3], optimizer='SGD-no-momentum',
    teacher_momentum=.99, pseudo_threshold=.5, match_tiou=.5, max_pseudo=32,
    classification_weight=1., localization_weight=1., prototype_weight=.1,
    maximum_relative_boundary_std=.1, minimum_support=2, temporal_crop_fraction=.1,
    source_calibration_videos_per_shift=8, source_bank='all-source-train-GT-final-EMA',
    update='last-shared-SGP-stem-block-and-shared-FPN-norms',
    final_prediction='three-view-unscaled-score-concatenation-original-NMS',
    negative_class_supervision=False, missing_predictions_are_background=False)


def unlabeled(video):
    # 不复制segments/labels后再依赖调用者“记得不用”；在接口处消除它们。
    return dict(feats=video['feats'], metadata=deepcopy(video['metadata']),
                segments=torch.empty(0, 2), labels=torch.empty(0, dtype=torch.long))


def assert_unlabeled(video):
    if video['segments'].numel() or video['labels'].numel():
        raise ValueError('TTA模型入口禁止目标GT')


def crop(video, begin, end):
    assert_unlabeled(video)
    length = video['feats'].shape[-1]
    if not 0 <= begin < end <= length:
        raise ValueError('时间窗口越界')
    result = unlabeled(video)
    result['feats'] = video['feats'][:, begin:end]
    result['metadata']['video_length'] = end-begin
    result['metadata']['duration'] = min(video['metadata']['duration'],
        ((end-begin)*video['metadata']['feat_stride']+.5*video['metadata']['feat_num_frames'])/video['metadata']['fps'])
    return result


def view_ranges(length, alignment):
    cut = int(length*SPEC['temporal_crop_fraction'])//alignment*alignment
    if cut < alignment or length-cut < alignment:
        return [(0, length)]
    return [(0, length), (0, length-cut), (cut, length)]


def dense(model, video):
    """与原target推理相同的分支，保留可微输出；不调用训练forward或GT分配。"""
    assert_unlabeled(video)
    if model.use_trident_head or model.cfg['model'].get('use_ce_cls', False):
        raise ValueError('当前冻结基座是非Trident、独立sigmoid检测头')
    x, mask = model.preprocessing([video], mode='test')
    x = model.dropout(x)
    feats, masks, _ = model.backbone_target(x, mask)
    fpn, masks = model.neck_target(feats, masks)
    logits, _ = model.cls_head_target(fpn, masks)
    offsets, _ = model.reg_head_target(fpn, masks)
    return dict(fpn=fpn, masks=[v.squeeze(1) for v in masks], points=model.point_generator(fpn),
                logits=[v.permute(0,2,1) for v in logits], offsets=[v.permute(0,2,1) for v in offsets])


def raw_prediction(model, output):
    return model.inference_single_video(output['points'], [v[0] for v in output['masks']],
        [v[0] for v in output['logits']], [v[0] for v in output['offsets']],
        [None]*len(output['points']), [None]*len(output['points']))


def postprocess(model, video, raw):
    meta = video['metadata']
    item = dict(raw, video_id=meta['video_id'], fps=meta['fps'], duration=meta['duration'],
        video_length=meta['video_length'], feat_stride=meta['feat_stride'],
        feat_num_frames=meta['feat_num_frames'], use_temporal_grid_feats=meta['use_temporal_grid_feats'])
    return model.postprocessing([item])[0]


@torch.no_grad()
def predict_full(model, video):
    return postprocess(model, video, raw_prediction(model, dense(model, video)))


@torch.no_grad()
def predict_multiview(model, video):
    parts = []
    for begin,end in view_ranges(video['feats'].shape[-1], model.max_div_factor):
        raw = raw_prediction(model, dense(model, crop(video,begin,end)))
        seg = raw['segments']+begin
        keep = torch.ones(len(seg),dtype=torch.bool,device=seg.device)
        if begin: keep &= seg[:,0] > begin+1
        if end < video['feats'].shape[-1]: keep &= seg[:,1] < end-1
        parts.append(dict(segments=seg[keep],scores=raw['scores'][keep],labels=raw['labels'][keep]))
    raw = {k:torch.cat([p[k] for p in parts]) for k in ('segments','scores','labels')}
    return postprocess(model, video, raw)


def tiou(a, b):
    inter=(torch.minimum(a[...,1],b[...,1])-torch.maximum(a[...,0],b[...,0])).clamp(min=0)
    union=(a[...,1]-a[...,0])+(b[...,1]-b[...,0])-inter
    return inter/union.clamp(min=1e-8)


def candidates(model, output, begin, end, total):
    """返回含FPN层身份的伪候选，纯前景筛选；与正式输出NMS独立。"""
    rows=[]
    with torch.no_grad():
        for level,(logit,offset,pts,mask) in enumerate(zip(output['logits'],output['offsets'],output['points'],output['masks'])):
            prob=logit[0,:,:10].sigmoid()
            choices=((prob >= SPEC['pseudo_threshold']) & mask[0,:,None]).nonzero()
            if not len(choices): continue
            scores=prob[choices[:,0],choices[:,1]]
            order=scores.argsort(descending=True)[:SPEC['max_pseudo']*4]
            choices=choices[order];scores=scores[order]
            points=pts[choices[:,0]];off=offset[0,choices[:,0]]
            segs=torch.stack((points[:,0]-off[:,0]*points[:,3],points[:,0]+off[:,1]*points[:,3]),dim=-1)+begin
            keep=(segs[:,1]>segs[:,0]) & (segs[:,0]>=begin) & (segs[:,1]<=end)
            if begin:keep &= segs[:,0]>begin+1
            if end<total:keep &= segs[:,1]<end-1
            # 小规模候选匹配在CPU完成，避免逐候选GPU同步和成千上万次小kernel。
            segs,scores,labels=segs[keep].cpu(),scores[keep].cpu(),choices[keep,1].cpu()
            rows.extend(dict(segment=seg,score=score,label=int(label),level=level)
                        for seg,score,label in zip(segs,scores,labels))
        rows.sort(key=lambda x:-float(x['score']))
        result=[]
        for row in rows:
            if any(row['label']==x['label'] and float(tiou(row['segment'],x['segment']))>.5 for x in result):continue
            result.append(row)
            if len(result)==SPEC['max_pseudo']:break
    return result


def match(base, other):
    # 一对一匹配，避免同一个观察结果同时给相邻动作背书。
    pairs=[]
    for i,a in enumerate(base):
        for j,b in enumerate(other):
            if a['label']!=b['label']:continue
            score=float(tiou(a['segment'],b['segment']))
            if score>=SPEC['match_tiou']:pairs.append((score,i,j))
    used_a,used_b,result=set(),set(),{}
    for _,i,j in sorted(pairs,reverse=True):
        if i not in used_a and j not in used_b:
            result[i]=j;used_a.add(i);used_b.add(j)
    return result


@torch.no_grad()
def consensus(model, video):
    total=video['feats'].shape[-1];views=view_ranges(total,model.max_div_factor)
    observations=[candidates(model,dense(model,crop(video,a,b)),a,b,total) for a,b in views]
    base=observations[0];groups=[[x] for x in base]
    for other in observations[1:]:
        for i,j in match(base,other).items():groups[i].append(other[j])
    result=[]
    for row,group in zip(base,groups):
        if len(group)<SPEC['minimum_support']:continue
        bounds=torch.stack([x['segment'] for x in group]);scores=torch.stack([x['score'] for x in group])
        segment=bounds.mean(0);duration=(segment[1]-segment[0]).clamp(min=1.)
        relative_std=float(bounds.std(0,unbiased=False).mean()/duration)
        result.append(dict(segment=segment.detach(),label=row['label'],level=row['level'],
            probability=float(scores.mean()),class_weight=float(scores.mean()*(1-scores.std(unbiased=False)).clamp(0,1)),
            boundary_std=relative_std,location_weight=1/(1+relative_std),support=len(group)))
    return result


def adaptable_parameters(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    modules=[model.backbone_target.stem[-1], model.neck_target.fpn_norms]
    allowed={id(p) for module in modules for p in module.parameters()}
    named={n:p for n,p in model.named_parameters() if id(p) in allowed}
    if not named:raise ValueError('没有可更新的共享检测参数')
    for p in named.values():p.requires_grad_(True)
    return named


def parameter_digest(named):
    h=hashlib.sha256()
    for name,p in sorted(named.items()):
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def adaptation_loss(model, output, rows, begin, end, variant, bank=None):
    zero=output['logits'][0].sum()*0
    cls,loc,pro,cls_weights,loc_weights=[],[],[],[],[]
    proto_rows=[]
    for row in rows:
        segment=row['segment'].to(zero.device)-begin
        if float(segment[0])<=1 or float(segment[1])>=end-begin-1:continue
        level=row['level'];points=output['points'][level]
        middle=segment.mean();idx=int((points[:,0]-middle).abs().argmin())
        if not output['masks'][level][0,idx]:continue
        logit=output['logits'][level][0,idx,row['label']]
        cls.append(F.binary_cross_entropy_with_logits(logit,logit.new_tensor(row['probability'])))
        weighted=variant=='uncertainty';cls_weights.append(row['class_weight'] if weighted else 1.)
        p=points[idx];off=output['offsets'][level][0,idx]
        pred=torch.stack((p[0]-off[0]*p[3],p[0]+off[1]*p[3]))
        duration=(segment[1]-segment[0]).clamp(min=1.)
        if not weighted or row['boundary_std']<=SPEC['maximum_relative_boundary_std']:
            loc.append(F.smooth_l1_loss(pred/duration,segment.detach()/duration))
            loc_weights.append(row['location_weight'] if weighted else 1.)
        proto_rows.append((segment.detach(),level,row['label']))
    def mean(values,weights):
        if not values:return zero
        weight=zero.new_tensor(weights);return (torch.stack(values)*weight).sum()/weight.sum().clamp(min=1e-8)
    if variant in ('prototype','uncertainty') and proto_rows:
        if bank is None:raise ValueError('原型TTA缺少冻结源参考')
        selected=[x for x in proto_rows if bool(bank['ready'][x[2]])]
        if selected:
            proposal=dict(segments=torch.stack([x[0] for x in selected]),
                levels=torch.tensor([x[1] for x in selected],device=zero.device,dtype=torch.long))
            encoded=model.rq3_capacity.branch.encode(output['fpn'],[proposal],model.fpn_strides)['embedding']
            centers=bank['centers'][[x[2] for x in selected]].detach()
            pro=1-(encoded*centers).sum(-1)
    prototype=pro.mean() if torch.is_tensor(pro) else zero
    classification,localization=mean(cls,cls_weights),mean(loc,loc_weights)
    loss=classification+localization+SPEC['prototype_weight']*prototype
    return loss,dict(classification=float(classification.detach()),localization=float(localization.detach()),
        prototype=float(prototype.detach()),positive_matches=len(cls),boundary_matches=len(loc))


def adapt_video(student, teacher, initial, video, variant, lr, steps, bank=None):
    if variant not in VARIANTS:raise ValueError('未知TTA方法')
    assert_unlabeled(video)
    named=adaptable_parameters(student)
    teacher_named=dict(teacher.named_parameters())
    # 每视频只需恢复可能更新的参数；eval路径不更新normalizer/原型/BN缓存。
    with torch.no_grad():
        for name,p in named.items():p.copy_(initial[name]);teacher_named[name].copy_(initial[name])
    before=parameter_digest(named)
    trace=[]
    first_rows=consensus(teacher,video)
    if variant!='multiview':
        optimizer=torch.optim.SGD(list(named.values()),lr=lr)
        for step in range(steps):
            rows=first_rows if step==0 else consensus(teacher,video)
            if not rows:
                trace.append(dict(step=step,reason='no_reliable_pseudo',updated=False));continue
            target=rows[step%len(rows)]['segment'].mean().item()
            total=video['feats'].shape[-1];maximum=student.max_seq_len
            begin=max(0,min(int(target-maximum/2),max(0,total-maximum)))
            begin=begin//student.max_div_factor*student.max_div_factor;end=min(total,begin+maximum)
            output=dense(student,crop(video,begin,end))
            loss,metrics=adaptation_loss(student,output,rows,begin,end,variant,bank)
            if not torch.isfinite(loss):raise ValueError('TTA loss非有限')
            if not metrics['positive_matches']:
                trace.append(dict(step=step,reason='no_fully_visible_positive',updated=False));continue
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradients=[p.grad for p in named.values() if p.grad is not None]
            norm=torch.stack([g.norm() for g in gradients]).norm() if gradients else loss.new_zeros(())
            if not torch.isfinite(norm):raise ValueError('TTA梯度非有限')
            if float(norm)>0:
                torch.nn.utils.clip_grad_norm_(list(named.values()),1.)
                optimizer.step()
                with torch.no_grad():
                    for name,p in named.items():teacher_named[name].lerp_(p,1-SPEC['teacher_momentum'])
            trace.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),
                updated=float(norm)>0,pseudo_count=len(rows),window=[begin,end],**metrics))
            del output,loss
        del optimizer
    # 所有方法共用相同输出融合与无伪标签回退，避免把多视图收益误算成梯度收益。
    changed=before!=parameter_digest(named)
    result=predict_multiview(student,video) if first_rows else predict_full(student,video)
    return result,dict(initial_parameter_sha256=before,final_parameter_sha256=parameter_digest(named),
        changed=changed,trace=trace,target_gt_seen=False)
