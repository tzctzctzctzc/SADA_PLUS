"""一维概率检测头与语义核验；VLM只提供片段类别，不提供端点GT。"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .prototype_capacity import KNOWN_LABELS, feature_interval


def specification():
    return dict(schema='temporal-probabilistic-teacher-v1',hidden_dim=128,
        variance_min=.001,variance_max=1.,variance_initial=.1,
        distribution='Gaussian-in-log1p-native-offset',source_nll_weight=.1,
        target_semantic_weight=.1,target_localization_weight=.1,
        target_start_epoch=5,mask_ratio=.1,mask_block=8,
        teacher_initialization='copy-student',teacher_sync_before_target=True,
        source_selection='original-source-cls-plus-reg',
        target_assignment='native-FPN-points-inside-frozen-selected-proposals',
        semantic_gate='known-unique-VLM-class-agrees-with-EMA',
        normalization='common-geometrically-routable-candidates',
        target_negative_classes=False,temporal_warp=False)


class TemporalVarianceHead(nn.Module):
    def __init__(self,channels):
        super().__init__();spec=specification()
        self.layers=nn.Sequential(nn.Conv1d(channels,spec['hidden_dim'],3,padding=1),
                                  nn.GELU(),nn.Conv1d(spec['hidden_dim'],2,1))
        nn.init.zeros_(self.layers[-1].weight)
        value=(spec['variance_initial']-spec['variance_min'])/(spec['variance_max']-spec['variance_min'])
        nn.init.constant_(self.layers[-1].bias,math.log(value/(1-value)))

    def forward(self,features):
        spec=specification()
        return [(spec['variance_min']+(spec['variance_max']-spec['variance_min'])*
                 self.layers(x).sigmoid()).transpose(1,2) for x in features]


def packet(head,features,masks,logits,offsets,points):
    return dict(logits=[x.transpose(1,2) for x in logits],
        offsets=[x.transpose(1,2) for x in offsets],variance=head(features),
        masks=[x[:,0].bool() for x in masks],points=points)


def shifted_gaussian_nll(mean,variance,target):
    # 加常数只令记录非负，梯度与高斯负对数似然相同。
    return .5*((mean-target).square()/variance+
                (variance/specification()['variance_min']).log())


def gaussian_kl(student_mean,student_variance,teacher_mean,teacher_variance):
    teacher_mean=teacher_mean.detach();teacher_variance=teacher_variance.detach()
    return .5*((student_variance/teacher_variance).log()+
        (teacher_variance+(teacher_mean-student_mean).square())/student_variance-1.)


def source_likelihood(model,raw,videos,return_rows=False):
    labels,offsets=model.label_points(raw['points'],[v['segments'] for v in videos],[v['labels'] for v in videos])
    gt=torch.stack(offsets);positive=torch.stack(labels).sum(-1)>0
    positive &= torch.cat(raw['masks'],dim=1)
    mean=torch.cat(raw['offsets'],dim=1).clamp_min(0).log1p()
    variance=torch.cat(raw['variance'],dim=1)
    target=gt.clamp_min(0).log1p()
    loss=shifted_gaussian_nll(mean,variance,target)[positive].mean() if positive.any() else variance.sum()*0.
    if not return_rows:return loss,int(positive.sum())
    return loss,dict(error_squared=(mean-target).square().detach(),variance=variance.detach(),
                     positive=positive.detach(),level_sizes=[len(p) for p in raw['points']])


def visible_geometry(cache,video):
    meta=video['metadata'];start=float(meta.get('crop_start',0))
    stop=start+float(meta.get('crop_feat_len',video['feats'].shape[-1]))
    result=[]
    for row in cache.get(str(meta['video_id']),[]):
        if not row.get('participation_selected',True):continue
        a,b=feature_interval(row,meta)
        if start<=a<b<=stop:result.append((row,a-start,b-start))
    return result


def vlm_class(row,classes):
    # 不能把BACKGROUND/UNKNOWN/ABSTAIN或平票强制改成某个已知动作。
    label=row.get('semantic_label');scores=row.get('class_probabilities',[])
    if label not in KNOWN_LABELS[:classes] or len(scores)!=classes:return None
    top=max(scores)
    if scores.count(top)!=1:return None
    index=scores.index(top)
    return index if KNOWN_LABELS[index]==label else None


def temporal_iou(left,right,start,end):
    intersection=(torch.minimum(right,right.new_tensor(end))-torch.maximum(left,left.new_tensor(start))).clamp_min(0)
    return intersection/((right-left).clamp_min(0)+(end-start)-intersection).clamp_min(1e-6)


def target_losses(student,teacher,videos,cache,use_vlm,variance_scale=(1.,1.)):
    zero=sum(x.sum()*0. for x in student['offsets'])+sum(x.sum()*0. for x in student['logits'])
    semantic,localization=zero,zero
    stats=dict(routable=0,accepted=0,vlm_rejected=0,native_points=0,weight_sum=0.)
    for batch,video in enumerate(videos):
        for row,start,end in visible_geometry(cache,video):
            level=int(row['source_fpn_levels'][0])
            if not 0<=level<len(student['points']):continue
            points=student['points'][level]
            if not torch.equal(points,teacher['points'][level]):raise ValueError('教师/学生时间格不一致')
            valid=student['masks'][level][batch]&teacher['masks'][level][batch]
            valid &= (points[:,0]>=start)&(points[:,0]<=end)
            if not valid.any():continue
            stats['routable']+=1
            probability=teacher['logits'][level][batch,valid].detach().sigmoid()
            # 只选择时间支撑最强的类别；语义与端点可信度不混为一个量。
            scores=probability.square().sum(0)/probability.sum(0).clamp_min(1e-8)
            label=int(scores.argmax());confidence=scores[label].detach()
            if use_vlm and vlm_class(row,probability.shape[1])!=label:
                stats['vlm_rejected']+=1;continue
            means=teacher['offsets'][level][batch,valid].detach().clamp_min(0)
            native=points[valid]
            left=native[:,0]-means[:,0]*native[:,3];right=native[:,0]+means[:,1]*native[:,3]
            weights=probability[:,label].square()*temporal_iou(left,right,start,end)
            total=weights.sum()
            if total<=0:continue
            weights=(weights/total).detach()
            current=student['logits'][level][batch,valid,label].sigmoid()
            bag=(weights*current).sum().clamp_min(1e-8)
            semantic=semantic-confidence*bag.log()
            smean=student['offsets'][level][batch,valid].clamp_min(0).log1p()
            tmean=means.log1p()
            scale=smean.new_tensor(variance_scale)
            kl=gaussian_kl(smean,student['variance'][level][batch,valid]*scale,tmean,
                           teacher['variance'][level][batch,valid]*scale)
            localization=localization+confidence*(weights[:,None]*kl).sum()/2.
            stats['accepted']+=1;stats['native_points']+=int(valid.sum());stats['weight_sum']+=float(confidence)
    denominator=max(stats['routable'],1)
    return semantic/denominator,localization/denominator,stats


def masked_videos(videos,generator):
    spec=specification();result=[]
    for video in videos:
        out=dict(video);length=video['feats'].shape[-1]
        blocks=math.ceil(length/spec['mask_block'])
        keep=(torch.rand(blocks,generator=generator)>=spec['mask_ratio']).repeat_interleave(spec['mask_block'])[:length]
        out['feats']=video['feats']*keep.to(video['feats'].device)[None,:]
        result.append(out)
    return result
