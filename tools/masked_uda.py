"""MIC启发的训练期时序一致性；保留源检测和原Prototype，不在测试视频上优化。"""
from copy import deepcopy
import math
import torch
from torch.nn import functional as F
from tools.tta_core import dense,unlabeled,assert_unlabeled
from tools.ttt_recovery import temporal_mask,seed_for

SPEC=dict(schema='prototype-training-masked-consistency-v1',
    variants=['continuation','ema_unmasked','ema_masked'],seed=123,batch=2,
    max_epochs=10,validate_every=2,patience_epochs=6,selection='full-source-val-EMA-map',
    warmstart='original-six-direction-source-selected-Prototype-EMA',
    optimizer='fresh-AdamW',learning_rate=1e-5,weight_decay=.01,clip_grad=1.,
    ema_momentum=.99,consistency_weight=.1,mask_fraction=.5,mask_block_tokens=16,
    top_points_per_level=16,pseudo_regression_weight=0.,
    target_loss='positive-class-soft-Bernoulli-KL-weighted-by-teacher-probability-squared',
    missing_predictions_are_background=False,target_GT=False,
    training='source-GT-detection-plus-original-sourceCE-targetPrototype-plus-consistency',
    reference_paper='https://openaccess.thecvf.com/content/CVPR2023/html/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.html',
    reference_code='https://github.com/lhoyer/MIC/tree/2f932a98b5dd9f598aaeb32411863ceea0809314/det',
    transfer='1D feature masking and positive-only sigmoid KL, not a reproduction of FasterRCNN MIC')


def trainable_parameters(model):
    for name,p in model.named_parameters():
        if any(x in name for x in ('disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings')):
            p.requires_grad_(False)
    return {n:p for n,p in model.named_parameters() if p.requires_grad}


def masked_video(video,step):
    assert_unlabeled(video)
    clean=unlabeled(video)
    mask=temporal_mask(clean['feats'].shape[-1],seed_for(clean['metadata']['video_id'],step),
        SPEC['mask_fraction'],SPEC['mask_block_tokens']).to(clean['feats'].device)
    clean['feats']=clean['feats'].clone().masked_fill(mask[None],0.)
    return clean


def positive_kl(student,teacher):
    """不将教师没找到的类别/区域变成背景负类；边界伪回归保持关闭。"""
    losses=[];weights=[];points=0
    for logits,targets,valid in zip(student['logits'],teacher['logits'],teacher['masks']):
        with torch.no_grad():
            prob=targets[0,:,:10].sigmoid();confidence,label=prob.max(-1)
            indices=valid[0].nonzero().flatten()
            if not len(indices):continue
            indices=indices[confidence[indices].argsort(descending=True)[:SPEC['top_points_per_level']]]
            label=label[indices];p=confidence[indices].clamp(1e-6,1-1e-6).detach()
            weight=p.square()
        logit=logits[0,indices,label]
        kl=F.binary_cross_entropy_with_logits(logit,p,reduction='none')+p*p.log()+(1-p)*(1-p).log()
        losses.append(kl);weights.append(weight);points+=len(indices)
    if not losses:return student['logits'][0].sum()*0,dict(points=0,teacher_probability=0.)
    values=torch.cat(losses);weight=torch.cat(weights)
    loss=(values*weight).sum()/weight.sum().clamp(min=1e-6)
    return loss,dict(points=points,teacher_probability=float(weight.sqrt().mean()),
                    pseudo_regression_loss=0.,target_gt_used=False)


def consistency(model,teacher,videos,variant,step):
    if variant=='continuation':
        return next(model.parameters()).sum()*0,dict(points=0,teacher_probability=0.,target_gt_used=False)
    if variant not in SPEC['variants']:raise ValueError('未知训练期UDA方法')
    losses=[];reports=[]
    for video in videos:
        assert_unlabeled(video)
        with torch.no_grad():weak=dense(teacher,video)
        strong=masked_video(video,step) if variant=='ema_masked' else video
        output=dense(model,strong)
        loss,report=positive_kl(output,weak);losses.append(loss);reports.append(report)
    return torch.stack(losses).mean(),dict(points=sum(x['points'] for x in reports),
        teacher_probability=sum(x['teacher_probability'] for x in reports)/len(reports),
        pseudo_regression_loss=0.,target_gt_used=False)


@torch.no_grad()
def update_teacher(teacher,student):
    parameters=dict(student.named_parameters())
    for name,p in teacher.named_parameters():
        current=parameters[name]
        if current.requires_grad:p.lerp_(current,1-SPEC['ema_momentum'])
        else:p.copy_(current)
    buffers=dict(student.named_buffers())
    for name,p in teacher.named_buffers():p.copy_(buffers[name])
    teacher.eval()


def select(history):
    if not history or any(not math.isfinite(x['map']) or not 0<=x['map']<=1 for x in history):
        raise ValueError('源验证mAP非法')
    expected=list(range(2,history[-1]['epoch']+1,2))
    if [x['epoch'] for x in history]!=expected:raise ValueError('源验证历史不完整')
    best=max(history,key=lambda x:x['map'])
    stop=history[-1]['epoch']-best['epoch']>=SPEC['patience_epochs'] or history[-1]['epoch']>=SPEC['max_epochs']
    return best,stop
