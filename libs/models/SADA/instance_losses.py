"""一对一源监督与显式实例监督接口；没有域对抗，也不强制依赖预测方差。"""
from copy import deepcopy
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def tiou(a,b):
    inter=(torch.minimum(a[:,None,1],b[None,:,1])-torch.maximum(a[:,None,0],b[None,:,0])).clamp_min(0)
    return inter/((a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None,:]-inter).clamp_min(1e-8)


def center_width(segments):return torch.stack((segments.mean(-1),segments[...,1]-segments[...,0]),-1)


def source_targets(videos,lengths,classes):
    targets=[]
    for video,length in zip(videos,lengths):
        segments=video['segments'];labels=video['labels']
        if segments is None:
            segments=length.new_empty(0,2,dtype=torch.float32);labels=length.new_empty(0,dtype=torch.long)
        segments=segments.to(length.device).clamp(0,int(length))/length
        labels=labels.to(length.device);valid=(segments[:,1]>segments[:,0])&(labels>=0)&(labels<classes)
        targets.append(dict(segments=segments[valid],labels=labels[valid]))
    return targets


@torch.no_grad()
def match(output,targets):
    result=[]
    for logits,segments,gt in zip(output['pred_logits'],output['pred_segments'],targets):
        if len(gt['labels'])>len(logits):raise ValueError('query数量不足，不能静默丢弃源GT')
        p=logits.sigmoid().clamp(1e-6,1-1e-6)
        cost=(.25*(1-p)**2*(-p.log())-.75*p**2*(-torch.log1p(-p)))[:,gt['labels']]
        cost=cost+5*torch.cdist(center_width(segments),center_width(gt['segments']),p=1)-2*tiou(segments,gt['segments'])
        a,b=linear_sum_assignment(cost.cpu().numpy())
        result.append((torch.as_tensor(a,device=logits.device),torch.as_tensor(b,device=logits.device)))
    return result


def source_losses(output,videos):
    targets=source_targets(videos,output['feature_lengths'],output['pred_logits'].shape[-1])
    normalizer=max(sum(len(x['labels']) for x in targets),1)
    def layer_loss(layer):
        assignments=match(layer,targets);logits=layer['pred_logits'];labels=torch.zeros_like(logits)
        l1=logits.sum()*0.;iou=l1
        for batch,(a,b) in enumerate(assignments):
            if not len(a):continue
            labels[batch,a,targets[batch]['labels'][b]]=1.
            pred=layer['pred_segments'][batch,a];gt=targets[batch]['segments'][b]
            l1=l1+F.l1_loss(center_width(pred),center_width(gt),reduction='sum')
            iou=iou+(1-tiou(pred,gt).diag()).sum()
        probability=logits.sigmoid();pt=probability*labels+(1-probability)*(1-labels)
        alpha=.25*labels+.75*(1-labels)
        cls=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*(1-pt)**2*alpha).sum()/normalizer
        return cls,l1/normalizer,iou/normalizer
    cls,l1,iou=layer_loss(output);aux=cls*0.
    for layer in output['aux_outputs']:
        a,b,c=layer_loss(layer);aux=aux+a+5*b+2*c
    quality_targets=[]
    with torch.no_grad():
        for segments,gt in zip(output['pred_segments'],targets):
            quality_targets.append(tiou(segments,gt['segments']).max(1).values if len(gt['labels']) else segments.new_zeros(len(segments)))
    quality=F.l1_loss(output['pred_quality_logits'].sigmoid(),torch.stack(quality_targets))
    return dict(loss=cls+5*l1+2*iou+quality+aux,classification=cls,localization_l1=l1,
        localization_iou=iou,quality=quality,auxiliary=aux,source_instances=sum(len(x['labels']) for x in targets))


def create_teacher(student):return deepcopy(student).eval().requires_grad_(False)


@torch.no_grad()
def update_teacher(teacher,student,decay=.999):
    for name,value in teacher.state_dict().items():
        other=student.state_dict()[name]
        if value.is_floating_point():value.mul_(decay).add_(other,alpha=1-decay)
        else:value.copy_(other)


@torch.no_grad()
def attach_vlm_labels(teacher,caches,minimum_overlap):
    """缓存须来自VLM，区间须已转换到当前窗口的归一化坐标；不读目标GT。"""
    labels=torch.full(teacher['pred_logits'].shape[:2],-1,device=teacher['pred_logits'].device,dtype=torch.long)
    for batch,cache in enumerate(caches):
        if cache['origin']!='vlm-prediction':raise ValueError('不能把GT oracle接入无监督训练')
        device=labels.device;classes=cache['labels'].to(device);segments=cache['segments'].to(device)
        valid=(classes>=0)&(classes<teacher['pred_logits'].shape[-1]);segments=segments[valid];classes=classes[valid]
        if not len(segments):continue
        overlap=tiou(teacher['pred_segments'][batch],segments)
        a,b=linear_sum_assignment(-overlap.cpu().numpy())
        for i,j in zip(a,b):
            if overlap[i,j]>=minimum_overlap:labels[batch,i]=classes[j]
    return labels


def distill_instances(student,teacher,semantic_mask,localization_mask,semantic_labels=None):
    """可靠性策略由源域诊断后冻结；类别与定位分别接入，不给未匹配query背景标签。"""
    cls=student['pred_logits'].sum()*0.;loc=student['pred_segments'].sum()*0.;semantic_count=localization_count=0
    for batch in range(len(student['pred_logits'])):
        selected=(semantic_mask[batch]|localization_mask[batch]).nonzero().flatten()
        if not len(selected):continue
        target_segments=teacher['pred_segments'][batch,selected].detach()
        with torch.no_grad():
            a,b=linear_sum_assignment((torch.cdist(center_width(student['pred_segments'][batch]),center_width(target_segments),p=1)-tiou(student['pred_segments'][batch],target_segments)).cpu().numpy())
        for i,j in zip(a,b):
            index=selected[j];label=int(teacher['pred_logits'][batch,index].detach().argmax()) if semantic_labels is None else int(semantic_labels[batch,index])
            if semantic_mask[batch,index] and label>=0:
                confidence=teacher['pred_logits'][batch,index,label].detach().sigmoid()
                cls=cls+confidence*F.softplus(-student['pred_logits'][batch,i,label]);semantic_count+=1
            if localization_mask[batch,index]:
                pred=student['pred_segments'][batch,i:i+1];target=target_segments[j:j+1]
                loc=loc+F.l1_loss(center_width(pred),center_width(target))+(1-tiou(pred,target)[0,0]);localization_count+=1
    return dict(semantic=cls/max(semantic_count,1),localization=loc/max(localization_count,1),
        semantic_instances=semantic_count,localization_instances=localization_count)
