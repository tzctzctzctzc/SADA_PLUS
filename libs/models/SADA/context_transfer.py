"""已知源动作移入未标注目标上下文；未知区域不参与监督。"""
import hashlib
import json
import math
from pathlib import Path
import torch

SCHEMA = 'prototype-source-context-transfer-v1'
POLICY = dict(context_fraction=.5, minimum_context_tokens=1, edge_ignore_strides=1,
    target_context_tokens_per_side=1,
    patches_per_pair=1, rescale=False, weight=1., onset='original_warmup_adv',
    selection='sha256_seed123_video_crop_epoch_pair', target_labels='always_empty',
    known_source_background=True, partial_source_instances='ignore_covered_points',
    normalizer='current_known_source_positive_points', extra_forward='one_shared_detector_forward',
    extra_forward_rng='fork_and_restore', teacher=False)


def stable_index(parts, size):
    digest=hashlib.sha256(json.dumps([123,*parts],separators=(',',':')).encode()).digest()
    return int.from_bytes(digest[:8],'little')%size


def feature_ground_truth(annotations, metadata, classes, device):
    rows=[a for a in annotations if 0<=int(a['label_id'])<classes]
    intervals=torch.tensor([a['segment'] for a in rows],dtype=torch.float32,device=device).reshape(-1,2)
    labels=torch.tensor([a['label_id'] for a in rows],dtype=torch.long,device=device)
    intervals=(intervals*float(metadata['fps'])-.5*float(metadata['feat_num_frames']))/float(metadata['feat_stride'])
    intervals-=float(metadata.get('crop_start',0))
    return intervals,labels


def paste_pair(source, target, intervals, labels, epoch, pair_index):
    if target['segments'].numel() or target['labels'].numel(): raise ValueError('真实目标GT必须为空')
    src, tgt=source['feats'],target['feats']; source_length=src.shape[-1];target_length=tgt.shape[-1]
    if src.shape[0]!=tgt.shape[0]: raise ValueError('源/目标特征通道不同')
    eligible=[]
    for index,(left,right) in enumerate(intervals.detach().cpu().tolist()):
        if not 0<=left<right<=source_length: continue
        margin=max(POLICY['minimum_context_tokens'],math.ceil((right-left)*POLICY['context_fraction']))
        start=math.floor(left)-margin; end=math.ceil(right)+margin
        if 0<=start<end<=source_length and left-start>=1 and end-right>=1 and end-start<=target_length-2:
            eligible.append((index,start,end))
    if not eligible: return None
    identity=[source['metadata']['video_id'],source['metadata'].get('crop_start',0),
              target['metadata']['video_id'],target['metadata'].get('crop_start',0),int(epoch),int(pair_index)]
    anchor,start,end=eligible[stable_index(identity+['instance'],len(eligible))]
    destination=1+stable_index(identity+['destination'],target_length-(end-start)-1)
    shift=destination-start
    complete=(intervals[:,0]>=start)&(intervals[:,1]<=end)
    intersects=(intervals[:,1]>start)&(intervals[:,0]<end)
    cut=intersects&~complete
    mixed=tgt.detach().clone(); mixed[:,destination:destination+end-start]=src[:,start:end].detach()
    if not torch.equal(mixed[:,destination:destination+end-start],src[:,start:end]):
        raise ValueError('复制源特征不精确')
    # 原数据对象、真实target labels以及元数据均不修改；GT独立传给赋值器。
    video=dict(target,feats=mixed,metadata=dict(target['metadata']))
    video['metadata']['video_id']='mixed_source_context_'+str(pair_index)
    payload=dict(segments=intervals[complete]+shift,labels=labels[complete],
        partial=intervals[cut].clamp(min=start,max=end)+shift,pasted=[destination,destination+end-start],
        anchor_segment=intervals[anchor]+shift,anchor_class_id=int(labels[anchor]),
        original_source_patch=[start,end],translation=shift,source_video=source['metadata']['video_id'],
        target_video=target['metadata']['video_id'])
    return video,payload


def known_points(points, masks, payloads):
    grid=torch.cat(points,0);valid=torch.cat([m.squeeze(1) for m in masks],-1).bool()
    result=[]
    for index,item in enumerate(payloads):
        left,right=item['pasted']; margin=grid[:,3]*POLICY['edge_ignore_strides']
        known=(grid[:,0]>=left+margin)&(grid[:,0]<=right-margin)&valid[index]
        for segment in item['partial']:
            known&=~((grid[:,0]>=segment[0])&(grid[:,0]<=segment[1]))
        result.append(known)
    return torch.stack(result)


class SourceContextTransfer:
    def __init__(self,cfg,classes):
        if cfg['context_transfer']!=dict(schema=SCHEMA,policy=POLICY):raise ValueError('上下文迁移配方改变')
        if classes!=10 or cfg['dataset']['action_subsampling']!='10_majority_classes':raise ValueError('仅配对原10类训练空间')
        body=Path(cfg['dataset']['json_file']).read_bytes()
        if hashlib.sha256(body).hexdigest()!=cfg['baseline']['source_manifest_sha256']:
            raise ValueError('源GT manifest SHA改变')
        subsets=set(cfg['src_train_split'])
        self.annotations={vid:record.get('annotations',[]) for vid,record in json.loads(body)['database'].items()
            if record['subset'].lower() in subsets}
        self.classes=classes

    def __call__(self,model,sources,targets,epoch):
        from .losses_aux import sigmoid_focal_loss,ctr_diou_loss_1d
        mixed=[];payloads=[]
        for index,source in enumerate(sources):
            vid=source['metadata']['video_id']
            if vid not in self.annotations:raise ValueError('混合监督只能使用source_train GT')
            intervals,labels=feature_ground_truth(self.annotations[vid],source['metadata'],self.classes,source['feats'].device)
            pair=paste_pair(source,targets[index%len(targets)],intervals,labels,epoch,index)
            if pair is not None:
                video,payload=pair; mixed.append(video);payloads.append(payload)
        zero=sources[0]['feats'].new_zeros(())
        stats=dict(pairs=len(mixed),complete_instances=sum(len(p['labels']) for p in payloads),
            partial_instances=sum(len(p['partial']) for p in payloads),positive_points=0,known_negative_points=0,
            source_gt_only=1,target_gt_used=0,copied_tokens=sum(p['pasted'][1]-p['pasted'][0] for p in payloads),
            cls_loss=zero,reg_loss=zero,feature_gradient_norm=zero,cls_head_gradient_norm=zero,reg_head_gradient_norm=zero)
        if not mixed:return dict(loss=zero,**stats)
        if model.use_trident_head:raise ValueError('当前配方只适配原普通左右回归头')
        devices=[sources[0]['feats'].device.index] if sources[0]['feats'].is_cuda else []
        # 新前向不改变下一次原source/target/dropout的随机流；选择过程本身不抽样。
        with torch.random.fork_rng(devices=devices):
            inputs,masks=model.preprocessing(mixed,mode='train');inputs=model.dropout(inputs)
            feats,masks,_=model.backbone_source(inputs,masks);fpn,masks=model.neck_source(feats,masks)
            logits,_=model.cls_head(fpn,masks);offsets,_=model.reg_head(fpn,masks)
            points=model.point_generator(fpn)
        labels,distances=model.label_points(points,[p['segments'] for p in payloads],[p['labels'] for p in payloads])
        labels=torch.stack(labels);distances=torch.stack(distances)
        known=known_points(points,masks,payloads);positive=known&labels.gt(0).any(-1)
        supported=positive.any(-1)
        known&=supported[:,None]  # 不把没有正点的混合对变成纯背景训练。
        stats['pairs']=int(supported.sum())
        count=int(positive.sum());stats.update(positive_points=count,known_negative_points=int((known&~positive).sum()))
        if not count:return dict(loss=zero,**stats)
        logit=torch.cat([x.permute(0,2,1) for x in logits],1)
        offset=torch.cat([x.permute(0,2,1) for x in offsets],1)
        cls=sigmoid_focal_loss(logit[known],labels[known],alpha=.25,
            gamma=model.cfg['model'].get('focal_loss_gamma',2.),reduction='sum')/count
        reg=ctr_diou_loss_1d(offset[positive],distances[positive],reduction='sum')/count
        loss=cls+model.cfg['model']['loss_weights']['weight_reg']*reg
        feature_count=len(fpn);cls_parameters=[];reg_parameters=[]
        if getattr(model,'context_gradient_probe',False):
            cls_parameters=list(model.cls_head.parameters());reg_parameters=list(model.reg_head.parameters())
        gradients=torch.autograd.grad(loss,[*fpn,*cls_parameters,*reg_parameters],retain_graph=True,allow_unused=True)
        norm=lambda values:sum((g.detach().square().sum() for g in values if g is not None),zero).sqrt()
        stats.update(cls_loss=cls.detach(),reg_loss=reg.detach(),feature_gradient_norm=norm(gradients[:feature_count]),
            cls_head_gradient_norm=norm(gradients[feature_count:feature_count+len(cls_parameters)]),
            reg_head_gradient_norm=norm(gradients[feature_count+len(cls_parameters):]))
        return dict(loss=loss,**stats)
