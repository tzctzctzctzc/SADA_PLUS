"""DocTTT启发的时序检测迁移：完整SGP、多尺度恢复、可微内循环。

原论文恢复RGB/SSIM；此处仅有SlowFast特征，因此恢复冻结原Proto的各FPN特征。
这是显式的TAL迁移，并非原论文复现。外层只用源GT，不读目标训练集。
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.stateless import functional_call

SPEC = dict(schema='prototype-meta-auxiliary-ttt-v1', seed=123,
    reference='original-source-selected-prototype-EMA-frozen',
    source_only_preparation=True, input='frozen-SlowFast-features',
    representation='all-shared-SGP-and-all-shared-FPN',
    reconstruction='per-level-masked-valid-position-mean-cosine-then-level-mean',
    mask_fraction=.3, block_tokens=16, decoder_hidden=256,
    inner_steps=5, inner_lr=1e-4, inner_optimizer='SGD-no-momentum-no-weight-decay',
    second_order=True, test_decoder='frozen', source_batch=2,
    outer_lr=1e-5, decoder_lr=1e-4, weight_decay=.01, clip=1.,
    max_epochs=50, validate_every=5, patience_epochs=15,
    checkpoint_selection='full-source-val-mAP-after-five-TTT-steps',
    calibration_steps=[5,10,20], calibration_selection='six-shift-full-source-val-mean-mAP-tie-fewer-steps',
    test_protocol='offline-whole-video-mask-adapt-predict-reset-each-video',
    target_GT_used=False, target_results_for_selection=False,
    arms=['joint','meta'], target_modes=['no_adapt','adapt'],
    ordinary_joint='source-detection-plus-same-multiscale-recovery',
    meta_objective='source-detection-after-exact-unrolled-recovery-SGD',
    native_prototype='original-weights-retained-frozen-no-new-Prototype-loss',
    source_task_heads='train-in-preparation-freeze-in-test')


def clean_video(video):
    # 兼容旧unlabeled()未清除metadata.annotations的问题，接口递归拒绝隐藏GT。
    metadata = deepcopy(video['metadata'])
    for key in ('annotations', 'segments', 'labels'):
        metadata.pop(key, None)
    return dict(feats=video['feats'], metadata=metadata,
                segments=torch.empty(0,2), labels=torch.empty(0,dtype=torch.long))


def check_clean(video):
    if video['segments'].numel() or video['labels'].numel():
        raise ValueError('恢复/测试接口禁止GT')
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('annotations','segments','labels') and len(item):
                    raise ValueError('metadata中藏有GT')
                walk(item)
        elif isinstance(value, (list,tuple)):
            for item in value: walk(item)
    walk(video['metadata'])


def seed_for(video_id, step):
    return int(hashlib.sha256(('123:%s:%d' % (video_id,step)).encode()).hexdigest()[:8],16)


def block_mask(length, seed):
    if length < 1: raise ValueError('空视频')
    gen=torch.Generator().manual_seed(seed)
    result=torch.zeros(length,dtype=torch.bool)
    wanted=max(1,min(length-1,round(length*SPEC['mask_fraction']))) if length>1 else 1
    starts=list(range(0,length,SPEC['block_tokens']))
    for i in torch.randperm(len(starts),generator=gen).tolist():
        begin=starts[i]; remaining=wanted-int(result.sum())
        if remaining<=0: break
        result[begin:min(length,begin+min(SPEC['block_tokens'],remaining))]=True
    return result


class Encoder(nn.Module):
    """只包装实际共享分支，避开全SADA多处别名对functional_call的影响。"""
    def __init__(self, model):
        super().__init__()
        self.backbone=model.backbone_target
        self.neck=model.neck_target
    def forward(self, x, valid):
        feats,masks,_=self.backbone(x,valid)
        return self.neck(feats,masks)


class Recovery(nn.Module):
    def __init__(self, dimension, levels, hidden=256):
        super().__init__()
        self.decoders=nn.ModuleList(nn.Sequential(
            nn.Conv1d(dimension+1,hidden,3,padding=1),nn.GELU(),
            nn.Conv1d(hidden,hidden,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv1d(hidden,dimension,1)) for _ in range(levels))
    def forward(self, features, marks):
        if len(features)!=len(self.decoders): raise ValueError('恢复层数错误')
        return [d(torch.cat((f,m.to(f.dtype)),1)) for d,f,m in zip(self.decoders,features,marks)]


def configure(model):
    model.eval().requires_grad_(False)
    encoder=Encoder(model).eval().requires_grad_(True)
    if model.backbone_source is not model.backbone_target or model.neck_source is not model.neck_target:
        raise ValueError('本协议要求原共享SGP/FPN')
    return encoder


def input_tensors(model, video, marked=None):
    check_clean(video)
    x,valid=model.preprocessing([video],mode='test')
    position=torch.zeros_like(valid,dtype=torch.bool)
    if marked is not None:
        if len(marked)!=video['feats'].shape[-1]: raise ValueError('mask长度不匹配')
        position[0,0,:len(marked)]=marked.to(valid.device)
        x=x.masked_fill(position,0.)
    return x,valid,position


def representations(encoder, x, valid, parameters=None):
    # 所有内循环参数均为无写入fast tensor，禁止load_state_dict/copy_切断高阶图。
    return encoder(x,valid) if parameters is None else functional_call(encoder,parameters,(x,valid))


def reference_targets(reference, video):
    with torch.no_grad():
        x,valid,_=input_tensors(reference,video)
        features,masks=Encoder(reference)(x,valid)
        return tuple(f.detach() for f in features),tuple(m.detach() for m in masks)


def recovery_loss(model, encoder, decoder, video, targets, parameters, step):
    clean=clean_video(video);check_clean(clean)
    marked=block_mask(clean['feats'].shape[-1],seed_for(clean['metadata']['video_id'],step))
    x,valid,position=input_tensors(model,clean,marked)
    features,masks=representations(encoder,x,valid,parameters)
    marks=[F.adaptive_max_pool1d(position.float(),f.shape[-1]).bool() for f in features]
    predictions=decoder(features,marks)
    values=[]; counts=[]
    for predicted,target,mask,tvalid,mark in zip(predictions,targets[0],masks,targets[1],marks):
        chosen=(mask.bool() & tvalid.bool() & mark).squeeze(1)
        counts.append(int(chosen.sum()))
        if chosen.any():
            values.append((1-F.cosine_similarity(predicted,target,dim=1))[chosen].mean())
    if not values: raise ValueError('所有尺度都没有合法遮挡位置')
    loss=torch.stack(values).mean()
    if not torch.isfinite(loss): raise ValueError('恢复损失非有限')
    return loss,dict(masked_per_level=counts,levels_used=len(values),target_GT_used=False)


def unroll(loss_function, parameters, steps, lr, second_order):
    fast=OrderedDict(parameters)
    for step in range(steps):
        loss=loss_function(fast,step)
        gradients=torch.autograd.grad(loss,tuple(fast.values()),create_graph=second_order)
        fast=OrderedDict((name,p-lr*g) for (name,p),g in zip(fast.items(),gradients))
        if not second_order:
            fast=OrderedDict((name,p.detach().requires_grad_(True)) for name,p in fast.items())
    return fast


def detection_packet(model, encoder, video, parameters=None):
    x,valid,_=input_tensors(model,clean_video(video))
    features,masks=representations(encoder,x,valid,parameters)
    logits,_=model.cls_head_target(features,masks)
    offsets,_=model.reg_head_target(features,masks)
    return dict(fpn=features,masks=[m.squeeze(1) for m in masks],
        points=model.point_generator(features),logits=[v.permute(0,2,1) for v in logits],
        offsets=[v.permute(0,2,1) for v in offsets])


def detection_loss(model, packet, source):
    from libs.models.SADA.losses_aux import sigmoid_focal_loss,ctr_diou_loss_1d
    if not source['segments'].numel():
        # 合法源背景也保留分类项，不能丢掉负例。
        pass
    labels,offsets=model.label_points(packet['points'],[source['segments'].to(model.device)],
        [source['labels'].to(model.device)])
    valid=torch.cat(packet['masks'],1);labels=torch.stack(labels);offsets=torch.stack(offsets)
    logits=torch.cat(packet['logits'],1);prediction=torch.cat(packet['offsets'],1)
    positive=(labels.sum(-1)>0)&valid; count=int(positive.sum())
    cls=sigmoid_focal_loss(logits[valid],labels[valid],reduction='sum',alpha=.25,
        gamma=model.cfg['model'].get('focal_loss_gamma',2.))/max(count,1)
    reg=ctr_diou_loss_1d(prediction[positive],offsets[positive],reduction='sum')/count if count else prediction.sum()*0
    weight=model.train_loss_reg_weight
    if weight<=0: weight=cls.detach()/max(float(reg.detach()),.01)
    return cls+weight*reg


def episode(model, reference, encoder, decoder, source, arm, offset=0):
    if arm not in SPEC['arms']: raise ValueError('未知准备组')
    clean=clean_video(source); targets=reference_targets(reference,clean)
    def auxiliary(fast,step):
        return recovery_loss(model,encoder,decoder,clean,targets,fast,offset+step)[0]
    named=OrderedDict(encoder.named_parameters())
    if arm=='meta':
        fast=unroll(auxiliary,named,SPEC['inner_steps'],SPEC['inner_lr'],True)
        loss=detection_loss(model,detection_packet(model,encoder,clean,fast),source)
    else:
        # 对照同样计算五个mask上的恢复，但不把恢复更新嵌入检测损失。
        auxiliary_mean=torch.stack([auxiliary(named,i) for i in range(SPEC['inner_steps'])]).mean()
        loss=detection_loss(model,detection_packet(model,encoder,clean),source)+auxiliary_mean
    return loss


def fast_adapt(model, reference, encoder, decoder, video, steps):
    check_clean(video)
    named=OrderedDict((n,p.detach().clone().requires_grad_(True)) for n,p in encoder.named_parameters())
    if not steps: return named
    targets=reference_targets(reference,video)
    def auxiliary(fast,step):
        return recovery_loss(model,encoder,decoder,video,targets,fast,step)[0]
    return unroll(auxiliary,named,steps,SPEC['inner_lr'],False)


def predict(model, reference, encoder, decoder, video, steps):
    from tools.tta_core import raw_prediction,postprocess
    check_clean(video)
    # functional fast weights保证视频之间零参数/优化器状态传播，即使预测失败也不污染初值。
    with torch.enable_grad(): fast=fast_adapt(model,reference,encoder,decoder,video,steps)
    with torch.no_grad(): packet=detection_packet(model,encoder,video,fast)
    return postprocess(model,video,raw_prediction(model,packet))


def choose_checkpoint(history):
    if not history or history[0]['epoch']!=0: raise ValueError('必须包含epoch0')
    if [x['epoch'] for x in history]!=list(range(0,history[-1]['epoch']+1,SPEC['validate_every'])):
        raise ValueError('源验证历史不连续')
    if any(not math.isfinite(x['map']) or not 0<=x['map']<=1 for x in history):
        raise ValueError('源mAP非法')
    best=max(history,key=lambda row:(row['map'],-row['epoch']))
    stop=history[-1]['epoch']-best['epoch']>=SPEC['patience_epochs'] or history[-1]['epoch']>=SPEC['max_epochs']
    return best,stop


def choose_steps(candidates):
    if {r['steps'] for r in candidates}!=set(SPEC['calibration_steps']) or len(candidates)!=3:
        raise ValueError('源校准预算不完整')
    for row in candidates:
        if set(row['source_maps'])!=set('S%d'%i for i in range(1,7)):
            raise ValueError('必须六方向完整源验证')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in row['source_maps'].values()):
            raise ValueError('源校准mAP非法')
    return max(candidates,key=lambda r:(sum(r['source_maps'].values())/6,-r['steps']))
