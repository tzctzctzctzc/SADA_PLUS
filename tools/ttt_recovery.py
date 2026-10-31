"""时间遮挡恢复；GT只允许进入源域检测损失，恢复与测试适应只接无标签视频。"""
from copy import deepcopy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from tools.tta_core import (assert_unlabeled, unlabeled, dense, crop,
                            adaptable_parameters, parameter_digest, predict_full)

SPEC = dict(schema='temporal-feature-recovery-ttt-v1', seed=123,
    preparation_variants=['source_only','reconstruction','temporal_change'],
    preparation_epochs=5, preparation_batch=1, preparation_lr=1e-4,
    decoder_lr=1e-3, reconstruction_weight=1., change_weight=.1,
    mask_fraction=.3, block_tokens=16, level=0, decoder_hidden=256,
    adaptive_modules='last-shared-SGP-stem-block-and-shared-FPN-norms',
    reference='frozen-original-prototype-EMA', preparation_selection='full-source-val-mAP-best-of-epochs-1-to-5',
    test_optimizer='Adam-no-momentum-history-between-videos', learning_rates=[1e-4,1e-3], steps=[1,3],
    calibration_videos_per_shift=8, test_protocol='offline-whole-video-reset-each-video',
    target_GT_for_adaptation=False, preparation_target_data=False)


def seed_for(video_id, step):
    return int(hashlib.sha256(('123:%s:%s'%(video_id,step)).encode()).hexdigest()[:8],16)


def temporal_mask(length, seed, fraction=.3, block=16):
    """精确约30%有效token，连续块而非逐点白噪声；与动作标注无关。"""
    generator=torch.Generator().manual_seed(seed)
    blocks=list(range(0,length,block))
    order=torch.randperm(len(blocks),generator=generator).tolist()
    result=torch.zeros(length,dtype=torch.bool)
    wanted=max(1,min(length-1,round(length*fraction))) if length>1 else 1
    for index in order:
        begin=blocks[index];remaining=wanted-int(result.sum())
        if remaining<=0:break
        result[begin:min(length,begin+min(block,remaining))]=True
    return result


class RecoveryHead(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden=256):
        super().__init__()
        self.mask_token=nn.Parameter(torch.zeros(1,input_dim,1))
        self.decoder=nn.Sequential(nn.Conv1d(feature_dim+1,hidden,3,padding=1),nn.GELU(),
            nn.Conv1d(hidden,hidden,3,padding=2,dilation=2),nn.GELU(),nn.Conv1d(hidden,feature_dim,1))

    def forward(self, encoded, marked):
        return self.decoder(torch.cat((encoded,marked.to(encoded.dtype)),dim=1))


def encoded_view(model, video, head=None, marked=None):
    assert_unlabeled(video)
    x,valid=model.preprocessing([video],mode='test')
    position=torch.zeros_like(valid)
    if marked is not None:
        position[0,0,:len(marked)]=marked.to(position.device)
        # preprocessing本身为no_grad，因此可学习token在其后替换，且标记不改变有效性mask。
        x=torch.where(position,head.mask_token.to(x.dtype),x)
    feats,masks,_=model.backbone_target(model.dropout(x),valid)
    fpn,masks=model.neck_target(feats,masks)
    level=SPEC['level'];feature=fpn[level]
    marked_level=F.interpolate(position.float(),size=feature.shape[-1],mode='nearest').bool()
    return feature,masks[level].bool(),marked_level


def recovery_loss(student, reference, head, video, seed, change=False):
    assert_unlabeled(video)
    marked=temporal_mask(video['feats'].shape[-1],seed,SPEC['mask_fraction'],SPEC['block_tokens'])
    with torch.no_grad():
        target,valid,_=encoded_view(reference,video)
    encoded,visible_mask,position=encoded_view(student,video,head,marked)
    prediction=head(encoded,position)
    selected=(valid & visible_mask & position).squeeze(1)
    distance=1-F.cosine_similarity(prediction,target.detach(),dim=1)
    reconstruction=distance[selected].mean()
    delta=reconstruction*0
    adjacent=(valid[:,:,1:] & valid[:,:,:-1] & (position[:,:,1:] | position[:,:,:-1])).squeeze(1)
    if change and adjacent.any():
        pred_delta=F.normalize(prediction,dim=1)[:,:,1:]-F.normalize(prediction,dim=1)[:,:,:-1]
        target_normal=F.normalize(target.detach(),dim=1)
        target_delta=target_normal[:,:,1:]-target_normal[:,:,:-1]
        delta=(pred_delta-target_delta).square().mean(1)[adjacent].mean()
    loss=reconstruction+SPEC['change_weight']*delta
    return loss,dict(reconstruction=float(reconstruction.detach()),change=float(delta.detach()),
        masked_positions=int(selected.sum()),valid_positions=int(valid.sum()),target_gt_seen=False)


def source_detection_loss(model, video):
    """复用官方point分配/focal/DIoU，等价于原mode=val的source downstream loss。"""
    from libs.models.SADA.losses_aux import sigmoid_focal_loss,ctr_diou_loss_1d
    output=dense(model,unlabeled(video))
    target,offset=model.label_points(output['points'],[video['segments'].to(model.device)],
                                     [video['labels'].to(model.device)])
    valid=torch.cat(output['masks'],dim=1)
    labels=torch.stack(target);offset=torch.stack(offset)
    logits=torch.cat(output['logits'],dim=1);pred=torch.cat(output['offsets'],dim=1)
    positive=(labels.sum(-1)>0)&valid;count=int(positive.sum())
    classification=sigmoid_focal_loss(logits[valid],labels[valid],reduction='sum',alpha=.25,
        gamma=model.cfg['model'].get('focal_loss_gamma',2.))/max(count,1)
    regression=ctr_diou_loss_1d(pred[positive],offset[positive],reduction='sum')/count if count else pred.sum()*0
    weight=model.train_loss_reg_weight
    if weight<=0:weight=classification.detach()/max(float(regression.detach()),.01)
    return classification+weight*regression


def initialize(model):
    named=adaptable_parameters(model)
    with torch.no_grad():
        # 不消耗模型/数据随机流；各准备变体的恢复头具有相同初始化。
        with torch.random.fork_rng(devices=[0] if next(model.parameters()).is_cuda else []):
            torch.manual_seed(SPEC['seed'])
            head=RecoveryHead(model.cfg['model']['input_dim'],
                              model.cfg['model']['fpn_dim'],SPEC['decoder_hidden'])
    return named,head.to(next(model.parameters()).device)


def restore(named, initial):
    with torch.no_grad():
        for name,p in named.items():p.copy_(initial[name])


def apply_prepared(model, head, payload):
    named=adaptable_parameters(model)
    if set(payload['adaptable'])!=set(named):raise ValueError('准备checkpoint参数名单改变')
    restore(named,payload['adaptable']);head.load_state_dict(payload['head'],strict=True)


def adapt(model,reference,head,initial,video,lr,steps,change=False):
    assert_unlabeled(video)
    named=adaptable_parameters(model);restore(named,initial)
    head.eval()
    for p in head.parameters():p.requires_grad_(False)
    before=parameter_digest(named);trace=[]
    optimizer=torch.optim.Adam(list(named.values()),lr=lr)
    total=video['feats'].shape[-1];maximum=model.max_seq_len
    for step in range(steps):
        # 均匀覆盖长视频的不同窗口，无GT/候选阈值门控。
        start=0 if total<=maximum else round((total-maximum)*(step+.5)/steps)
        start=start//model.max_div_factor*model.max_div_factor
        window=crop(video,start,min(total,start+maximum))
        seed=seed_for(video['metadata']['video_id'],'test%d'%step)
        loss,metrics=recovery_loss(model,reference,head,window,seed,change)
        if not torch.isfinite(loss):raise ValueError('恢复loss非有限')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grads=[p.grad for p in named.values() if p.grad is not None]
        norm=torch.stack([g.norm() for g in grads]).norm() if grads else loss.new_zeros(())
        if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('恢复loss未产生有效共享梯度')
        torch.nn.utils.clip_grad_norm_(list(named.values()),1.);optimizer.step()
        with torch.no_grad():after,_=recovery_loss(model,reference,head,window,seed,change)
        trace.append(dict(step=step,loss=float(loss.detach()),loss_after=float(after),
                          gradient_norm=float(norm),window=[start,min(total,start+maximum)],**metrics))
    prediction=predict_full(model,video)
    return prediction,dict(initial_parameter_sha256=before,final_parameter_sha256=parameter_digest(named),
        changed=before!=parameter_digest(named),trace=trace,target_gt_seen=False)
