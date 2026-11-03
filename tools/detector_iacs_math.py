"""将源定位IoU直接作为原分类头目标；不再另训不参与检测分数的BG/质量小头。"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from tools.native_sada_math import NativeSADA
from tools.background_four_routes_math import source_symbols


def restore_optimizer(optimizer,state):
    # AdamW的CPU step标量可能与传入state共享存储，恢复分支必须各自持有副本。
    optimizer.load_state_dict(deepcopy(state))


@lru_cache(None)
def original_varifocal():
    def reduce(loss,weight=None,reduction='mean',avg_factor=None):
        if weight is not None:loss=loss*weight
        if avg_factor is not None:return loss.sum()/avg_factor
        return loss if reduction=='none' else loss.sum() if reduction=='sum' else loss.mean()
    ns=dict(F=F,weight_reduce_loss=reduce,mmcv=SimpleNamespace(jit=lambda **kwargs:lambda fn:fn))
    source_symbols(Path(__file__).with_name('iacs_reference')/'varifocal_loss.py',['varifocal_loss'],ns)
    return ns['varifocal_loss']


def iou_targets(classes,offsets,predicted,ap_aligned=False):
    # 每个anchor相同中心，offset均为同层stride单位；质量只作detach监督。
    intersection=torch.minimum(offsets[:,0],predicted[:,0])+torch.minimum(offsets[:,1],predicted[:,1])
    union=offsets.sum(-1)+predicted.sum(-1)-intersection
    quality=(intersection/union.clamp_min(1e-8)).clamp(0,1).detach()
    if ap_aligned:
        quality=(quality[:,None]>=quality.new_tensor([.1,.2,.3,.4,.5])[None]).to(quality.dtype).mean(-1)
    return classes.detach()*quality[:,None]


class QualityDetector(NativeSADA):
    def __init__(self,detector):
        super().__init__(detector);del self.head
        self.reg_head=detector.reg_head;self.cls_head.requires_grad_(True);self.reg_head.requires_grad_(True)
        for name in ('loss_normalizer','loss_normalizer_pseudo_labels','loss_normalizer_momentum','train_label_smoothing','train_loss_reg_weight','cls_loss_type','use_trident_head','num_bins','iou_weight_power'):
            if hasattr(detector,name):setattr(self,name,deepcopy(getattr(detector,name)))
        assert not self.use_trident_head
        for name,value in vars(detector).items():
            if name.startswith('test_'):setattr(self,name,deepcopy(value))
        from libs.models.SADA.meta_archs import SADA
        for name in ('losses','decode_offset','inference','inference_single_video','postprocessing'):setattr(self,name,getattr(SADA,name).__get__(self))

    def forward_heads(self,raw,metadata):
        f,m=self.features(raw,metadata);cls,_=self.cls_head([x[None] for x in f],[x[None,None] for x in m]);reg,act=self.reg_head([x[None] for x in f],[x[None,None] for x in m])
        return f,m,[x[0].T for x in cls],[x[0].T for x in reg],[x[0].T for x in act]

    def source_loss(self,features,masks,cls,reg,actionness,segments,labels,quality):
        points=self.point_generator([x[None] for x in features]);gcls,goff=self.label_points(points,[segments],[labels]);lengths=[len(x) for x in cls]
        source_classes=list(gcls[0].split(lengths));source_offsets=list(goff[0].split(lengths))
        # 直接调用原源域检测损失/归一化/DIoU，暂关闭其中DA，另用已核的共享FPN条件分支。
        modes=self.cfg['model']['adv_method'];self.cfg['model']['adv_method']=['None']
        try:
            loss=self.losses([x[None].expand(2,-1) for x in masks],[torch.stack((x,x.detach())) for x in cls],[torch.stack((x,x.detach())) for x in reg],[torch.stack((x,x.detach())) for x in actionness],
                [gcls[0],torch.zeros_like(gcls[0])],[goff[0],torch.zeros_like(goff[0])],None,None,[],None,[x[None] for x in source_classes],mode='train',domain='src',curr_epoch=0)
        finally:self.cfg['model']['adv_method']=modes
        valid=torch.cat(masks);logits=torch.cat(cls)[valid];pred=torch.cat(reg)[valid]
        targets=iou_targets(gcls[0][valid],goff[0][valid],pred,ap_aligned=True)
        vfl=original_varifocal()(logits,targets,alpha=.75,gamma=2.,iou_weighted=True,reduction='sum')/self.loss_normalizer
        reg_weight=self.train_loss_reg_weight if self.train_loss_reg_weight>0 else loss['cls_loss'].detach()/max(loss['reg_loss'].item(),.01)
        total=(vfl if quality else loss['cls_loss'])+reg_weight*loss['reg_loss']
        if not quality:assert torch.equal(total,loss['downstream_loss'])
        return total,dict(cls=vfl if quality else loss['cls_loss'],reg=loss['reg_loss'],focal=loss['cls_loss'],positive_anchors=int((gcls[0].sum(-1)>0)[valid].sum()),quality_target_sum=targets.sum().detach()),source_classes

    def decode(self,raw,metadata):
        f,m,c,r,_=self.forward_heads(raw,metadata);video=[dict(feats=raw.T,segments=raw.new_empty((0,2)),labels=torch.empty(0,dtype=torch.long,device=raw.device),metadata=metadata)]
        return self.inference(video,self.point_generator([x[None] for x in f]),[x[None] for x in m],[x[None] for x in c],[x[None] for x in r],None,None)[0]


@torch.no_grad()
def ema_step(ema,model,decay=.999):
    # 规范化模块没有source/target参数别名，避免同一权重被EMA更新两次。
    a=ema.state_dict();b=model.state_dict()
    for name,value in a.items():
        if value.is_floating_point():value.mul_(decay).add_(b[name],alpha=1-decay)
        else:value.copy_(b[name])
