"""局部头部适配：Trident相对边界分布，或轻量分类/定位特征交互。

不是完整TOOD复现，不更换样本分配、focal、DIoU、SADA或Prototype。
Trident参考 https://github.com/dingfengshi/TriDet 。
"""
import torch
from torch import nn

SPEC = dict(schema='sada-prototype-head-replacement-v1', variants=['trident','interaction'],
    trident_bins=16, interaction_width=128, extra_loss=False, keep_original_focal=True,
    initialization_seed=11042026, target_GT=False, new_VLM_queries=0,
    trident='raw signed bin logits + class-dependent start/end relative boundary distributions',
    interaction='shared masked temporal stem, bidirectional task gates, zero-initialized residual outputs',
    references=['https://arxiv.org/abs/2303.07347','https://arxiv.org/abs/2108.07755'])


class TaskInteraction(nn.Module):
    def __init__(self, channels, width=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Conv1d(channels,width,1),nn.GELU(),
            nn.Conv1d(width,width,3,padding=1,groups=width),nn.GELU())
        self.cls_features=nn.Conv1d(width,width,1)
        self.reg_features=nn.Conv1d(width,width,1)
        self.cls_gate=nn.Conv1d(width*2,width,1)
        self.reg_gate=nn.Conv1d(width*2,width,1)
        self.cls_output=nn.Conv1d(width,channels,1,bias=False)
        self.reg_output=nn.Conv1d(width,channels,1,bias=False)
        nn.init.zeros_(self.cls_output.weight)
        nn.init.zeros_(self.reg_output.weight)

    def forward(self, features, masks):
        classification, regression=[],[]
        for x,mask in zip(features,masks):
            valid=mask.to(x.dtype)
            shared=self.shared(x*valid)*valid
            c,r=self.cls_features(shared)*valid,self.reg_features(shared)*valid
            both=torch.cat((c,r),1)
            dc=self.cls_output(c+self.cls_gate(both).sigmoid()*r)*valid
            dr=self.reg_output(r+self.reg_gate(both).sigmoid()*c)*valid
            # 零初始化时原有效位置逐位相等；padding继续屏蔽。
            classification.append(x+dc)
            regression.append(x+dr)
        return classification,regression


def install(model, channels, head_dim, kernel, layers, with_ln):
    from tools.baseline_engine import tensor_digest
    from libs.models.SADA.meta_archs import ClsHead
    from libs.models.SADA.blocks import MaskedConv1D
    cfg=model.cfg['head_replacement']
    assert cfg['specification']==SPEC and cfg['variant'] in SPEC['variants']
    # 构造新部件前，原模型的完整初始化必须与历史基线一致。
    model.head_replacement_base_sha=tensor_digest(model.state_dict())
    devices=[model.device.index or 0] if model.device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SPEC['initialization_seed'])
        if cfg['variant']=='trident':
            assert not model.use_trident_head
            model.use_trident_head=True
            model.num_bins=SPEC['trident_bins']
            model.start_head=ClsHead(channels,head_dim,model.num_classes,kernel_size=kernel,
                prior_prob=model.train_cls_prior_prob,with_ln=with_ln,num_layers=layers,
                empty_cls=model.train_head_empty_cls,detach_feat=True)
            model.end_head=ClsHead(channels,head_dim,model.num_classes,kernel_size=kernel,
                prior_prob=model.train_cls_prior_prob,with_ln=with_ln,num_layers=layers,
                empty_cls=model.train_head_empty_cls,detach_feat=True)
            model.reg_head.offset_head=MaskedConv1D(head_dim,2*(model.num_bins+1),kernel,
                                                  stride=1,padding=kernel//2)
            # 这里保留有符号的分布logits，不能沿用距离回归的ReLU。
            model.reg_head.distribution_offsets=True
        else:
            model.task_interaction=TaskInteraction(channels,SPEC['interaction_width'])


def head_inputs(model,features,masks):
    if getattr(model,'task_interaction',None) is None:
        return features,features
    return model.task_interaction(features,masks)
