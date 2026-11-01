from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from tools.prototype_regression_gram import SPEC,install,balanced_indices,paired_config,assert_pair
from tools.regression_gram import gram_alignment


class MaskedConv(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.conv=nn.Conv1d(cin,cout,3,padding=1)
    def forward(self,x,m):return self.conv(x)*m,m


class Head(nn.Module):
    def __init__(self,output):
        super().__init__();self.head=nn.ModuleList([MaskedConv(2,2)]);self.offset_head=MaskedConv(2,output)
    def forward(self,features,masks):
        out=[]
        for f,m in zip(features,masks):
            f,_=self.head[0](f,m);f=torch.relu(f);f,_=self.offset_head(f,m);out.append(f+2)
        return out,[]


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Conv1d(2,2,1);self.reg_head=Head(2);self.cls_head=Head(1)
        self.reg_head_target=self.reg_head_pl=self.reg_head
        self.use_trident_head=False;self.device='cpu'
    def point_generator(self,f):return [torch.arange(x.shape[-1])[:,None] for x in f]
    def label_points(self,p,segments,labels):return [torch.ones(len(p[0]),1) for _ in labels],None
    def forward(self,video_list_src,video_list_tgt,mode,curr_epoch):
        sf=[self.backbone(torch.stack([v['feats'] for v in video_list_src]))]
        tf=[self.backbone(torch.stack([v['feats'] for v in video_list_tgt]))]
        mask=[torch.ones(sf[0].shape[0],1,sf[0].shape[-1],dtype=torch.bool)]
        for f in [sf,tf,tf]:self.cls_head(f,mask)
        offsets=[]
        for f in [sf,tf,tf]:offsets.append(self.reg_head(f,mask)[0][0])
        return (dict(downstream_loss=offsets[0].square().mean()),None,None,None,None,None,None)


def inputs():
    make=lambda shift:dict(feats=torch.randn(2,20)+shift,segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),metadata=dict(video_id=str(shift)))
    return [make(0),make(1)],[make(2),make(3)]


def config():
    return dict(regression_gram=deepcopy(SPEC),train_cfg=dict(warmup_adv=5),model=dict(loss_weights=dict(weight_downstream=.1)))


def test_balanced_point_selection_ignores_padding_and_is_round_robin():
    mask=torch.tensor([[1,1,1,1,0,0],[0,1,0,0,1,0]],dtype=torch.bool)
    rng=torch.get_rng_state().clone();v,p=balanced_indices(mask,3)
    assert torch.equal(v,torch.tensor([0,1,0,1,0]))
    assert mask[v,p].all() and len(p)==5 and torch.equal(rng,torch.get_rng_state())


def test_warmup_disabled_and_validation_paths_exactly_preserve_original():
    torch.manual_seed(123);model=Toy();s,t=inputs();kw=dict(video_list_src=s,video_list_tgt=t,mode='train',curr_epoch=1)
    expected=model(**kw)[0]['downstream_loss'];rng=torch.get_rng_state().clone();ctx=SimpleNamespace(model=model);install(ctx,config())
    result=model(**kw)[0];assert torch.equal(expected,result['downstream_loss']) and result['gram_events']==0
    assert torch.equal(rng,torch.get_rng_state())
    assert 'gram_events' not in model(**dict(kw,mode='validation'))[0]


def test_native_three_call_hooks_deliver_target_gradient_and_restore_counters():
    torch.manual_seed(123);model=Toy();ctx=SimpleNamespace(model=model);install(ctx,config());s,t=inputs()
    for v in t:v['feats'].requires_grad_()
    result=model(video_list_src=s,video_list_tgt=t,mode='train',curr_epoch=5)[0]
    assert result['gram_events']==1 and result['gram_target_fpn_gradient_norm']>0 and result['gram_regression_gradient_norm']>0
    result['downstream_loss'].backward()
    assert all(v['feats'].grad is not None and torch.isfinite(v['feats'].grad).all() for v in t)
    assert sum(float(v['feats'].grad.abs().sum()) for v in t)>0
    saved=deepcopy(model.regression_gram_state);model.regression_gram_state=deepcopy(saved)
    model(video_list_src=s,video_list_tgt=t,mode='train',curr_epoch=5)
    assert model.regression_gram_state['step']==saved['step']+1
    assert model.regression_gram_state['events']==saved['events']+1


def test_target_gt_rejected_before_original_forward():
    model=Toy();ctx=SimpleNamespace(model=model);install(ctx,config());s,t=inputs();t[0]['segments']=torch.ones(1,2)
    with pytest.raises(ValueError):model(video_list_src=s,video_list_tgt=t,mode='train',curr_epoch=5)


def test_normalized_gram_loss_is_invariant_to_repeated_rows():
    torch.manual_seed(14);a=torch.randn(20,6,dtype=torch.float64);b=torch.randn(20,6,dtype=torch.float64)*2
    x,_=gram_alignment(a,b,normalize=True);y,_=gram_alignment(a.repeat(2,1),b.repeat(2,1),normalize=True)
    assert torch.allclose(x,y,rtol=1e-9,atol=1e-9)


def test_full_config_pair_rejects_adversary_budget_and_prototype_changes():
    old=dict(seed=123,loader=dict(batch_size=2),prototype_capacity=dict(variant='control'),
             baseline=dict(code='old',tmpdir='old'),model=dict(adv_method=['conditional_rev']))
    new=paired_config(old,'new','tmp');assert_pair(new,old)
    new['model']['adv_method']=['None']
    with pytest.raises(ValueError):assert_pair(new,old)
