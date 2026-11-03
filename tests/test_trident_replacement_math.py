from types import SimpleNamespace
import torch
from libs.models.SADA.meta_archs import SADA,RegHead


def test_trident_distribution_decoder_matches_independent_loop_and_has_gradients():
    torch.manual_seed(13)
    bins,classes,steps=4,3,7
    offsets=torch.randn(1,steps,2*(bins+1),requires_grad=True)
    start=torch.randn(1,steps,classes,bins+1,requires_grad=True)
    end=torch.randn(1,steps,classes,bins+1,requires_grad=True)
    fake=SimpleNamespace(use_trident_head=True,device=torch.device('cpu'))
    actual=SADA.decode_offset(fake,[offsets],[start],[end],'train')
    expected=torch.empty(1,steps,classes,2)
    for t in range(steps):
        for c in range(classes):
            expected[0,t,c,0]=((offsets[0,t,:bins+1]+start[0,t,c]).softmax(0)*torch.arange(bins,-1,-1)).sum()
            expected[0,t,c,1]=((offsets[0,t,bins+1:]+end[0,t,c]).softmax(0)*torch.arange(bins+1)).sum()
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    actual.sum().backward()
    for value in (offsets,start,end):assert torch.isfinite(value.grad).all() and value.grad.norm()>0
    assert actual.min()>=0 and actual.max()<=bins


def test_bin_logits_preserve_sign_and_legacy_distances_stay_nonnegative():
    head=RegHead(4,4,1,num_layers=1,with_ln=False,num_bins=4)
    with torch.no_grad():
        head.offset_head.conv.weight.zero_();head.offset_head.conv.bias.fill_(-2)
    x=[torch.ones(1,4,8)];mask=[torch.ones(1,1,8,dtype=torch.bool)]
    legacy=head(x,mask)[0][0]
    head.distribution_offsets=True
    distributed=head(x,mask)[0][0]
    assert legacy.min()==0 and distributed.max()<0


def test_installer_accepts_original_string_device_without_disturbing_RNG():
    from copy import deepcopy
    from torch import nn
    from tools.head_replacements import install,SPEC
    for variant in ('trident','interaction'):
        model=nn.Module();model.device='cpu';model.cfg=dict(head_replacement=dict(variant=variant,specification=SPEC))
        model.use_trident_head=False;model.num_classes=3;model.train_cls_prior_prob=.01;model.train_head_empty_cls=[]
        model.reg_head=RegHead(4,4,1,num_layers=1,with_ln=False,num_bins=0)
        before=torch.get_rng_state().clone()
        install(model,4,4,3,1,False)
        assert torch.equal(before,torch.get_rng_state()) and model.head_replacement_base_sha
        if variant=='trident':assert model.use_trident_head and model.reg_head.distribution_offsets
        else:assert model.task_interaction is not None
