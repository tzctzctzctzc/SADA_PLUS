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
