import torch
from tools.uncertainty_dual_head import EndpointHead,point_context,objective,fuse

def test_real_neighbours_and_padding():
    f=[torch.tensor([[[1.,2.,3.]]])];m=[torch.ones(1,1,3,dtype=torch.bool)]
    x,valid,levels=point_context(f,m)
    assert torch.equal(x[:,:5],torch.tensor([[0.,0.,1.,2.,3.],[0.,1.,2.,3.,0.],[1.,2.,3.,0.,0.]]))
    assert valid.all() and levels.tolist()==[0,0,0]

def test_positive_offsets_source_gradient_and_restore():
    torch.manual_seed(123);h=EndpointHead(2,1);x=torch.randn(16,11);x[:,-1]=1;gt=torch.ones(16,2)*2
    loss,b,s=objective(h,x,gt);loss.mean().backward()
    assert (b>0).all() and (s>0).all() and torch.isfinite(loss).all()
    assert sum(p.grad.abs().sum() for p in h.parameters())>0
    copy=EndpointHead(2,1);copy.load_state_dict(h.state_dict(),strict=True)
    assert torch.equal(copy(x)[0],b)

def test_endpoint_uncertainty_independent_and_background_identity():
    a=torch.tensor([[2.,2.],[2.,2.]]);b=torch.tensor([[3.,3.],[3.,3.]])
    scales=torch.tensor([[.01,100.],[.01,100.]])
    cal=dict(a_scale=torch.ones(1,2),b_ratio=torch.ones(1,2))
    out,w=fuse(a,b,scales,torch.zeros(2,dtype=torch.long),cal,torch.tensor([1.,0.]))
    assert w[0,0]>w[0,1] and torch.equal(out[1],a[1]) and (out>0).all()

def test_source_pretrained_three_tap_to_five_tap_mapping():
    torch.manual_seed(123);conv=torch.nn.Conv1d(2,2,3,padding=1)
    f=[torch.randn(1,2,10)];mask=[torch.ones(1,1,10,dtype=torch.bool)]
    h=EndpointHead(2,1);h.initialize(conv.weight,conv.bias,torch.tensor([1.2]))
    x,_,_=point_context(f,mask)
    expected=(1.2*conv(f[0])[0].T).relu().clamp_min(1e-6)
    assert torch.allclose(h(x)[0],expected,atol=1e-6)
    assert h.offset.weight.data_ptr()!=conv.weight.data_ptr()


def test_masked_tower_bias_preserved_at_valid_neighbour():
    from types import SimpleNamespace
    class Masked(torch.nn.Module):
        def forward(self,x,m):return x*m,m
    class Bias(torch.nn.Module):
        def forward(self,x):return x+.3
    reg=SimpleNamespace(head=[Masked()],norm=[Bias()],act=torch.nn.ReLU())
    f=[torch.ones(1,2,4)];mask=[torch.tensor([[[True,True,False,False]]])]
    conv=torch.nn.Conv1d(2,2,3,padding=1);h=EndpointHead(2,1)
    h.initialize(conv.weight,conv.bias,torch.ones(1));x,valid,_=point_context(f,mask,reg)
    expected=conv((f[0]*mask[0]+.3).relu())[0].T.relu().clamp_min(1e-6)
    assert torch.allclose(h(x)[0][valid],expected[valid],atol=1e-6)


def test_initialization_candidate_and_first_early_stop():
    import pytest
    from tools.uncertainty_dual_head import audit_history
    h=[dict(epoch=i,source_validation=1.+i*.1) for i in range(16)]
    assert audit_history(h)==(0,15)
    with pytest.raises(ValueError):audit_history(h[:-1])
    with pytest.raises(ValueError):audit_history(h+[dict(epoch=16,source_validation=3.)])
