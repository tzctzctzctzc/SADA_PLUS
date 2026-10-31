import torch
from tools.uncertainty_dual_head import EndpointHead,point_context,objective,fuse

def test_real_neighbours_and_padding():
    f=[torch.tensor([[[1.,2.,3.]]])];m=[torch.ones(1,1,3,dtype=torch.bool)]
    x,valid,levels=point_context(f,m)
    assert torch.equal(x[:,:3],torch.tensor([[0.,1.,2.],[1.,2.,3.],[2.,3.,0.]]))
    assert valid.all() and levels.tolist()==[0,0,0]

def test_positive_offsets_source_gradient_and_restore():
    torch.manual_seed(123);h=EndpointHead(2,1);x=torch.randn(16,7);gt=torch.ones(16,2)*2
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
