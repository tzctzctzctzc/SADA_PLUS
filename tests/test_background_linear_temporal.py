import torch
from tools.background_linear_temporal import MODES,LinearBackground,descriptors,region_samples,temporal_descriptor,objective
from tools.proposal_foreground_math import sample_regions


def fixture():
    torch.manual_seed(17)
    return torch.randn(5,24),torch.ones(24,dtype=torch.bool),torch.tensor([[3.,19.],[-2.,3.],[22.,29.]]),torch.tensor([0,1,2]),torch.tensor([.3,.6,.9])


def test_max_and_low_capacity_metadata():
    f,v,b,c,s=fixture();d=descriptors(f,v,b,c,s)
    assert torch.equal(d['max_linear'][:,:18],sample_regions(f,v,b))
    assert torch.equal(d['max_linear'][:,-12:],d['temporal_abs'][:,-12:])
    assert sum(p.numel() for p in LinearBackground('max_linear').parameters())==3104
    assert sum(p.numel() for p in LinearBackground('temporal_abs').parameters())==4130
    assert sum(p.numel() for p in LinearBackground('temporal_signed').parameters())==4130


def test_reversal_isolates_direction():
    f,v,b,c,s=fixture();values,masks=region_samples(f,v,b)
    a=temporal_descriptor(values,masks,False);t=temporal_descriptor(values,masks,True)
    rv=[values[0],values[1].flip(1),values[2]];rm=[masks[0],masks[1].flip(1),masks[2]]
    # mean浮点求和顺序改变允许舍入误差；两分支的变化向量绝对值原本逐值相同。
    assert torch.allclose(a,temporal_descriptor(rv,rm,False),atol=1e-6)
    z=temporal_descriptor(rv,rm,True)
    assert torch.allclose(t[:,15:20],-z[:,15:20],atol=1e-6)
    assert torch.equal(t[:,15:20].abs(),a[:,15:20])


def test_invalid_and_empty():
    f,v,b,c,s=fixture();v.zero_();d=descriptors(f,v,b,c,s)
    for m in MODES:assert torch.isfinite(d[m]).all() and (d[m][:,:-12]==0).all()
    d=descriptors(f,v,b[:0],c[:0],s[:0])
    for m in MODES:assert d[m].shape[0]==0


def test_cached_gradients_and_update():
    f,v,b,c,s=fixture();d=descriptors(f,v,b,c,s)
    bg=torch.tensor([0.,1.,0.]);q=torch.tensor([.8,0.,.2])
    for m in MODES:
        a=LinearBackground(m,5);z=LinearBackground(m,5)
        oa=torch.optim.AdamW(a.parameters(),lr=3e-4);oz=torch.optim.AdamW(z.parameters(),lr=3e-4)
        la=objective(a(d[m]),bg,q);lz=objective(z(descriptors(f,v,b,c,s)[m]),bg,q)
        la.backward();lz.backward()
        for p,r in zip(a.parameters(),z.parameters()):assert torch.equal(p.grad,r.grad)
        oa.step();oz.step()
        for p,r in zip(a.parameters(),z.parameters()):assert torch.equal(p,r)
