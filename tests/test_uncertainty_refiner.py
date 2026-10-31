import torch
from tools.uncertainty_refiner import Refiner,context,objective,aligned_iou

def test_zero_residual_and_legal_duration():
    m=Refiner(2);b=torch.tensor([[1.,3.],[2.,9.]])
    p,s=m(torch.randn(2,20),b)
    assert torch.equal(p,b) and (s>0).all()
    with torch.no_grad():m.out.bias[:2].fill_(20)
    p,_=m(torch.randn(2,20),b)
    assert (p[:,1]>p[:,0]).all()

def test_masked_real_point_pool():
    f=[torch.arange(40.).view(1,2,20)];mask=[torch.ones(1,1,20,dtype=torch.bool)]
    a,v=context(f,mask,torch.tensor([[5.,10.]]),torch.tensor([0]),[1])
    assert v.all() and a.shape==(1,20)
    mask[0].zero_();a,v=context(f,mask,torch.tensor([[5.,10.]]),torch.tensor([0]),[1])
    assert not v.any() and a.count_nonzero()==0

def test_learns_known_signed_boundary_error_and_strict_restore():
    torch.manual_seed(123);m=Refiner(2);x=torch.randn(8,20)
    b=torch.tensor([[2.,8.]]).repeat(8,1);g=torch.tensor([[3.,9.]]).repeat(8,1)
    opt=torch.optim.AdamW(m.parameters(),lr=.01)
    before=aligned_iou(b,g).mean()
    for _ in range(50):
        loss,_,_=objective(m,x,b,g);opt.zero_grad();loss.backward();opt.step()
    _,pred,_=objective(m,x,b,g)
    assert torch.isfinite(loss) and aligned_iou(pred,g).mean()>before+.15
    clone=Refiner(2);clone.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(m(x,b)[0],clone(x,b)[0])

def test_gain_gate_reject_is_exact_identity():
    from tools.uncertainty_refiner import gated_boxes,GainGate,gate_input
    b=torch.tensor([[1.,3.],[2.,9.]]);p=b+1
    actual=gated_boxes(b,p,torch.tensor([-1.,1.]))
    assert torch.equal(actual[0],b[0]) and torch.equal(actual[1],p[1])
    model=Refiner(2);z,p=gate_input(model,torch.randn(2,20),b)
    assert z.shape==(2,260) and GainGate()(z).shape==(2,)
