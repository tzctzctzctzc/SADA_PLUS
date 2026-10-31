import torch
from tools.uncertainty_interval_energy import IntervalEnergy,candidates,select_interval,objective


def test_joint_grid_identity_legal_and_complete_interval_selection():
    b=torch.tensor([[2.,8.],[5.,9.]]);grid,delta=candidates(b)
    assert torch.equal(grid[:,4],b) and (grid[:,:,1]>grid[:,:,0]).all()
    scores=torch.zeros(2,9);valid=torch.ones(2,9,dtype=torch.bool)
    assert select_interval(scores,valid).tolist()==[4,4]
    scores[0,0]=-2;valid[0,0]=False;scores[0,8]=-1
    assert select_interval(scores,valid).tolist()==[8,4]
    valid[0,4]=False;assert select_interval(scores,valid)[0]==4


def test_energy_source_gradient_parameter_update_and_strict_restore():
    torch.manual_seed(123);h=IntervalEnergy(2,3);a=torch.randn(8,20);x=torch.randn(8,9,20);d=torch.randn(8,9,2);labels=torch.arange(8)%3
    q=torch.rand(8,9);valid=torch.ones(8,9,dtype=torch.bool);before=h.score[-1].weight.detach().clone()
    loss,energy=objective(h,a,x,d,labels,q,valid)
    assert select_interval(energy,valid).eq(4).all();loss.mean().backward();assert sum(float(p.grad.abs().sum()) for p in h.parameters() if p.grad is not None)>0
    opt=torch.optim.AdamW(h.parameters(),lr=.01);opt.step();assert not torch.equal(before,h.score[-1].weight)
    clone=IntervalEnergy(2,3);clone.load_state_dict(h.state_dict(),strict=True);assert torch.equal(h(a,x,d,labels),clone(a,x,d,labels))


def test_invalid_candidate_gets_no_quality_probability_or_gradient():
    torch.manual_seed(123);h=IntervalEnergy(2,1);a=torch.randn(2,20);x=torch.randn(2,9,20);d=torch.zeros(2,9,2);labels=torch.zeros(2,dtype=torch.long)
    q=torch.rand(2,9);valid=torch.ones(2,9,dtype=torch.bool);valid[:,0]=False
    loss,_=objective(h,a,x,d,labels,q,valid);q[:,0]=1000
    again,_=objective(h,a,x,d,labels,q,valid);assert torch.equal(loss,again) and torch.isfinite(loss).all()


def test_source_quality_ranking_can_learn_joint_peak():
    torch.manual_seed(123);h=IntervalEnergy(2,1);a=torch.randn(4,20);x=torch.randn(4,9,20)
    d=torch.linspace(-1,1,9)[None,:,None].repeat(4,1,2);labels=torch.zeros(4,dtype=torch.long)
    q=torch.zeros(4,9);q[:,8]=1;valid=torch.ones(4,9,dtype=torch.bool);opt=torch.optim.AdamW(h.parameters(),lr=.01)
    for _ in range(25):
        loss,_=objective(h,a,x,d,labels,q,valid);opt.zero_grad();loss.mean().backward();opt.step()
    assert select_interval(h(a,x,d,labels),valid).eq(8).all()


def test_pairwise_quality_ordering_sign_matches_map():
    class Fixed(torch.nn.Module):
        def __init__(self,energy):super().__init__();self.energy=energy
        def forward(self,*args):return self.energy
    q=torch.linspace(0,1,9)[None];valid=torch.ones(1,9,dtype=torch.bool)
    args=(torch.zeros(1,20),torch.zeros(1,9,20),torch.zeros(1,9,2),torch.zeros(1,dtype=torch.long),q,valid)
    good,_=objective(Fixed(-q),*args);bad,_=objective(Fixed(q),*args)
    assert good.item()<bad.item() and select_interval(-q,valid).item()==8
