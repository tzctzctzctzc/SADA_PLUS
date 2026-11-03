import numpy as np
import torch
from tools.style_robust_math import mix_style,StyledEvidence,StyleDraws,group_update
from tools.dense_revisit_math import DenseEvidence


def test_style_stats_and_detached_partner():
    torch.manual_seed(2);x=torch.randn(31,8,requires_grad=True);p=(torch.randn(23,8)*3+5).requires_grad_(True);z=mix_style(x,p,.25)
    a=x.detach().numpy();b=p.detach().numpy();mu=a.mean(0);sig=np.sqrt(a.var(0,ddof=1)+1e-6);om=b.mean(0);os=np.sqrt(b.var(0,ddof=1)+1e-6)
    ref=(a-mu)/sig*(.25*sig+.75*os)+(.25*mu+.75*om)
    assert np.allclose(z.detach().numpy(),ref,atol=2e-6)
    z.sum().backward();assert p.grad is None and torch.isfinite(x.grad).all()
    assert torch.allclose(z.mean(0),.25*x.mean(0)+.75*p.mean(0),atol=2e-6)


def test_disabled_network_and_rng_exact():
    torch.manual_seed(1);a=DenseEvidence();b=StyledEvidence();b.load_state_dict(a.state_dict());x=torch.randn(19,2304)
    assert torch.equal(a(x)[0],b(x)[0])
    g=StyleDraws();state=g.state();draws=[g.draw() for _ in range(4)];g.restore(state);assert draws==[g.draw() for _ in range(4)]


def test_dro_closed_form():
    q=torch.tensor([.5,.5]);loss=torch.tensor([.1,.8]);out=group_update(q,loss)
    ref=q*torch.exp(.01*loss);ref/=ref.sum();assert torch.equal(out,ref) and out[1]>out[0]


def test_training_entrypoint_import():
    from tools.run_style_robust import SPEC
    assert len(SPEC['modes'])==4 and SPEC['target_train_GT_read'] is False
