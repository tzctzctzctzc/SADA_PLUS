from copy import deepcopy
import torch
from libs.models.SADA.prototype_capacity import VLMProposalCoAdaptationBranch
from libs.models.SADA.prototype_temporal_roles import TemporalRoleBranch, sample_masked_tokens


def test_initial_all_valid_matches_original_and_preserves_shared_parameters():
    torch.manual_seed(0); old=VLMProposalCoAdaptationBranch(8,10)
    torch.manual_seed(0); new=TemporalRoleBranch(8,10)
    for k,v in old.state_dict().items():
        assert torch.equal(v,new.state_dict()[k])
    f=[torch.randn(1,8,64)]
    p=[dict(segments=torch.tensor([[16.,40.]]),levels=torch.tensor([0]))]
    a=old.encode(f,p,[1]);b=new.encode_masked(f,p,[1],[torch.ones(1,1,64,dtype=torch.bool)])
    torch.testing.assert_close(a['embedding'],b['embedding'],rtol=1e-5,atol=1e-6)


def test_order_information_is_available_without_changing_multiset_or_mean():
    branch=TemporalRoleBranch(8,10)
    with torch.no_grad():
        branch.role_residual.weight.zero_()
        branch.role_residual.weight[:,:128].copy_(torch.eye(128))
    tokens=torch.randn(2,7,128);valid=torch.ones(2,7,dtype=torch.bool)
    a=branch.fuse_roles(tokens,valid);b=branch.fuse_roles(tokens.flip(1),valid)
    assert not torch.allclose(a['embedding'],b['embedding'])
    a=branch.fuse_roles(tokens,valid,True);b=branch.fuse_roles(tokens.flip(1),valid,True)
    torch.testing.assert_close(a['embedding'],b['embedding'],rtol=1e-5,atol=1e-6)


def test_invalid_tokens_and_absent_roles_do_not_create_information():
    branch=TemporalRoleBranch(8,10)
    a=torch.randn(2,7,8);mask=torch.tensor([[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]],dtype=torch.bool)
    b=a.clone();b[~mask]=123456.
    x=branch.encode_tokens(a,mask);y=branch.encode_tokens(b,mask)
    torch.testing.assert_close(x['embedding'],y['embedding'],rtol=0,atol=0)
    assert x['embedding'][1].count_nonzero()==0 and torch.isfinite(x['embedding']).all()
    tokens=torch.randn(1,7,128);valid=torch.tensor([[1,0,0,0,0,0,0]],dtype=torch.bool)
    branch.role_residual.weight.data.fill_(1.)
    out=branch.fuse_roles(tokens,valid)
    assert out['statistics']['role_residual_norm']==0


def test_selected_level_gradients_and_real_role_update_strict_restore():
    branch=TemporalRoleBranch(8,10)
    f0=torch.randn(1,8,64,requires_grad=True);f1=torch.randn(1,8,32,requires_grad=True)
    prop=[dict(segments=torch.tensor([[16.,40.]]),levels=torch.tensor([1]),targets=torch.nn.functional.one_hot(torch.tensor([2]),10).float())]
    masks=[torch.ones(1,1,64,dtype=torch.bool),torch.ones(1,1,32,dtype=torch.bool)]
    opt=torch.optim.AdamW(branch.parameters(),lr=.001,weight_decay=0)
    out=branch.encode_masked([f0,f1],prop,[1,2],masks);loss=branch.source_loss(out,prop)
    loss.backward();assert branch.role_residual.weight.grad.norm()>0
    assert f0.grad is None and f1.grad.norm()>0
    opt.step();assert branch.role_residual.weight.norm()>0
    restored=deepcopy(branch);restored.load_state_dict(branch.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in branch.state_dict().items())


def test_empty_and_real_masked_sampling_units():
    f=[torch.arange(64.).reshape(1,1,64)]
    p=[dict(segments=torch.tensor([[16.,40.]]),levels=torch.tensor([0]))]
    rows,valid=sample_masked_tokens(f,p,[2],[torch.ones(1,1,64,dtype=torch.bool)],[-.25,0,.25,.5,.75,1,1.25])
    torch.testing.assert_close(rows[0,:,0],torch.tensor([5.,8.,11.,14.,17.,20.,23.]))
    assert valid.all()
    rows,valid=sample_masked_tokens(f,[dict(segments=torch.empty(0,2),levels=torch.empty(0,dtype=torch.long))],[2],[torch.ones(1,1,64,dtype=torch.bool)],TemporalRoleBranch.sample_ratios)
    out=TemporalRoleBranch(1,10).encode_tokens(rows,valid)
    assert out['embedding'].shape==(0,128)


def test_true_sada_source_loss_detector_rng_optimizer_and_target_gradient(tmp_path,monkeypatch):
    from runpy import run_path
    from tools.baseline_protocol import CODE
    integration=run_path(str(CODE/'tests/test_prototype_capacity.py'))['test_true_sada_integration_keeps_detector_initialization_and_source_loss']
    integration(tmp_path,monkeypatch,'temporal_roles')
