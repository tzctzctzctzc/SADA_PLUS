import copy
import importlib.util
from pathlib import Path
import sys
import torch
import pytest

from test_rq3_revised import MODULE as OLD, prototype_data
from tools.rq3_enhancement import ORDER, specification, validate, assert_reference_pair

spec = importlib.util.spec_from_file_location('rq3_revised_cpu.rq3_enhancement',
    Path(__file__).resolve().parents[1]/'libs/models/SADA/rq3_enhancement.py')
NEW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NEW)


def build(name):
    from test_rq3_revised import entry
    return NEW.EnhancedPrototype(4,10,{'v':[dict(entry(),participation_selected=True)]},[1],[(0,99)],0.,specification(name))


@pytest.mark.parametrize('name',ORDER)
def test_real_branch_backward_target_gt_isolation_and_strict_restore(name):
    torch.set_num_threads(1)
    torch.manual_seed(6)
    before=torch.random.get_rng_state().clone()
    model=build(name)
    assert torch.equal(before,torch.random.get_rng_state())
    data=prototype_data()
    data[1][0]['labels']=object()
    data[1][0]['segments']=object()
    result=model(*data,epoch=5)
    assert torch.isfinite(result['loss']) and result['target_gradient_norm']>0
    result['loss'].backward()
    assert data[2][0].grad.abs().sum()>0 and data[3][0].grad.abs().sum()>0
    other=build(name)
    other.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),other.state_dict().values()))


def test_control_exact_old_state_and_loss():
    from test_rq3_revised import entry
    old=OLD.RQ3ProposalPrototype(4,10,{'v':[dict(entry(),participation_selected=True)]},[1],[(0,99)],0.)
    new=build('control')
    assert all(torch.equal(old.state_dict()[k],v) for k,v in new.state_dict().items())
    torch.manual_seed(6)
    data=prototype_data()
    a=old(*data,epoch=5)
    b=new(*data,epoch=5)
    for k in a:
        assert torch.equal(torch.as_tensor(a[k]),torch.as_tensor(b[k]))


def test_parameter_counts_and_pool_scope():
    for name, expected in [('control',200576),('wide',663296),('deep',1190400)]:
        model=NEW.EnhancedPrototype(512,10,{},[1],[(0,99)],0.,specification(name))
        assert sum(p.numel() for p in model.parameters())==expected
    assert build('pool').branch.sample_ratios == build('control').branch.sample_ratios


def test_source_supcon_correct_pairs_and_no_positive_case():
    x=torch.nn.functional.normalize(torch.randn(6,12),dim=-1).requires_grad_()
    target=torch.nn.functional.one_hot(torch.tensor([0,0,1,1,2,3]),10).float()
    loss,n=NEW.source_supcon(x,target,.1)
    assert n==4 and torch.isfinite(loss)
    loss.backward()
    assert x.grad.abs().sum()>0
    loss,n=NEW.source_supcon(x[:2],torch.eye(10)[:2],.1)
    assert n==0 and loss==0


def test_multiple_prototypes_source_only_and_unit_length():
    model=build('multiprototype')
    data=prototype_data()
    model(*data,epoch=5)
    branch=model.branch
    assert branch.subprototype_ready[:2].any()
    norms=branch.subprototypes.norm(dim=-1)[branch.subprototype_ready]
    assert torch.allclose(norms,torch.ones_like(norms),atol=1e-6)
    assert not branch.subprototypes.requires_grad


def test_warmup_reaches_frozen_weight_and_no_target_gradient_before_start():
    model=build('warmup')
    for epoch,expected in [(0,0.),(1,.02),(5,.1),(50,.1)]:
        out=model(*prototype_data(),epoch=epoch)
        assert abs(out['alignment_weight']-expected)<1e-8
        assert (out['target_gradient_norm']>0)==(epoch>0)


def test_spec_tampering_and_nonbranch_changes_rejected():
    cfg={'rq3_enhancement':specification('deep')}
    validate(cfg)
    cfg['rq3_enhancement']['embedding_dim']=512
    with pytest.raises(ValueError):
        validate(cfg)
    reference={'seed':6,'loader':{'batch_size':2},'rq3_auxiliary':{'selected_coordinates':[['v',2]]}}
    cfg=copy.deepcopy(reference)
    cfg['rq3_enhancement']=specification('wide')
    assert_reference_pair(cfg,reference)
    cfg['rq3_auxiliary']['selected_coordinates']=[['v',3]]
    with pytest.raises(ValueError):
        assert_reference_pair(cfg,reference)
