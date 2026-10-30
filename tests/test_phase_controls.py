"""验证干预只改角色信息，及其独立随机控制状态可恢复。"""
import copy,subprocess
import pytest
import torch
from torch import nn
from libs.models.SADA.phase_alignment import (balanced_domain_losses,conditioned,
    permute_role_vectors,model_role_permuter,control_specification,canonical_initial_state)
from tools import baseline_protocol as p


def q_values():
    return torch.tensor([[1.,0.,0.],[.5,.5,0.],[0.,1.,0.],[0.,.5,.5],[0.,0.,1.],[.2,.6,.2]])


def test_semantic_helper_keeps_previous_loss_and_gradients():
    old_source=subprocess.check_output(['git','show','f71c0e146ba43501330dd1e5112398e9b76559e6:libs/models/SADA/phase_alignment.py'],cwd=p.CODE,text=True)
    old={};exec(compile(old_source,'previous_phase_alignment.py','exec'),old)
    torch.manual_seed(17);source=torch.randn(6,4);target=torch.randn(6,4)
    disc=nn.Linear(12,1);q=q_values();results=[]
    for function in [old['balanced_domain_losses'],balanced_domain_losses]:
        s=source.clone().requires_grad_();t=target.clone().requires_grad_()
        a,b,n=function(disc,s,t,q,q.flip(0));grads=torch.autograd.grad(a+b,(s,t))
        results.append((a,b,n,grads))
    a,b=results
    assert a[2]==b[2] and torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    assert all(torch.equal(x,y) for x,y in zip(a[3],b[3]))


def test_blind_preserves_weight_rule_and_removes_discriminator_role_input():
    torch.manual_seed(18);s=torch.randn(6,4,requires_grad=True);t=torch.randn(6,4,requires_grad=True)
    disc=nn.Linear(12,1);qs=q_values();qt=qs.flip(0);stats={}
    a,b,n=balanced_domain_losses(disc,s,t,qs,qt,role_mode='role_blind',statistics=stats)
    roles=((qs.sum(0)>=1)&(qt.sum(0)>=1)).nonzero().flatten()
    ws=sum(qs[:,r]/qs[:,r].sum() for r in roles)/len(roles)
    wt=sum(qt[:,r]/qt[:,r].sum() for r in roles)/len(roles)
    ps=disc(conditioned(s,torch.full((6,3),1/3))).flatten()
    pt=disc(conditioned(t,torch.full((6,3),1/3))).flatten()
    expected_s=(torch.nn.functional.binary_cross_entropy_with_logits(ps,torch.zeros_like(ps),reduction='none')*ws).sum()
    expected_t=(torch.nn.functional.binary_cross_entropy_with_logits(pt,torch.ones_like(pt),reduction='none')*wt).sum()
    torch.testing.assert_close(a,expected_s);torch.testing.assert_close(b,expected_t)
    assert n==len(roles) and stats['nonuniform_input']==0 and stats['condition_calls']==2*n
    semantic=balanced_domain_losses(disc,s,t,qs,qt)
    assert not torch.allclose(a+b,semantic[0]+semantic[1])
    (a+b).backward();assert s.grad.abs().sum()>0 and t.grad.abs().sum()>0


def test_permutation_preserves_vectors_mass_support_and_original_rng():
    q=q_values();rng=torch.get_rng_state().clone()
    actual,counter=permute_role_vectors(q,123,8)
    assert counter==9 and torch.equal(rng,torch.get_rng_state())
    assert sorted(map(tuple,actual.tolist()))==sorted(map(tuple,q.tolist()))
    torch.testing.assert_close(actual.sum(0),q.sum(0))
    assert not torch.equal(actual,q)
    replay,_=permute_role_vectors(q,123,8);assert torch.equal(actual,replay)
    model=type('Model',(),dict(cfg={'seed':123},phase_permutation_counter=8))()
    s=torch.randn(6,4);t=torch.randn(6,4);stats={}
    result=balanced_domain_losses(nn.Linear(12,1),s,t,q,q.flip(0),role_mode='role_permuted',
        permuter=model_role_permuter(model),statistics=stats)
    assert model.phase_permutation_counter==10 and result[2]==3
    assert stats['permutation_l1']>0 and stats['nonuniform_input']>0


@pytest.mark.parametrize('mode',['role_blind','role_permuted'])
def test_true_sada_controls_preserve_source_and_initial_state_and_restore_counter(tmp_path,monkeypatch,mode):
    from runpy import run_path
    from libs.models.SADA.meta_archs import SADA
    from tools.baseline_engine import controls
    helper=run_path(str(p.CODE/'tests/test_prototype_capacity.py'))
    cfg=run_path(str(p.CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['seed']=123;cfg['model']['pl_threshold']=0.;cfg['prototype_capacity']=helper['configuration'](tmp_path,'control')
    from libs.models.SADA.phase_alignment import SCHEMA,POLICY
    cfg['phase_alignment']=dict(schema=SCHEMA,policy=POLICY)
    native=torch.tensor
    def cpu_copy(*a,**kw):
        value=native(*a,**kw);return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    torch.manual_seed(123);base=SADA(copy.deepcopy(cfg),'cpu');rng=torch.get_rng_state().clone()
    modified=copy.deepcopy(cfg);modified['phase_control']=control_specification(mode)
    torch.manual_seed(123);model=SADA(modified,'cpu')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    for m in [base,model]:
        with torch.no_grad():m.cls_head.cls_head.conv.bias.fill_(-10);m.cls_head.cls_head.conv.bias[0]=10
    source,target=helper['videos']();losses=[]
    for m in [base,model]:
        torch.manual_seed(456);losses.append(m(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0])
    for name in ['cls_loss','reg_loss','rq3_loss','downstream_loss']:assert torch.equal(losses[0][name],losses[1][name])
    (losses[1]['downstream_loss']+losses[1]['loss_conditional_adv_rev']).backward()
    assert model.disc_dom_conditional_adv[0].initial_layers[0].weight.grad.abs().sum()>0
    state=controls(model);assert 'phase_permutation_counter' in state
    restored=SADA(modified,'cpu');restored.load_state_dict(model.state_dict(),strict=True)
    restored.phase_permutation_counter=state['phase_permutation_counter']
    if mode=='role_permuted':
        assert state['phase_permutation_counter']>0
        q=q_values();expected=model_role_permuter(model)(q);actual=model_role_permuter(restored)(q)
        assert torch.equal(expected,actual) and model.phase_permutation_counter==restored.phase_permutation_counter


def test_control_config_cannot_change_sampling_or_loss_weight():
    from tools.phase_control_protocol import assert_pair
    base=dict(seed=123,loader={'batch_size':2},baseline=dict(code={},tmpdir='/old',target_gt_used=False),model={'weight':.1})
    cfg=copy.deepcopy(base);cfg['baseline'].update(code={'commit':'new'},tmpdir='/new');cfg['phase_control']=control_specification('role_blind')
    assert_pair(cfg,base)
    cfg['model']['weight']=.2
    with pytest.raises(ValueError):assert_pair(cfg,base)
