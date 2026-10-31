import torch
from libs.models.SADA.boundary_distribution import (RelativeBoundaryDistribution, source_distribution_loss,
    absolute_boundary_support, wasserstein_1d, common_grid_indices, completely_visible, temporal_distribution_loss)


def test_initial_mean_preserves_original_distances_and_model_initialization_rng():
    torch.manual_seed(6);before=torch.get_rng_state().clone();head=RelativeBoundaryDistribution(4)
    assert torch.equal(before,torch.get_rng_state())
    base=torch.tensor([[[0.,.1,5.,100.],[.2,2.,20.,200.]]]);got=head(torch.ones(1,4,4),base)
    torch.testing.assert_close(got['offsets'],base)
    torch.testing.assert_close(got['probability'].sum(-1),torch.ones(1,2,4))


def test_source_ce_updates_head_then_features_and_does_not_move_label_bins():
    head=RelativeBoundaryDistribution(3);feature=torch.ones(1,3,2,requires_grad=True)
    base=torch.ones(1,2,2,requires_grad=True);truth=torch.ones_like(base)*2;positive=torch.ones(1,2,dtype=torch.bool)
    result=head(feature,base);loss=source_distribution_loss(result['logits'],base,truth,positive,head.factors)['loss'];loss.backward()
    assert base.grad is None and head.logits.weight.grad.abs().sum()>0
    with torch.no_grad():head.logits.weight-=.1*head.logits.weight.grad
    feature.grad=None;head.zero_grad();result=head(feature,base)
    source_distribution_loss(result['logits'],base,truth,positive,head.factors)['loss'].backward()
    assert feature.grad.abs().sum()>0


def test_no_source_positive_is_zero_loss_zero_grad_and_censoring_is_reported():
    head=RelativeBoundaryDistribution(2);base=torch.ones(1,2,2);logits=torch.zeros(1,2,2,17,requires_grad=True)
    got=source_distribution_loss(logits,base,base*100,torch.zeros(1,2,dtype=torch.bool),head.factors)
    assert got['loss']==0;got['loss'].backward();assert logits.grad.abs().sum()==0
    got=source_distribution_loss(logits,base,base*100,torch.ones(1,2,dtype=torch.bool),head.factors)
    assert got['upper_tail_fraction']==1


def test_wasserstein_point_masses_have_known_distance_and_permutation_invariance():
    x=torch.tensor([[1.,4.]]);p=torch.tensor([[1.,0.]])
    y=torch.tensor([[3.,8.]]);q=torch.tensor([[1.,0.]])
    torch.testing.assert_close(wasserstein_1d(x,p,y,q),torch.tensor([2.]))
    torch.testing.assert_close(wasserstein_1d(x.flip(-1),p.flip(-1),y,q),torch.tensor([2.]))


def test_equal_means_do_not_imply_equal_boundary_distributions():
    x=torch.tensor([[0.,2.]]);p=torch.tensor([[.5,.5]]);y=torch.tensor([[1.,1.]])
    torch.testing.assert_close(wasserstein_1d(x,p,y,p),torch.tensor([1.]))


def test_crop_coordinates_and_fpn_stride_are_both_required():
    factors=torch.tensor([.5,1.,1.5]);base=torch.tensor([[2.,3.]])
    a=absolute_boundary_support(base,torch.tensor([8.]),torch.tensor([4.]),factors,16)
    b=absolute_boundary_support(base,torch.tensor([24.]),torch.tensor([4.]),factors,0)
    torch.testing.assert_close(a,b)
    assert a[0,0,1]==16 and a[0,1,1]==36


def test_only_real_common_grid_points_match_without_interpolation_or_padding():
    a,b=common_grid_indices(torch.tensor([0.,4.,8.,12.]),torch.tensor([0.,4.,8.]),
        torch.tensor([True,True,True,False]),torch.tensor([True,True,True]),4,8)
    assert a.tolist()==[1,2] and b.tolist()==[0,1]
    a,b=common_grid_indices(torch.tensor([0.,4.]),torch.tensor([0.,4.]),torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.bool),0,1)
    assert a.numel()==b.numel()==0


def test_full_distribution_visibility_not_only_its_mean():
    support=torch.tensor([[[1.,4.],[6.,9.]],[[3.,4.],[6.,7.]]])
    assert completely_visible(support,0,10,2,6).tolist()==[False,True]


def test_temporal_loss_detaches_weak_view_but_updates_strong_view_and_handles_empty():
    weak=torch.tensor([[[1.,2.],[5.,6.]]],requires_grad=True);strong=(weak.detach()+1).requires_grad_()
    wp=torch.full_like(weak,.5,requires_grad=True);sp=torch.full_like(strong,.5,requires_grad=True)
    loss=temporal_distribution_loss(weak,wp,strong,sp,torch.tensor([True]),torch.tensor([4.]))
    torch.testing.assert_close(loss,torch.tensor(.25));loss.backward()
    assert weak.grad is wp.grad is None and strong.grad.abs().sum()>0 and sp.grad is not None
    strong.grad=None;loss=temporal_distribution_loss(weak,wp,strong,sp,torch.tensor([False]),torch.tensor([4.]))
    assert loss==0;loss.backward();assert strong.grad.abs().sum()==0


def test_real_sada_connects_source_loss_inference_optimizer_and_strict_restore(tmp_path,monkeypatch):
    import copy
    from pathlib import Path
    from runpy import run_path
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    from libs.models.SADA.boundary_equivariance import SCHEMA,POLICY
    code=Path(__file__).resolve().parents[1]
    cfg=run_path(str(code/'tests/test_corrected_baseline.py'))['small_config']()
    proto=run_path(str(code/'tests/test_prototype_capacity.py'))
    cfg['prototype_capacity']=proto['configuration'](tmp_path,'control');cfg['model']['pl_threshold']=0
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    torch.manual_seed(123);old=SADA(copy.deepcopy(cfg),'cpu');rng=torch.get_rng_state().clone()
    new_cfg=copy.deepcopy(cfg);new_cfg['boundary_equivariance']=dict(schema=SCHEMA,policy=copy.deepcopy(POLICY))
    torch.manual_seed(123);model=SADA(new_cfg,'cpu')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.state_dict().items())
    source,target=proto['videos']();results=[];states=[]
    for m in (old,model):
        torch.manual_seed(456)
        results.append(m(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0])
        states.append(torch.get_rng_state().clone())
    assert torch.equal(states[0],states[1])
    torch.testing.assert_close(results[0]['cls_loss'],results[1]['cls_loss'])
    torch.testing.assert_close(results[0]['reg_loss'],results[1]['reg_loss'])
    assert results[1]['boundary_source_distribution_loss']>0 and results[1]['boundary_target_rng_restored']==1
    head=model.boundary_equivariance.head.logits.weight;before=head.detach().clone()
    opt=make_optimizer(model,cfg['opt'],.0001,exclude_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    assert any(p is head for g in opt.param_groups for p in g['params'])
    opt.zero_grad();results[1]['downstream_loss'].backward();assert head.grad.abs().sum()>0;opt.step()
    assert not torch.equal(before,head)
    restored=SADA(copy.deepcopy(new_cfg),'cpu');restored.load_state_dict(model.state_dict(),strict=True)
    model.eval();old.eval()
    with torch.no_grad():
        output=model(video_list_src=[source],video_list_tgt=[target],mode='test')
    assert len(output[0])==len(output[1])==1
