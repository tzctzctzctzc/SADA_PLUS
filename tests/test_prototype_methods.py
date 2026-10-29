from copy import deepcopy
import json
from pathlib import Path
from runpy import run_path

import pytest
import torch
from torch.nn import functional as F

from libs.models.SADA.prototype_capacity import build_capacity,VLMProposalCoAdaptationBranch
from libs.models.SADA.prototype_methods import TemporalPoolingBranch,MultiPrototypeBranch,method_specification
from tools.baseline_protocol import CODE


helpers=run_path(str(CODE/'tests/test_prototype_capacity.py'))
configuration,videos=helpers['configuration'],helpers['videos']


@pytest.mark.parametrize('variant',['temporal_pool','multi_prototype'])
def test_real_shared_feature_gradients_optimizer_and_strict_state(tmp_path,variant):
    cfg=configuration(tmp_path,variant);torch.manual_seed(123);before=torch.get_rng_state().clone()
    model=build_capacity(cfg,8,10,[1],[[0,10000]],'cpu')
    assert torch.equal(before,torch.get_rng_state())
    assert model.branch.classifier.in_features==128 and len(model.branch.encoder.layers)==1
    s,t=videos();sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()]
    masks=[torch.ones(1,1,64,dtype=torch.bool)]
    out=model([s],[t],sf,tf,5,source_masks=masks,target_masks=masks)
    assert out['source_loss']>0 and out['alignment_loss']>0 and out['target_gradient_norm']>0
    original={k:v.detach().clone() for k,v in model.named_parameters()}
    opt=torch.optim.AdamW(model.parameters(),lr=.001);out['loss'].backward();opt.step()
    assert sf[0].grad.abs().sum()>0 and tf[0].grad.abs().sum()>0
    assert any(not torch.equal(original[k],v) for k,v in model.named_parameters())
    restored=build_capacity(cfg,8,10,[1],[[0,10000]],'cpu');restored.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())


def test_uniform_attention_initially_matches_old_mean_when_all_valid():
    torch.manual_seed(0);old=VLMProposalCoAdaptationBranch(8,10)
    torch.manual_seed(0);new=TemporalPoolingBranch(8,10)
    feats=[torch.randn(1,8,64)]
    prop=[dict(segments=torch.tensor([[15.,35.]]),levels=torch.tensor([0]))]
    a=old.encode(feats,prop,[1]);b=new.encode_masked(feats,prop,[1],[torch.ones(1,1,64,dtype=torch.bool)])
    torch.testing.assert_close(a['embedding'],b['embedding'],rtol=1e-5,atol=1e-6)
    assert b['statistics']['temporal_valid_tokens']==7


def test_padding_cannot_leak_into_attention_or_visual_embedding():
    model=TemporalPoolingBranch(8,10);feats=torch.randn(1,8,32)
    prop=[dict(segments=torch.tensor([[4.,14.]]),levels=torch.tensor([0]))]
    mask=torch.arange(32)[None,None,:]<16
    changed=feats.clone();changed[:,:,16:]=123456.
    a=model.encode_masked([feats],prop,[1],[mask]);b=model.encode_masked([changed],prop,[1],[mask])
    torch.testing.assert_close(a['embedding'],b['embedding'],rtol=0,atol=0)
    assert a['statistics']['temporal_context_valid']==1


def test_all_invalid_or_empty_context_is_finite_and_does_not_fake_points():
    model=TemporalPoolingBranch(8,10);features=[torch.randn(1,8,32)]
    prop=[dict(segments=torch.tensor([[4.,14.]]),levels=torch.tensor([0]))]
    out=model.encode_masked(features,prop,[1],[torch.zeros(1,1,32,dtype=torch.bool)])
    assert not out['valid'].any() and out['embedding'].count_nonzero()==0
    assert all(torch.isfinite(out[k]).all() for k in ['embedding','logits'])
    with pytest.raises(ValueError):model.encode_masked(features,prop,[1],None)


def test_temporal_stays_on_original_selected_pyramid_level():
    model=TemporalPoolingBranch(8,10)
    f0=torch.randn(1,8,64,requires_grad=True);f1=torch.randn(1,8,32,requires_grad=True)
    prop=[dict(segments=torch.tensor([[15.,35.]]),levels=torch.tensor([1]))]
    out=model.encode_masked([f0,f1],prop,[1,2],[torch.ones(1,1,64,dtype=torch.bool),torch.ones(1,1,32,dtype=torch.bool)])
    out['logits'].sum().backward()
    assert f0.grad is None and f1.grad.abs().sum()>0


def test_multimodal_source_centers_are_distinct_and_duplicates_do_not_fill_slots():
    model=MultiPrototypeBranch(8,10)
    x=F.normalize(torch.randn(1,128),dim=-1);gt=F.one_hot(torch.zeros(4,dtype=torch.long),10).float()
    model._update_source_prototypes(x.repeat(4,1),gt)
    assert model.source_prototype_ready[0].sum()==1
    vectors=torch.eye(128)[:3]
    model._update_source_prototypes(vectors,F.one_hot(torch.zeros(3,dtype=torch.long),10).float())
    assert model.source_prototype_ready[0].sum()==3
    centers=model.source_prototypes[0]
    assert (centers@centers.T-torch.eye(3)).abs().max()<1
    assert model.source_prototype_counts.sum()==7 and not model.source_prototype_ready[1:].any()


def test_multicenter_only_uses_vlm_selected_class_and_target_never_updates_centers():
    model=MultiPrototypeBranch(8,10)
    src=F.normalize(torch.randn(3,128),dim=-1);source=dict(embedding=src)
    gt=F.one_hot(torch.tensor([0,0,1]),10).float()
    model._update_source_prototypes(src,gt)
    target=F.normalize(torch.randn(1,128),dim=-1).requires_grad_()
    selection=dict(labels=torch.tensor([0]),selected=torch.tensor([True]))
    model.prototype_alignment(source,gt,dict(embedding=target),selection)
    before=model.source_prototypes.clone();counts=model.source_prototype_counts.clone()
    # 无源GT时改变目标向量，不能写入原型或创建目标所指的新类别。
    empty=src[:0];unknown=dict(labels=torch.tensor([5]),selected=torch.tensor([True]))
    loss,classes,_=model.prototype_alignment(dict(embedding=empty),gt[:0],dict(embedding=-target),unknown)
    assert classes==0 and loss==0 and torch.equal(before,model.source_prototypes)
    assert torch.equal(counts,model.source_prototype_counts) and not model.source_prototype_ready[5].any()
    loss,_,_=model.prototype_alignment(dict(embedding=empty),gt[:0],dict(embedding=target),selection)
    expected=1-(target[:,None]*before[0][None]).sum(-1).masked_fill(~model.source_prototype_ready[0][None],-2.).max()
    torch.testing.assert_close(loss,expected);loss.backward();assert target.grad.abs().sum()>0


@pytest.mark.parametrize('variant',['temporal_pool','multi_prototype'])
def test_actual_sada_optimizer_masks_initialization_and_detection_loss_unchanged(tmp_path,monkeypatch,variant):
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    native=torch.tensor
    def copy_leaf(*a,**kw):
        value=native(*a,**kw)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',copy_leaf)
    cfg=small();cfg['prototype_capacity']=configuration(tmp_path,variant)
    plain=deepcopy(cfg);plain.pop('prototype_capacity')
    torch.manual_seed(123);base=SADA(plain,'cpu')
    torch.manual_seed(123);model=SADA(cfg,'cpu')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    opt=make_optimizer(model,cfg['opt'],.0001,exclude_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    ids={id(p) for group in opt.param_groups for p in group['params']}
    assert all(id(p) in ids for p in model.rq3_capacity.parameters())
    s,t=videos()
    torch.manual_seed(777);a=base(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);b=model(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    assert torch.equal(a['cls_loss'],b['cls_loss']) and torch.equal(a['reg_loss'],b['reg_loss'])
    assert b['rq3_target_gradient_norm']>0
    torch.testing.assert_close(b['downstream_loss'],a['downstream_loss']+b['rq3_loss'])
    b['downstream_loss'].backward();assert model.rq3_capacity.branch.input_projection.weight.grad.abs().sum()>0


def test_acceptance_rejects_untrained_pooling_and_nominal_multicenters():
    from tools.run_prototype_methods import acceptance_check
    aux=dict(rq3_target_gradient_norm=.1,rq3_alignment_loss=.1,rq3_source_loss=.1,rq3_selected=2,
             rq3_target_temporal_context_valid=2,rq3_prototype_classes_with_multiple=1,rq3_prototype_source_updates=5)
    phase=dict(target_gt_losses=[0.,0.,0.],branch_parameters_updated=True,auxiliary=aux,
               pool_submodules_updated=dict(inside_attention=True,outside_attention=True,pool_fusion=True))
    report=dict(status='passed',rng_roundtrip={'status':'passed'},initial_detector_sha='x',phases={'1':phase,'5':deepcopy(phase)})
    for variant in ['temporal_pool','multi_prototype']:acceptance_check(report,variant,'x')
    bad=deepcopy(report);bad['phases']['5']['pool_submodules_updated']['outside_attention']=False
    with pytest.raises(ValueError):acceptance_check(bad,'temporal_pool','x')
    bad=deepcopy(report);bad['phases']['5']['auxiliary']['rq3_prototype_classes_with_multiple']=0
    with pytest.raises(ValueError):acceptance_check(bad,'multi_prototype','x')


def test_methods_remain_independent_original_capacity_and_original_sampling():
    a,b=[method_specification(v) for v in ['temporal_pool','multi_prototype']]
    for s in [a,b]:
        assert (s['embedding_dim'],s['layers'],s['hidden_dim'])==(128,1,256)
        assert s['sample_ratios']==[-.25,0.,.25,.5,.75,1.,1.25]
        assert s['context_distance']=='original_quarter_duration'
    assert a['prototypes_per_class']==1 and b['pooling']=='mean'


def test_per_direction_gate_never_retrains_control_or_overtakes_missing_reference(tmp_path):
    from tools.run_prototype_methods import next_ready_job
    pending=[('temporal_pool','S1'),('multi_prototype','S1'),('temporal_pool','S2')]
    assert next_ready_job(pending,'temporal_pool',tmp_path) is None
    ready=tmp_path/'formal/control/S2/complete.json';ready.parent.mkdir(parents=True);ready.write_text('{}')
    assert next_ready_job(pending,'temporal_pool',tmp_path) is None
    ready=tmp_path/'formal/control/S1/complete.json';ready.parent.mkdir(parents=True);ready.write_text('{}')
    assert next_ready_job(pending,'temporal_pool',tmp_path)==('temporal_pool','S1')
    assert next_ready_job(pending,'multi_prototype',tmp_path)==('multi_prototype','S1')
    pending.remove(('temporal_pool','S1'))
    assert next_ready_job(pending,'temporal_pool',tmp_path)==('temporal_pool','S2')
