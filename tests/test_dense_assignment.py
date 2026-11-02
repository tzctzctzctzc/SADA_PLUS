import math
import torch
import pytest
from tools.dense_assignment_math import (map_time,feature_view,linear_sample,decoded,
    interval_iou,offset_iou,union_contains,class_thresholds,quality_focal,source_quality_targets,task_separation)


def point_tensor(xs,level=0):
    return torch.tensor([[x,0,100,2**level] for x in xs],dtype=torch.float32)


def test_center_coordinate_inverse_and_subgrid():
    view=dict(lo=5,width=80,ratio=.9,length=72)
    x=torch.tensor([5.,5.05,6.2,70.],requires_grad=True)
    y=map_time(x,view)
    assert torch.allclose(map_time(y,view,True),x,atol=1e-5)
    y.sum().backward();assert torch.allclose(x.grad,torch.full_like(x,.9))
    assert map_time(torch.tensor(5.),view).item()==pytest.approx(-.05)


def test_linear_sampling_masks_and_endpoint_derivative():
    x=torch.arange(5.).reshape(5,1).requires_grad_();coords=torch.tensor([.5,1.5,3.5,-.1],requires_grad=True)
    values,valid=linear_sample(x,torch.tensor([1,1,1,1,0],dtype=torch.bool),coords,1)
    assert valid.tolist()==[True,True,False,False]
    assert values[:2,0].tolist()==[.5,1.5]
    values[valid].sum().backward();assert torch.all(coords.grad[:2]>0) and x.grad.abs().sum()>0


def test_feature_view_has_no_GT_and_exact_length_ratio():
    v=dict(feats=torch.arange(24.).reshape(2,12),segments=torch.tensor([[1.,3.]]),labels=torch.tensor([1]))
    y,w=feature_view(v,3,0)
    assert not y['segments'].numel() and not y['labels'].numel()
    assert y['feats'].shape[-1]==w['length'] and w['ratio']==w['length']/w['width']
    assert torch.equal(v['segments'],torch.tensor([[1.,3.]]))


def test_union_and_short_interval_survive():
    boxes=torch.tensor([[-.2,.05],[3.,5.],[1.,4.]])
    p=torch.tensor([-.3,-.1,.025,.1,.9,2.,4.5,5.1])
    assert union_contains(p,boxes).tolist()==[False,True,True,False,False,True,True,False]
    assert union_contains(p,boxes[:0]).sum()==0


def test_decode_preserves_point_not_half_stride():
    p=point_tensor([0,2],1);b=decoded(p,torch.tensor([[.025,.025],[.1,.2]]))
    assert torch.allclose(b,torch.tensor([[-.05,.05],[1.8,2.4]]))
    assert offset_iou(torch.tensor([[1.,2.]]),torch.tensor([[1.,2.]])).item()==1
    assert interval_iou(b,b).tolist()==[1.,1.]


def test_quality_labels_detached_and_two_heads_learn_jointly():
    c=torch.tensor([[1.,0.],[0.,0.]])
    off=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
    truth=torch.tensor([[2.,2.],[0.,0.]],requires_grad=True)
    soft,iou,pos=source_quality_targets(c,off,truth,torch.tensor([True,True]))
    assert not soft.requires_grad and not iou.requires_grad
    assert iou[0].item()==pytest.approx(.75) and pos.tolist()==[True,False]
    l=torch.zeros(2,2,requires_grad=True);q=torch.zeros(2,requires_grad=True)
    loss=quality_focal(l.sigmoid()*q.sigmoid()[:,None],soft,torch.tensor([True,True]),1.)
    loss.backward();assert l.grad.abs().sum()>0 and q.grad.abs().sum()>0
    assert off.grad is None and truth.grad is None


def test_ignore_has_zero_classification_and_quality_gradient():
    l=torch.tensor([[.3,.2],[.5,.2]],requires_grad=True);q=torch.zeros(2,requires_grad=True)
    target=torch.tensor([[1.,0.],[.3,.2]],requires_grad=True)
    loss=quality_focal(l.sigmoid()*q.sigmoid()[:,None],target,torch.tensor([True,False]),1.)
    loss.backward();assert torch.equal(l.grad[1],torch.zeros(2)) and q.grad[1]==0
    assert target.grad is None


def tsa(prob,quality,boxes,xs=None,levels=None,protected=None,length=8):
    n=len(prob);p=point_tensor(xs or [2.]*n)
    return task_separation(torch.tensor(prob),torch.tensor(quality),torch.tensor(boxes),p,
        torch.tensor(levels or [0]*n),torch.ones(n,dtype=torch.bool),
        torch.tensor(protected or [False]*n),length)


def test_no_positive_does_not_fabricate_top10():
    r=tsa([[.01,.02],[.02,.01]],[.5,.5],[[0.,3.],[1.,4.]])
    assert r['positive'].sum()==0 and r['regression'].sum()==0 and r['negative'].sum()==2


def test_high_class_low_quality_is_soft_not_background():
    r=tsa([[.9,.01]],[.01],[[0.,4.]])
    assert r['soft'].item() and r['active'].item() and not r['negative'].item()
    assert r['classes'][0,0].item()==pytest.approx(.009)


def test_background_inside_possible_action_is_ignored():
    r=tsa([[.01,.01]],[.5],[[0.,4.]],protected=[True])
    assert r['ignored'].item() and not r['active'].item()


def test_tsa_mines_same_instance_and_keeps_weighted_boundaries():
    r=tsa([[.9,.01],[.25,.01]],[.9,.5],[[0.,5.],[.2,4.8]],xs=[2.,2.5])
    assert r['positive'].tolist()==[True,False] and r['mined'].tolist()==[False,True]
    assert r['regression'].tolist()==[True,True]
    assert torch.allclose(r['boxes'][1],torch.tensor([0.,5.]))


def test_tsa_rejects_cross_level_mining():
    r=tsa([[.9,.01],[.25,.01]],[.9,.5],[[0.,5.],[.2,4.8]],xs=[2.,2.5],levels=[0,2])
    assert not r['mined'][1]


def test_tsa_never_averages_conflicting_same_class_instances():
    r=tsa([[.9,.01],[.85,.01],[.25,.01]],[.9,.9,.5],
          [[0.,4.],[2.,6.],[1.,5.]],xs=[2.,3.,3.])
    assert r['conflict'][2] and not r['mined'][2] and not r['regression'][2]


def test_truncated_or_unrepresentable_endpoint_not_regressed():
    r=tsa([[.9,.01],[.9,.01]],[.9,.9],[[-1.,4.],[4.,6.]],xs=[2.,2.])
    assert r['positive'].sum()==2 and r['regression'].sum()==0


def test_adaptive_thresholds_are_per_class_and_bounded():
    p=torch.tensor([[.8,.1],[.7,.2],[.01,.6]])
    t,state=class_thresholds(p,torch.tensor([True,True,True]),(torch.zeros(2),torch.zeros(2)))
    assert torch.all((t>=.25)&(t<=.35)) and state[0].tolist()==pytest.approx([.2,.1])
    assert all(not x.requires_grad for x in state)


def test_empty_quality_loss_is_finite_and_differentiable():
    p=torch.full((3,2),.1,requires_grad=True)
    loss=quality_focal(p,torch.zeros_like(p),torch.zeros(3,dtype=torch.bool),250.)
    loss.backward();assert torch.isfinite(loss) and p.grad.abs().sum()==0


def test_no_python_import_of_oracle_truth_or_model_replacement():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=(root/'tools/dense_assignment.py').read_text(encoding='utf-8')
    assert 'oracle_train_truth' not in source and 'torch.load' not in source
    assert 'backbone_source =' not in source and 'reg_head =' not in source


def test_source_AP_selection_can_keep_late_checkpoint_without_target(tmp_path):
    import json
    from tools.run_dense_assignment import source_ap_selection
    for epoch,value in [(5,.5),(10,.6),(15,.65),(20,.7)]:
        p=tmp_path/('signals/e%03d/evaluation.json'%epoch);p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'results':[{'domain':'source','epoch':epoch,'map':value}]}))
    result=source_ap_selection(tmp_path,[[5,.4],[10,.3],[15,.32],[20,.35]])
    assert result['epoch']==20 and len(result['candidates'])==4


def test_source_signal_reports_multilabel_shortest_assignment_not_occupancy():
    import numpy as np
    from tools.dense_assignment_signals import assign_cpu
    points=np.array([[1.,0.,100.,1.],[4.,0.,100.,1.]],np.float32)
    boxes=np.array([[0.,2.],[0.,2.],[0.,10.]],np.float32)
    classes=assign_cpu(points,boxes,np.array([0,1,2]),1.5)
    assert classes[0].tolist()==[1.,1.,0.,0.,0.,0.,0.,0.,0.,0.]
    assert classes[1,2]==1 and classes[1].sum()==1
