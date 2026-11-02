"""监督风险的标签、梯度隔离、覆盖与保存点恢复测试。"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from tools.supervision_risk import (SupervisionRisk,attach,risk_labels,source_loss,source_truth,
    select,positive_negative_loss)
from tools.supervision_risk_signals import stable_crop,truth_arrays,ranking_metrics,aggregate,readiness


def test_detached_features_and_logits_only_update_small_head():
    torch.manual_seed(5);head=SupervisionRisk(8)
    x=torch.randn(2,8,15,requires_grad=True);p=torch.randn(2,10,15,requires_grad=True)
    mask=torch.ones(2,1,15,dtype=torch.bool)
    out=head([x],[p],[mask])[0];out.square().mean().backward()
    assert x.grad is None and p.grad is None
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())


def test_masked_padding_cannot_change_valid_risk():
    head=SupervisionRisk(8);x=torch.randn(1,8,9);p=torch.randn(1,10,9)
    mask=torch.tensor([[[1,1,1,1,1,0,0,0,0]]],dtype=torch.bool)
    a=head([x],[p],[mask])[0]
    x[:,:,5:]=1e5;p[:,:,5:]=-1e5
    b=head([x],[p],[mask])[0]
    assert torch.equal(a,b) and torch.count_nonzero(a[:,5:])==0


def test_background_excludes_action_edges_and_detector_positives():
    cls=torch.zeros(6,10);cls[5,2]=1
    bg=risk_labels(cls,torch.arange(6.),torch.tensor([[1.,3.]]),torch.ones(6,dtype=torch.bool))
    assert bg.tolist()==[True,False,False,False,True,False]


def test_safe_background_full_source_GT_keeps_removed_crop_instance():
    class Model:
        risk_source_truth={'v':dict(segments=[[10.,12.]])}
        def label_points(self,*args):return [torch.zeros(4,10)],None
    points=torch.tensor([[0.,0.,100.,1.],[1.,0.,100.,1.],[2.,0.,100.,1.],[3.,0.,100.,1.]])
    pred=dict(points=[points],cls=torch.full((1,4,10),-8.),risk=torch.zeros(1,4,2,requires_grad=True),
              valid=torch.ones(1,4,dtype=torch.bool))
    video=dict(segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),metadata=dict(video_id='v',crop_start=9))
    loss,stats=source_loss(Model(),pred,[video]);loss.backward()
    assert stats['source_safe_background']==1
    assert pred['risk'].grad[0,1,1]>0 and pred['risk'].grad[0,0,1]<0


def test_source_truth_filters_target_val_and_preserves_grid_center(tmp_path):
    p=tmp_path/'source.json';p.write_text(json.dumps(dict(database={
        'train':dict(subset='src_train',fps=30,annotations=[dict(label_id=2,segment=[0.,2.])]),
        'val':dict(subset='src_val',fps=30,annotations=[]),
        'target':dict(subset='tgt_train',fps=30,annotations=[])})))
    cfg=dict(dataset=dict(json_file=str(p),feat_stride=16,num_frames=32,default_fps=30),src_train_split=['src_train'])
    d=source_truth(cfg)
    assert set(d)=={'train'} and d['train']['segments']==[[-1.,2.75]]


def test_conflict_is_ignored_and_low_quality_not_automatic_background():
    p=torch.full((4,10),.01);p[0,3]=.4;p[1,2]=.5;p[2,4]=.8
    r=torch.tensor([[.9,.99],[.9,.1],[.2,.1],[.1,.99]])
    s=select(p,r,torch.ones(4,dtype=torch.bool))
    assert s['conflict'].tolist()==[True,False,False,False]
    assert s['positive'].tolist()==[False,True,False,False]
    assert s['negative'].tolist()==[False,False,False,True]
    assert s['ignored'].tolist()==[True,False,True,False]


def test_target_positive_does_not_push_other_classes_negative():
    logits=torch.zeros(3,10,requires_grad=True)
    loss=positive_negative_loss(logits,torch.tensor([2,4,1]),torch.tensor([1,0,0],dtype=torch.bool),
        torch.tensor([0,1,0],dtype=torch.bool),1.,1.)
    loss.backward()
    assert logits.grad[0,2]<0 and torch.count_nonzero(logits.grad[0])==1
    assert (logits.grad[1]>0).all() and torch.count_nonzero(logits.grad[2])==0


def test_all_ignored_target_loss_has_finite_zero_gradient():
    logits=torch.randn(8,10,requires_grad=True);mask=torch.zeros(8,dtype=torch.bool)
    loss=positive_negative_loss(logits,torch.arange(8),mask,mask,1.,1.)
    loss.backward()
    assert float(loss)==0 and torch.count_nonzero(logits.grad)==0


def test_attach_preserves_global_rng_and_original_weights():
    from torch import nn
    class Detector(nn.Module):
        def __init__(self):
            super().__init__();self.cls_head=nn.Module();self.cls_head.cls_head=nn.Module()
            self.cls_head.cls_head.conv=nn.Conv1d(8,10,1)
    m=Detector();old=deepcopy(m.state_dict());rng=torch.get_rng_state().clone();attach(m)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in old.items())
    state=deepcopy(m.supervision_risk.state_dict());attach(m)
    assert all(torch.equal(v,m.supervision_risk.state_dict()[k]) for k,v in state.items())


def test_multilabel_detection_truth_not_forced_to_argmax():
    points=np.array([[2.,0.,100.,1.]],np.float32)
    cfg=dict(model=dict(center_sample_radius=10,reg_distance=True))
    c,b,o=truth_arrays(cfg,points,np.array([[1.,3.],[1.,3.]],np.float32),np.array([2,4]),np.ones(1,bool))
    assert c[0,2]==c[0,4]==1 and not b[0] and o[0]


def test_readiness_rejects_all_ignore_and_bad_background():
    good=dict(class_ranking_gain=.01,background_ranking_gain=.001,positive=20,negative=30,candidates=100,valid=1000,
              positive_accuracy=.9,negative_safe_precision=.99)
    assert readiness(good)
    assert not readiness(dict(good,positive=0))
    assert not readiness(dict(good,negative_safe_precision=.9))
    assert not readiness(dict(good,background_ranking_gain=-.001))


def test_same_count_rank_comparison_and_video_macro():
    p=np.full((20,10),.01,np.float32);p[:10,0]=np.linspace(.2,.9,10)
    cls=np.zeros((20,10),np.float32);cls[:5,0]=1
    r=np.zeros((20,2),np.float32);r[:5,0]=.9;r[10:,1]=.99
    valid=np.ones(20,bool);bg=np.arange(20)>=10
    row=ranking_metrics(p,r,valid,cls,bg,~bg)
    assert row['class_ranking_gain']>0 and row['background_ranking_gain']==0
    assert all(x['count']>0 for x in row['class_equal_count'])
    macro=aggregate([row,dict(row,class_ranking_gain=0)])
    assert macro['class_ranking_gain']==row['class_ranking_gain']/2


def test_crop_immutable_and_identical_across_epochs():
    v=dict(feats=torch.arange(200.).reshape(2,100),segments=torch.tensor([[20.,60.]]),labels=torch.tensor([1]),metadata=dict(video_id='v'))
    a=stable_crop(v,30);b=stable_crop(v,30)
    assert torch.equal(a['feats'],b['feats']) and a['metadata']==b['metadata']
    assert 'crop_start' not in v['metadata'] and v['feats'].shape[-1]==100


def test_head_adamw_checkpoint_continuation_exact():
    torch.manual_seed(123);m=SupervisionRisk(4);o=torch.optim.AdamW(m.parameters(),lr=1e-3)
    def step():
        x=torch.randn(1,4,12);p=torch.randn(1,10,12);mask=torch.ones(1,1,12,dtype=torch.bool)
        o.zero_grad();loss=m([x],[p],[mask])[0].square().mean();loss.backward();o.step();return loss.detach()
    step();saved=deepcopy((m.state_dict(),o.state_dict(),torch.get_rng_state()))
    expected=[step() for _ in range(3)];weights=deepcopy(m.state_dict())
    m.load_state_dict(saved[0]);o.load_state_dict(saved[1]);torch.set_rng_state(saved[2])
    actual=[step() for _ in range(3)]
    assert all(torch.equal(x,y) for x,y in zip(expected,actual))
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in weights.items())
