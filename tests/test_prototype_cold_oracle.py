from types import SimpleNamespace
import pytest
import torch
from tools.prototype_cold_oracle import crop_truth, assigned_instances, detection_losses, MODES, SPEC


def test_truth_crop_keeps_short_and_truncated_class_instances_without_false_boundaries():
    record=dict(fps=30.,feat_stride=16,feat_num_frames=32,
        segments=[[9.98,10.03],[8.,12.],[12.,18.],[21.,24.]],labels=[1,2,3,4])
    meta=dict(fps=30.,feat_stride=16,feat_num_frames=32,crop_start=10)
    boxes,labels,visible,ids=crop_truth(record,meta,10,'cpu')
    assert labels.tolist()==[1,2,3]
    assert visible.tolist()==[False,False,True]
    assert ids.tolist()==[0,1,2]
    assert 0<float(boxes[0,1]-boxes[0,0])<.05
    assert boxes.tolist()[1]==[0.,2.]


def test_truth_crop_requires_exact_physical_units():
    record=dict(fps=30,feat_stride=16,feat_num_frames=32,segments=[],labels=[])
    with pytest.raises(AssertionError):crop_truth(record,dict(fps=30,feat_stride=8,feat_num_frames=32),10,'cpu')
    boxes,labels,visible,ids=crop_truth(record,dict(fps=30,feat_stride=16,feat_num_frames=32),10,'cpu')
    assert boxes.shape==(0,2) and labels.numel()==visible.numel()==ids.numel()==0


@pytest.mark.parametrize('center',['radius','none'])
@pytest.mark.parametrize('use_range',[True,False])
def test_instance_assignment_matches_original_detection_labels_and_offsets(center,use_range):
    from libs.models.SADA.meta_archs import SADA
    stub=SimpleNamespace(num_class_orig=10,train_center_sample=center,
        train_center_sample_radius=1.5,train_reg_distance=use_range)
    points=torch.tensor([[float(i),0.,5.,1.] for i in range(14)]+[[float(i),4.,20.,2.] for i in range(0,14,2)])
    segments=torch.tensor([[1.,12.],[2.,5.],[2.,5.],[7.,9.],[4.98,5.03]])
    labels=torch.tensor([0,1,2,3,4])
    classes,offsets=SADA.label_points_single_video(stub,points,segments,labels)
    index=assigned_instances(points,segments,center,1.5,use_range)
    pos=index>=0
    assert torch.equal(pos,classes.sum(-1)>0)
    expected=torch.stack([points[pos,0]-segments[index[pos],0],segments[index[pos],1]-points[pos,0]],-1)/points[pos,3,None]
    assert torch.equal(expected,offsets[pos])
    assert (classes.sum(-1)>1).any()


def test_empty_and_short_instance_assignment():
    p=torch.tensor([[1.,0.,100.,1.],[1.01,0.,100.,1.]])
    assert assigned_instances(p,torch.empty(0,2),'radius',1.5,True).tolist()==[-1,-1]
    index=assigned_instances(p,torch.tensor([[.99,1.02]]),'radius',1.5,True)
    assert index.tolist()==[0,0]


@pytest.mark.parametrize('mode',MODES)
def test_only_selected_oracle_channel_has_gradient(mode):
    logits=torch.zeros(1,3,10,requires_grad=True)
    offsets=torch.full((1,3,2),1.,requires_grad=True)
    classes=torch.zeros_like(logits);classes[0,1,2]=1
    valid=torch.ones(1,3,dtype=torch.bool);mask=torch.tensor([[False,True,False]])
    targets=torch.full_like(offsets,2.)
    cls,reg=detection_losses(logits,offsets,valid,classes,targets,mask,[1.,1.])
    aux=(cls if mode!='localization' else cls*0)+(reg if mode!='classification' else reg*0)
    gc,gr=torch.autograd.grad(aux,(logits,offsets))
    assert torch.isfinite(aux) and torch.isfinite(gc).all() and torch.isfinite(gr).all()
    assert bool(gc.abs().sum()>0)==(mode!='localization')
    assert bool(gr.abs().sum()>0)==(mode!='classification')


def test_background_is_supervised_and_padding_is_ignored():
    logits=torch.zeros(1,2,10,requires_grad=True);offsets=torch.ones(1,2,2,requires_grad=True)
    cls,reg=detection_losses(logits,offsets,torch.tensor([[True,False]]),torch.zeros_like(logits),
        offsets.detach(),torch.zeros(1,2,dtype=torch.bool),[1.,1.])
    cls.backward()
    assert (logits.grad[0,0]>0).all() and (logits.grad[0,1]==0).all() and float(reg)==0


def test_exact_offsets_have_zero_regression_loss_and_labels_are_detached():
    logits=torch.zeros(1,1,10,requires_grad=True);offsets=torch.tensor([[[1.,2.]]],requires_grad=True)
    labels=torch.zeros_like(logits);labels[0,0,0]=1
    cls,reg=detection_losses(logits,offsets,torch.ones(1,1,dtype=torch.bool),labels,
        offsets.detach().clone(),torch.ones(1,1,dtype=torch.bool),[1.,1.])
    assert abs(float(reg))<1e-6 and not labels.requires_grad


def test_fixed_scope_is_cold_full_target_without_teacher_candidate_filter():
    assert set(MODES)=={'joint','classification','localization'}
    assert SPEC['onset_epoch']==1 and SPEC['target_weight']==1.
    assert SPEC['max_epochs']==50 and SPEC['patience']==15
    assert 'not_mature' in SPEC['initialization']
