from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.phase_endpoint_oracle import crop_truth,choose_truth,select_endpoint,oracle_mask,oracle_loss


def test_full_grid_to_original_crop_preserves_censored_true_endpoints():
    meta=dict(fps=30,feat_stride=16,feat_num_frames=32,crop_start=10)
    record=dict(fps=30,feat_stride=16,feat_num_frames=32,segments=[[5.,20.],[25.,40.],[50.,70.]],labels=[0,1,2])
    boxes,labels=crop_truth(record,meta,25)
    assert boxes.tolist()==[[-5.,10.],[15.,30.]] and labels.tolist()==[0,1]
    assert choose_truth(np.array([[0,20],[5,15],[6,14]]),np.array([0,0,1]),0,10)==1
    assert choose_truth(boxes,labels,9,10) is None


def test_grid_oracle_does_not_invent_gt_outside_support_and_gate_keeps_raw():
    assert select_endpoint([6,14],[1,20],0,'gate',32)==6
    assert select_endpoint([6,14],[1,20],0,'grid',32)==4
    assert select_endpoint([6,14],[1,20],0,'gt',32)==1
    assert select_endpoint([6,14],[6,14],0,'grid',32)==6


def test_mask_requires_better_same_units_visible_and_representable():
    v=dict(lo=4,width=20,ratio=.9,length=18)
    teacher=torch.tensor([5.,15.]);gt=torch.tensor([5.,15.])
    student=torch.tensor([3.,12.],requires_grad=True)
    mask=oracle_mask(teacher,student,gt,7,32,v)
    assert mask.tolist()==[True,True]
    assert not oracle_mask(teacher,student,torch.tensor([-1.,15.]),7,32,v)[0]
    assert not oracle_mask(teacher,student,gt,-1,32,v)[0]


@pytest.mark.parametrize('mode',['control','gate','grid','gt'])
def test_actual_raw_proposals_oracle_loss_gradient_and_gt_isolation(mode):
    f=[torch.randn(1,512,32,requires_grad=True)];m=[torch.ones(1,1,32,dtype=torch.bool)]
    c=[torch.full((1,97,32),-20.)];c[0][0,0,10]=5.
    teacher=[torch.full((1,2,32),4.,requires_grad=True)];student=[torch.full((1,2,32),3.,requires_grad=True)]
    view=[dict(lo=0,width=32,ratio=1.,length=32,original_length=32)]
    meta=dict(video_id='train',fps=30,feat_stride=16,feat_num_frames=32,crop_start=5)
    truth={'train':dict(fps=30,feat_stride=16,feat_num_frames=32,segments=[[10.,20.]],labels=[0])}
    rows=[];loss,stats=oracle_loss(SimpleNamespace(fpn_strides=[1]),(f,m,c,teacher),(f,m,c,student),view,[dict(metadata=meta)],truth if mode!='control' else {},mode,rows)
    loss.backward();assert teacher[0].grad is None
    if mode=='control':
        assert loss==0 and stats['left']==stats['right']==0 and not stats['target_train_GT_used']
    else:
        assert stats['left']==stats['right']==1 and stats['target_train_GT_used'] and student[0].grad.norm()>0
        if mode in ['grid','gt']:assert rows[0]['target']==[5.,15.]
