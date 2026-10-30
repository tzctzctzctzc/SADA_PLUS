from copy import deepcopy
import torch
from tools.audit_mature_endpoint_teacher import point_record,summary


def test_identical_teacher_has_exactly_zero_advantage():
    points=[torch.tensor([[10.,0.,100.,2.]])]
    model=dict(points=points,masks=[torch.ones(1,1,1,dtype=torch.bool)],offsets=[torch.tensor([[[2.,3.]]])])
    row,reason=point_record(model,deepcopy(model),[torch.tensor([[[5.],[0.]]])],
        [torch.tensor([[1.,0.]])],[torch.tensor([[2.,3.]])],(4.,18.),0,32)
    assert reason is None and row['student_error']==row['teacher_error']==[0.,0.]
    row.update(video_id='a')
    report=summary([row],['a'])
    assert report['teacher_better']==0 and report['video_equal_teacher_advantage']==0


def test_single_side_visibility_and_original_stride_units():
    s=dict(points=[torch.tensor([[10.,0.,100.,2.]])],masks=[torch.ones(1,1,1,dtype=torch.bool)],offsets=[torch.tensor([[[2.,3.]]])])
    t=deepcopy(s);t['offsets']=[torch.tensor([[[6.,2.]]])]
    row,_=point_record(s,t,[torch.tensor([[[5.],[0.]]])],[torch.tensor([[1.,0.]])],
        [torch.tensor([[2.,3.]])],(4.,18.),0,32)
    assert row['teacher_boundary']==[-2.,14.] and row['allowed']==[False,True]
    row.update(video_id='a');assert summary([row],['a'])['endpoints']==1


def test_summary_does_not_let_long_video_dominate():
    a=dict(video_id='a',duration=10.,student_error=[10.,10.],teacher_error=[0.,0.],allowed=[True,True])
    b=dict(video_id='b',duration=10.,student_error=[0.,0.],teacher_error=[10.,10.],allowed=[True,True])
    report=summary([deepcopy(a) for _ in range(100)]+[b],['a','b'])
    assert report['video_equal_teacher_advantage']==0


def test_filter_video_and_no_source_positive_rejected():
    assert summary([],['a'])['endpoints']==0
    s=dict(points=[torch.tensor([[10.,0.,100.,2.]])],masks=[torch.ones(1,1,1,dtype=torch.bool)],offsets=[torch.tensor([[[2.,3.]]])])
    row,reason=point_record(s,s,[torch.tensor([[[5.],[0.]]])],[torch.tensor([[0.,0.]])],
        [torch.tensor([[2.,3.]])],(4.,18.),0,32)
    assert row is None and reason=='not_source_assigned_positive'
