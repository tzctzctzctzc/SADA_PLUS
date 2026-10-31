"""只核实例政策、坐标及门禁；真实SADA与NMS还须Linux数据验证。"""
import pytest
import torch
from tools.instance_task_source import interval_from_offsets, rows_from_dense, source_metrics, source_decision, fully_visible


def test_decode_seconds_uses_original_frame_units_and_clips():
    meta = dict(feat_stride=16, feat_num_frames=32, fps=32, duration=10)
    assert interval_from_offsets(4, 2, [1, 2], meta) == [1.5, 4.5]
    assert interval_from_offsets(0, 1, [2, 40], meta) == [0., 10]


def test_raw_policy_filters_padding_and_keeps_highest_coordinate_level():
    points = [torch.tensor([[0.,0,100,2**level],[2.,0,100,2**level]]) for level in range(3)]
    masks = [torch.tensor([[[True,False]]]) for _ in range(3)]
    logits = [torch.tensor([[[1.+level,9.],[0.,9.]]]) for level in range(3)]
    offsets = [torch.ones(1,2,2) for _ in range(3)]
    rows = rows_from_dense(points,masks,logits,offsets,
        dict(video_id='P01_01',feat_stride=16,feat_num_frames=32,fps=32,duration=10),.6)
    assert len(rows)==1 and rows[0]['level']==2 and rows[0]['class_id']==0
    assert rows[0]['candidate_id']=='P01_01-p0000000'


def test_raw_policy_no_200_or_30_cap():
    count=220
    points=[torch.stack([torch.arange(count),torch.zeros(count),torch.ones(count)*100,torch.ones(count)],-1) for _ in range(3)]
    masks=[torch.ones(1,1,count,dtype=torch.bool) for _ in range(3)]
    logits=[torch.ones(1,2,count)*4 for _ in range(3)]; offsets=[torch.ones(1,2,count) for _ in range(3)]
    rows=rows_from_dense(points,masks,logits,offsets,
        dict(video_id='P01_01',feat_stride=16,feat_num_frames=32,fps=32,duration=1000),.6)
    assert len(rows)==count


def test_precision_counts_duplicates_once_and_requires_correct_class():
    truth={'a':[dict(interval=[0,2],class_id=1),dict(interval=[4,6],class_id=1)]}
    pred=[dict(video_id='a',interval=[0,2],class_id=1,score=.9),
          dict(video_id='a',interval=[0,2],class_id=1,score=.8),
          dict(video_id='a',interval=[4,6],class_id=0,score=.7)]
    got=source_metrics(pred,truth)
    assert got['precision05']==got['precision07']==1/3 and got['coverage05']==.5


def test_localization_threshold_and_empty_predictions():
    truth={'a':[dict(interval=[0,2],class_id=1)]}
    row=dict(video_id='a',interval=[0,1],class_id=1,score=.9)
    got=source_metrics([row],truth)
    assert got['precision05']==1 and got['precision07']==0 and got['coverage05']==1
    assert source_metrics([],truth)['precision05']==0


def test_fully_visible_uses_global_seconds_and_crop_origin_without_clipping():
    meta=dict(feat_stride=16,feat_num_frames=32,fps=32,crop_start=10,crop_feat_len=8)
    rows=[dict(interval=[5.5,9.5]),dict(interval=[5.,8.]),dict(interval=[7.,10.])]
    selected=fully_visible(rows,meta,8)
    assert len(selected)==1 and selected[0]['feature_interval']==[0.,8.]
    assert 'feature_interval' not in rows[0]


def test_source_gate_requires_exact_six_and_equal_direction_weights():
    results=[dict(shift='S'+str(i),metrics=dict(precision05=.8,precision07=.5,coverage05=.1)) for i in range(1,7)]
    assert source_decision(results)['status']=='passed'
    results[0]['metrics']['coverage05']=0
    assert source_decision(results)['status']=='rejected'
    with pytest.raises(ValueError): source_decision(results[:-1])
    with pytest.raises(ValueError): source_decision(results[:-1]+[results[0]])
