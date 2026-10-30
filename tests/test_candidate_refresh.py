from copy import deepcopy
from tools.check_candidate_refresh import budget,interval_from_offsets,pool_metrics,fixed_metrics,raw_video


def test_budget_deterministic_capped_and_does_not_read_gt_or_score():
    rows=[dict(candidate_id=f'{v}-{i}',video_id=v,score=i) for v in ['a','b','c','d','e','f','g','h'] for i in range(50)]
    chosen=budget(rows);assert len(chosen)==200
    assert max(sum(x['video_id']==v for x in chosen) for v in ['a','b','c','d','e','f','g','h'])<=30
    changed=deepcopy(rows)
    for x in changed:x.update(score=-x['score'],gt_tiou=1.)
    assert [x['candidate_id'] for x in budget(changed)]==[x['candidate_id'] for x in chosen]


def test_seconds_use_fpn_stride_and_original_feat_num_frames_units():
    meta=dict(feat_stride=16,feat_num_frames=32,fps=32,duration=100)
    assert interval_from_offsets(20,4,[2,3],meta)==[6.5,16.5]
    assert interval_from_offsets(0,4,[2,3],meta)[0]==0.


def test_coverage_counts_all_gt_not_only_videos_with_proposals():
    db={'a':dict(annotations=[dict(segment=[0.,2.],label_id=0)]),
        'b':dict(annotations=[dict(segment=[0.,2.],label_id=0)])}
    p=[dict(candidate_id='a-1',video_id='a',interval=[0.,2.],class_id=1,score=1.)]
    result=pool_metrics(p,db)
    assert result['mean_best_gt_tiou']==1.
    assert result['class_aware_mean_best_gt_tiou']==0.
    assert result['recall_at']['0.5']==.5 and result['class_aware_recall_at']['0.5']==0.


def test_fixed_comparison_cannot_change_to_another_gt_instance():
    db={'a':dict(annotations=[dict(segment=[0.,2.],label_id=0),dict(segment=[8.,10.],label_id=0)])}
    old=[dict(candidate_id='a-1',video_id='a',proposal_start_sec=0.,proposal_end_sec=2.)]
    pred=[dict(candidate_id='a-1',video_id='a',interval=[8.,10.])]
    result=fixed_metrics(pred,old,db)
    assert result['historical_mean_tiou']==1. and result['current_mean_tiou']==0. and result['degraded']==1


def test_export_keeps_fixed_anchor_level_while_pool_uses_strongest_level():
    import torch
    class Model:
        use_trident_head=False
        cfg=dict(model=dict(pl_threshold=.6))
        def preprocessing(self,*a,**kw):return None,None
        def backbone_target(self,*a):return None,None,None
        def neck_target(self,*a):return [None,None],[torch.ones(1,1,1,dtype=torch.bool)]*2
        def cls_head_target(self,*a):return [torch.tensor([[[1.],[0.]]]),torch.tensor([[[3.],[0.]]])],None
        def reg_head_target(self,*a):return [torch.ones(1,2,1),torch.ones(1,2,1)*2],None
        def point_generator(self,*a):return [torch.tensor([[4.,0.,100.,1.]]),torch.tensor([[4.,0.,100.,2.]])]
    video=dict(metadata=dict(video_id='v',feat_stride=16,feat_num_frames=32,fps=32,duration=20))
    anchor=dict(candidate_id='v-p0000004',absolute_feature_index=4,source_fpn_levels=[0])
    pool,fixed=raw_video(Model(),video,[anchor])
    assert pool[0]['level']==1 and fixed[0]['level']==0
    assert pool[0]['interval']==[.5,4.5] and fixed[0]['interval']==[2.,3.]


def test_actual_batch_device_interface_preserves_video_dictionary_and_empty_gt():
    import torch
    from tools.check_candidate_refresh import to_device
    video=dict(feats=torch.ones(3,4),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),metadata={'video_id':'v'})
    result=to_device(video,'cpu')
    assert isinstance(result,dict) and result is not video
    assert result['metadata']==video['metadata'] and result['segments'].shape==(0,2)
