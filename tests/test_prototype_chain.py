import pytest
from tools.audit_prototype_chain import union_length,truth_for,coverage,auc,semantic_summary,training_audit
from tools.baseline_protocol import save


def test_union_does_not_double_count_overlapping_gt():
    assert union_length([(0.,4.),(2.,5.),(6.,8.),(7.,9.)])==8.
    assert union_length([])==0.
    x=truth_for([0.,10.],[dict(label_id=0,segment=[0.,4.]),dict(label_id=0,segment=[2.,6.])])
    assert x['per_class_fraction'][0]==.6 and x['unique_majority']==0


def test_unknown_action_is_not_called_background_and_mixed_labels_not_forced():
    x=truth_for([0.,10.],[dict(label_id=12,segment=[0.,10.])])
    assert x['unique_majority'] is None and x['known_fraction']==0 and x['any_annotated_fraction']==1
    x=truth_for([0.,10.],[dict(label_id=0,segment=[0.,7.]),dict(label_id=1,segment=[3.,10.])])
    assert x['unique_majority'] is None and set(x['majority_labels'])=={0,1}


def test_coverage_denominator_includes_videos_without_candidates():
    db={'a':dict(annotations=[dict(label_id=0,segment=[0.,2.])]),
        'b':dict(annotations=[dict(label_id=0,segment=[0.,2.])])}
    out=coverage([dict(video_id='a',interval=[0.,2.],class_id=1)],db)
    assert out['known_gt_instances']==2 and out['class_agnostic_gt_recall']['0.5']==.5
    assert out['videos']==1
    assert out['class_aware_gt_recall']['0.5']==0


def test_auc_tie_handling():
    assert auc([.9,.1],[True,False])==1
    assert auc([.1,.9],[True,False])==0
    assert auc([.5,.5],[True,False])==.5
    assert auc([.5],[True]) is None


def test_vlm_student_complementarity_is_on_same_gt_interval():
    anns=[dict(label_id=0,segment=[0.,2.])]
    truth=truth_for([0.,2.],anns)
    rows=[dict(video_id='a',vlm_class=0,student_class=1,truth=truth),
          dict(video_id='a',vlm_class=1,student_class=0,truth=truth)]
    out=semantic_summary(rows)
    assert out['complementarity']==dict(vlm_correct_student_wrong=1,student_correct_vlm_wrong=1)
    assert out['vlm_accuracy_on_majority']==.5


def test_zero_batch_statistic_is_a_lower_bound_not_unique_coverage(tmp_path):
    save(tmp_path/'config.json',dict(baseline=dict(epoch_budget={'iterations_per_epoch':10}),
        train_cfg=dict(warmup_adv=5),model=dict(loss_weights={'weight_downstream':2.})))
    a=dict(rq3_selected=.3,rq3_candidate_count=2.,rq3_fully_visible_count=1.5,
        rq3_source_loss=.1,rq3_alignment_loss=.01,rq3_target_gradient_norm=.001)
    (tmp_path/'metrics.jsonl').write_text(__import__('json').dumps(dict(epoch=5,auxiliary=a))+'\n')
    out=training_audit(tmp_path);row=out['epoch_statistics'][0]
    assert row['selected_exposures']==3 and row['minimum_fraction_batches_without_selected']==.7
    assert row['effective_target_coefficient']==.2 and out['unique_exposures_not_recorded']


def test_gradient_groups_distinguish_backbone_from_detection_heads_without_updating_weights():
    import torch
    from torch import nn
    from tools.audit_prototype_gradients import parameter_groups,gradients
    m=nn.Module();m.backbone_source=nn.Linear(4,3);m.neck_source=nn.Identity()
    m.cls_head=nn.Linear(3,2);m.cls_head_target=m.cls_head
    m.reg_head=nn.Linear(3,2);m.reg_head_target=m.reg_head;m.rq3_capacity=nn.Linear(3,2)
    before={k:v.clone() for k,v in m.state_dict().items()}
    params,groups=parameter_groups(m);x=m.backbone_source(torch.ones(2,4))
    auxiliary=m.rq3_capacity(x).square().mean();detection=m.reg_head(x).square().mean()
    a=gradients(auxiliary,params,groups,True);b=gradients(detection,params,groups,False)
    assert a['norms']['shared_fpn']>0 and a['norms']['auxiliary']>0
    assert a['norms']['detection_regression']==a['norms']['detection_classification']==0
    assert b['norms']['detection_regression']>0 and len(groups['detection_regression'])>0
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_gradient_cosine_does_not_treat_zero_as_agreement():
    import torch
    from tools.audit_prototype_gradients import cosine
    x=torch.tensor([1.,0.]);assert cosine(x,x)==1. and cosine(x,-x)==-1.
    assert cosine(x,torch.zeros(2)) is None
