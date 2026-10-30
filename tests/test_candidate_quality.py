from tools.diagnose_vlm_candidates import evaluate_candidates,summarize,tiou


def row(name,start,end,student=0,vlm=1):
    return dict(candidate_id=name,video_id='v',proposal_start_sec=start,proposal_end_sec=end,
        participation_selected=True,student_class=student,class_id=vlm,student_score=.8)


def test_class_quality_separates_vlm_from_detector_and_counts_unique_coverage():
    db={'v':dict(annotations=[dict(segment=[0,10],label_id=1)])}
    results=evaluate_candidates([row('a',0,10),row('b',0,10)],db);s=summarize(results,db)
    assert s['student_majority_accuracy']==0 and s['vlm_majority_accuracy']==1
    assert s['thresholds']['0.5']['covered_unique_gt']==1
    assert s['thresholds']['0.5']['student_joint_pass_fraction']==0
    assert s['thresholds']['0.5']['vlm_joint_pass_fraction']==1


def test_background_other_classes_and_mixed_candidates_are_not_conflated():
    db={'v':dict(annotations=[dict(segment=[0,1],label_id=0),dict(segment=[1,2],label_id=1),dict(segment=[4,5],label_id=20)])}
    results=evaluate_candidates([row('mixed',0,2),row('other',4,5),row('bg',6,7)],db);s=summarize(results,db)
    assert s['unique_majority_count']==0 and s['student_majority_accuracy'] is None
    assert s['kinds']==dict(no_known_majority=1,other_annotated_class_only=1,no_annotated_action_overlap=1)


def test_localization_iou_is_not_proposal_purity():
    db={'v':dict(annotations=[dict(segment=[0,10],label_id=1)])}
    results=evaluate_candidates([row('short',4,5)],db);s=summarize(results,db)
    assert s['vlm_majority_accuracy']==1 and abs(s['mean_best_tiou']-.1)<1e-10
    assert s['thresholds']['0.5']['localization_pass_count']==0
    assert tiou([0,10],[0,10])==1 and tiou([0,1],[2,3])==0
