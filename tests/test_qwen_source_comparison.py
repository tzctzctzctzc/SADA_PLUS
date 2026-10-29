from tools.compare_qwen_source_holdout import evaluation_label,score_predictions


def test_gt_is_only_scoring_and_ambiguous_or_low_overlap_is_reported():
    rows=[dict(segment=[1.,3.],label_id=2),dict(segment=[5.,7.],label_id=4)]
    assert evaluation_label([1.,3.],rows)==(2,1.)
    assert evaluation_label([10.,11.],rows)==(None,0.)
    assert evaluation_label([1.,3.],rows+[dict(segment=[1.,3.],label_id=4)])==(None,1.)


def test_source_scoring_exposes_unmatched_candidates_and_class_balance():
    rows=[dict(candidate_id='a'),dict(candidate_id='b'),dict(candidate_id='c')]
    labels={'a':dict(label=0),'b':dict(label=1),'c':dict(label=None)}
    predictions={'a':dict(class_probabilities=[.9,.1]+[0.]*8),
                 'b':dict(class_probabilities=[.7,.3]+[0.]*8)}
    out=score_predictions(rows,labels,predictions)
    assert out['total_candidates']==3 and out['gt_match_tiou_at_least_half']==2
    assert out['matched_accuracy']==.5 and out['matched_macro_accuracy']==.5


def test_same_candidate_has_same_prompt_and_token_mapping_for_both_models():
    from tools.qwen_source_interface import build_prompt,digit_to_label,normalize_digit_logits
    p=build_prompt('S1-fixed-source-candidate',8,'baseline','native_video')
    assert p==build_prompt('S1-fixed-source-candidate',8,'baseline','native_video')
    assert 'ground truth' not in p.lower()
    mapping=digit_to_label('S1-fixed-source-candidate')
    assert len(set(mapping))==10
    result=normalize_digit_logits([1.]*10,mapping)
    assert all(abs(x-.1)<1e-9 for x in result['class_probabilities'])
