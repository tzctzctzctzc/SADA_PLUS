import pytest
from tools.run_qwen_gt_oracle import balanced,cid,prompt,option_order,OPTIONS,safe_row,metrics,shard_id


def test_balanced_selection_deterministic_and_fills_budget():
    rows=[dict(label=k,sample_id='%03d'%i) for k in range(10) for i in range(25)]
    chosen=balanced(rows);assert len(chosen)==200
    assert all(sum(r['label']==k for r in chosen)==20 for k in range(10))
    assert balanced(rows[::-1])==chosen


def test_prompt_and_identity_do_not_depend_on_ground_truth_class():
    key=cid('video',1.,2.);text=prompt(key)
    assert len(set(option_order(key)))==14 and set(option_order(key))==set(OPTIONS)
    assert all(x in text for x in ('OTHER_ACTION','MIXED_ACTIONS','NO_ACTION','UNCERTAIN'))
    assert 'ground truth' not in text.lower()
    row=dict(candidate_id=key,video_id='video',start=1.,end=2.,frame_paths=['x']*8,frame_sha256=['h']*8,actual_times=[1.5]*8,unique_frames=1)
    safe_row(row)
    with pytest.raises(ValueError):safe_row(dict(row,label=0))


def test_abstention_cannot_inflate_overall_accuracy():
    rows=[dict(candidate_id='a',label=0),dict(candidate_id='b',label=1)]
    result=metrics(rows,{'a':dict(prediction='TAKE',confidence=.9),'b':dict(prediction='UNCERTAIN',confidence=.8)})
    assert result['accuracy']==.5 and result['known_prediction_coverage']==.5 and result['accepted_accuracy']==1.
    assert metrics(rows,{})['unavailable']==2


def test_four_shards_are_disjoint_exhaustive_and_stable():
    keys=[cid('video',float(i),float(i+1)) for i in range(200)]
    sets=[{k for k in keys if shard_id(k)==s} for s in range(4)]
    assert set.union(*sets)==set(keys) and sum(map(len,sets))==len(keys)
    assert all(sets) and [shard_id(k) for k in keys]==[shard_id(k) for k in keys]
