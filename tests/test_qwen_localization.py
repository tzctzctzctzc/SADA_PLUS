from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from runpy import run_path

import pytest

from tools.qwen_localization import (OPTIONS,POLICY,build_prompt,frozen_class,sample_indices,
    option_scores,combine,top_twenty,reference_instance,score_subset,audit_predictions,reuse_preparation)


def row(cid='S1-a',shift='S1'):
    return dict(candidate_id=cid,shift=shift,video_id='video',class_id=3,proposal=[2.,12.],
        frame_times_sec=list(map(float,range(16))),input_status='ready')


def answer(index,probability=.5):
    values=[-20.]*19;values[index]=0.
    return option_scores(values,math.log(1/probability))


def test_prompt_defines_real_time_gaps_and_uses_cached_class_only():
    r=row();p=build_prompt(r,'START')
    assert 'OPEN (opening an object)' in p and 'frame 0 (0.000000 s)' in p
    assert 'O: between frame 14' in p and 'S: UNCERTAIN' in p and 'Do not copy' in p
    r.update(ground_truth=[6,8],gt_class_id=9)
    assert build_prompt(r,'START')==p
    assert 'BOUNDARY TO LOCATE: END' in build_prompt(r,'END')
    with pytest.raises(ValueError):build_prompt(r,'OTHER')


def test_real_pts_sampling_never_creates_duplicate_or_interpolated_frames():
    pts=[i*.02 for i in range(200)]
    indices=sample_indices(pts)
    assert len(indices)==len(set(indices))==16 and indices==sorted(indices)
    assert sample_indices(pts[:15]) is None
    with pytest.raises(ValueError):sample_indices([0,1,1,2])


def test_frozen_class_refuses_tie_and_corrupt_probabilities():
    assert frozen_class([1.]+[0.]*9)==0
    assert frozen_class([.5,.5]+[0.]*8) is None
    with pytest.raises(ValueError):frozen_class([.2]*10)


def test_confidence_uses_full_vocab_mass_not_renormalized_option_certainty():
    a=answer(3,.01);b=answer(3,.9)
    assert a['conditional_probability']==b['conditional_probability']
    assert a['probability']==pytest.approx(.01) and b['probability']==pytest.approx(.9)
    assert combine(row(),a,answer(8,.8))['confidence']==pytest.approx(.01)
    with pytest.raises(ValueError):option_scores([float('nan')]*19,10.)


@pytest.mark.parametrize('start,end',[(5,5),(7,3),(15,9),(1,16),(17,18)])
def test_same_gap_reversed_outside_or_abstain_are_not_valid_boxes(start,end):
    pred=combine(row(),answer(start),answer(end))
    assert pred['interval'] is None and pred['confidence']==-1 and pred['status']=='unresolved'


def test_interval_midpoints_and_lowest_boundary_probability():
    pred=combine(row(),answer(3,.8),answer(9,.2))
    assert pred['interval']==[3.5,9.5] and pred['confidence']==pytest.approx(.2)


def test_top20_is_per_shift_including_failures_and_ignores_ground_truth():
    rows=[row(f'S{s}-{i:03d}',f'S{s}') for s,n in [(1,95),(2,200)] for i in range(n)]
    predictions={r['candidate_id']:dict(confidence=-1.) for r in rows}
    first=top_twenty(rows,predictions)
    assert len(first)==59 and sum(x.startswith('S1-') for x in first)==19
    assert first==top_twenty(list(reversed(rows)),predictions)
    for p in predictions.values():p['ground_truth_iou']=1.
    assert top_twenty(rows,predictions)==first
    with pytest.raises(ValueError):top_twenty(rows+rows[:1],predictions)


def test_reference_is_unique_original_overlap_not_best_teacher_output_match():
    anns=[dict(segment=[2,5],label_id=3),dict(segment=[8,11],label_id=3)]
    assert reference_instance([2,5],anns)['interval']==[2.,5.]
    assert reference_instance([0,13],anns)['status']=='no_unique_known_reference'
    assert reference_instance([2,5],anns+[anns[0]])['status']=='matched'
    assert reference_instance([30,31],anns)['status']=='no_unique_known_reference'


def test_evaluation_keeps_abstention_in_denominator_and_compares_same_candidates():
    a=row('a');b=row('b')
    labels={x['candidate_id']:dict(status='matched',class_id=3,interval=[3.5,9.5],original_tiou=.6) for x in [a,b]}
    predictions={'a':combine(a,answer(3),answer(9)),'b':combine(b,answer(18),answer(18))}
    report=score_subset([a,b],labels,predictions)['correct_class_only_localization']
    assert report['count']==2 and report['vlm_mean_tiou']==.5
    assert report['original_mean_tiou']==.6 and report['mean_gain']==pytest.approx(-.1)
    assert report['success_at']['0.7']==.5 and report['valid_localization_count']==1


def test_prediction_audit_rejects_changed_confidence_or_prompt(tmp_path):
    r=row();r['prompt_sha256']={b:hashlib.sha256(build_prompt(r,b).encode()).hexdigest() for b in ['START','END']}
    start=answer(3);end=answer(9)
    start['prompt_sha256']=r['prompt_sha256']['START'];end['prompt_sha256']=r['prompt_sha256']['END']
    pred=combine(r,start,end)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(candidates=[r])))
    output=tmp_path/'predictions.jsonl';output.write_text(json.dumps(pred)+'\n')
    assert audit_predictions(tmp_path)['count']==1
    pred['confidence']=.99;output.write_text(json.dumps(pred)+'\n')
    with pytest.raises(ValueError):audit_predictions(tmp_path)


def test_fixed_letters_must_be_individual_token_ids():
    resolve=run_path(str(Path(__file__).resolve().parents[1]/'tools/run_qwen_localization.py'))['choice_tokens']
    class Tokenizer:
        def encode(self,text,**kwargs):return [ord(text)]
    assert resolve(Tokenizer())==[ord(x) for x in OPTIONS]
    class BadTokenizer:
        def encode(self,text,**kwargs):return [1,2]
    with pytest.raises(ValueError):resolve(BadTokenizer())


def test_prepared_inputs_reused_byte_exact_and_changed_gt_or_cache_rejected(tmp_path):
    prepared=tmp_path/'cpu';prepared.mkdir();previous=tmp_path/'previous';previous.mkdir()
    identity=dict(commit='frozen');manifest=prepared/'manifest.json';labels=prepared/'evaluation_labels.json'
    manifest.write_text(json.dumps(dict(candidates=[])));labels.write_text('{}')
    cache=previous/'cache.json';cache.write_text('{}')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (prepared/'inputs.json').write_text(json.dumps(dict(previous_root=str(previous),
        manifest_sha256=digest(manifest),evaluation_labels_sha256=digest(labels),
        previous_sha256={'cache.json':digest(cache)})))
    (prepared/'plan.json').write_text(json.dumps(dict(code=identity,policy=POLICY)))
    (prepared/'state.json').write_text(json.dumps(dict(phase='prepared_only')))
    output=tmp_path/'inference';output.mkdir()
    assert reuse_preparation(output,prepared,identity)==dict(candidates=[])
    assert (output/'manifest.json').read_bytes()==manifest.read_bytes()
    cache.write_text('{"changed":true}')
    with pytest.raises(ValueError):reuse_preparation(output,prepared,identity)
