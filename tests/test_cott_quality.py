import copy
import json
import math
import pytest
from tools.cott_protocol import *
from tools.cott_frames import sample_times

def event(v='v',start=0,end=1,label='TAKE',confidence=.9):
    return dict(video_id=v,start=start,end=end,label=label,description='take cup',confidence=confidence)

@pytest.mark.parametrize('n',[2,11,12,13,22,23,35,346])
def test_windows_cover_every_transition_without_padding(n):
    ww=windows(n)
    assert set((a,b) for w in ww for a,b in zip(w,w[1:]))==set((i,i+1) for i in range(n-1))
    assert all(len(w)==min(n,12) for w in ww)

def test_selection_independent_of_gt_and_existing_scores():
    db={f'v{i}':dict(duration=30+i,annotations=[i]) for i in range(6)}
    a=select_videos(db,'S1')
    for x in db.values():x['annotations']=[999];x['score']=1000
    assert select_videos(db,'S1')==a
    with pytest.raises(AssertionError):no_annotations(db)

def test_frame_schedule_includes_full_tail():
    t=sample_times(38.123)
    assert t[0]==0 and abs(t[-1]-38.122)<1e-9
    assert max(b-a for a,b in zip(t,t[1:]))<=.5

def test_missing_pair_or_hallucinated_extra_pair_rejected():
    x=dict(transitions=[dict(pair=i,event='none',description='no interaction') for i in range(11)])
    validate(x,'transition',0,5.5,11)
    for broken in [x['transitions'][:-1],x['transitions']+[x['transitions'][0]]]:
        with pytest.raises(ValueError):validate(dict(transitions=broken),'transition',0,5.5,11)

def test_refinement_can_reject_instead_of_creating_false_action():
    assert validate(dict(valid=False,evidence='no visible hand-object contact'),'refine',0,10)['valid'] is False
    x=event(start=-1);x.update(valid=True,evidence='cup moved')
    with pytest.raises(ValueError):validate(x,'refine',0,10)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,-.1,1.1])
def test_bad_confidence_rejected(bad):
    with pytest.raises(ValueError):validate(dict(intervals=[event(confidence=bad)]),'direct',0,2)

def test_grouping_preserves_paper_nonoverlap_and_vocabulary():
    a=event();b=event(start=.5,end=2)
    with pytest.raises(ValueError):validate(dict(intervals=[a,b]),'grouping',0,3)
    with pytest.raises(ValueError):validate(dict(intervals=[event(label='WRONG')]),'grouping',0,3)

def test_duplicate_false_positives_and_wrong_class_are_penalized():
    gt=[event()];pred=[event(),event(confidence=.8),event(label='PUT',confidence=.7)]
    r=metrics(pred,gt)['0.5']
    assert (r['tp'],r['fp'],r['fn'])==(1,2,0)
    assert r['f1']==.5 and r['precision']==1/3

def test_empty_predictions_do_not_drop_missed_gt():
    r=metrics([], [event()])['0.5']
    assert r['fn']==1 and r['map']==0 and r['matched_start_mae_sec'] is None

def test_matching_is_video_scoped_and_class_aware():
    pred=[event(v='other'),event(label='PUT')];gt=[event()]
    assert len(match_predictions(pred,gt,.5)[1])==0
    assert len(match_predictions(pred,gt,.5,False)[1])==1

def test_ap_accounts_for_ranking_and_missed_instances():
    gt=[event(),event(start=3,end=4)]
    pred=[event(start=8,end=9,confidence=.99),event(),event(start=3,end=4,confidence=.8)]
    rows,_=match_predictions(pred,gt,.5)
    assert abs(average_precision(rows,2)-2/3)<1e-10

def test_nms_same_class_only_preserves_raw_elsewhere():
    rows=[event(),event(confidence=.8),event(label='PUT')];original=copy.deepcopy(rows)
    kept,removed=deduplicate(rows)
    assert len(kept)==2 and len(removed)==1 and rows==original

def test_json_fences_supported_but_prose_or_nonobjects_rejected():
    assert parse_object('```json\n{"intervals":[]}\n```')==dict(intervals=[])
    for s in ['[]','hello {"intervals":[]}','{"intervals":[]} trailing']:
        with pytest.raises(ValueError):parse_object(s)

def test_global_grouping_prompt_contains_all_windows_and_can_cross_boundary():
    rows=[dict(pair=i,start=i*.5,end=(i+1)*.5,observations=[dict(description='same cut continues')]) for i in range(30)]
    prompt=grouping_prompt(rows,15)
    assert '"pair": 29' in prompt and 'across sliding-window boundaries' in prompt
    validate(dict(intervals=[event(start=0,end=15,label='CUT')]),'grouping',0,15)

def test_raw_images_are_not_40_fpn_tokens_or_feature_indices():
    frames=[dict(time_sec=10+i*.5) for i in range(12)]
    p=transition_prompt(frames)
    assert '15.5' in p and '"pair": 10' in p

def test_transition_contract_has_exactly_k_minus_one_explicit_slots():
    frames=[dict(time_sec=i*.5) for i in range(12)]
    p=transition_prompt(frames)
    template=p.split('OUTPUT TEMPLATE:\n')[1].split('\nEXACT PAIR')[0]
    assert [r['pair'] for r in json.loads(template)['transitions']]==list(range(11))
    assert 'exactly 11 ADJACENT FRAME PAIRS' in p
    assert 'pair 11 does not exist' in p

def test_extra_frame_caption_error_is_specific_and_never_truncated():
    rows=[dict(pair=i,description='visible change',event='continuing') for i in range(12)]
    with pytest.raises(ValueError,match='Expected exactly 11'):
        validate(dict(transitions=rows),'transition',0,5.5,11)
    assert len(rows)==12

def test_temporal_iou_values():
    assert iou(event(),event())==1
    assert iou(event(),event(start=1,end=2))==0
    assert iou(event(),event(start=0,end=2))==.5

def test_grouping_overlap_error_identifies_conflicting_numbers_without_repairing_rows():
    rows=[event(start=2.002,end=6.006),event(start=5.5055,end=8.008,label='INSERT')]
    with pytest.raises(ValueError,match='previous_end.*6.006.*next_start.*5.5055'):
        validate(dict(intervals=rows),'grouping',0,25)
    assert rows[0]['end']==6.006 and rows[1]['start']==5.5055
    prompt=grouping_prompt([],25)
    assert 'end[i] <= start[i+1]' in prompt

def test_successful_prefix_only_reuses_identical_non_grouping_request(tmp_path):
    from tools.run_cott_quality import Generator
    gen=Generator.__new__(Generator);gen.root=tmp_path;gen.model_sha='model'
    req=dict(stage='direct',system=SYSTEM,prompt='same',frames=[],lo=0.,hi=5.,pair_count=0,model_sha256='model',spec_sha256='previous')
    d=tmp_path/'calls/v/direct_0000';save(d/'request.json',req);parsed=dict(intervals=[])
    result=dict(request_sha256=digest(req),request_file_sha256=sha(d/'request.json'),attempt_sha256={},parsed=parsed);save(d/'result.json',result)
    save(tmp_path/'accepted_prefix_reuse.json',dict(calls={'v/direct_0000':dict(result_sha256=sha(d/'result.json'),request_sha256=digest(req))}))
    assert gen.call('v','direct_0000','direct','same',[],0.,5.)==parsed
    with pytest.raises(AssertionError):gen.call('v','direct_0000','direct','changed',[],0.,5.)

def test_recorded_call_resume_preserves_exact_request_and_rejects_changed_prompt(tmp_path):
    from tools.run_cott_quality import Generator
    gen=Generator.__new__(Generator);gen.root=tmp_path;gen.model_sha='model'
    req=dict(stage='grouping',system=SYSTEM,prompt='frozen prompt',frames=[],lo=0.,hi=5.,
        pair_count=0,model_sha256='model',spec_sha256=digest(SPEC))
    entry=tmp_path/'calls/v/grouping';save(entry/'request.json',req)
    parsed=dict(intervals=[event(start=0,end=3)])
    save(entry/'attempt_0.json',dict(status='valid',request_sha256=digest(req),parsed=parsed))
    # Simulate crash after immutable attempt and before result receipt, then resume without model calls.
    assert gen.call('v','grouping','grouping','frozen prompt',[],0.,5.)==parsed
    assert (entry/'result.json').exists()
    assert gen.call('v','grouping','grouping','frozen prompt',[],0.,5.)==parsed
    with pytest.raises(AssertionError):gen.call('v','grouping','grouping','changed prompt',[],0.,5.)

def test_actual_video_pts_and_local_refinement(tmp_path):
    pytest.importorskip('av')
    import shutil,subprocess
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg unavailable')
    from tools.cott_frames import probe,full_stream,refined_stream
    path=tmp_path/'synthetic.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc=size=96x64:rate=25:duration=3.12',
        '-pix_fmt','yuv420p',str(path)],check=True)
    duration=probe(path)['duration'];frames=full_stream(path,duration,tmp_path/'full')
    assert frames[0]['time_sec']==0 and frames[-1]['time_sec']>=duration-.05
    assert all(abs(f['time_sec']-f['requested_time_sec'])<=.041 for f in frames)
    local=refined_stream(path,1.01,1.16,tmp_path/'local')
    assert len(local)==12 and len({x['pts'] for x in local})<12
    assert all(.97<=x['time_sec']<=1.20 for x in local)

def test_all_three_stages_run_across_complete_video(tmp_path,monkeypatch):
    import tools.run_cott_quality as runner
    frames=[dict(path='unused',time_sec=i*.5,pts=i,sha256='unused') for i in range(25)]
    row=dict(video_id='v',frames=frames,duration=12.1,path='video.mp4',video_sha256='sha')
    monkeypatch.setattr(runner,'refined_stream',lambda *a,**k:frames[:12])
    class MockGenerator:
        root=tmp_path
        calls=[]
        def call(self,v,rid,stage,prompt,frames,lo,hi,pair_count=0):
            self.calls.append(stage)
            if stage=='transition':
                parsed=dict(transitions=[dict(pair=i,event='continuing',description='cut vegetable') for i in range(pair_count)])
            elif stage=='direct':parsed=dict(intervals=[])
            elif stage=='grouping':parsed=dict(intervals=[event(start=1,end=10,label='CUT')])
            else:parsed=dict(event(start=1,end=10,label='CUT'),valid=True,evidence='cutting')
            p=tmp_path/'calls'/v/rid
            save(p/'result.json',parsed)
            save(p/'attempt_0.json',dict(input_tokens=1,output_tokens=1,elapsed_sec=.1,status='valid'))
            return parsed
    gen=MockGenerator();runner.run_video(gen,row)
    assert gen.calls.count('transition')==3 and gen.calls.count('direct')==3
    assert gen.calls.count('grouping')==1 and gen.calls.count('refine')==1
    assert len(read(tmp_path/'videos/v/transitions.json'))==24
    assert read(tmp_path/'videos/v/predictions.json')['cott'][0]['end']==10
    old=list(gen.calls);runner.run_video(gen,row);assert gen.calls==old
