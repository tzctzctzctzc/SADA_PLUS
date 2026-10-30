from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.prototype_refresh import (POLICY,CACHE_SCHEMA,SCHEMA,selected_coordinates,
    sanitized_rows,config_for,assert_pair,validate_cache)
from tools.baseline_protocol import save,sha
from libs.models.SADA.prototype_capacity import specification


def pool():
    return [dict(candidate_id=f'v-p{i:07d}',video_id='v',absolute_feature_index=i,
        interval=[float(i),float(i)+1],level=i%3,class_id=9,gt_label='must not be read') for i in range(200)]


def test_vlm_inputs_drop_student_classes_and_all_diagnostic_gt():
    result=sanitized_rows(pool(),{'v':{}})
    assert len(result)==200 and all(set(r)=={'candidate_id','video_id','absolute_feature_index',
        'proposal_start_sec','proposal_end_sec','source_fpn_levels'} for r in result)
    assert result[17]['source_fpn_levels']==[2] and result[17]['proposal_start_sec']==17.
    bad=pool();bad[-1]['candidate_id']=bad[0]['candidate_id']
    with pytest.raises(ValueError):sanitized_rows(bad,{'v':{}})


def entries():
    rows=[]
    for i in range(200):
        row=sanitized_rows(pool(),{'v':{}})[i]
        score=.5+i/1000
        row.update(class_probabilities=[score]+[(1-score)/9]*9,class_id=0,
                   semantic_label='TAKE',participation_selected=i>=160)
        rows.append(row)
    return rows


def test_top20_is_original_probability_ranking_and_ties_do_not_refill_pool():
    rows=entries();assert selected_coordinates(rows)==[['v',i] for i in range(199,159,-1)]
    for r in rows:r['class_probabilities']=[.1]*10
    assert selected_coordinates(rows)==[]


def test_fresh_config_changes_only_cache_and_identity(tmp_path):
    old=dict(seed=123,loader={'batch_size':2},opt={'learning_rate':.0001},
        baseline=dict(regime='uda',shift='S1',target_gt_used=False,code={'commit':'old'},tmpdir='/old'),
        prototype_capacity=dict(variant='control',specification=specification('control'),
            prepared_cache='/old.json',prepared_cache_sha256='old'))
    path=tmp_path/'cache.json';save(path,{})
    new=config_for(old,path,{'commit':'new'},tmp_path);assert_pair(new,old)
    assert new['prototype_capacity']['specification']==old['prototype_capacity']['specification']
    for mutate in [lambda c:c['opt'].update(learning_rate=.0002),
                   lambda c:c.update(seed=6),lambda c:c['baseline'].update(target_gt_used=True),
                   lambda c:c['prototype_capacity'].update(variant='wide_deep')]:
        changed=deepcopy(new);mutate(changed)
        with pytest.raises(ValueError):assert_pair(changed,old)


def test_cache_budget_and_sha_are_verified(tmp_path):
    evidence=tmp_path/'predictions.jsonl';evidence.write_text('frozen')
    path=tmp_path/'prepared.json'
    payload=dict(schema=CACHE_SCHEMA,target_gt_used=False,policy=POLICY,selected_count=40,
        by_video={'v':entries()},source_artifacts={str(evidence):sha(evidence)})
    save(path,payload);assert validate_cache(path)['selected_count']==40
    payload['by_video']['v'][0]['participation_selected']=True;save(path,payload)
    with pytest.raises(ValueError):validate_cache(path)
    payload['by_video']['v'][0]['participation_selected']=False;save(path,payload)
    evidence.write_text('changed')
    with pytest.raises(ValueError):validate_cache(path)


def test_reused_prompt_is_class_blind_and_native_video():
    from tools.qwen_source_interface import build_prompt,build_native_video_messages
    prompt=build_prompt('v-p0000001',8,'baseline','native_video')
    assert 'video above contains 8 frames' in prompt and 'Choose exactly one option' in prompt
    msg=build_native_video_messages([Path('a.png')]*8,prompt,224*224,448*448,8.)
    assert msg[0]['content'][0]['type']=='video' and msg[0]['content'][0]['sample_fps']==8.


def test_original_frame_rule_uses_eight_midpoints_and_no_interpolation():
    from tools.refresh_frames import proposal_frame_indices
    assert proposal_frame_indices(1.,3.,30.)==[33,41,48,56,63,71,78,86]
    with pytest.raises(ValueError):proposal_frame_indices(3.,1.,30.)


def test_teacher_predictions_are_recomputed_and_frozen(tmp_path):
    from tools.prototype_refresh import freeze_scores,SHIFTS
    from tools.qwen_source_interface import build_prompt,normalize_digit_logits,digit_to_label
    shifts={};pred=[]
    for s in SHIFTS:
        rows=sanitized_rows(pool(),{'v':{}})
        for x in rows:
            x.update(frames=list(range(8)),frame_sha256=['pixel']*8)
            digit=[float(i) for i in range(10)];result=normalize_digit_logits(digit,digit_to_label(x['candidate_id']))
            pred.append(dict(shift=s,candidate_id=x['candidate_id'],frame_sha256=x['frame_sha256'],
                digit_logits=digit,class_id=result['class_id'],class_probabilities=result['class_probabilities'],
                prompt_sha256=hashlib.sha256(build_prompt(x['candidate_id'],8,'baseline','native_video').encode()).hexdigest()))
        path=tmp_path/'candidates'/s/'manifest.json';save(path,dict(candidates=rows))
        shifts[s]=dict(manifest=str(path),sha256=sha(path))
    save(tmp_path/'candidate_audit.json',dict(shifts=shifts))
    teacher=tmp_path/'teacher';teacher.mkdir()
    for name in ['model_identity.json','device_map.json','first_cuda_four.json']:save(teacher/name,{})
    prediction=teacher/'predictions.jsonl';prediction.write_text(''.join(json.dumps(x)+'\n' for x in pred))
    save(teacher/'inference_complete.json',dict(status='passed',count=1200,
        predictions_sha256=sha(prediction),candidate_audit_sha256=sha(tmp_path/'candidate_audit.json'),
        policy=POLICY,target_gt_used=False,model_identity_sha256=sha(teacher/'model_identity.json'),
        device_map_sha256=sha(teacher/'device_map.json'),guard_path=str(teacher/'first_cuda_four.json'),guard_sha256=sha(teacher/'first_cuda_four.json')))
    out=freeze_scores(tmp_path);assert len(out)==6
    for row in out.values():assert validate_cache(row['path'])['selected_count']==40
    # 不能用float32/低精度转换绕过原始概率的严格复算。
    import struct
    original=deepcopy(pred[0]['class_probabilities'])
    pred[0]['class_probabilities']=[struct.unpack('f',struct.pack('f',x))[0] for x in original]
    prediction.write_text(''.join(json.dumps(x)+'\n' for x in pred))
    done=json.loads((teacher/'inference_complete.json').read_text());done['predictions_sha256']=sha(prediction)
    save(teacher/'inference_complete.json',done)
    with pytest.raises(ValueError):freeze_scores(tmp_path)
    pred[0]['class_probabilities']=original
    pred[0]['digit_logits'][0]=99.
    prediction.write_text(''.join(json.dumps(x)+'\n' for x in pred))
    done=json.loads((teacher/'inference_complete.json').read_text());done['predictions_sha256']=sha(prediction)
    save(teacher/'inference_complete.json',done)
    with pytest.raises(ValueError):freeze_scores(tmp_path)


def test_prepared_only_reuse_preserves_frames_and_rejects_prior_scoring(tmp_path):
    from tools.run_prototype_refresh import reuse_frames
    from tools.prototype_refresh import SHIFTS
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir()
    save(old/'plan.json',dict(policy=POLICY,code={'commit':'old'}))
    save(old/'state.json',dict(phase='waiting_four_gpu_teacher',pid=99999999))
    save(old/'monitor/stopped_for_single_teacher.json',dict(remaining_pids=[],scored=0,formal_completed=0))
    shifts={}
    for s in SHIFTS:
        path=old/'candidates'/s/'manifest.json'
        rows=[dict(candidate_id=str(i),frames=list(range(8))) for i in range(200)]
        for row in rows:
            for i in row['frames']:
                p=path.parent/'frames'/row['candidate_id']/(str(i)+'.png');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'preserved')
        save(path,dict(candidates=rows,target_gt_used=False,frame_protocol=POLICY['frame_protocol']))
        shifts[s]=dict(manifest=str(path),sha256=sha(path),candidate_count=200)
    save(old/'candidate_audit.json',dict(status='passed',count=1200,shifts=shifts))
    before=sha(old/'candidate_audit.json');reuse_frames(old,new)
    assert sha(new/'candidate_audit.json')==before==sha(old/'candidate_audit.json')
    p=old/'teacher/predictions.jsonl';p.parent.mkdir();p.write_text('one existing prediction')
    with pytest.raises(ValueError):reuse_frames(old,new)


def test_freeze_uses_the_actual_teacher_interpreter():
    from tools.run_prototype_refresh import freeze_command
    teacher=Path('/vepfs/vlms/envs/rq1-cu121/bin/python')
    cmd=freeze_command(teacher,Path('/result'))
    assert cmd[0]==str(teacher) and '--freeze-only' in cmd and cmd[-1]=='/result'


def test_scoring_recovery_copies_raw_probabilities_and_refuses_formal_retraining(tmp_path):
    from tools.run_prototype_refresh import reuse_scores
    from tools.prototype_refresh import SHIFTS
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir()
    save(old/'plan.json',dict(policy=POLICY,code={'commit':'scoring'}))
    save(old/'state.json',dict(pid=99999999,child_pid=99999998))
    shifts={}
    for s in SHIFTS:
        p=old/'candidates'/s/'manifest.json';save(p,{})
        shifts[s]=dict(manifest=str(p),sha256=sha(p))
    save(old/'candidate_audit.json',dict(status='passed',count=1200,shifts=shifts))
    for name in ['model_identity.json','device_map.json','guard.json']:save(old/'teacher'/name,{})
    raw=old/'teacher/predictions.jsonl';raw.write_text('{"class_probabilities":[0.9225347256207934]}\n')
    save(old/'teacher/inference_complete.json',dict(status='passed',count=1200,code={'commit':'scoring'},
        policy=POLICY,target_gt_used=False,candidate_audit_sha256=sha(old/'candidate_audit.json'),
        predictions_sha256=sha(raw),model_identity_sha256=sha(old/'teacher/model_identity.json'),
        device_map_sha256=sha(old/'teacher/device_map.json'),guard_path=str(old/'teacher/guard.json'),
        guard_sha256=sha(old/'teacher/guard.json')))
    reuse_scores(old,new)
    assert (new/'teacher/predictions.jsonl').read_bytes()==raw.read_bytes()
    assert json.loads((new/'predecessor.json').read_text())['new_teacher_queries']==0
    p=old/'formal/single_refresh/S1/metrics.jsonl';p.parent.mkdir(parents=True);p.write_text('one complete epoch')
    with pytest.raises(ValueError):reuse_scores(old,new)


def test_real_python312_probability_fixture_reproduces_without_tolerance():
    import subprocess,os
    teacher=Path('/vepfs/vlms/envs/rq1-cu121/bin/python')
    if not teacher.exists():pytest.skip('需要实际评分环境的Python3.12')
    from tools.baseline_protocol import CODE
    expected=[0.0005102395376594811,0.00018770663597774496,0.9225347256207934,
        0.0001138498297520208,0.0003973749514831954,0.00014618607508716693,
        0.07572626168308169,0.0001138498297520208,0.0002410200914763782,2.8785744936940133e-05]
    script="""import sys,json
sys.path.insert(0,sys.argv[1])
from tools.qwen_source_interface import normalize_digit_logits,digit_to_label
d=normalize_digit_logits([25.,30.75,25.5,33.25,24.75,24.25,22.875,25.75,24.5,24.25],digit_to_label('P27_103-p0004548'))
print(json.dumps(d['class_probabilities']))
"""
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    actual=json.loads(subprocess.check_output([str(teacher),'-B','-c',script,str(CODE)],env=env,text=True))
    assert actual==expected
