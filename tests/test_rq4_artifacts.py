import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import rq4_protocol as p
from tools import rq4_artifacts as a
from tools.prepare_rq4_target_candidates import original_candidate, paired_candidates
from tools.rq4_source_acceptance import summarize, calibrate


def write(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


def request(row, clip):
    return dict(candidate=row, clip_path=str(clip), clip_sha256=p.file_sha(clip),
                clip_start_sec=row['window'][0], clip_end_sec=row['window'][1])


def response(r):
    row = r['candidate']
    return dict(candidate_id=row['candidate_id'], video_id=row['video_id'], teacher='timelens2_8b',
        candidate_sha256=p.digest(row), request_sha256=p.digest(r), class_names_sha256='c'*64,
        raw_text='[[2,6]]', metadata={}, error='')


def fixture(tmp):
    clip = tmp/'clip.mkv'; clip.write_bytes(b'unit-test-only')
    raw = [dict(video_id='target', candidate_id='p'+str(i), teacher_top1_class=1,
        proposal_start_sec=2., proposal_end_sec=6., absolute_center_time=3., source_fpn_levels=[1]) for i in range(10)]
    proposals = tmp/'proposals.jsonl'
    proposals.write_text('\n'.join(json.dumps(r) for r in raw))
    metadata = tmp/'metadata.json'
    write(metadata, dict(domain='target', split='train', seed=6, selection_uses_gt=False,
        max_per_video=30, max_proposals=200, selected_proposals=10, expected_video_ids=['target'],
        selection_teacher_provenance=dict(selection_teacher_checkpoint_sha256='a'*64)))
    pool = [original_candidate(r, 10., 'S1') for r in raw]
    budget = p.build_budget(pool, 'S1'); write(tmp/'budget.json', budget)
    requests = [request(r, clip) for row in p.select_candidates(pool, .2) for r in paired_candidates(row)]
    target = dict(schema='rq4-target-requests-v1', split='target_train', target_gt_used=False,
        seed=6, shift='S1', proposals_path=str(proposals), proposals_sha256=p.file_sha(proposals),
        metadata_path=str(metadata), metadata_sha256=p.file_sha(metadata), target_video_ids=['target'],
        checkpoint_sha256='a'*64, budget_path=str(tmp/'budget.json'),
        budget_file_sha256=p.file_sha(tmp/'budget.json'), budget_payload_sha256=budget['payload_sha256'],
        rejected_geometry=[], videos={'target':dict(probe={'format':dict(duration='10')})}, requests=requests)
    write(tmp/'target.json', target)
    write(tmp/'target.responses.json', [response(r) for r in requests])
    identity = dict(teacher='timelens2_8b', seed=6, weights={'w':'b'*64}, processor_sha256='d'*64,
                    requests_sha256=p.file_sha(tmp/'target.json'))
    write(tmp/'target.identity.json', identity)
    source = tmp/'source'; source.mkdir()
    row = dict(pool[0], video_id='source')
    src_requests = [request(r, clip) for r in paired_candidates(row)]
    write(tmp/'source.requests.json', dict(diagnostic_only=True, split='source_validation', seed=6,
        shift='S1', source_subsets=['source_val'], videos={'source':{}}, requests=src_requests))
    write(source/'responses.json', [response(r) for r in src_requests])
    write(source/'identity.json', dict(identity, requests_sha256=p.file_sha(tmp/'source.requests.json')))
    write(tmp/'annotations.json', dict(database={'source':dict(subset='source_val',
        annotations=[dict(label_id=1, segment=[2.,6.])])}))
    summarize(SimpleNamespace(requests=tmp/'source.requests.json', responses=source/'responses.json',
                              output=source/'diagnostics.json'))
    calibrate(SimpleNamespace(requests=tmp/'source.requests.json', diagnostics=source/'diagnostics.json',
                              annotations=tmp/'annotations.json', output=source/'source_calibration.json'))
    return dict(requests=str(tmp/'target.json'), responses=str(tmp/'target.responses.json'),
        identity=str(tmp/'target.identity.json'), source_requests=str(tmp/'source.requests.json'),
        source_folder=str(source), annotations=str(tmp/'annotations.json'), output=str(tmp/'output'))


def test_full_paired_chain_and_formal_gate(tmp_path):
    args = fixture(tmp_path)
    cache, gate = a.build_artifacts(**args)
    a.audit_evidence(gate['paired_evidence'], cache, gate)
    assert len(cache['records']) == 2 and gate['threshold'] == .75
    for strategy in ('direct', 'gated'):
        spec = a.read(Path(args['output'])/(strategy+'.json'))
        spec['protocol_frozen'] = True
        p.assert_launch_ready({'rq4_boundary':spec})


def test_source_curves_must_replay_not_just_have_sha(tmp_path):
    args = fixture(tmp_path)
    path = Path(args['source_folder'])/'source_calibration.json'
    data = a.read(path); data['curves'][0]['mean_delta'] = 99.; write(path, data)
    with pytest.raises(ValueError, match='calibration does not replay'):
        a.build_artifacts(**args)


def test_source_membership_cannot_be_claimed(tmp_path):
    args = fixture(tmp_path)
    path = Path(args['source_folder'])/'source_calibration.json'
    data = a.read(path); data['source_video_ids'] = ['fake_source']; write(path, data)
    with pytest.raises(ValueError, match='calibration does not replay'):
        a.build_artifacts(**args)


def test_target_scores_cannot_be_replaced(tmp_path):
    args = fixture(tmp_path)
    cache, gate = a.build_artifacts(**args)
    gate['scores'] = {k:0.123 for k in gate['scores']}
    with pytest.raises(ValueError, match='frozen consistency'):
        p.validate_gate(gate, cache)


def test_duplicate_response_rejected(tmp_path):
    args = fixture(tmp_path)
    responses = a.read(args['responses']); responses[1] = responses[0]
    write(Path(args['responses']), responses)
    with pytest.raises(ValueError, match='duplicate'):
        a.build_artifacts(**args)


def test_timeple_missing_token_is_invalid_not_text_fallback():
    from test_rq4_boundary import candidate, identity
    row = candidate()
    r = dict(candidate_id=row['candidate_id'], video_id=row['video_id'], teacher='timeple_8b',
        candidate_sha256=p.digest(row), raw_text='from 2 to 4 seconds', metadata={},
        error='native interval missing', output_status='native_interval_missing')
    cache = p.build_cache([row], [r], teacher='timeple_8b', shift='S1', identity=identity(),
                           min_iou=0., ambiguity_margin=0.)
    assert cache['records'][0]['reason'] == 'invalid_output'
    r['raw_text'] = '<|TIMESPAN|>'
    with pytest.raises(ValueError, match='inconsistent native'):
        p.build_cache([row], [r], teacher='timeple_8b', shift='S1', identity=identity(),
                       min_iou=0., ambiguity_margin=0.)
