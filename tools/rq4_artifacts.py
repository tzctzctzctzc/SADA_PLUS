"""RQ4双上下文工件接线：校验原请求，再构建Direct缓存和一致性门控。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from tools import rq4_protocol as p


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=p.file_sha(path))


def checked(ref):
    if p.file_sha(ref['path']) != ref['sha256']:
        raise ValueError('evidence file SHA mismatch: '+ref['path'])
    return read(ref['path'])


def check_responses(manifest, responses, teacher):
    requests = manifest['requests']
    keys = lambda r: (r['video_id'], r['candidate_id'])
    lookup = {keys(r): r for r in responses}
    if len(lookup) != len(responses) or len(responses) != len(requests):
        raise ValueError('missing/duplicate teacher responses')
    for request in requests:
        row = request['candidate']
        r = lookup.pop(keys(row))
        if (r['teacher'] != teacher or r['candidate_sha256'] != p.digest(row)
                or r['request_sha256'] != p.digest(request)):
            raise ValueError('teacher response/request identity mismatch')
    if lookup:
        raise ValueError('extra teacher responses')


def audit_target_requests(path, verify_clips=True):
    from tools.prepare_rq4_target_candidates import check_input, original_candidate, paired_candidates
    manifest = read(path)
    if (manifest.get('schema') != 'rq4-target-requests-v1' or manifest['split'] != 'target_train'
            or manifest['target_gt_used'] is not False or manifest['seed'] != 6):
        raise ValueError('not formal target-only requests')
    for filename, sha in ((manifest['proposals_path'], manifest['proposals_sha256']),
                          (manifest['metadata_path'], manifest['metadata_sha256']),
                          (manifest['budget_path'], manifest['budget_file_sha256'])):
        if p.file_sha(filename) != sha:
            raise ValueError('target input artifact changed')
    rows = [json.loads(s) for s in Path(manifest['proposals_path']).read_text().splitlines() if s.strip()]
    meta = read(manifest['metadata_path'])
    check_input(meta, rows)
    if (sorted(meta['expected_video_ids']) != manifest['target_video_ids'] or
            meta['selection_teacher_provenance']['selection_teacher_checkpoint_sha256'] != manifest['checkpoint_sha256']):
        raise ValueError('target membership/checkpoint binding differs')
    pool, rejected = [], []
    for row in rows:
        duration = float(manifest['videos'][row['video_id']]['probe']['format']['duration'])
        try:
            pool.append(original_candidate(row, duration, manifest['shift']))
        except ValueError as exc:
            rejected.append(dict(candidate_id=row['candidate_id'], reason=str(exc)))
    budget = p.build_budget(pool, manifest['shift'])
    if (budget != read(manifest['budget_path']) or rejected != manifest['rejected_geometry']
            or budget['payload_sha256'] != manifest['budget_payload_sha256']):
        raise ValueError('target budget differs from original proposal pool')
    selected = p.select_candidates(pool, .2)
    expected = [item for row in selected for item in paired_candidates(row)]
    if expected != [r['candidate'] for r in manifest['requests']]:
        raise ValueError('paired requests changed fixed candidate selection')
    for r in manifest['requests']:
        if [r['clip_start_sec'], r['clip_end_sec']] != r['candidate']['window']:
            raise ValueError('target clip origin mismatch')
        if verify_clips and p.file_sha(r['clip_path']) != r['clip_sha256']:
            raise ValueError('target clip hash changed')
    return manifest, budget, selected


def audit_source_chain(requests, folder, annotations, teacher):
    """从原响应和源标注重算全部诊断/曲线；不把摘要字段当证明。"""
    from tools.rq4_source_acceptance import summarize, calibrate
    folder = Path(folder)
    manifest, identity = read(requests), read(folder/'identity.json')
    if (manifest.get('split') != 'source_validation' or manifest.get('diagnostic_only') is not True
            or identity['teacher'] != teacher or identity['seed'] != 6
            or identity['requests_sha256'] != p.file_sha(requests)):
        raise ValueError('source evidence identity mismatch')
    responses = read(folder/'responses.json')
    check_responses(manifest, responses, teacher)
    original = read(folder/'source_calibration.json')
    if original['annotation_sha256'] != p.file_sha(annotations):
        raise ValueError('source annotations changed')
    with tempfile.TemporaryDirectory(prefix='rq4_source_replay_') as temp:
        diagnostics = Path(temp)/'diagnostics.json'
        calibration = Path(temp)/'calibration.json'
        summarize(SimpleNamespace(requests=requests, responses=folder/'responses.json', output=diagnostics))
        if read(diagnostics) != read(folder/'diagnostics.json'):
            raise ValueError('source diagnostics do not replay')
        calibrate(SimpleNamespace(requests=requests, diagnostics=diagnostics,
                                  annotations=annotations, output=calibration))
        if read(calibration) != original:
            raise ValueError('source calibration does not replay')
    return dict(requests=reference(requests), responses=reference(folder/'responses.json'),
                identity=reference(folder/'identity.json'), diagnostics=reference(folder/'diagnostics.json'),
                calibration=reference(folder/'source_calibration.json'), annotations=reference(annotations))


def paired_cache(requests_path, responses_path, identity_path):
    manifest, budget, selected = audit_target_requests(requests_path)
    identity = read(identity_path)
    if identity['requests_sha256'] != p.file_sha(requests_path) or identity['seed'] != 6:
        raise ValueError('target teacher identity mismatch')
    teacher = identity['teacher']
    responses = read(responses_path)
    check_responses(manifest, responses, teacher)
    by_id = {(r['video_id'], r['candidate_id']): r for r in responses}
    primary, scores = [], {}
    for row in selected:
        pair = []
        for scale in (2, 3):
            response = by_id[(row['video_id'], row['candidate_id']+'-context'+str(scale))]
            request = next(q for q in manifest['requests'] if q['candidate']['video_id'] == row['video_id']
                           and q['candidate']['candidate_id'] == response['candidate_id'])
            qrow = request['candidate']
            if response.get('error') and response.get('output_status') != 'native_interval_missing':
                raise RuntimeError(response['error'])
            try:
                decoded = p.decode_response(teacher, response['raw_text'], response['metadata'],
                                            qrow['window'][1]-qrow['window'][0])
                target, _ = p.match_response(qrow, decoded, 0., 0.)
            except (ValueError, TypeError):
                target = None
            pair.append(target)
            if scale == 2:
                adapted = deepcopy(response)
                adapted.update(candidate_id=row['candidate_id'], candidate_sha256=p.digest(row))
                primary.append(adapted)
        scores[p.digest(row)] = p.interval_iou(*pair) if all(pair) else -1.
    ids = dict(proposal_checkpoint_sha256=manifest['checkpoint_sha256'],
               teacher_weights_sha256=p.digest(identity['weights']),
               processor_sha256=identity['processor_sha256'],
               input_manifest_sha256=p.file_sha(requests_path),
               class_names_sha256=responses[0]['class_names_sha256'])
    if any(r['class_names_sha256'] != ids['class_names_sha256'] for r in responses):
        raise ValueError('teacher class vocabulary differs')
    cache = p.build_cache(selected, primary, teacher=teacher, shift=manifest['shift'], identity=ids,
                          min_iou=0., ambiguity_margin=0., budget=budget)
    return cache, scores


def build_artifacts(requests, responses, identity, source_requests, source_folder, annotations, output):
    from tools.rq4_source_acceptance import save
    cache, scores = paired_cache(requests, responses, identity)
    source = audit_source_chain(source_requests, source_folder, annotations, cache['teacher'])
    source_identity = checked(source['identity'])
    if p.digest(source_identity['weights']) != cache['identity']['teacher_weights_sha256']:
        raise ValueError('source/target teacher weights or processor assets differ')
    proof = dict(schema='rq4-paired-evidence-v1', requests=reference(requests), responses=reference(responses),
                 identity=reference(identity), source=source,
                 scorer_code_sha256=p.file_sha(__file__))
    gate = dict(schema='rq4-source-gate-v1', fit_split='source_validation', target_gt_used=False,
        teacher_identity=cache['identity'], cache_sha256=cache['payload_sha256'],
        source_calibration_sha256=source['calibration']['sha256'], scorer_code_sha256=p.file_sha(__file__),
        score_name='paired-context-absolute-tiou', source_video_ids=checked(source['calibration'])['source_video_ids'],
        threshold=.75, scores=scores, interpretation='consistency filter, not a guarantee of accuracy',
        paired_evidence=proof)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output/'cache.json', cache)
    save(output/'gate.json', gate)
    save(output/'evidence.json', proof)
    for strategy in ('direct', 'gated'):
        spec = dict(schema='rq4-boundary-training-v1', strategy=strategy, weight=.1,
            normalizer='common-routable-candidates', activation='sada-adaptation-phase',
            early_stop_patience=15, protocol_frozen=False,
            cache_path=str(output/'cache.json'), cache_file_sha256=p.file_sha(output/'cache.json'),
            evidence_path=str(output/'evidence.json'), evidence_file_sha256=p.file_sha(output/'evidence.json'))
        if strategy == 'gated':
            spec.update(gate_path=str(output/'gate.json'), gate_file_sha256=p.file_sha(output/'gate.json'))
        save(output/(strategy+'.json'), spec)
    return cache, gate


def audit_evidence(proof, cache, gate=None):
    if proof['schema'] != 'rq4-paired-evidence-v1' or proof['scorer_code_sha256'] != p.file_sha(__file__):
        raise ValueError('paired evidence scorer identity differs')
    for key in ('requests', 'responses', 'identity'):
        checked(proof[key])
    rebuilt, scores = paired_cache(*(proof[k]['path'] for k in ('requests', 'responses', 'identity')))
    if rebuilt != cache:
        raise ValueError('training cache differs from paired teacher evidence')
    src = proof['source']
    for ref in src.values():
        checked(ref)
    replay = audit_source_chain(src['requests']['path'], Path(src['identity']['path']).parent,
                                src['annotations']['path'], cache['teacher'])
    if replay != src or p.digest(checked(src['identity'])['weights']) != cache['identity']['teacher_weights_sha256']:
        raise ValueError('source provenance differs')
    if gate is not None and (gate['threshold'] != .75 or gate['scores'] != scores
            or gate['source_calibration_sha256'] != src['calibration']['sha256']
            or gate['source_video_ids'] != checked(src['calibration'])['source_video_ids']):
        raise ValueError('gate differs from frozen consistency rule or source evidence')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('requests', 'responses', 'identity', 'source-requests', 'source-folder', 'annotations', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    cache, gate = build_artifacts(**vars(args))
    audit_evidence(gate['paired_evidence'], cache, gate)
    print('RQ4_PAIRED_ARTIFACTS_READY', len(cache['records']), 'threshold', gate['threshold'])


if __name__ == '__main__':
    main()
