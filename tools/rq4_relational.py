"""关系式辅助的固定协议与派生工件；复用原响应，不重新查询教师。

借鉴 NPL (ICCV 2023) 的实例内绝对边界一致性；这里由 VLM 双上下文
确定内部区域，不使用目标 GT，也不把教师端点作为学生的回归标签。
"""
from copy import deepcopy
from pathlib import Path

from tools import rq4_protocol as p
from tools.rq4_artifacts import (read, checked, reference, audit_evidence,
                                 audit_target_requests, check_responses)

VARIANT = 'rq4-relational-boundary-v1'
STRATEGIES = ('relational', 'proposal_relation')
RULE = dict(threshold=.75, max_points=3, min_points=2,
            selection='first-middle-last-valid-original-fpn',
            loss='adjacent-absolute-boundary-l1-over-original-duration')


def core_interval(proposal, pair):
    if not all(pair) or p.interval_iou(*pair) < RULE['threshold']:
        return None
    start = max(proposal[0], *(x[0] for x in pair))
    stop = min(proposal[1], *(x[1] for x in pair))
    return [start, stop] if start < stop else None


def relation_rows(cache, manifest, responses):
    """用原生解码重建3x绝对区间，不从tIoU分数反推时间坐标。"""
    teacher = cache['teacher']
    check_responses(manifest, responses, teacher)
    requests = {(r['candidate']['video_id'], r['candidate']['candidate_id']): r['candidate']
                for r in manifest['requests']}
    replies = {(r['video_id'], r['candidate_id']): r for r in responses}
    result = []
    for record in cache['records']:
        row, pair = record['candidate'], []
        for scale in (2, 3):
            key = (row['video_id'], row['candidate_id']+'-context'+str(scale))
            candidate = requests[key]
            # build_cache区分合法无效输出与codec/加载技术错误，后者不可吞掉。
            single = p.build_cache([candidate], [replies[key]], teacher=teacher,
                shift=cache['shift'], identity=cache['identity'], min_iou=0., ambiguity_margin=0.)
            pair.append(single['records'][0]['target'])
        if pair[0] != record['target']:
            raise ValueError('2x interval differs from original paired cache')
        result.append(dict(candidate_sha256=p.digest(row), pair=pair,
                           core=core_interval(row['proposal'], pair)))
    return result


def self_cache(requests):
    manifest, budget, selected = audit_target_requests(requests)
    replies = [dict(candidate_id=r['candidate_id'], video_id=r['video_id'],
                    candidate_sha256=p.digest(r), teacher='sada_self') for r in selected]
    identity = dict(proposal_checkpoint_sha256=manifest['checkpoint_sha256'],
        teacher_weights_sha256=p.digest('no-vlm'), processor_sha256=p.file_sha(__file__),
        input_manifest_sha256=p.file_sha(requests), class_names_sha256=p.digest('no-vlm'))
    return p.build_cache(selected, replies, teacher='sada_self', shift=manifest['shift'],
                        identity=identity, min_iou=0., ambiguity_margin=0., budget=budget)


def build_regions(cache, proof, strategy):
    if strategy == 'relational':
        audit_evidence(proof, cache)
        rows = relation_rows(cache, checked(proof['requests']), checked(proof['responses']))
    elif strategy == 'proposal_relation':
        if set(proof) != {'schema', 'requests'} or proof['schema'] != 'rq4-proposal-relation-evidence-v1':
            raise ValueError('no-VLM control cannot depend on teacher evidence')
        checked(proof['requests'])
        if cache != self_cache(proof['requests']['path']):
            raise ValueError('control cache differs from original fixed proposal budget')
        rows = [dict(candidate_sha256=p.digest(r['candidate']), pair=None,
                     core=list(r['candidate']['proposal'])) for r in cache['records']]
    else:
        raise ValueError('not a relational strategy')
    value = dict(schema=VARIANT, strategy=strategy, rule=RULE,
                 builder_sha256=p.file_sha(__file__), cache_sha256=cache['payload_sha256'], rows=rows)
    value['payload_sha256'] = p.digest(value)
    return value


def validate_spec(spec, cache):
    if (spec.get('variant') != VARIANT or spec.get('strategy') not in STRATEGIES
            or spec.get('relation_rule') != RULE or spec.get('weight') != .1
            or spec.get('early_stop_patience') != 15 or cache['seed'] != 6
            or 'budget' not in cache or spec.get('gate_path')):
        raise ValueError('relational protocol differs from fixed first version')
    if (spec['strategy'] == 'proposal_relation') != (cache['teacher'] == 'sada_self'):
        raise ValueError('no-VLM control/teacher identity mismatch')
    regions = checked(dict(path=spec['regions_path'], sha256=spec['regions_file_sha256']))
    if (regions.get('schema') != VARIANT or regions.get('strategy') != spec['strategy']
            or regions.get('rule') != RULE or regions.get('builder_sha256') != p.file_sha(__file__)
            or regions.get('cache_sha256') != cache['payload_sha256']
            or regions.get('payload_sha256') != p.digest({k:v for k,v in regions.items() if k!='payload_sha256'})):
        raise ValueError('relational regions identity changed')
    if len(regions['rows']) != len(cache['records']):
        raise ValueError('relational regions candidate count mismatch')
    for relation, record in zip(regions['rows'], cache['records']):
        row = record['candidate']
        if relation['candidate_sha256'] != p.digest(row):
            raise ValueError('relational region/candidate binding mismatch')
        if spec['strategy'] == 'relational':
            pair = relation['pair']
            if len(pair) != 2 or pair[0] != record['target']:
                raise ValueError('invalid paired regions')
            for interval in pair:
                if interval is not None: p.span(interval, row['video_duration'])
            expected = core_interval(row['proposal'], pair)
        else:
            if relation['pair'] is not None:
                raise ValueError('no-VLM control has teacher intervals')
            expected = row['proposal']
        if relation['core'] != expected:
            raise ValueError('region intersection differs')
    return regions


def audit_spec(spec, cache):
    regions = validate_spec(spec, cache)
    proof = checked(dict(path=spec['evidence_path'], sha256=spec['evidence_file_sha256']))
    if regions != build_regions(cache, proof, spec['strategy']):
        raise ValueError('regions do not replay from original responses')
    return regions


def prepare(output, *, original_spec=None, requests=None):
    """两个显式入口：VLM用旧工件；无VLM对照只接原候选请求，不接教师输出。"""
    from tools.rq4_source_acceptance import save
    if (original_spec is None) == (requests is None):
        raise ValueError('provide exactly one of original-spec or proposal-requests')
    if original_spec is not None:
        old = read(original_spec)
        cache = p.load_cache(old['cache_path'], old['cache_file_sha256'])
        proof = checked(dict(path=old['evidence_path'], sha256=old['evidence_file_sha256']))
        strategy = 'relational'
    else:
        cache = self_cache(requests)
        proof = dict(schema='rq4-proposal-relation-evidence-v1', requests=reference(requests))
        strategy = 'proposal_relation'
    regions = build_regions(cache, proof, strategy)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('cache',cache), ('evidence',proof), ('regions',regions)):
        save(output/(name+'.json'), value)
    spec = dict(schema='rq4-boundary-training-v1', variant=VARIANT, strategy=strategy,
        relation_rule=deepcopy(RULE), weight=.1, normalizer='common-routable-candidates',
        activation='sada-adaptation-phase', early_stop_patience=15, protocol_frozen=False)
    for name in ('cache','evidence','regions'):
        spec[name+'_path'] = str(output/(name+'.json'))
        spec[name+'_file_sha256'] = p.file_sha(spec[name+'_path'])
    save(output/'spec.json', spec)
    return spec


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--original-spec')
    group.add_argument('--proposal-requests', dest='requests')
    parser.add_argument('--output', required=True)
    spec = prepare(**vars(parser.parse_args()))
    print('RQ4_RELATIONAL_ARTIFACTS_READY', spec['strategy'], 'NOT_A_TRAINING_RESULT')


if __name__ == '__main__': main()
