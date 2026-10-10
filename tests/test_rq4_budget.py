"""CPU预算验收，不导入CUDA或改变教师响应。"""
import copy
import json
import pytest
from tools import rq4_protocol as p


def pool():
    return [dict(candidate_id='p'+str(i), video_id='v'+str(v), class_id=1,
                 proposal=[2., 6.], window=[1., 8.], video_duration=10.,
                 anchor_sec=4., fpn_level=0) for v in range(5) for i in range(4)]


def cache(rows, budget=None):
    identity = {k: 'a'*64 for k in ('proposal_checkpoint_sha256', 'teacher_weights_sha256',
                 'processor_sha256', 'input_manifest_sha256', 'class_names_sha256')}
    responses = [dict(video_id=r['video_id'], candidate_id=r['candidate_id'],
                     candidate_sha256=p.digest(r), teacher='timelens2_8b',
                     raw_text='[[1.5, 4.5]]', metadata={}) for r in rows]
    return p.build_cache(rows, responses, teacher='timelens2_8b', shift='S1',
                         identity=identity, min_iou=.1, ambiguity_margin=.05, budget=budget)


def test_shift_floor_not_per_video():
    rows = pool()
    chosen = p.select_candidates(rows, .2)
    assert len(chosen) == 4  # 旧逐视频floor会选0。
    assert chosen == p.select_candidates(rows[::-1], .2)
    assert p.build_budget(rows, 'S1') == p.build_budget(rows[::-1], 'S1')


def test_budget_cache_roundtrip():
    b = p.build_budget(pool(), 'S1')
    result = cache(p.select_candidates(pool(), .2), b)
    assert p.validate_cache(result) is result
    assert b['pool_count'] == 20 and b['selected_count'] == 4


def test_full_pool_cannot_claim_twenty_percent():
    with pytest.raises(ValueError, match='fixed shift budget'):
        cache(pool(), p.build_budget(pool(), 'S1'))


@pytest.mark.parametrize('field,value', [('selected_count', 20), ('pool_sha256', 'f'*64),
                                        ('shift', 'S2'), ('fraction', 1.)])
def test_budget_recomputed_even_after_rehash(field, value):
    b = p.build_budget(pool(), 'S1')
    b[field] = value
    b['payload_sha256'] = p.digest({k: v for k, v in b.items() if k != 'payload_sha256'})
    with pytest.raises(ValueError, match='recomputed'):
        cache(p.select_candidates(pool(), .2), b)


def test_contexts_cannot_inflate_candidate_count():
    rows = pool()
    rows[0]['candidate_id'] += '-context2'
    with pytest.raises(ValueError, match='original candidate'):
        p.build_budget(rows, 'S1')


def test_duplicate_rejected_and_selection_does_not_mutate():
    rows = pool()
    with pytest.raises(ValueError, match='duplicate'):
        p.build_budget(rows + [rows[0]], 'S1')
    before = copy.deepcopy(rows)
    p.select_candidates(rows, .2)[0]['proposal'][0] = 0
    assert rows == before


def test_formal_launch_rejects_legacy_unbudgeted_cache(tmp_path):
    path = tmp_path / 'cache.json'
    path.write_text(json.dumps(cache(pool())), encoding='utf-8')
    with pytest.raises(ValueError, match='fixed shift budget'):
        p.assert_launch_ready({'rq4_boundary': dict(protocol_frozen=True, strategy='direct',
            cache_path=str(path), cache_file_sha256=p.file_sha(path))})


def test_formal_launch_accepts_budgeted_cache(tmp_path):
    rows = p.select_candidates(pool(), .2)
    path = tmp_path / 'cache.json'
    path.write_text(json.dumps(cache(rows, p.build_budget(pool(), 'S1'))), encoding='utf-8')
    p.assert_launch_ready({'rq4_boundary': dict(protocol_frozen=True, strategy='direct',
        cache_path=str(path), cache_file_sha256=p.file_sha(path))})
