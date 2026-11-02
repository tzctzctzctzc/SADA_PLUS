from tools.vlm_background_short import select


def test_fixed_count_and_insertion_order_independence():
    mapping = {str(i): 'case'+str(i) for i in range(40)}
    a = {'S1_source_val': {'candidate_map': mapping, 'irrelevant_result': 'positive'}}
    b = {'S1_source_val': {'candidate_map': dict(reversed(list(mapping.items()))), 'irrelevant_result': 'negative'}}
    assert select(a) == select(b)
    assert len(select(a)['S1_source_val']) == 8


def test_twelve_groups_fixed_budget():
    groups = {f'S{s}_{d}': {'candidate_map': {str(i):f'{s}_{d}_{i}' for i in range(18)}}
              for s in range(1,7) for d in ('source_val','target_train')}
    chosen = select(groups)
    assert len(chosen) == 12 and sum(map(len, chosen.values())) == 96
