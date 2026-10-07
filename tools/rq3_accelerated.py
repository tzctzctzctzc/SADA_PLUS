"""显式的大batch探索协议；旧batch2基线仅作参考，不宣称严格配对。"""
from copy import deepcopy

SCHEMA = 'rq3-accelerated-batch-v1'


def apply_batch(cfg, batch):
    if batch == 2:
        return
    aux = cfg.get('rq3_auxiliary', {})
    from tools.rq3_protocol import REVISION
    if batch not in (4, 8) or aux.get('schema') != REVISION or aux.get('strategy') not in ('verify', 'prototype'):
        raise ValueError('accelerated batch requires revised Verify/Prototype and batch4/8')
    cfg['loader']['batch_size'] = batch
    aux['acceleration'] = dict(schema=SCHEMA, batch_size=batch, reference_batch_size=2,
                               comparison='exploratory-unmatched-batch', learning_rate='unchanged')
    from tools.rq2_formal import build_epoch_budget
    cfg['rq2_formal']['epoch_budget'] = build_epoch_budget(cfg)


def valid_batch(cfg):
    batch = cfg['loader']['batch_size']
    marker = cfg.get('rq3_auxiliary', {}).get('acceleration')
    if marker is None:
        return batch == 2
    return batch in (4, 8) and marker == dict(schema=SCHEMA, batch_size=batch, reference_batch_size=2,
                                             comparison='exploratory-unmatched-batch', learning_rate='unchanged')


def reference_config(candidate, baseline):
    """只归一化用户授权改变的batch及其派生步数；其余配置仍严格比较。"""
    if not valid_batch(candidate):
        raise ValueError('invalid accelerated batch provenance')
    normalized = deepcopy(candidate)
    normalized['loader']['batch_size'] = baseline['loader']['batch_size']
    normalized['rq2_formal']['epoch_budget'] = deepcopy(baseline['rq2_formal']['epoch_budget'])
    return normalized
