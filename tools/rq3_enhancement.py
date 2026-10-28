"""固定顺序的Prototype增强；只改新增分支，旧v5证据和停止协议保持不变。"""
from copy import deepcopy
from pathlib import Path
import os

SCHEMA = 'rq3-prototype-enhancement-v1'
ORDER = ('control', 'wide', 'deep', 'pool', 'multiprototype', 'supcon', 'warmup')


def specification(name):
    if name not in ORDER:
        raise ValueError('unknown enhancement')
    i = ORDER.index(name)
    return dict(schema=SCHEMA, variant=name, embedding_dim=128 if i == 0 else 256,
                hidden_dim=256 if i == 0 else 512, layers=1 if i < 2 else 2,
                heads=4, pooling='split-attention' if i >= 3 else 'mean',
                prototypes_per_class=3 if i >= 4 else 1,
                source_supcon_weight=.1 if i >= 5 else 0., temperature=.1,
                alignment_warmup_epochs=5 if i >= 6 else 0,
                alignment_weight=.1, source_weight=1., parent=ORDER[i-1] if i else None,
                target_gt_used=False)


def configure(cfg, name):
    if name is None:
        return
    aux = cfg.get('rq3_auxiliary', {})
    if (aux.get('strategy') != 'prototype' or aux.get('participation_fraction') != .2
            or aux.get('teacher_model') != 'qwen3_vl_8b' or cfg.get('seed') != 6
            or cfg['loader']['batch_size'] != 2):
        raise ValueError('enhancement requires the frozen Qwen/seed6/batch2/top20 protocol')
    cfg['rq3_enhancement'] = specification(name)


def validate(cfg):
    spec = cfg.get('rq3_enhancement')
    if spec is not None and spec != specification(spec['variant']):
        raise ValueError('enhancement parameters differ from the fixed experiment')


def assert_reference_pair(cfg, reference):
    """比较完整展开配置；只忽略运行名、签名和显式新分支身份。"""
    def normalize(value):
        value = deepcopy(value)
        for key in ('name', 'unique_id', 'gpu', 'config_path', 'rq3_enhancement'):
            value.pop(key, None)
        value.get('rq2_formal', {}).pop('run_signature', None)
        value.get('dataset', {}).pop('stage1_context', None)
        value.get('train_cfg', {}).pop('head_empty_cls', None)
        for key in ('training_initialization', 'initialization_inputs'):
            value.get('benchmark_audit', {}).pop(key, None)
        return value
    validate(cfg)
    def same(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, str) and isinstance(b, str) and a != b:
            # 迁移只改变路径解析；必须确为同一工件，不能仅比较basename。
            return Path(a).exists() and Path(b).exists() and os.path.samefile(a, b)
        return a == b
    a, b = normalize(cfg), normalize(reference)
    if not same(a, b):
        changed = [k for k in set(a) | set(b) if not same(a.get(k), b.get(k))]
        raise ValueError('reference configuration changed outside branch: ' + str(changed))
