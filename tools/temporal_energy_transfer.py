"""冻结源缓存按视频搬运；不缓存联合训练中会变化的FPN。"""
import torch
from copy import deepcopy
import math


def source_record_to_device(record, device):
    result = dict(record)
    for key in ('features', 'masks'):
        result[key] = [value.to(device) for value in record[key]]
    for key in ('segments', 'gt_labels', 'anchors', 'truths', 'classes', 'instances', 'levels'):
        result[key] = record[key].to(device)
    return result


def restore_source_checkpoint(head, optimizer, generator, saved, specification):
    """只恢复完整epoch保存点；矩阵阶段选择与原工件绑定由入口另行处理。"""
    required = {'state_dict', 'optimizer', 'epoch', 'history', 'specification',
                'generator', 'cpu_rng', 'cuda_rng'}
    if not required.issubset(saved) or saved['specification'] != specification:
        raise ValueError('源保存点不完整或科学配置不同')
    history = saved['history']
    if not history or [x['epoch'] for x in history] != list(range(saved['epoch'] + 1)):
        raise ValueError('源保存点history不连续')
    if not all(math.isfinite(x['source_validation']) for x in history):
        raise ValueError('源保存点包含非有限选模指标')
    device = next(head.parameters()).device
    if (device.type == 'cuda') != (saved['cuda_rng'] is not None):
        raise ValueError('源保存点CUDA RNG与恢复设备不匹配')
    best = min(history, key=lambda x: x['source_validation'])
    for row in history[:-1]:
        prefix = history[:row['epoch'] + 1]
        chosen = min(prefix, key=lambda x: x['source_validation'])['epoch']
        if row['epoch'] - chosen >= specification['patience']:
            raise ValueError('源保存点错过首次早停')
    head.load_state_dict(saved['state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    generator.set_state(saved['generator'])
    torch.set_rng_state(saved['cpu_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(saved['cuda_rng'], device)
    return dict(next_epoch=saved['epoch'] + 1, history=deepcopy(history),
                best_epoch=best['epoch'], best_loss=best['source_validation'])
