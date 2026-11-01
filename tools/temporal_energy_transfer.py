"""冻结源缓存按视频搬运；不缓存联合训练中会变化的FPN。"""
import torch


def source_record_to_device(record, device):
    result = dict(record)
    for key in ('features', 'masks'):
        result[key] = [value.to(device) for value in record[key]]
    for key in ('segments', 'gt_labels', 'anchors', 'truths', 'classes', 'instances', 'levels'):
        result[key] = record[key].to(device)
    return result
