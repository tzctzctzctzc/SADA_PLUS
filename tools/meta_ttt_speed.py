"""相同逐通道卷积的可微展开，避开Torch1.13二阶反传逐通道发射卷积。

不改变参数、mask、内循环步数或数值精度；无梯度推理仍走原Conv1d。
"""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

BACKEND = 'depthwise-unfold-fp32-v1'


def depthwise_forward(module, x):
    if not torch.is_grad_enabled():
        return nn.Conv1d.forward(module, x)
    kernel = module.kernel_size[0]
    if kernel == 1:
        out = x * module.weight[:, 0, 0][None, :, None]
    else:
        windows = F.pad(x, (module.padding[0], module.padding[0])).unfold(-1, kernel, 1)
        out = (windows * module.weight[:, 0, :][None, :, None, :]).sum(-1)
    if module.bias is not None:
        out = out + module.bias[None, :, None]
    return out


def install(encoder):
    names = []
    for name, module in encoder.backbone.named_modules():
        if isinstance(module, nn.Conv1d) and module.groups == module.in_channels == module.out_channels:
            if module.stride != (1,) or module.dilation != (1,) or module.padding_mode != 'zeros':
                raise ValueError('优化仅支持已核验的SGP逐通道卷积')
            if module.padding != (module.kernel_size[0] // 2,):
                raise ValueError('优化要求same padding')
            module.forward = MethodType(depthwise_forward, module)
            names.append(name)
    if not names:
        raise ValueError('没有SGP逐通道卷积')
    return names


def restore_training(saved, model, decoder, optimizer, generator):
    """epoch边界恢复；调用前完成模型、数据集和optimizer构造，最后恢复随机流。"""
    import random
    import numpy as np
    model.load_state_dict(saved['state_dict'], strict=True)
    decoder.load_state_dict(saved['decoder'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    random.setstate(saved['rng']['python'])
    np.random.set_state(saved['rng']['numpy'])
    torch.set_rng_state(saved['rng']['torch_cpu'])
    torch.cuda.set_rng_state(saved['rng']['torch_cuda'], 0)
    generator.set_state(saved['rng']['source_loader'])


def check_resume(saved, job, cfg, initial, head_initial):
    from tools.meta_ttt import SPEC, choose_checkpoint
    if (saved['specification'] != SPEC or saved['config'] != cfg or saved['arm'] != job['arm']
            or saved['parent_checkpoint_sha256'] != job['input_sha256']['checkpoint']
            or saved['initial_model_sha256'] != initial or saved['initial_decoder_sha256'] != head_initial):
        raise ValueError('恢复点与冻结任务身份不一致')
    epoch = saved['epoch']
    if [r['epoch'] for r in saved['epochs']] != list(range(1, epoch + 1)):
        raise ValueError('恢复点epoch历史不连续')
    history = saved['history']
    if history[-1]['epoch'] != epoch // SPEC['validate_every'] * SPEC['validate_every']:
        raise ValueError('恢复点源验证历史缺失')
    if choose_checkpoint(history)[1]:
        raise ValueError('已合法结束的训练不能重开')
    budget = cfg['baseline']['epoch_budget']['iterations_per_epoch']
    if any(r['steps'] != budget or r['target_GT_used'] for r in saved['epochs']):
        raise ValueError('恢复点预算或GT隔离不符')
    if set(saved['rng']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'source_loader'}:
        raise ValueError('缺少完整随机流')
    if not saved['optimizer']['state']:
        raise ValueError('缺少已训练optimizer状态')
    return epoch
