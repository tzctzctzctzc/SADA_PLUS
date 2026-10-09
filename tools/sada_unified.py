"""RQ2/RQ3共用停止与选模协议；不改变SADA训练目标或对齐参数。"""
from copy import deepcopy
import math


POLICY = dict(schema='sada-unified-source-patience-v1', max_epochs=50,
              patience_epochs=15, validation_every_epochs=5, min_delta=0.0,
              selection_split='source_validation',
              selection_loss='source_gt_downstream_loss', state_key='ema_state_dict')


def apply_policy(cfg):
    if cfg.get('rq3_auxiliary', {}).get('early_stopping'):
        raise ValueError('cannot combine legacy RQ3 and unified stopping policies')
    cfg['sada_unified'] = deepcopy(POLICY)
    validate(cfg)


def validate(cfg):
    if cfg.get('sada_unified') != POLICY:
        raise ValueError('invalid unified SADA policy')
    train = cfg['train_cfg']
    if (cfg.get('rq2_formal', {}).get('regime') not in ('so', 'uda')
            or cfg['loader']['batch_size'] != 2 or cfg['opt']['epochs'] != 50
            or cfg.get('seed') != 6 or cfg.get('init_rand_seed') != 6
            or train['valid_freq'] != 5 or train['early_stop'] > 0
            or not train.get('ema_active') or train.get('evaluate_student', False)):
        raise ValueError('unified SADA requires matched batch2/seed6/max50/source-EMA/valid5')
    if cfg.get('rq3_auxiliary', {}).get('early_stopping'):
        raise ValueError('cannot combine legacy RQ3 and unified stopping policies')
    return True


def selection_loss(losses, domain):
    if domain != 'src':
        raise ValueError('unified model selection must use source validation')
    # 此字段在辅助损失加入之前保存，不能由Prototype额外分类损失替代。
    return losses['source_gt_downstream_loss']


def audit_termination(checkpoint):
    from tools.rq3_early_stop import decision
    cfg = checkpoint['config']
    cfg = getattr(cfg, 'config', cfg)
    validate(cfg)
    report = checkpoint.get('sada_termination', {})
    history = report.get('source_validation_history', [])
    epoch = checkpoint['epoch']
    if (not history or [row[0] for row in history] != list(range(5, epoch + 1, 5))
            or epoch % 5 or epoch > 50):
        raise ValueError('unified termination requires complete validation history')
    result = decision(history)
    selection = checkpoint['selection_provenance']
    if (result['reason'] == 'running' or report.get('decision') != result
            or result['epoch'] != epoch
            or any(decision(history[:i])['reason'] != 'running' for i in range(1, len(history)))
            or selection['split'] != 'source_validation'
            or selection['metric'] != 'validation_loss'
            or selection['state_key'] != 'ema_state_dict'
            or selection['selected_epoch'] != result['best_epoch']
            or not math.isfinite(selection['metric_value'])
            or selection['metric_value'] != result['best_loss']):
        raise ValueError('unified termination or source-selected best is invalid')
    return report
