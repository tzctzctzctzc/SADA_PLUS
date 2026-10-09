"""RQ3源域早停：按epoch而非验证次数；不读取任何目标域指标。"""
import math

POLICY = dict(schema='rq3-source-loss-patience-v1', patience_epochs=15,
              max_epochs=50, min_delta=0.0, split='source_validation',
              metric='validation_loss', comparison='mixed-stopping-budget')


def apply_policy(cfg, patience):
    if not patience:
        return
    from tools.rq3_protocol import REVISION
    aux = cfg.get('rq3_auxiliary', {})
    if (patience != 15 or aux.get('schema') != REVISION
            or aux.get('strategy') not in ('verify', 'prototype')
            or cfg['opt']['epochs'] != 50 or cfg['loader']['batch_size'] != 2
            or cfg['train_cfg']['early_stop'] > 0):
        raise ValueError('RQ3 early stopping requires revised batch2/50 and no legacy patience')
    aux['early_stopping'] = dict(POLICY)


def enabled(cfg):
    if cfg.get('sada_unified') is not None:
        from tools.sada_unified import validate
        return validate(cfg)
    policy = cfg.get('rq3_auxiliary', {}).get('early_stopping')
    if policy is not None and policy != POLICY:
        raise ValueError('invalid RQ3 early-stopping policy')
    return policy is not None


def decision(history):
    """严格下降才更新best；只在完成源域验证后判断。"""
    best, best_epoch = float('inf'), None
    for epoch, loss in history:
        if not math.isfinite(loss):
            raise ValueError('nonfinite source validation loss')
        if loss < best:
            best, best_epoch = loss, epoch
    if best_epoch is None:
        raise ValueError('missing source validation history')
    epoch = history[-1][0]
    return dict(epoch=epoch, best_epoch=best_epoch, best_loss=best,
                reason=('source_patience_exhausted' if epoch - best_epoch >= 15
                        else 'max_epochs' if epoch == 50 else 'running'))


def audit_termination(checkpoint):
    cfg = checkpoint.get('config', {})
    cfg = getattr(cfg, 'config', cfg)
    if cfg.get('sada_unified') is not None:
        from tools.sada_unified import audit_termination as audit_unified
        return audit_unified(checkpoint)
    report = checkpoint.get('rq3_termination', {})
    history = report.get('source_validation_history', [])
    epochs = [row[0] for row in history]
    if not epochs or epochs != sorted(set(epochs)) or epochs[0] < 1 or epochs[-1] > 50:
        raise ValueError('invalid early-stop validation history')
    expected = decision(history)
    if (expected['reason'] == 'running' or report.get('decision') != expected
            or checkpoint['epoch'] != expected['epoch']
            or checkpoint['selection_provenance']['selected_epoch'] != expected['best_epoch']
            or checkpoint['selection_provenance']['split'] != 'source_validation'
            or checkpoint['selection_provenance']['metric_value'] != expected['best_loss']):
        raise ValueError('invalid source-only early-stop termination')
    if any(decision(history[:i])['reason'] != 'running' for i in range(1, len(history))):
        raise ValueError('training continued past first early-stop boundary')
    return report
