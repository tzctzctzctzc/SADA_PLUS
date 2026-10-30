"""不改官方单epoch循环：新头在主优化/裁剪之后独立更新，随后沿用原EMA更新。"""
from copy import deepcopy
import math
import torch


def attach(ctx, cfg):
    from tools.baseline_protocol import read,sha
    from libs.models.SADA.optimizer import make_scheduler
    head=ctx.model.endpoint_risk;teacher=ctx.model_ema.module
    if head is None or teacher.endpoint_risk is None or teacher is ctx.model:
        raise ValueError('缺少真实学生与独立EMA误差头')
    if any(a is b for a,b in zip(ctx.model.parameters(),teacher.parameters())):
        raise ValueError('EMA与学生意外共享参数')
    ids={id(p) for p in head.parameters()}
    for optimizer in (ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment,ctx.optimizer_disc):
        if any(id(p) in ids for g in optimizer.param_groups for p in g['params']):
            raise ValueError('新头混入原优化器/梯度路径')
    opt=cfg['endpoint_risk']
    ctx.endpoint_risk_optimizer=torch.optim.AdamW(head.parameters(),lr=opt['head_learning_rate'],weight_decay=0.)
    ctx.endpoint_risk_scheduler=make_scheduler(ctx.endpoint_risk_optimizer,cfg['opt'],
        cfg['baseline']['epoch_budget']['iterations_per_epoch'])
    ctx.endpoint_risk_generator=torch.Generator(device='cpu').manual_seed(cfg['seed']+982451653)
    ctx.endpoint_risk_pending=False
    ctx.endpoint_risk_last_losses=None
    if opt.get('calibration'):
        if sha(opt['calibration'])!=opt['calibration_sha256']:raise ValueError('源校准工件改变')
        certificate=read(opt['calibration'])
        head.install_calibration(certificate);teacher.endpoint_risk.install_calibration(certificate)
    original=ctx.model.forward

    def forward(*args, **kwargs):
        if kwargs.get('mode')!='train':return original(*args,**kwargs)
        if args:raise ValueError('端点辅助训练要求显式source/target关键字')
        ctx.endpoint_risk_optimizer.zero_grad(set_to_none=True)
        if ctx.endpoint_risk_pending:raise ValueError('上个风险头更新未完成')
        teacher.eval()
        target=kwargs['video_list_tgt']
        # 即使head-only也执行同一教师前向，以便对照隔离计算路径。
        with torch.no_grad():observation=teacher.endpoint_observation(target,mode='train')
        result=original(**dict(kwargs,risk_teacher=observation,risk_generator=ctx.endpoint_risk_generator))
        ctx.endpoint_risk_pending=True;ctx.endpoint_risk_last_losses=result[0]
        return result
    ctx.model.forward=forward

    for optimizer in (ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment):
        old_step=optimizer.step
        def step(*args,_old=old_step,**kwargs):
            result=_old(*args,**kwargs)
            if not ctx.endpoint_risk_pending:raise ValueError('没有本次source风险监督')
            ctx.endpoint_risk_pending=False
            source_loss=head.last_source_loss
            if source_loss is None or not torch.isfinite(source_loss):raise ValueError('源风险监督丢失/非有限')
            # 与主图完全断开，因此不需要retain_graph，也不会再写共享特征梯度。
            source_loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),opt['head_clip'])
            if not math.isfinite(float(norm)):raise ValueError('误差头梯度非有限')
            ctx.endpoint_risk_optimizer.step();ctx.endpoint_risk_scheduler.step()
            ctx.endpoint_risk_last_losses['endpoint_head_gradient_norm']=source_loss.detach().new_tensor(float(norm))
            return result
        optimizer.step=step


def capture_extra(ctx):
    if ctx.endpoint_risk_pending:raise ValueError('不能在一次未完成的更新中保存')
    return dict(optimizer=ctx.endpoint_risk_optimizer.state_dict(),
        scheduler=ctx.endpoint_risk_scheduler.state_dict(),
        generator=ctx.endpoint_risk_generator.get_state(),
        calibration=deepcopy(ctx.model.endpoint_risk.calibration),
        options=deepcopy(ctx.cfg['endpoint_risk']))


def restore_extra(ctx, state):
    if state['options']!=ctx.cfg['endpoint_risk']:raise ValueError('恢复时风险方法/参数改变')
    ctx.endpoint_risk_optimizer.load_state_dict(state['optimizer'])
    ctx.endpoint_risk_scheduler.load_state_dict(state['scheduler'])
    ctx.endpoint_risk_generator.set_state(state['generator'].cpu())
    for model in (ctx.model,ctx.model_ema.module):
        if state['calibration'] is not None:model.endpoint_risk.install_calibration(state['calibration'])
        else:model.endpoint_risk.calibration=None
    ctx.endpoint_risk_pending=False
