"""真实保存点、真实整epoch、无诊断checkpoint；失败不得生成放行凭证。"""
from copy import deepcopy
from pathlib import Path
import torch
from tools.baseline_protocol import save,sha


def probe(cfg, checkpoint, output, identity, guard):
    from tools import baseline_engine as engine
    from libs.models.SADA.endpoint_risk import SCHEMA
    output=Path(output)
    if output.exists():raise ValueError('不能覆盖已有GPU验收')
    output.mkdir(parents=True)
    saved=torch.load(checkpoint,map_location='cpu')
    if saved.get('schema')!=SCHEMA or saved['config']!=cfg:
        raise ValueError('真实验收必须恢复当前配置的原始预热保存点')
    cert=saved.get('endpoint_risk',{}).get('calibration')
    if not cert or cert['status']!='passed' or cert['code']!=identity:
        raise ValueError('真实验收缺当前代码源校准/教师优势证明')
    from tools.endpoint_risk_protocol import validate_installed_certificate
    validate_installed_certificate(cert)
    ctx=engine.build(cfg);engine.restore(ctx,saved)
    reference=engine.tensor_digest(saved)
    if engine.tensor_digest(engine.capture(ctx,cfg,saved['epoch'],saved['history']))!=reference:
        raise ValueError('真实验收恢复状态不精确')
    before=engine.tensor_digest(ctx.model.endpoint_risk.state_dict())
    epoch=saved['epoch']+1
    if epoch<=cfg['endpoint_risk']['warmup_epochs']:raise ValueError('不能用关闭定位辅助的预热轮当验收')
    engine.epoch_setup(ctx,epoch);ctx._train_epoch(epoch)
    metrics=ctx.train_metrics.result()
    updated=before!=engine.tensor_digest(ctx.model.endpoint_risk.state_dict())
    teacher_detached=all(p.grad is None for p in ctx.model_ema.module.parameters())
    real_gradient=metrics.get('t_endpoint_pseudo_gradient_norm',0)>0
    if not updated or not teacher_detached or ctx.checked_steps!=cfg['baseline']['epoch_budget']['iterations_per_epoch']:
        raise ValueError('真实整epoch没有误差头更新/教师隔离/原预算')
    mode=cfg['endpoint_risk']['mode']
    if mode!='head_only' and (not real_gradient or metrics.get('t_endpoint_selected_endpoints',0)<=0):
        raise ValueError('真实target端点/定位梯度没有作用；不放宽源门槛补样本')
    changed_gate=metrics.get('t_endpoint_random_changed_endpoints',0)>0
    if mode=='random' and not changed_gate:
        raise ValueError('随机负对照未发生不同候选间的实际交换；不能把单候选恒等置乱当证据')
    engine.restore(ctx,saved)
    if engine.tensor_digest(engine.capture(ctx,cfg,saved['epoch'],saved['history']))!=reference:
        raise ValueError('GPU整状态/RNG恢复不精确')
    report=dict(status='passed',schema=SCHEMA,code=identity,device='cuda',epoch=epoch,
        batch=cfg['loader']['batch_size'],seed=cfg['seed'],
        metrics={k:float(v) for k,v in metrics.items()},risk_head_updated=updated,
        teacher_detached=teacher_detached,target_gt_zero=True,
        pseudo_gradient_nonzero=bool(real_gradient),rng_roundtrip=True,
        random_gate_actually_changed=bool(changed_gate),
        head_only_no_pseudo=mode=='head_only',config_sha256=engine.tensor_digest(cfg),
        calibration_sha256=cert['calibration_sha256'],checkpoint_sha256=sha(checkpoint),
        checkpoint_path=str(Path(checkpoint).resolve()),
        gpu_guard=guard,gpu_guard_sha256=sha(guard),
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    save(output/'report.json',report)
    acceptance=deepcopy(report);acceptance['reports']={str(output/'report.json'):sha(output/'report.json')}
    save(output/'acceptance.json',acceptance)
    return report
