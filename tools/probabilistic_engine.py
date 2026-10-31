"""新概率检测三组共用训练器；源验证选模，目标GT从不进入训练。"""
from copy import deepcopy
import itertools
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tools import baseline_engine as base
from tools.baseline_protocol import read,save,sha,assert_no_target_gt,decision
from libs.trainer.da_trainer import move_to_device
from libs.models.SADA.probabilistic_temporal import specification,source_likelihood,target_losses,masked_videos

SCHEMA='temporal-probabilistic-teacher-s1-v1'
VARIANTS=('source','ema','vlm')


def build(cfg):
    ctx=base.build(cfg)
    ctx.model_ema.module.load_state_dict(ctx.model.state_dict(),strict=True)
    ctx.model_ema.module.requires_grad_(False);ctx.model_ema.module.eval()
    ctx.generators['perturbation']=torch.Generator().manual_seed(cfg['seed']+3000133)
    ctx.cache={}
    if cfg['probabilistic_teacher']['variant']!='source':
        path=cfg['probabilistic_teacher']['prepared_cache']
        if sha(path)!=cfg['probabilistic_teacher']['prepared_cache_sha256']:raise ValueError('教师缓存改变')
        payload=read(path)
        if payload.get('schema')=='prototype-gt-oracle-cache-v1':raise ValueError('不能用GT oracle冒充VLM')
        ctx.cache=payload['by_video']
    return ctx


def gradient_norm(loss,parameters):
    gradients=torch.autograd.grad(loss,list(parameters),retain_graph=True,allow_unused=True)
    return math.sqrt(sum(float(g.detach().square().sum()) for g in gradients if g is not None))


def clear_packets(ctx):
    ctx.model.pt_packet=None;ctx.model_ema.module.pt_packet=None


def one_epoch(ctx,epoch,limit=None):
    cfg=ctx.cfg;spec=specification();variant=cfg['probabilistic_teacher']['variant']
    ctx.model.train();ctx.model_ema.module.eval()
    base.epoch_setup(ctx,epoch)
    if epoch==spec['target_start_epoch']:
        # 各组在相同源热身边界同步；不会用随机独立EMA给目标视频造标签。
        ctx.model_ema.module.load_state_dict(ctx.model.state_dict(),strict=True)
    iterations=cfg['baseline']['epoch_budget']['iterations_per_epoch'] if limit is None else limit
    sources=itertools.cycle(iter(ctx.train_loader_src));targets=itertools.cycle(iter(ctx.train_loader_tgt))
    values=[];started=time.monotonic()
    for step in range(iterations):
        source=move_to_device(next(sources),'cuda:0');target=move_to_device(next(targets),'cuda:0')
        assert_no_target_gt(target)
        for opt in base.optimizers(ctx):opt.zero_grad(set_to_none=True)
        active=variant!='source' and epoch>=spec['target_start_epoch']
        teacher=None
        if active:
            with torch.no_grad():
                teacher=ctx.model_ema.module(video_list_src=target,video_list_tgt=target,
                    mode='pt_raw',curr_epoch=epoch)['target']
            strong=masked_videos(target,ctx.generators['perturbation'])
        else:strong=target
        losses=ctx.model(video_list_src=source,video_list_tgt=strong,mode='train',curr_epoch=epoch)[0]
        raw=ctx.model.pt_packet
        nll,positives=source_likelihood(ctx.model,raw['source'],source)
        semantic=localization=nll*0.
        stats=dict(routable=0,accepted=0,vlm_rejected=0,native_points=0,weight_sum=0.)
        if active:semantic,localization,stats=target_losses(raw['target'],teacher,target,ctx.cache,variant=='vlm',
            cfg['probabilistic_teacher']['variance_scale'])
        source_loss=losses['downstream_loss']+spec['source_nll_weight']*nll
        weight=1. if epoch<spec['target_start_epoch'] else cfg['model']['loss_weights']['weight_downstream']
        total=weight*source_loss+spec['target_semantic_weight']*semantic+spec['target_localization_weight']*localization
        if not torch.isfinite(total):raise ValueError('新结构损失非有限')
        src_norm=gradient_norm(source_loss,ctx.model.cls_head.parameters())
        uncertainty_norm=gradient_norm(nll,ctx.model.pt_uncertainty.parameters())
        cls_norm=gradient_norm(semantic,ctx.model.cls_head.parameters()) if stats['accepted'] else 0.
        reg_norm=gradient_norm(localization,ctx.model.reg_head.parameters()) if stats['accepted'] else 0.
        total.backward()
        if any(p.grad is not None for p in ctx.model_ema.module.parameters()):raise ValueError('教师不能接收反传梯度')
        if any(p.grad is not None for n,p in ctx.model.named_parameters() if any(k in n for k in
                ('disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'))):raise ValueError('存在对抗梯度')
        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),cfg['train_cfg']['clip_grad_l2norm'])
        (ctx.optimizer_gen_source if epoch<spec['target_start_epoch'] else ctx.optimizer_gen_source_alignment).step()
        ctx.model_ema.update(ctx.model)
        for scheduler in base.schedulers(ctx):scheduler.step()
        row=dict(train_loss=float(total.detach()),source_cls_loss=float(losses['cls_loss'].detach()),
            source_reg_loss=float(losses['reg_loss'].detach()),source_nll=float(nll.detach()),source_positive_points=positives,
            source_gradient_norm=src_norm,source_uncertainty_gradient_norm=uncertainty_norm,
            target_semantic_loss=float(semantic.detach()),target_localization_loss=float(localization.detach()),
            target_classifier_gradient_norm=cls_norm,target_regressor_gradient_norm=reg_norm,
            target_gt_cls_loss=0.,target_gt_reg_loss=0.,target_gt_downstream_loss=0.,**stats)
        if not all(math.isfinite(v) for v in row.values()):raise ValueError('非有限训练统计')
        values.append(row);clear_packets(ctx)
        if step%10==0:print('PT_STEP',epoch,step,iterations,json.dumps(row),flush=True)
    torch.cuda.synchronize()
    average={key:sum(x[key] for x in values)/len(values) for key in values[0]}
    return dict(average,epoch=epoch,steps=iterations,seconds=time.monotonic()-started)


def validate(ctx,epoch):
    ctx.valid_metrics_src.reset()
    value=float(ctx._valid_epoch(epoch,ctx.val_loader_src,domain='src'))
    clear_packets(ctx)
    if not math.isfinite(value) or value<=0:raise ValueError('源验证无效')
    return value


def checkpoint_state(ctx,cfg,epoch,history):
    saved=base.capture(ctx,cfg,epoch,history);saved['schema']=SCHEMA
    saved['teacher_initialization']='copy-student'
    saved['probabilistic_specification']=specification()
    return saved


def train(cfg,entry):
    entry=Path(entry);ctx=build(cfg)
    save(entry/'runtime_config.json',ctx.cfg);save(entry/'seed_proof.json',ctx.cuda_seed_proof)
    save(entry/'initialization.json',dict(initial_state_sha256=ctx.initial_state_sha,
        student_sha256=base.tensor_digest(ctx.model.state_dict()),
        teacher_sha256=base.tensor_digest(ctx.model_ema.module.state_dict()),specification=specification()))
    rows=[];history=[];best=float('inf')
    for epoch in range(1,51):
        row=one_epoch(ctx,epoch)
        if epoch%5==0:
            value=validate(ctx,epoch);row['source_validation_loss']=value;history.append([epoch,value])
            if value<best:
                best=value;base.checkpoint(entry/'best.pt',checkpoint_state(ctx,cfg,epoch,history))
        rows.append(row)
        (entry/'metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        print('PT_EPOCH',json.dumps(row),flush=True)
        if epoch%5==0:
            base.checkpoint(entry/'latest.pt',checkpoint_state(ctx,cfg,epoch,history))
            result=decision(history)
            if result['reason']!='running':
                save(entry/'termination.json',dict(history=history,decision=result,initial_state_sha=ctx.initial_state_sha));break


def probe(cfg,entry,checkpoint=None):
    entry=Path(entry);ctx=build(cfg)
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu')
        ctx.model.load_state_dict(saved['ema_state_dict'],strict=True)
        ctx.model_ema.module.load_state_dict(saved['ema_state_dict'],strict=True)
    initial=base.tensor_digest(ctx.model.state_dict())
    uncertainty_before=base.tensor_digest(dict(ctx.model.pt_uncertainty.named_parameters()))
    reports={}
    # 目标缓存稀疏，必须遍历整个原始epoch，不能用前几个batch代表有无路由。
    epochs=(1,) if cfg['probabilistic_teacher']['variant']=='source' else (5,)
    for epoch in epochs:reports[str(epoch)]=one_epoch(ctx,epoch,limit=None if checkpoint else 3)
    validation=validate(ctx,5)
    state=checkpoint_state(ctx,cfg,5,[[5,validation]])
    buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0)
    state=torch.load(buffer,map_location='cpu');buffer.close()
    digest=base.tensor_digest(state);base.restore(ctx,state)
    if base.tensor_digest(checkpoint_state(ctx,cfg,5,[[5,validation]]))!=digest:raise ValueError('完整GPU状态恢复失败')
    if base.tensor_digest(ctx.model.state_dict())==initial:raise ValueError('真实模型没有更新')
    uncertainty_after=base.tensor_digest(dict(ctx.model.pt_uncertainty.named_parameters()))
    if uncertainty_before==uncertainty_after:raise ValueError('概率头未被实际优化器更新')
    if any(p.grad is not None for p in ctx.model_ema.module.parameters()):raise ValueError('EMA教师有梯度')
    for row in reports.values():
        if row['source_gradient_norm']<=0 or row['source_nll']<0 or row['source_uncertainty_gradient_norm']<=0:
            raise ValueError('源监督/概率头未生效')
        if checkpoint and (row['accepted']<=0 or row['target_classifier_gradient_norm']<=0 or row['target_regressor_gradient_norm']<=0):
            raise ValueError('目标语义/定位学习未真实生效')
    save(entry/'result.json',dict(status='passed',variant=cfg['probabilistic_teacher']['variant'],reports=reports,
        source_validation_loss=validation,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        initial_state_sha256=initial,rng_roundtrip_sha256=digest,cuda_seed_proof=ctx.cuda_seed_proof,
        uncertainty_parameters_updated=True,uncertainty_before_sha256=uncertainty_before,uncertainty_after_sha256=uncertainty_after,
        checkpoint_sha256=sha(checkpoint) if checkpoint else None))


def uncertainty_audit(cfg,entry,output):
    entry=Path(entry);saved=torch.load(entry/'best.pt',map_location='cpu')
    ctx=build(cfg);model=ctx.model_ema.module;model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval()
    videos=[];without_positive=[]
    with torch.no_grad():
        for sample in ctx.val_loader_src:
            source=move_to_device(sample,'cuda:0')
            raw=model(video_list_src=source,video_list_tgt=source,mode='pt_raw')['source']
            _,data=source_likelihood(model,raw,source,return_rows=True)
            valid=data['positive'][0]
            if not valid.any():
                without_positive.append(source[0]['metadata']['video_id']);model.pt_packet=None;continue
            error=data['error_squared'][0,valid].cpu().numpy();variance=data['variance'][0,valid].cpu().numpy()
            videos.append(dict(video_id=source[0]['metadata']['video_id'],error_squared=error.tolist(),variance=variance.tolist()))
            model.pt_packet=None
    from scipy.stats import spearmanr
    from hashlib import sha256
    ordered=sorted(videos,key=lambda v:sha256(('pt-source-confirm-v1:'+v['video_id']).encode()).hexdigest())
    middle=len(ordered)//2;calibration,confirmation=ordered[:middle],ordered[middle:]
    if len(calibration)<2 or len(confirmation)<2:raise ValueError('源留出视频不足')
    # 视频等权；标定只使用前半源验证视频，后半核验，不查看目标GT。
    c=np.asarray([np.asarray(v['error_squared']).mean(0) for v in calibration]).mean(0).clip(1e-8)
    ratios=np.asarray([(np.asarray(v['error_squared'])/np.asarray(v['variance'])).mean(0) for v in calibration]).mean(0).clip(1e-8)
    differences=[];correlations=[]
    for v in confirmation:
        error=np.asarray(v['error_squared']);variance=np.asarray(v['variance'])*ratios
        if not len(error):continue
        learned=.5*(error/variance+np.log(variance));constant=.5*(error/c+np.log(c))
        differences.append(float((constant-learned).mean()))
        for side in range(2):
            corr=spearmanr(variance[:,side],error[:,side])[0]
            if math.isfinite(corr):correlations.append(float(corr))
    improvement=float(np.mean(differences)) if differences else float('nan')
    correlation=float(np.mean(correlations)) if correlations else float('nan')
    passed=math.isfinite(improvement) and math.isfinite(correlation) and improvement>0 and correlation>0
    save(output,dict(status='passed' if passed else 'failed',checkpoint_sha256=sha(entry/'best.pt'),
        criterion='video-equal-confirmation-NLL-better-than-constant-and-positive-error-ranking',
        source_only=True,target_gt_used=False,calibration_videos=[v['video_id'] for v in calibration],
        confirmation_videos=[v['video_id'] for v in confirmation],variance_scale=ratios.tolist(),
        confirmation_nll_improvement=improvement if math.isfinite(improvement) else None,
        confirmation_error_rank_correlation=correlation if math.isfinite(correlation) else None,
        videos_without_positive_points=without_positive,
        rows=videos,interpretation='source-diagnostic-not-a-target-reliability-guarantee'))
