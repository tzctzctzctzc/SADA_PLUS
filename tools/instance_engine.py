"""新实例检测器的独立源域训练/验证，不调用SADA的前向、旧检测头或对抗训练器。"""
from copy import deepcopy
import io
import itertools
import json
import math
from pathlib import Path
import random
import numpy as np
import torch

from tools import baseline_engine as data_utils
from tools.baseline_protocol import read,save,sha,decision,validate_cuda_rng
from libs.models.SADA.instance_detector import InstanceTemporalDetector,specification
from libs.models.SADA.instance_losses import source_losses,create_teacher,update_teacher
from libs.models.SADA.optimizer import make_optimizer,make_scheduler

SCHEMA='instance-query-source-v1'


def query_budget(cfg):
    """只统计本方向source_train。整个窗口相交动作数是裁剪后监督数量的上界。"""
    db=read(cfg['dataset']['json_file'])['database'];maximum=0;videos=0
    width=cfg['dataset']['max_seq_len']*cfg['dataset']['feat_stride']/cfg['dataset']['default_fps']
    for record in db.values():
        if record['subset'].lower() not in cfg['src_train_split']:continue
        videos+=1;events=[]
        for a in record.get('annotations',[]):
            if 0<=int(a['label_id'])<10:
                start,end=a['segment'];events.extend(((max(0.,start-width),1),(end,-1)))
        current=0
        for _,delta in sorted(events,key=lambda x:(x[0],x[1])):current+=delta;maximum=max(maximum,current)
    return dict(source_training_videos=videos,max_window_intersections=maximum,
        num_queries=max(128,32*math.ceil(maximum*1.25/32)),source_manifest_sha256=sha(cfg['dataset']['json_file']))


def build(cfg,spec):
    from types import SimpleNamespace
    data_utils.fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
    model=InstanceTemporalDetector(cfg,spec).cuda();teacher=create_teacher(model)
    optimizer=make_optimizer(model,cfg['opt'],cfg['opt']['learning_rate'])
    scheduler=make_scheduler(optimizer,cfg['opt'],cfg['baseline']['epoch_budget']['iterations_per_epoch'])
    source=data_utils.dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],True)
    validation=data_utils.dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    generators={k:torch.Generator().manual_seed(cfg['seed']+offset) for k,offset in [('source',0),('validation',2000006)]}
    loaders=[data_utils.loader(ds,training,generators[name],cfg['loader']['batch_size'] if training else 1,cfg['loader']['num_workers'])
        for ds,training,name in [(source,True,'source'),(validation,False,'validation')]]
    return SimpleNamespace(model=model,teacher=teacher,optimizer=optimizer,scheduler=scheduler,
        source=loaders[0],validation=loaders[1],generators=generators,cfg=cfg,spec=spec)


def state(ctx,epoch,history):
    return dict(schema=SCHEMA,config=ctx.cfg,specification=ctx.spec,epoch=epoch,history=deepcopy(history),
        model=ctx.model.state_dict(),teacher=ctx.teacher.state_dict(),optimizer=ctx.optimizer.state_dict(),scheduler=ctx.scheduler.state_dict(),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=[torch.cuda.get_rng_state(0)],
            loaders={k:g.get_state() for k,g in ctx.generators.items()}))


def restore(ctx,saved):
    if saved['config']!=ctx.cfg or saved['specification']!=ctx.spec:raise ValueError('新结构恢复配置不一致')
    ctx.model.load_state_dict(saved['model'],strict=True);ctx.teacher.load_state_dict(saved['teacher'],strict=True)
    ctx.optimizer.load_state_dict(saved['optimizer']);ctx.scheduler.load_state_dict(saved['scheduler'])
    rng=saved['rng'];validate_cuda_rng(rng['cuda'])
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'].cpu());torch.cuda.set_rng_state(rng['cuda'][0].cpu())
    for k,g in ctx.generators.items():g.set_state(rng['loaders'][k].cpu())


def gradient_norm(module):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None))


def step(ctx,videos):
    ctx.model.train();ctx.teacher.eval();ctx.optimizer.zero_grad(set_to_none=True)
    output=ctx.model(videos);losses=source_losses(output,videos)
    if not torch.isfinite(losses['loss']):raise ValueError('源检测损失非有限')
    losses['loss'].backward()
    modules={'encoder':ctx.model.encoder,'decoder':ctx.model.decoder,'classification':ctx.model.class_heads,
        'localization':ctx.model.segment_heads,'quality':ctx.model.quality_head,'query_content':ctx.model.query_content,
        'query_position':ctx.model.query_position,'reference':ctx.model.initial_reference}
    gradients={k:gradient_norm(v) for k,v in modules.items()}
    if not all(math.isfinite(x) for x in gradients.values()):raise ValueError('梯度非有限')
    if any(p.grad is not None for p in ctx.teacher.parameters()):raise ValueError('EMA教师有梯度')
    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),ctx.cfg['train_cfg']['clip_grad_l2norm'])
    ctx.optimizer.step();ctx.scheduler.step();update_teacher(ctx.teacher,ctx.model,ctx.cfg['train_cfg']['ema_decay'])
    return dict(losses={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in losses.items()},gradients=gradients,
        target_gt_losses=[0.,0.,0.],target_data_read=False)


def validation_windows(video,size):
    for start in range(0,video['feats'].shape[-1],size):
        clip=dict(video);clip['feats']=video['feats'][:,start:start+size]
        if video['segments'] is None:yield clip;continue
        segments=video['segments']-start;keep=(segments[:,1]>0)&(segments[:,0]<clip['feats'].shape[-1])
        clip['segments']=segments[keep].clamp(0,clip['feats'].shape[-1]);clip['labels']=video['labels'][keep];yield clip


@torch.no_grad()
def validate(ctx):
    ctx.teacher.eval();values=[]
    for batch in ctx.validation:
        for video in batch:
            for clip in validation_windows(video,ctx.model.max_length):
                losses=source_losses(ctx.teacher([clip]),[clip])
                values.append(float(losses['classification']+5*losses['localization_l1']+2*losses['localization_iou']))
    return sum(values)/len(values)


def probe(cfg,spec,root):
    ctx=build(cfg,spec);initial=data_utils.tensor_digest(ctx.model.state_dict());reports=[]
    quality_before=data_utils.tensor_digest(ctx.model.quality_head.state_dict())
    for i,videos in enumerate(ctx.source):
        reports.append(step(ctx,videos))
        if i==2:break
    if len(reports)!=3 or not all(any(r['gradients'][k]>0 for r in reports) for k in reports[0]['gradients']):
        raise ValueError('实际结构中存在未学习的必要模块')
    if initial==data_utils.tensor_digest(ctx.model.state_dict()) or quality_before==data_utils.tensor_digest(ctx.model.quality_head.state_dict()):
        raise ValueError('新模型或质量头没有被优化器更新')
    saved=state(ctx,1,[]);digest=data_utils.tensor_digest(saved);buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu');restore(ctx,saved)
    if digest!=data_utils.tensor_digest(state(ctx,1,[])):raise ValueError('完整GPU状态恢复失败')
    video=next(iter(ctx.validation))[0];clip=next(validation_windows(video,ctx.model.max_length))
    ctx.teacher.eval();prediction=ctx.teacher.predict_video(clip)
    if not torch.isfinite(prediction['scores']).all() or not torch.isfinite(prediction['segments']).all():raise ValueError('预测非有限')
    save(root/'probe.json',dict(status='passed',reports=reports,full_state_roundtrip_sha256=digest,
        parameter_count=sum(p.numel() for p in ctx.model.parameters()),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        prediction_count=len(prediction['scores']),quality_parameters_updated=True,no_formal_checkpoint=True,
        model_modules=list(dict(ctx.model.named_children())),specification=spec))


def train(cfg,spec,root):
    ctx=build(cfg,spec);rows=[];history=[];best=float('inf')
    save(root/'initialization.json',dict(student=data_utils.tensor_digest(ctx.model.state_dict()),teacher=data_utils.tensor_digest(ctx.teacher.state_dict())))
    for epoch in range(1,51):
        iterator=itertools.cycle(iter(ctx.source));reports=[]
        for _ in range(cfg['baseline']['epoch_budget']['iterations_per_epoch']):reports.append(step(ctx,next(iterator)))
        row=dict(epoch=epoch,loss=sum(x['losses']['loss'] for x in reports)/len(reports),target_gt_losses=[0.,0.,0.],source_steps=len(reports))
        if epoch%5==0:
            value=validate(ctx);history.append([epoch,value]);row['source_validation_detection_loss']=value
            if value<best:best=value;data_utils.checkpoint(root/'best.pt',state(ctx,epoch,history))
        rows.append(row);(root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print('INSTANCE_EPOCH',row,flush=True)
        if epoch%5==0:
            data_utils.checkpoint(root/'latest.pt',state(ctx,epoch,history));result=decision(history)
            if result['reason']!='running':save(root/'termination.json',dict(history=history,decision=result));break
