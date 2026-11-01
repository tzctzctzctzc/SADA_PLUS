"""原Prototype上联合学习定位回归几何；原检测头/对抗/优化阶段保持。

迁移依据DARE-GRAM https://arxiv.org/abs/2303.13325，非作者TAL实现。
按FPN层分开、等视频取真实回归输入、Gram按点数归一化。
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from tools.regression_gram import SPEC as KERNEL_SPEC,gram_alignment
from tools.baseline_protocol import assert_no_target_gt

SCHEMA='prototype-joint-regression-gram-v1'
SPEC=dict(schema=SCHEMA,kernel={k:KERNEL_SPEC[k] for k in ['explained_variance','angle_weight','scale_weight','rank_rule','numerical_precision','zero_columns']},source='original-assigned-positive-points',
    target='detached-current-student-foreground-ge-.5',per_video_level_cap=64,minimum_points=4,
    matching='equal-domain-count-per-level-round-robin-videos',level_weight='mean-supported-levels',
    normalization='augmented-feature-matrix-divided-by-sqrt-point-count',
    onset='original-warmup-adv',loss_location='downstream-before-original-stage-weight',
    original_adversary='unchanged',prototype='original128-fixed200-pool-selected40',
    target_gt=False,extra_trainable_parameters=0,extra_inference_parameters=0,
    degeneracy='skip-level-if-retained-eigen-ratio-below-1e-10; fail-nonfinite',
    inference='exact-original-detector-path',random_sampling=False)


def paired_config(old,identity,tmp):
    cfg=deepcopy(old);cfg['baseline'].update(code=identity,tmpdir=str(tmp))
    cfg['regression_gram']=deepcopy(SPEC)
    assert_pair(cfg,old)
    return cfg


def assert_pair(cfg,old):
    plain=deepcopy(cfg)
    if plain.pop('regression_gram')!=SPEC:raise ValueError('定位几何协议改变')
    plain['baseline']['code']=old['baseline']['code'];plain['baseline']['tmpdir']=old['baseline']['tmpdir']
    if plain!=old:raise ValueError('除声明定位几何及运行身份外改变了原Prototype配置')
    if cfg['seed']!=123 or cfg['loader']['batch_size']!=2 or cfg['prototype_capacity']['variant']!='control':
        raise ValueError('必须匹配seed123/batch2/原128维Prototype')


def balanced_indices(eligible,cap):
    """每视频等距最多cap点，按轮次交错；不消费原训练随机流。"""
    indices=[];order=[];batch=eligible.shape[0]
    for video in range(batch):
        ids=eligible[video].nonzero(as_tuple=True)[0]
        if len(ids)>cap:ids=ids[torch.linspace(0,len(ids)-1,cap,device=ids.device).long()]
        indices.extend([(video,ids)])
        order.append(torch.arange(len(ids),device=eligible.device)*batch+video)
    videos=torch.cat([torch.full_like(ids,v) for v,ids in indices])
    points=torch.cat([ids for _,ids in indices]);order=torch.cat(order).argsort()
    return videos[order],points[order]


def install(ctx,cfg):
    if cfg['regression_gram']!=SPEC:raise ValueError('定位几何spec改变')
    model=ctx.model
    if model.use_trident_head or model.reg_head is not model.reg_head_target or model.reg_head is not model.reg_head_pl:
        raise ValueError('定位几何要求当前共享普通回归塔')
    model.regression_gram_state=dict(step=0,attempts=0,events=0,digest='0'*64)
    capture=dict(active=False,reg=[],cls=[],current=None)
    def reg_start(module,inputs):
        if capture['active']:capture['current']=dict(features=inputs[0],masks=inputs[1],taps=[])
    def taps(module,inputs):
        if capture['active'] and len(capture['reg'])<2:
            capture['current']['taps'].append(F.unfold(inputs[0].unsqueeze(2),(1,3),padding=(0,1)).transpose(1,2))
    def reg_end(module,inputs,output):
        if capture['active']:capture['reg'].append(capture['current']);capture['current']=None
    def cls_end(module,inputs,output):
        if capture['active']:capture['cls'].append(output[0])
    ctx.regression_gram_handles=[model.reg_head.register_forward_pre_hook(reg_start),
        model.reg_head.offset_head.register_forward_pre_hook(taps),model.reg_head.register_forward_hook(reg_end),
        model.cls_head.register_forward_hook(cls_end)]
    original=model.forward;ctx.regression_gram_gradient_epoch=None
    names=['events','supported_levels','paired_points','loss','effective_loss','angle','scale',
           'target_fpn_gradient_norm','regression_gradient_norm','degenerate_levels']
    def forward(*args,**kwargs):
        training=kwargs.get('mode')=='train';epoch=kwargs.get('curr_epoch',0)
        if not training:return original(*args,**kwargs)
        assert_no_target_gt(kwargs['video_list_tgt'])
        state=model.regression_gram_state;state['step']+=1
        active=epoch>=cfg['train_cfg']['warmup_adv']
        capture.update(active=active,reg=[],cls=[],current=None)
        try:
            output=original(*args,**kwargs);metrics={k:0. for k in names}
            if active:
                if len(capture['reg'])!=3 or len(capture['cls'])!=3:raise ValueError('原source/target/pl共享头调用次序改变')
                source,target=capture['reg'][:2];logits=capture['cls'][1]
                points=model.point_generator(source['features'])
                classes,_=model.label_points(points,[v['segments'] for v in kwargs['video_list_src']],
                                              [v['labels'] for v in kwargs['video_list_src']])
                pos=torch.stack(classes).sum(-1)>0;start=0;losses=[];details=[]
                for level,(sx,tx,sm,tm,logit) in enumerate(zip(source['taps'],target['taps'],source['masks'],target['masks'],logits)):
                    src=sm.squeeze(1).bool()&pos[:,start:start+sx.shape[1]];start+=sx.shape[1]
                    tgt=tm.squeeze(1).bool()&(logit.detach().sigmoid().amax(1)>=.5)
                    sv,sp=balanced_indices(src,SPEC['per_video_level_cap']);tv,tp=balanced_indices(tgt,SPEC['per_video_level_cap'])
                    n=min(len(sp),len(tp))
                    if n<SPEC['minimum_points']:continue
                    try:loss,info=gram_alignment(sx[sv[:n],sp[:n]],tx[tv[:n],tp[:n]],normalize=True)
                    except ValueError as error:
                        if str(error)!='共同子空间退化，不能安全迁移':raise
                        metrics['degenerate_levels']+=1;continue
                    if not torch.isfinite(loss):raise ValueError('定位Gram损失非有限')
                    losses.append(loss);details.append(dict(level=level,**info,
                        source_video_indices=sv[:n].tolist(),source_point_indices=sp[:n].tolist(),
                        target_video_indices=tv[:n].tolist(),target_point_indices=tp[:n].tolist()))
                if losses:
                    loss=torch.stack(losses).mean().float();output[0]['downstream_loss']=output[0]['downstream_loss']+loss
                    metrics.update(events=1.,supported_levels=float(len(losses)),paired_points=float(sum(d['source_points'] for d in details)),
                        loss=float(loss.detach()),effective_loss=float(loss.detach())*cfg['model']['loss_weights']['weight_downstream'],
                        angle=sum(d['angle'] for d in details)/len(details),scale=sum(d['scale'] for d in details)/len(details))
                    if ctx.regression_gram_gradient_epoch!=epoch:
                        grads=torch.autograd.grad(loss,target['features'],retain_graph=True,allow_unused=True)
                        metrics['target_fpn_gradient_norm']=sum(float(g.square().sum()) for g in grads if g is not None)**.5
                        grads=torch.autograd.grad(loss,tuple(model.reg_head.head.parameters()),retain_graph=True,allow_unused=True)
                        metrics['regression_gradient_norm']=sum(float(g.square().sum()) for g in grads if g is not None)**.5
                        if not all(math.isfinite(metrics[k]) and metrics[k]>0 for k in ['target_fpn_gradient_norm','regression_gradient_norm']):
                            raise ValueError('定位几何没有真实目标FPN/回归塔梯度')
                        ctx.regression_gram_gradient_epoch=epoch
                    state['events']+=1
                state['attempts']+=1
                event=dict(step=state['step'],epoch=epoch,source_videos=[v['metadata']['video_id'] for v in kwargs['video_list_src']],
                    target_videos=[v['metadata']['video_id'] for v in kwargs['video_list_tgt']],levels=details,
                    loss=metrics['loss'],target_gt_used=False)
                state['digest']=hashlib.sha256((state['digest']+json.dumps(event,sort_keys=True,separators=(',',':'))).encode()).hexdigest()
                if getattr(ctx,'regression_gram_ledger',None):
                    with Path(ctx.regression_gram_ledger).open('a') as f:f.write(json.dumps(dict(event,digest=state['digest']))+'\n')
            for k,v in metrics.items():output[0]['gram_'+k]=torch.as_tensor(v,device=model.device).detach()
            return output
        finally:capture.update(active=False,reg=[],cls=[],current=None)
    model.forward=forward


def audit(entry):
    from tools import baseline_protocol as protocol
    e=Path(entry);read,sha=protocol.read,protocol.sha;r=read(e/'complete.json');cfg=read(e/'config.json')
    ref=Path(r['reference']);assert sha(ref/'complete.json')==r['reference_complete_sha256']
    assert_pair(cfg,read(ref/'config.json'))
    result=protocol.audit_entry(e,expected_schema=SCHEMA)
    root=e.parents[2];acceptance=read(root/'acceptance.json');report=acceptance['reports'][r['shift']]
    assert sha(report['path'])==report['sha256']
    probe=read(report['path']);assert probe['status']=='passed' and probe['full_state_roundtrip']=='passed'
    assert probe['initial_state_sha']==read(ref/'termination.json')['initial_state_sha']
    assert sha(probe['guard'])==probe['guard_sha256']
    for receipt in [probe]+[read(e/(name+'_gpu.json')) for name in ['train','evaluate']]:
        guard=read(receipt['guard']);lease=read(guard['lease']);assert sha(guard['lease'])==guard['lease_sha256']
        assert guard['owner_pid']==lease['owner_pid'] and lease['commit']==cfg['baseline']['code']['commit']
    if read(e/'detector_initialization.json')!=read(ref/'detector_initialization.json'):
        raise ValueError('冷启动检测器或原128维Prototype初始化改变')
    if read(e/'termination.json')['initial_state_sha']!=read(ref/'termination.json')['initial_state_sha']:
        raise ValueError('完整冷启动状态不匹配')
    cap=cfg['prototype_capacity'];assert sha(cap['prepared_cache'])==cap['prepared_cache_sha256']
    rows=[json.loads(z) for z in (e/'metrics.jsonl').read_text().splitlines()]
    if not any(z['regression_gram']['gram_target_fpn_gradient_norm']>0 and z['regression_gram']['gram_regression_gradient_norm']>0 for z in rows):
        raise ValueError('正式训练没有有效目标FPN/回归塔几何梯度')
    if any(z['regression_gram']['gram_events']!=0 for z in rows if z['epoch']<cfg['train_cfg']['warmup_adv']):
        raise ValueError('提前启动定位几何')
    latest=torch.load(e/'latest.pt',map_location='cpu');state=latest['model_controls']['regression_gram_state']
    assert state['step']==len(rows)*cfg['baseline']['epoch_budget']['iterations_per_epoch']
    ledger=[json.loads(z) for z in (e/'regression_gram_events.jsonl').read_text().splitlines()]
    assert len(ledger)==state['attempts'] and sum(bool(z['levels']) for z in ledger)==state['events'] and state['events']>0
    digest='0'*64;source=set(read(cfg['dataset']['json_file'])['database']);target=set(read(cfg['baseline']['target_manifest'])['database'])
    for item in ledger:
        given=item.pop('digest');assert not item['target_gt_used'] and item['epoch']>=cfg['train_cfg']['warmup_adv']
        assert set(item['source_videos'])<=source and set(item['target_videos'])<=target
        for level in item['levels']:
            n=level['source_points'];assert n==level['target_points'] and n>=SPEC['minimum_points']
            assert len(level['source_point_indices'])==len(level['target_point_indices'])==n
        digest=hashlib.sha256((digest+json.dumps(item,sort_keys=True,separators=(',',':'))).encode()).hexdigest();assert digest==given
    assert digest==state['digest']
    expected_attempts=max(0,len(rows)-cfg['train_cfg']['warmup_adv']+1)*cfg['baseline']['epoch_budget']['iterations_per_epoch']
    assert state['attempts']==expected_attempts
    for domain in ['source','target']:
        manifest=read(e/(domain+'_val_unlabeled.json'));assert all('annotations' not in v for v in manifest['database'].values())
        evaluation=read(e/'evaluation'/(domain+'_val.json'));assert evaluation['prediction_sha256']==sha(e/(domain+'_val.npz'))
    original=read(ref/'evaluation/target_val.json')['map_0.1_to_0.5']
    return dict(result,original_prototype_map=original,delta_pp=100*(result['map']-original))
