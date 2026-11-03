"""调用官方单epoch训练/源验证函数，只重写数据隔离、停止与工件管理。"""
from copy import deepcopy
import itertools
import io
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from libs.datasets import make_dataset
from libs.datasets.data_utils import trivial_batch_collator, worker_init_reset_seed
from libs.models.SADA import make_meta_arch
from libs.models.SADA.optimizer import make_optimizer, make_scheduler
from libs.trainer.da_trainer import Trainer, move_to_device
from libs.utils import fix_random_seed, ModelEma2
from libs.evaluation.metrics import MetricTracker
from tools.baseline_protocol import (SCHEMA, POLICY, assert_no_target_gt, decision, random_seeds, validate_cuda_rng,
                                      read, save, sha, termination)


class BudgetLoader:
    def __init__(self, loader, length):
        self.loader, self.length = loader, length
    def __iter__(self):
        return iter(self.loader)
    def __len__(self):
        return self.length


class StepLoader:
    """测速/真实数据验收限定step，不保存正式checkpoint。"""
    def __init__(self, loader, steps):
        self.loader, self.steps = loader, steps
    def __iter__(self):
        return itertools.islice(itertools.cycle(self.loader), self.steps)
    def __len__(self):
        return self.steps


def loader(dataset, training, generator, batch, workers, released=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=workers,
        shuffle=training, drop_last=training, generator=generator,
        collate_fn=trivial_batch_collator, persistent_workers=released,
        worker_init_fn=worker_init_reset_seed if training else None)


def dataset(cfg, path, splits, training, unlabeled=False):
    options = deepcopy(cfg['dataset'])
    options.update(json_file=str(path), unlabeled=unlabeled)
    return make_dataset('epic', training, splits, **options)


def build(cfg):
    seeds = random_seeds(cfg['seed'])
    release_generator = fix_random_seed(seeds['model'], include_cuda=True)
    released = bool(cfg.get('release_data_protocol'))
    # 旧PyTorch的UUID/NVML计数可能为0，直接播种当前逻辑卡，不依赖seed_all的枚举。
    torch.cuda.manual_seed(seeds['model'])
    ctx = object.__new__(Trainer)
    ctx.cfg = deepcopy(cfg)
    ctx.ema_active = True
    initial_cuda_state=torch.cuda.get_rng_state(0)
    ctx.cuda_seed_proof=dict(requested_seed=seeds['model'],
        actual_cuda_seed=torch.cuda.default_generators[0].initial_seed(),
        state_sha256=tensor_digest(initial_cuda_state),
        uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if ctx.cuda_seed_proof['actual_cuda_seed']!=seeds['model']:raise ValueError('实际GPU播种失败')
    ctx.model = make_meta_arch('sada', ctx.cfg, device='cuda:0').to('cuda:0')
    # 保留官方独立构造EMA的初始化方式，不暗中换成另一套训练算法。
    ema_model = make_meta_arch('sada', ctx.cfg, device='cuda:0').to('cuda:0')
    ctx.model_ema = ModelEma2(ema_model, decay=cfg['train_cfg']['ema_decay'])
    ctx.initial_state_sha = tensor_digest(ctx.model.state_dict())
    ctx.initial_detector_sha=tensor_digest({k:v for k,v in ctx.model.state_dict().items() if not k.startswith('rq3_capacity.')})
    if cfg['baseline']['regime'] == 'so':
        # 模型参数构造与UDA完全相同；仅关闭对抗目标，保持检测与优化器阶段。
        ctx.cfg['model']['adv_method'] = ['None']
    # 作者的fix_random_seed返回CPU默认generator；模型初始化也会推进这条流。
    ctx.generators = ({k: release_generator for k in ('source', 'target', 'validation')}
                      if released else {k: torch.Generator().manual_seed(seeds[k])
                      for k in ('source', 'target', 'validation')})
    ctx.stream_records = []
    source = dataset(cfg, cfg['dataset']['json_file'], cfg['src_train_split'], True)
    ctx.train_loader_src = loader(source, True, ctx.generators['source'], cfg['loader']['batch_size'],
                                  cfg['loader']['num_workers'], released)
    if cfg['baseline']['regime'] == 'uda':
        target = dataset(cfg, cfg['baseline']['target_manifest'], cfg['tgt_train_split'], True, not released)
    else:
        target = dataset(cfg, cfg['dataset']['json_file'], cfg['src_train_split'], True, True)
    target_loader = loader(target, True, ctx.generators['target'], cfg['loader']['batch_size'],
                           cfg['loader']['num_workers'], released)
    budget = cfg['baseline']['epoch_budget']['iterations_per_epoch']
    ctx.train_loader_tgt = BudgetLoader(target_loader, budget)
    val = dataset(cfg, cfg['dataset']['json_file'], cfg['src_val_split'], False)
    ctx.val_loader_src = loader(val, False, None if released else ctx.generators['validation'],
                                1, cfg['loader']['num_workers'], released)
    exclude = ['disc_dom_adv', 'disc_dom_conditional_adv', 'learnable_embeddings']
    for key in ('source', 'alignment', 'discriminator'):
        is_disc = key == 'discriminator'
        opt = make_optimizer(ctx.model, cfg['opt'], cfg['opt']['disc_learning_rate' if is_disc else 'learning_rate'],
                             include_params=exclude if is_disc else [], exclude_params=[] if is_disc else exclude)
        sched = make_scheduler(opt, cfg['opt'], budget)
        setattr(ctx, {'source':'optimizer_gen_source', 'alignment':'optimizer_gen_source_alignment',
                       'discriminator':'optimizer_disc'}[key], opt)
        setattr(ctx, {'source':'lr_scheduler_gen_source', 'alignment':'lr_scheduler_gen_source_alignment',
                       'discriminator':'lr_scheduler_disc'}[key], sched)
    ctx.train_metrics = MetricTracker('t_loss')
    ctx.valid_metrics_src = MetricTracker('v_loss_src')
    ctx.valid_metrics_tgt = MetricTracker('v_loss_tgt')
    original = ctx.model.forward
    ctx.checked_steps = 0
    ctx.first_gradient_norm = None
    for opt in (ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment):
        step = opt.step
        def checked_step(*args, _opt=opt, _step=step, **kwargs):
            if ctx.first_gradient_norm is None:
                grads=[p.grad for group in _opt.param_groups for p in group['params'] if p.grad is not None]
                norm=torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads])).item()
                if not math.isfinite(norm) or norm<=0:raise ValueError('源监督梯度无效')
                ctx.first_gradient_norm=norm
            return _step(*args,**kwargs)
        opt.step=checked_step
    def checked_forward(*args, **kwargs):
        if kwargs.get('mode') == 'train':
            if not released:
                assert_no_target_gt(kwargs['video_list_tgt'])
            else:
                def record(v):
                    m = v['metadata']
                    return dict(video=m['video_id'], crop_start=m.get('crop_start', 0),
                        crop_length=v['feats'].shape[-1], label_count=v['labels'].numel(),
                        GT_sha256=tensor_digest((v['segments'], v['labels'])))
                ctx.stream_records.append(dict(source=[record(v) for v in kwargs['video_list_src']],
                    target=[record(v) for v in kwargs['video_list_tgt']]))
            if kwargs.get('domain', 'src') != 'src':
                raise ValueError('正式训练只能监督源域')
            ctx.checked_steps += 1
        result = original(*args, **kwargs)
        if kwargs.get('mode') != 'test':
            for name, value in result[0].items():
                if not torch.isfinite(value).all():
                    raise ValueError('非有限loss: '+name)
        return result
    ctx.model.forward = checked_forward
    return ctx


def tensor_digest(value):
    import hashlib
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x: visit(y)
        elif torch.is_tensor(x):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def controls(model):
    fields = ('loss_normalizer', 'loss_normalizer_pseudo_labels', 'train_label_smoothing',
              'cls_thres_prob', 'cls_thres_ti', 'ema_src_centroids', 'ema_tgt_centroids')
    return {k: deepcopy(getattr(model, k)) for k in fields}


def optimizers(ctx):
    return [ctx.optimizer_gen_source, ctx.optimizer_gen_source_alignment, ctx.optimizer_disc]


def schedulers(ctx):
    return [ctx.lr_scheduler_gen_source, ctx.lr_scheduler_gen_source_alignment, ctx.lr_scheduler_disc]


def capture(ctx, cfg, epoch, history):
    return dict(schema=cfg.get('release_data_protocol',{}).get('schema',SCHEMA), config=cfg, epoch=epoch, history=deepcopy(history),
        state_dict=ctx.model.state_dict(), ema_state_dict=ctx.model_ema.module.state_dict(),
        model_controls=controls(ctx.model), ema_controls=controls(ctx.model_ema.module),
        optimizers=[x.state_dict() for x in optimizers(ctx)],
        schedulers=[x.state_dict() for x in schedulers(ctx)],
        loader_resume=('unsupported_persistent_worker_RNG_not_captured' if cfg.get('release_data_protocol')
                       else 'epoch_boundary_generator_states'),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                 cuda=[torch.cuda.get_rng_state(0)], loaders={k:g.get_state() for k,g in ctx.generators.items()}))


def checkpoint(path, state):
    tmp = Path(str(path)+'.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def restore(ctx, saved):
    ctx.model.load_state_dict(saved['state_dict'], strict=True)
    ctx.model_ema.module.load_state_dict(saved['ema_state_dict'], strict=True)
    for m, key in [(ctx.model, 'model_controls'), (ctx.model_ema.module, 'ema_controls')]:
        for name, value in saved[key].items():
            current = getattr(m, name)
            if torch.is_tensor(current): current.data.copy_(value)
            elif isinstance(current, list):
                for a,b in zip(current,value): a.data.copy_(b)
            else: setattr(m, name, value)
    for current, value in zip(optimizers(ctx), saved['optimizers']): current.load_state_dict(value)
    for current, value in zip(schedulers(ctx), saved['schedulers']): current.load_state_dict(value)
    rng = saved['rng']
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['cpu'].cpu())
    validate_cuda_rng(rng['cuda'])
    torch.cuda.set_rng_state(rng['cuda'][0].cpu(),device=0)
    for name,g in ctx.generators.items(): g.set_state(rng['loaders'][name].cpu())


def epoch_setup(ctx, epoch):
    ctx.stream_records = []
    warmup = ctx.cfg['train_cfg']['warmup_adv']
    if epoch == warmup and ctx.cfg['model'].get('src_regularization') == 'label_smoothing':
        ctx.model.train_label_smoothing = 0
    ctx.train_metrics.reset()
    ctx.first_gradient_norm = None


def probe(cfg, output, warmup_steps=2, timed_steps=6):
    ctx = build(cfg)
    # 固定40/200名单不能用前6批覆盖来代表；诊断遍历完整原始epoch，不补抽候选。
    if cfg.get('prototype_capacity'):
        timed_steps=max(len(ctx.train_loader_src),len(ctx.train_loader_tgt))
    reports = {}
    for epoch in (1, 5):
        epoch_setup(ctx, epoch)
        originals = ctx.train_loader_src, ctx.train_loader_tgt
        before = tensor_digest(ctx.model.state_dict())
        if cfg.get('prototype_capacity'):
            branch_before=tensor_digest(dict(ctx.model.rq3_capacity.named_parameters()))
        ctx.train_loader_src, ctx.train_loader_tgt = [StepLoader(x, warmup_steps) for x in originals]
        ctx._train_epoch(epoch)
        ctx.train_metrics.reset()
        ctx.train_loader_src, ctx.train_loader_tgt = [StepLoader(x, timed_steps) for x in originals]
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic(); ctx._train_epoch(epoch); torch.cuda.synchronize()
        seconds = time.monotonic()-started
        changed = before != tensor_digest(ctx.model.state_dict())
        loss = float(ctx.train_metrics.result()['t_final_loss'])
        if not changed or not math.isfinite(loss) or loss <= 0:
            raise ValueError('真实数据验收没有有效优化')
        reports[str(epoch)] = dict(seconds=seconds, steps=timed_steps, loss=loss,
            samples_per_second=timed_steps*cfg['loader']['batch_size']*2/seconds,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            model_updated=changed, source_gradient_norm=ctx.first_gradient_norm,target_gt_losses=[0.,0.,0.])
        if cfg.get('prototype_capacity'):
            values=ctx.train_metrics.result()
            reports[str(epoch)]['auxiliary']={k[2:]:float(v) for k,v in values.items() if k.startswith('t_rq3_')}
            reports[str(epoch)]['branch_parameters_updated']=branch_before!=tensor_digest(dict(ctx.model.rq3_capacity.named_parameters()))
        ctx.train_loader_src, ctx.train_loader_tgt = originals
    ctx.valid_metrics_src.reset()
    validation = float(ctx._valid_epoch(5, ctx.val_loader_src, domain='src'))
    if not math.isfinite(validation) or validation <= 0:
        raise ValueError('真实源验证失败')
    # 内存序列化真实模型/EMA/优化器，不写诊断checkpoint；检查GPU随机流可真正接续。
    saved=capture(ctx,cfg,5,[[5,validation]])
    buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu');buffer.close()
    validate_cuda_rng(saved['rng']['cuda'])
    def draws():
        return dict(python=random.random(),numpy=np.random.random(4),cpu=torch.rand(4),
            cuda=torch.rand(4,device='cuda:0'),loaders={k:torch.randperm(8,generator=g) for k,g in ctx.generators.items()})
    restore(ctx,saved);expected_draw=tensor_digest(draws())
    restore(ctx,saved)
    actual=capture(ctx,cfg,5,[[5,validation]])
    if tensor_digest(actual)!=tensor_digest(saved) or tensor_digest(draws())!=expected_draw:
        raise ValueError('真实GPU完整状态恢复或随机流重放失败')
    restore(ctx,saved)
    save(output, dict(status='passed', batch=cfg['loader']['batch_size'], phases=reports,
                      source_validation_loss=validation,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      checked_target_batches=ctx.checked_steps, initial_state_sha=ctx.initial_state_sha,
                      initial_detector_sha=ctx.initial_detector_sha,
                      cuda_seed_proof=ctx.cuda_seed_proof,
                      rng_roundtrip=dict(status='passed',cuda_rng_count=1,full_state_sha=tensor_digest(saved),
                                         replayed_draw_sha=expected_draw),
                      diagnostic_steps_per_phase=timed_steps))


def train(cfg, entry, resume=None):
    if resume and cfg.get('release_data_protocol'):
        raise ValueError('本对照不支持重建常驻worker随机状态，禁止不精确续跑')
    entry = Path(entry)
    ctx = build(cfg)
    save(entry/'runtime_config.json',ctx.cfg)
    save(entry/'seed_proof.json',ctx.cuda_seed_proof)
    if cfg.get('prototype_capacity'):
        save(entry/'detector_initialization.json',dict(sha256=ctx.initial_detector_sha,
            specification=ctx.model.rq3_capacity.specification,
            auxiliary_parameters=sum(p.numel() for p in ctx.model.rq3_capacity.parameters())))
    history, start, rows = [], 1, []
    if resume:
        saved = torch.load(resume, map_location='cpu')
        if saved['config'] != cfg: raise ValueError('恢复配置必须完全相同')
        restore(ctx, saved)
        history, start = saved['history'], saved['epoch']+1
        rows = [json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
        rows = [x for x in rows if x['epoch'] < start]
        actual = capture(ctx, cfg, saved['epoch'], history)
        expected = {k:saved[k] for k in actual}
        if tensor_digest(actual) != tensor_digest(expected): raise ValueError('完整恢复状态不一致')
        save(entry/'restore_proof.json', dict(status='passed', start_epoch=start,
                                             checkpoint_sha256=sha(resume), full_state_sha=tensor_digest(actual)))
    best = min([v for _,v in history], default=float('inf'))
    for epoch in range(start, 51):
        if history and decision(history)['reason'] != 'running': break
        print('BASELINE_EPOCH', cfg['baseline']['regime'], cfg['baseline']['shift'], epoch, flush=True)
        epoch_setup(ctx, epoch)
        previous_checked=ctx.checked_steps
        started = time.monotonic(); ctx._train_epoch(epoch); torch.cuda.synchronize()
        metrics = ctx.train_metrics.result()
        if ctx.checked_steps-previous_checked != cfg['baseline']['epoch_budget']['iterations_per_epoch']:
            raise ValueError('真实更新次数与冻结预算不同')
        row = dict(epoch=epoch, train_loss=float(metrics['t_final_loss']), seconds=time.monotonic()-started,
                   source_cls_loss=float(metrics['t_cls_loss']), source_reg_loss=float(metrics['t_reg_loss']),
                   source_gradient_norm=ctx.first_gradient_norm,
                   target_gt_cls_loss=0., target_gt_reg_loss=0., target_gt_downstream_loss=0.)
        if cfg.get('prototype_capacity'):
            row['auxiliary']={k[2:]:float(v) for k,v in metrics.items() if k.startswith('t_rq3_')}
        if cfg.get('release_data_protocol'):
            stream_path=entry/'streams'/('epoch%03d.json'%epoch)
            save(stream_path,ctx.stream_records)
            row['data_stream_sha256']=sha(stream_path)
        if epoch%5 == 0:
            ctx.valid_metrics_src.reset()
            value = float(ctx._valid_epoch(epoch, ctx.val_loader_src, domain='src'))
            history.append([epoch,value]); row['source_validation_loss'] = value
            if value < best:
                best = value; checkpoint(entry/'best.pt', capture(ctx,cfg,epoch,history))
        rows.append(row)
        (entry/'metrics.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
        print('BASELINE_METRICS', json.dumps(row,allow_nan=False), flush=True)
        if epoch%5 == 0:
            checkpoint(entry/'latest.pt', capture(ctx,cfg,epoch,history))
            result = decision(history)
            if result['reason'] != 'running':
                save(entry/'termination.json', dict(history=history, decision=result,
                                                    initial_state_sha=ctx.initial_state_sha))
                print('BASELINE_TERMINATED',result,flush=True); break
    if not (entry/'termination.json').exists():
        raise ValueError('未生成合法停止证明')


def evaluate(cfg, entry, annotation):
    from libs.evaluation import ANETdetection
    entry = Path(entry)
    saved = torch.load(entry/'best.pt', map_location='cpu')
    terminal = read(entry/'termination.json')
    if cfg.get('release_data_protocol'):
        from tools.release_data_protocol import release_decision
        selected=release_decision(terminal['history'])
    else:selected=termination(terminal['history'])
    if saved['epoch'] != selected['best_epoch']:
        raise ValueError('不是源域选定的best EMA')
    fix_random_seed(cfg['seed'], include_cuda=True)
    torch.cuda.manual_seed(cfg['seed'])
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'], strict=True); model.eval()
    results = {}
    for domain, splits in [('source',cfg['src_val_split']), ('target',cfg['tgt_val_split'])]:
        ds = dataset(cfg, annotation, splits, False)
        dl = loader(ds,False,torch.Generator().manual_seed(cfg['seed']),1,cfg['loader']['num_workers'])
        columns = {'video-id':[], 't-start':[], 't-end':[], 'label':[], 'score':[]}
        with torch.no_grad():
            for video in dl:
                video = move_to_device(video,'cuda:0')
                prediction = model(video_list_src=video,video_list_tgt=video,mode='test',domain=domain,
                                   curr_epoch=saved['epoch'])[0 if domain=='source' else 1]
                for item in prediction:
                    n = len(item['scores']); columns['video-id'].extend([item['video_id']]*n)
                    for k,v in [('t-start',item['segments'][:,0]),('t-end',item['segments'][:,1]),
                                ('label',item['labels']),('score',item['scores'])]:
                        columns[k].extend(v.detach().cpu().tolist())
        evaluation_split=splits[0] if cfg.get('release_data_protocol') else splits
        evaluator = ANETdetection(str(annotation),evaluation_split,num_workers=4,action_subsampling='10_majority_classes')
        checkpoint(entry/'evaluation'/('%s_predictions.pt'%domain),columns)
        ap,mean,*_ = evaluator.evaluate({k:np.asarray(v) for k,v in columns.items()},verbose=True)
        result = dict(map_by_tiou=[float(x) for x in ap],
                      videos=len(ds),prediction_count=len(columns['score']),selected_epoch=saved['epoch'],
                      checkpoint_sha256=sha(entry/'best.pt'),state='ema_state_dict',split=evaluation_split,
                      prediction_splits=splits)
        result['map_0.1_to_0.5']=float(mean)
        save(entry/'evaluation'/('%s_val.json' % domain),result);results[domain]=result
    save(entry/'evaluation/manifest.json',dict(config_sha256=sha(entry/'config.json'),
         checkpoint_sha256=sha(entry/'best.pt'),annotation_sha256=sha(annotation),results=results))
