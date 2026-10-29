"""调用官方单epoch训练/源验证函数，只重写数据隔离、停止与工件管理。"""
from copy import deepcopy
import itertools
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
from tools.baseline_protocol import (SCHEMA, POLICY, assert_no_target_gt, decision,
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


def loader(dataset, training, generator, batch, workers):
    return torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=workers,
        shuffle=training, drop_last=training, generator=generator,
        collate_fn=trivial_batch_collator, persistent_workers=False,
        worker_init_fn=worker_init_reset_seed if training else None)


def dataset(cfg, path, splits, training, unlabeled=False):
    options = deepcopy(cfg['dataset'])
    options.update(json_file=str(path), unlabeled=unlabeled)
    return make_dataset('epic', training, splits, **options)


def build(cfg):
    fix_random_seed(6, include_cuda=True)
    ctx = object.__new__(Trainer)
    ctx.cfg = deepcopy(cfg)
    ctx.ema_active = True
    ctx.model = make_meta_arch('sada', ctx.cfg, device='cuda:0').to('cuda:0')
    # 保留官方独立构造EMA的初始化方式，不暗中换成另一套训练算法。
    ema_model = make_meta_arch('sada', ctx.cfg, device='cuda:0').to('cuda:0')
    ctx.model_ema = ModelEma2(ema_model, decay=cfg['train_cfg']['ema_decay'])
    ctx.initial_state_sha = tensor_digest(ctx.model.state_dict())
    if cfg['baseline']['regime'] == 'so':
        # 模型参数构造与UDA完全相同；仅关闭对抗目标，保持检测与优化器阶段。
        ctx.cfg['model']['adv_method'] = ['None']
    ctx.generators = {k: torch.Generator().manual_seed(seed)
                      for k, seed in [('source', 6), ('target', 1000009), ('validation', 2000012)]}
    source = dataset(cfg, cfg['dataset']['json_file'], cfg['src_train_split'], True)
    ctx.train_loader_src = loader(source, True, ctx.generators['source'], cfg['loader']['batch_size'],
                                  cfg['loader']['num_workers'])
    if cfg['baseline']['regime'] == 'uda':
        target = dataset(cfg, cfg['baseline']['target_manifest'], cfg['tgt_train_split'], True, True)
    else:
        target = dataset(cfg, cfg['dataset']['json_file'], cfg['src_train_split'], True, True)
    target_loader = loader(target, True, ctx.generators['target'], cfg['loader']['batch_size'],
                           cfg['loader']['num_workers'])
    budget = cfg['baseline']['epoch_budget']['iterations_per_epoch']
    ctx.train_loader_tgt = BudgetLoader(target_loader, budget)
    val = dataset(cfg, cfg['dataset']['json_file'], cfg['src_val_split'], False)
    ctx.val_loader_src = loader(val, False, ctx.generators['validation'], 1, cfg['loader']['num_workers'])
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
            assert_no_target_gt(kwargs['video_list_tgt'])
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
    return dict(schema=SCHEMA, config=cfg, epoch=epoch, history=deepcopy(history),
        state_dict=ctx.model.state_dict(), ema_state_dict=ctx.model_ema.module.state_dict(),
        model_controls=controls(ctx.model), ema_controls=controls(ctx.model_ema.module),
        optimizers=[x.state_dict() for x in optimizers(ctx)],
        schedulers=[x.state_dict() for x in schedulers(ctx)],
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all(), loaders={k:g.get_state() for k,g in ctx.generators.items()}))


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
    torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    for name,g in ctx.generators.items(): g.set_state(rng['loaders'][name].cpu())


def epoch_setup(ctx, epoch):
    warmup = ctx.cfg['train_cfg']['warmup_adv']
    if epoch == warmup and ctx.cfg['model'].get('src_regularization') == 'label_smoothing':
        ctx.model.train_label_smoothing = 0
    ctx.train_metrics.reset()
    ctx.first_gradient_norm = None


def probe(cfg, output, warmup_steps=2, timed_steps=6):
    ctx = build(cfg)
    reports = {}
    for epoch in (1, 5):
        epoch_setup(ctx, epoch)
        originals = ctx.train_loader_src, ctx.train_loader_tgt
        before = tensor_digest(ctx.model.state_dict())
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
        ctx.train_loader_src, ctx.train_loader_tgt = originals
    ctx.valid_metrics_src.reset()
    validation = float(ctx._valid_epoch(5, ctx.val_loader_src, domain='src'))
    if not math.isfinite(validation) or validation <= 0:
        raise ValueError('真实源验证失败')
    save(output, dict(status='passed', batch=cfg['loader']['batch_size'], phases=reports,
                      source_validation_loss=validation,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      checked_target_batches=ctx.checked_steps, initial_state_sha=ctx.initial_state_sha))


def train(cfg, entry, resume=None):
    entry = Path(entry)
    ctx = build(cfg)
    history, start, rows = [], 1, []
    if resume:
        saved = torch.load(resume, map_location='cuda:0')
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
    if saved['epoch'] != termination(terminal['history'])['best_epoch']:
        raise ValueError('不是源域选定的best EMA')
    fix_random_seed(6, include_cuda=True)
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'], strict=True); model.eval()
    results = {}
    for domain, splits in [('source',cfg['src_val_split']), ('target',cfg['tgt_val_split'])]:
        ds = dataset(cfg, annotation, splits, False)
        dl = loader(ds,False,torch.Generator().manual_seed(6),1,cfg['loader']['num_workers'])
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
        evaluator = ANETdetection(str(annotation),splits,num_workers=4,action_subsampling='10_majority_classes')
        ap,mean,*_ = evaluator.evaluate({k:np.asarray(v) for k,v in columns.items()},verbose=True)
        result = dict(map_by_tiou=[float(x) for x in ap],
                      videos=len(ds),prediction_count=len(columns['score']),selected_epoch=saved['epoch'],
                      checkpoint_sha256=sha(entry/'best.pt'),state='ema_state_dict',split=splits)
        result['map_0.1_to_0.5']=float(mean)
        save(entry/'evaluation'/('%s_val.json' % domain),result);results[domain]=result
    save(entry/'evaluation/manifest.json',dict(config_sha256=sha(entry/'config.json'),
         checkpoint_sha256=sha(entry/'best.pt'),annotation_sha256=sha(annotation),results=results))
