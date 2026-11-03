"""S1/S2各AE/VAE，从原seed123初始化，私有lease四卡队列，不恢复旧任务。"""
import argparse
from copy import deepcopy
import gc
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity, termination
from tools.domain_factor_vae import SPEC

ORIGIN = Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
DATA = Path('/vepfs/vlms/SADA_PLUS')
TASKS = ('S1_ae', 'S1_vae', 'S2_ae', 'S2_vae')


def configuration(original, variant, identity, tmpdir):
    cfg = deepcopy(original)
    cfg['baseline'].update(code=identity, tmpdir=str(tmpdir))
    cfg['model']['adv_method'] = ['None']
    cfg['model']['loss_weights']['weight_adv'] = 0.
    cfg['domain_factor'] = dict(variant=variant, specification=SPEC)
    return cfg


def pair_check(ae, vae):
    expected = deepcopy(ae)
    expected['domain_factor']['variant'] = 'vae'
    assert expected == vae, 'AE/VAE只有预先固定的变分机制不同，不能改变数据和预算'


def clone_state(base, ctx, cfg, epoch):
    stream = io.BytesIO()
    import torch
    torch.save(base.capture(ctx, cfg, epoch, []), stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu')


def probe(cfg, entry, binding):
    """真实检测器更新、全部新参数接线、完整optimizer/EMA/独立噪声RNG下一步重放。"""
    import random
    import numpy as np
    import torch
    from tools import baseline_engine as base
    from libs.trainer.da_trainer import move_to_device
    from tools.domain_factor_vae import DomainFactor
    ctx = base.build(cfg)
    shared = {k: v for k, v in ctx.model.state_dict().items() if not k.startswith('domain_factor.')}
    assert base.tensor_digest(shared) == binding['historical_initial_state_sha']
    initial = ctx.initial_state_sha
    branch = ctx.model.domain_factor
    common = {k: v for k, v in branch.state_dict().items()
              if not k.startswith(('action_logvar.', 'domain_logvar.', 'action_prior.', 'domain_prior.'))}
    common_sha = base.tensor_digest(common)
    # 各新参数必须确实进入两个生成器优化器；不存在挂上分支却未训练的问题。
    for opt in (ctx.optimizer_gen_source, ctx.optimizer_gen_source_alignment):
        ids = {id(p) for group in opt.param_groups for p in group['params']}
        assert all(id(p) in ids for p in branch.parameters() if p.requires_grad)
    reports = {}
    for epoch in (1, cfg['train_cfg']['warmup_adv']):
        base.epoch_setup(ctx, epoch)
        originals = ctx.train_loader_src, ctx.train_loader_tgt
        steps = min(8, cfg['baseline']['epoch_budget']['iterations_per_epoch'])
        ctx.train_loader_src, ctx.train_loader_tgt = [base.StepLoader(x, steps) for x in originals]
        before = base.tensor_digest(dict(branch.named_parameters()))
        torch.cuda.synchronize()
        started = time.monotonic()
        ctx._train_epoch(epoch)
        torch.cuda.synchronize()
        seconds = time.monotonic()-started
        metrics = ctx.train_metrics.result()
        assert before != base.tensor_digest(dict(branch.named_parameters()))
        assert metrics['t_factor_source_fg'] > 0
        assert metrics['t_factor_task'] > 0 and metrics['t_factor_reconstruction'] > 0
        if branch.variant == 'vae':
            assert metrics['t_factor_kl'] > 0 and metrics['t_factor_posterior_variance'] > 0
        reports[str(epoch)] = dict(steps=steps, seconds=seconds, seconds_per_step=seconds/steps,
            projected_epoch_seconds=seconds/steps*cfg['baseline']['epoch_budget']['iterations_per_epoch'],
            metrics={k: float(v) for k, v in metrics.items()},
            updated=True, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        ctx.train_loader_src, ctx.train_loader_tgt = originals
    # 固定同一真实batch，不重开数据iterator来冒称多worker中间状态可恢复。
    source = move_to_device(next(iter(ctx.train_loader_src)), ctx.model.device)
    target = move_to_device(next(iter(ctx.train_loader_tgt)), ctx.model.device)
    epoch = cfg['train_cfg']['warmup_adv']
    def step():
        draw = dict(python=random.random(), numpy=np.random.rand(3), cpu=torch.rand(3),
                    cuda=torch.rand(3, device='cuda:0'),
                    loaders={k: torch.randperm(7, generator=g) for k, g in ctx.generators.items()})
        for opt in base.optimizers(ctx):
            opt.zero_grad(set_to_none=True)
        ctx.model.train()
        losses = ctx.model(video_list_src=source, video_list_tgt=target, mode='train', curr_epoch=epoch)[0]
        final = losses['downstream_loss']*cfg['model']['loss_weights']['weight_downstream']
        final.backward()
        gradients = {k: p.grad.detach().clone() for k, p in branch.named_parameters() if p.grad is not None}
        for name in ('action_mean.weight', 'domain_mean.weight', 'task_projection.0.weight',
                     'reconstruction.0.weight', 'task_classifier.weight', 'style_classifier.weight'):
            assert name in gradients and gradients[name].norm().item() > 0, name
        if branch.variant == 'vae':
            for name in ('action_logvar.weight', 'domain_logvar.weight', 'action_prior.weight', 'domain_prior.weight'):
                assert name in gradients and gradients[name].norm().item() > 0, name
        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), cfg['train_cfg']['clip_grad_l2norm'])
        ctx.optimizer_disc.step()
        ctx.optimizer_gen_source_alignment.step()
        ctx.model_ema.update(ctx.model)
        for scheduler in base.schedulers(ctx):
            scheduler.step()
        return base.tensor_digest(dict(draw=draw, loss=final.detach(), gradients=gradients))
    saved = clone_state(base, ctx, cfg, epoch)
    base.checkpoint(entry/'gpu_probe_state.pt', saved)
    first = step()
    expected = base.tensor_digest(base.capture(ctx, cfg, epoch, []))
    base.restore(ctx, saved)
    assert first == step()
    assert expected == base.tensor_digest(base.capture(ctx, cfg, epoch, []))
    # 独立小kernel覆盖真实冷启动还没有高置信目标前景时的条件对齐路径。
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(9104)
        tiny = DomainFactor(16, 3, branch.variant, 'cuda:0').cuda()
        fs = [torch.randn(2, 16, 8, device='cuda:0')]
        ms = [torch.ones(2, 1, 8, dtype=torch.bool, device='cuda:0')]
        ss, sc = tiny.transform(fs, ms, True)
        tt, tc = tiny.transform(fs, ms, True)
        labels = [torch.zeros(2, 8, 3, device='cuda:0')]
        labels[0][:, :4, 0] = 1.
        logits = [torch.full((2, 8, 3), -4., device='cuda:0')]
        logits[0][:, :4, 0] = 4.
        aux = tiny.auxiliary(sc, tc, labels, logits, epoch, epoch)
        assert aux['aligned_classes'] > 0
        aux['loss'].backward()
        assert tiny.action_domain_classifier[0].weight.grad.norm().item() > 0
    save(entry/'probe.json', dict(passed=True, shared_initialization_matches_historical=True,
        initial_state_sha=initial, common_branch_initial_sha=common_sha,
        parameter_count=sum(p.numel() for p in branch.parameters()),
        phases=reports, complete_next_step_replay_passed=True, replay_sha256=expected,
        probe_checkpoint_sha256=sha(entry/'gpu_probe_state.pt'), target_GT=False,
        conditional_domain_synthetic_gradient_passed=True,
        caveat='8-step timing includes loader startup; estimate only, not full-epoch throughput'))
    del ctx
    gc.collect()
    torch.cuda.empty_cache()
    return initial


def worker(entry, stage):
    from tools.baseline_gpu import first_cuda_guard
    cfg, binding = read(entry/'config.json'), read(entry/'origin_binding.json')
    assert cfg['baseline']['code'] == code_identity()
    for path, digest in binding['inputs'].items():
        assert sha(path) == digest, path
    os.chdir(DATA)
    guard = first_cuda_guard()
    started = time.monotonic()
    from tools import baseline_engine as base
    if stage == 'train':
        initial = probe(cfg, entry, binding)
        original_build = base.build
        def checked_build(config):
            ctx = original_build(config)
            assert ctx.initial_state_sha == initial, 'probe参数不得转移到正式训练'
            save(entry/'formal_initialization.json', dict(passed=True, sha256=initial,
                loaded_trained_checkpoint=False, same_as_probe_before_updates=True))
            return ctx
        base.build = checked_build
        try:
            base.train(cfg, entry)
        finally:
            base.build = original_build
        termination(read(entry/'termination.json')['history'])
    else:
        base.evaluate(cfg, entry, cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(stage+'_complete.json'), dict(passed=True, seconds=time.monotonic()-started,
        guard=guard, guard_sha256=sha(guard), config_sha256=sha(entry/'config.json')))


def run(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    identity = code_identity()
    root.mkdir(parents=True, exist_ok=False)
    save(root/'plan.json', dict(code=identity, specification=SPEC, tasks=TASKS,
        protocol='original seed123/batch2/3optimizers/schedules/EMA/source every5 patience15 max50',
        initialization='fresh original random initialization, no trained weights',
        comparison='AE vs VAE same mean/decoder/task objectives; VAE additionally has learned variance and priors',
        baseline='reuse historical complete SADA+Prototype; no repeat of old on/off experiments',
        target_GT='not loaded in training; targetVAL only final evaluation',
        interpretation='S1/S2 development, not six-direction final improvement',
        literature=['https://proceedings.mlr.press/v121/ilse20a.html', 'https://arxiv.org/abs/1608.06019'],
        not_a_full_reproduction='two latent spaces and conditional foreground GRL differ from three-space DIVA'))
    for shift in ('S1', 'S2'):
        original = read(ORIGIN/shift/'config.json')
        term = read(ORIGIN/shift/'termination.json')
        inputs = {p: sha(p) for p in (original['dataset']['json_file'], original['baseline']['target_manifest'],
                                    original['prototype_capacity']['prepared_cache'])}
        assert inputs[original['dataset']['json_file']] == original['baseline']['source_manifest_sha256']
        assert inputs[original['baseline']['target_manifest']] == original['baseline']['target_manifest_sha256']
        assert inputs[original['prototype_capacity']['prepared_cache']] == original['prototype_capacity']['prepared_cache_sha256']
        configs = {v: configuration(original, v, identity, root/'tmp') for v in ('ae', 'vae')}
        pair_check(configs['ae'], configs['vae'])
        for variant, cfg in configs.items():
            entry = root/(shift+'_'+variant)
            entry.mkdir()
            save(entry/'config.json', cfg)
            save(entry/'origin_binding.json', dict(origin=str(ORIGIN/shift), inputs=inputs,
                origin_config_sha256=sha(ORIGIN/shift/'config.json'),
                historical_initial_state_sha=term['initial_state_sha'],
                historical_target_map=read(ORIGIN/shift/'evaluation/target_val.json')['map_0.1_to_0.5']))
    (root/'tmp').mkdir()
    pending, waiting, active, completed, failed = list(TASKS), [], {}, [], {}
    def stop(sig, frame):
        save(root/'stop.json', dict(signal=sig, automatic_restart_authorized=False))
        for item in active.values():
            if item['p'].poll() is None:
                os.killpg(item['p'].pid, signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def start(task, stage, gpu, lease, lock):
        entry = root/task
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=lease['observed']['uuid'],
            PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(CODE), PYTHONHASHSEED='123',
            OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUBLAS_WORKSPACE_CONFIG=':4096:8',
            SADA_BASELINE_LEASE=lease['path'], SADA_BASELINE_LEASE_SHA256=sha(lease['path']),
            TMPDIR=str(root/'tmp'), TMP=str(root/'tmp'), TEMP=str(root/'tmp'))
        with (entry/(stage+'.log')).open('x') as out:
            proc = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__)), '--entry', str(entry),
                '--stage', stage], cwd=CODE, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        active[task] = dict(p=proc, stage=stage, gpu=gpu, lease=lease, lock=lock)
    try:
        while pending or waiting or active:
            for task, item in list(active.items()):
                if item['p'].poll() is None:
                    continue
                del active[task]
                if item['p'].returncode:
                    failed[task] = dict(stage=item['stage'], exit_code=item['p'].returncode)
                    item['lock'].close()
                    continue
                if item['stage'] == 'train':
                    waiting.append((task, item))
                    continue
                files = ['config.json', 'origin_binding.json', 'probe.json', 'formal_initialization.json',
                         'train_complete.json', 'evaluate_complete.json', 'best.pt', 'latest.pt',
                         'metrics.jsonl', 'termination.json', 'evaluation/source_val.json',
                         'evaluation/target_val.json', 'evaluation/source_predictions.pt', 'evaluation/target_predictions.pt']
                save(root/task/'complete.json', dict(passed=True, artifacts={f: sha(root/task/f) for f in files}))
                completed.append(task)
                item['lock'].close()
            while waiting and sum(x['stage'] == 'evaluate' for x in active.values()) < 2:
                task, item = waiting.pop(0)
                start(task, 'evaluate', item['gpu'], item['lease'], item['lock'])
            occupied = [x['gpu'] for x in active.values()]+[x['gpu'] for _, x in waiting]
            for gpu in range(4):
                if not pending or gpu in occupied:
                    continue
                observed = snapshot(gpu)
                if observed['free_mib'] < 16384:
                    continue
                lockpath = root/('private_gpu_'+str(gpu)+'.lock')
                lock = lockpath.open('a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                path = root/'leases'/(str(gpu)+'_'+str(time.time_ns())+'.json')
                lease = dict(owner_pid=os.getpid(), lock=str(lockpath), observed=observed,
                             minimum_free_mib=16384, commit=identity['commit'])
                save(path, lease)
                lease['path'] = str(path)
                start(pending.pop(0), 'train', gpu, lease, lock)
            save(root/'state.json', dict(owner_pid=os.getpid(), pending=pending,
                waiting_evaluate=[t for t, _ in waiting],
                active={t: dict(pid=x['p'].pid, gpu=x['gpu'], stage=x['stage']) for t, x in active.items()},
                completed=completed, failed=failed))
            time.sleep(5)
        results = {task: dict(target=read(root/task/'evaluation/target_val.json'),
                   source=read(root/task/'evaluation/source_val.json'),
                   complete_sha256=sha(root/task/'complete.json')) for task in completed}
        save(root/'final.json', dict(passed=len(completed) == 4, completed=completed, failed=failed, results=results))
    finally:
        for item in active.values():
            if item['p'].poll() is None:
                os.killpg(item['p'].pid, signal.SIGTERM)
        for item in list(active.values())+[x for _, x in waiting]:
            item['lock'].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--stage', choices=['train', 'evaluate'])
    args = parser.parse_args()
    worker(args.entry, args.stage) if args.entry else run(args.root)
