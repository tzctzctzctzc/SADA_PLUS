"""独立原Prototype候选前景分支：prepare/probe/train/infer/CPU评估，默认只验收。"""
import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity, decision
from tools.proposal_foreground import SPEC, SCHEMA, BASE_COMMIT, split_source
REFERENCE = Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
SHIFTS = tuple('S%d' % i for i in range(1, 7))


def prepare(root):
    from tools import baseline_engine as base
    root.mkdir(parents=True, exist_ok=False)
    identity = code_identity()
    if subprocess.check_output(['git', 'diff', BASE_COMMIT, '--', 'libs', 'tools/baseline_engine.py'], cwd=CODE):
        raise ValueError('本方向必须保持原Prototype训练核心逐字节不变')
    for shift in SHIFTS:
        reference = REFERENCE/shift
        cfg = read(reference/'config.json')
        assert cfg['seed'] == 123 and cfg['loader']['batch_size'] == 2 and cfg['prototype_capacity']['variant'] == 'control'
        assert sha(cfg['prototype_capacity']['prepared_cache']) == cfg['prototype_capacity']['prepared_cache_sha256']
        assert sha(cfg['dataset']['json_file']) == cfg['baseline']['source_manifest_sha256']
        assert sha(cfg['baseline']['target_manifest']) == cfg['baseline']['target_manifest_sha256']
        target = read(cfg['baseline']['target_manifest'])['database']
        assert all('annotations' not in v for v in target.values())
        train = base.dataset(cfg, cfg['dataset']['json_file'], cfg['src_train_split'], False)
        validation = base.dataset(cfg, cfg['dataset']['json_file'], cfg['src_val_split'], False)
        records = {}
        for item in train.data_list:
            # 与原__getitem__完全相同的major10映射和SlowFast半窗中心；不需要重新读大FPN。
            segments, labels = [], []
            for box, label in zip(item['segments'], item['labels']):
                if 0 <= label < 10:
                    segments.append(((box * item['fps'] - .5 * train.num_frames) /
                                     (train.feat_stride * train.downsample_rate)).tolist())
                    labels.append(int(label))
            records[item['id']] = dict(segments=segments, labels=labels)
        val_ids = sorted(x['id'] for x in validation.data_list)
        assert not set(records) & set(val_ids) and not (set(records) | set(val_ids)) & set(target)
        truth = root/'inputs'/shift/'source_train_truth.json'
        save(truth, dict(records=records, roles=split_source(list(records)), source_val_ids=val_ids,
                         source_manifest_sha256=sha(cfg['dataset']['json_file']), target_GT=False))
        cfg['baseline'].update(code=identity, tmpdir=str(root/'tmp'))
        cfg['proposal_foreground'] = dict(specification=SPEC, source_truth=str(truth), source_truth_sha256=sha(truth),
            reference=str(reference), reference_complete_sha256=sha(reference/'complete.json'))
        for folder in ('probes', 'formal'):
            save(root/folder/shift/'config.json', cfg)
        annotation = read(cfg['baseline']['annotation_for_offline_evaluation'])
        for domain, subsets in (('source', cfg['src_val_split']), ('target', cfg['tgt_val_split'])):
            unlabeled = dict(database={k:{name:v for name,v in row.items() if name != 'annotations'}
                for k,row in annotation['database'].items() if row['subset'].lower() in subsets})
            save(root/'inputs'/shift/(domain+'_val_unlabeled.json'), unlabeled, sort_keys=False)
    (root/'tmp').mkdir()
    save(root/'plan.json', dict(schema=SCHEMA, code=identity, specification=SPEC, shifts=SHIFTS,
        base=BASE_COMMIT, active_negative_experiment_touched=False, formal_training_authorized_in_this_command=False))


def build(cfg):
    from tools import baseline_engine as base
    from tools.proposal_foreground import enable_controls, install
    enable_controls()
    ctx = base.build(cfg)
    reference = Path(cfg['proposal_foreground']['reference'])
    assert ctx.initial_state_sha == read(reference/'termination.json')['initial_state_sha']
    assert ctx.initial_detector_sha == read(reference/'detector_initialization.json')['sha256']
    install(ctx, cfg)
    return ctx


def probe(cfg, entry, guard):
    import random
    import numpy as np
    import torch
    from tools import baseline_engine as base
    ctx = build(cfg)
    ctx.proposal_fg_probe_gradients = True
    source = base.move_to_device(next(iter(ctx.train_loader_src)), 'cuda:0')
    target = base.move_to_device(next(iter(ctx.train_loader_tgt)), 'cuda:0')
    kwargs = dict(video_list_src=source, video_list_tgt=target, mode='train', curr_epoch=1)
    initial = deepcopy(base.capture(ctx, cfg, 0, []))
    ctx.model.train()
    expected = {k:v.detach().clone() for k,v in ctx.model.proposal_fg_native_forward(**kwargs)[0].items()}
    base.restore(ctx, initial); ctx.proposal_fg_enabled = False
    actual = ctx.model(**kwargs)[0]
    assert all(torch.equal(v, actual[k].detach()) for k,v in expected.items())
    base.restore(ctx, initial); ctx.proposal_fg_enabled = True
    ctx.proposal_fg_ledger = str(entry/'probe_events.jsonl')
    ctx.proposal_fg_probe_observations = []
    # 真实原单epoch/三优化器更新，不借另写近似训练循环冒充。
    report = {}
    for epoch in (1, 5):
        base.epoch_setup(ctx, epoch)
        modules = dict(SGP=ctx.model.backbone_source, cls=ctx.model.cls_head, reg=ctx.model.reg_head,
                       foreground=ctx.model.proposal_foreground)
        before = {k:base.tensor_digest(m.state_dict()) for k,m in modules.items()}
        loaders = ctx.train_loader_src, ctx.train_loader_tgt
        ctx.train_loader_src, ctx.train_loader_tgt = [base.StepLoader(x, 2) for x in loaders]
        torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
        ctx._train_epoch(epoch); torch.cuda.synchronize()
        updated = {k:before[k] != base.tensor_digest(m.state_dict()) for k,m in modules.items()}
        assert all(updated.values())
        gradients = ctx.proposal_fg_aux_gradients
        assert all(math.isfinite(x) for x in gradients.values())
        assert gradients['SGP'] > 0 and gradients['foreground'] > 0 and gradients['cls'] == gradients['reg'] == 0
        report[str(epoch)] = dict(updated=updated, auxiliary_gradients=gradients,
            seconds=time.monotonic()-start, peak_mib=torch.cuda.max_memory_reserved()/2**20)
        ctx.train_loader_src, ctx.train_loader_tgt = loaders
    # 完整source预测：关闭额外头与默认尚未校准的阈值1.01应逐数组精确相同。
    ds = base.dataset(cfg, cfg['dataset']['json_file'], cfg['src_val_split'], False, True)
    video = base.move_to_device([ds[0]], 'cuda:0')
    ctx.model.eval()
    with torch.no_grad():
        kw = dict(video_list_src=video, video_list_tgt=video, mode='test')
        old = ctx.model.proposal_fg_native_forward(**kw)
        ctx.proposal_fg_enabled = False
        off = ctx.model(**kw)
        ctx.proposal_fg_enabled = True
        on = ctx.model(**kw)
        assert base.tensor_digest(old) == base.tensor_digest(off) == base.tensor_digest(on)
    observation_path = entry/'probe_source_observations.pt'
    base.checkpoint(observation_path, ctx.proposal_fg_probe_observations)
    # 新head实际属于EMA与两个生成器optimizer，discriminator不含新head。
    ids = {id(p) for p in ctx.model.proposal_foreground.parameters()}
    for opt in (ctx.optimizer_gen_source, ctx.optimizer_gen_source_alignment):
        assert ids <= {id(p) for g in opt.param_groups for p in g['params']}
    assert not ids & {id(p) for g in ctx.optimizer_disc.param_groups for p in g['params']}
    saved = deepcopy(base.capture(ctx, cfg, 5, []))
    stream = io.BytesIO(); torch.save(saved, stream); stream.seek(0)
    loaded = torch.load(stream, map_location='cpu'); stream.close(); base.restore(ctx, loaded)
    assert base.tensor_digest(base.capture(ctx, cfg, 5, [])) == base.tensor_digest(saved)
    def draws():
        return [random.random(), np.random.rand(3), torch.rand(3), torch.rand(3, device='cuda:0'),
                {k:torch.randperm(8, generator=g) for k,g in ctx.generators.items()}]
    base.restore(ctx, loaded); expected_draw = base.tensor_digest(draws())
    base.restore(ctx, loaded); assert expected_draw == base.tensor_digest(draws())
    parameters = sum(p.numel() for p in ctx.model.proposal_foreground.parameters())
    del ctx, source, target, actual, initial, old, off, on
    import gc
    gc.collect(); torch.cuda.empty_cache()
    rebuilt = build(cfg); base.restore(rebuilt, loaded)
    assert base.tensor_digest(base.capture(rebuilt, cfg, 5, [])) == base.tensor_digest(saved)
    save(entry/'result.json', dict(schema=SCHEMA, status='passed', config_sha256=sha(entry/'config.json'),
        code=code_identity(), original_cold_identity_exact=True, disabled_all_native_losses_exact=True,
        original_complete_NMS_voting_predictions_exact=True, full_state_and_all_RNG_restore=True,
        parameters=parameters, phases=report, reconstruct_context_restore_exact=True,
        source_observation_sha256=sha(observation_path), ledger_sha256=sha(entry/'probe_events.jsonl'),
        guard=guard, guard_sha256=sha(guard), source_fit_calibration_video_isolation=True,
        target_new_loss=0, scientific_effectiveness_not_claimed=True))


def train(cfg, entry):
    import torch
    from tools import baseline_engine as base
    from tools.proposal_foreground_source import calibrate
    ctx = build(cfg); ctx.proposal_fg_ledger = str(entry/'foreground_events.jsonl')
    save(entry/'runtime_config.json', ctx.cfg); save(entry/'seed_proof.json', ctx.cuda_seed_proof)
    history, rows, best = [], [], float('inf')
    for epoch in range(1, 51):
        base.epoch_setup(ctx, epoch); before = ctx.checked_steps; started = time.monotonic()
        ctx._train_epoch(epoch); torch.cuda.synchronize()
        assert ctx.checked_steps-before == cfg['baseline']['epoch_budget']['iterations_per_epoch']
        metrics = ctx.train_metrics.result()
        row = dict(epoch=epoch, seconds=time.monotonic()-started, train_loss=float(metrics['t_final_loss']),
                   source_cls_loss=float(metrics['t_cls_loss']), source_reg_loss=float(metrics['t_reg_loss']))
        if epoch % 5 == 0:
            ctx.valid_metrics_src.reset()
            value = float(ctx._valid_epoch(epoch, ctx.val_loader_src, domain='src'))
            assert math.isfinite(value)
            history.append([epoch, value]); row['source_validation_loss'] = value
            row['calibration'] = calibrate(ctx, cfg, entry, epoch)
        rows.append(row)
        base.checkpoint(entry/'latest.pt', base.capture(ctx, cfg, epoch, history))
        if epoch % 5 == 0:
            os.link(entry/'latest.pt', entry/('epoch%03d.pt' % epoch))
            if value < best:
                best = value
                temporary = entry/'best.pt.tmp'; os.link(entry/'latest.pt', temporary); temporary.replace(entry/'best.pt')
        save(entry/'metrics.json', rows)
        print('PROPOSAL_FOREGROUND_EPOCH', json.dumps(row), flush=True)
        if epoch % 5 == 0 and decision(history)['reason'] != 'running':
            save(entry/'termination.json', dict(history=history, decision=decision(history),
                                               cold_initial_state_sha=ctx.initial_state_sha))
            break


def inference(cfg, entry, checkpoint):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    from tools.proposal_foreground import attach, bind_forward
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['config'] == cfg
    fix_random_seed(123, include_cuda=True); torch.cuda.manual_seed(123)
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').cuda(); attach(model)
    model.load_state_dict(saved['ema_state_dict'], strict=True)
    model.proposal_fg_state = deepcopy(saved['ema_controls']['proposal_fg_state'])
    bind_forward(model, SimpleNamespace(proposal_fg_enabled=True), {}, set()); model.eval()
    root = entry.parents[1]; outputs = {}
    for domain, subsets in (('source', cfg['src_val_split']), ('target', cfg['tgt_val_split'])):
        manifest = root/'inputs'/cfg['baseline']['shift']/(domain+'_val_unlabeled.json')
        assert all('annotations' not in v for v in read(manifest)['database'].values())
        ds = base.dataset(cfg, manifest, subsets, False, True)
        columns = {k:[] for k in ('video-id', 't-start', 't-end', 'label', 'score')}
        with torch.no_grad():
            for i in range(len(ds)):
                video = base.move_to_device([ds[i]], 'cuda:0')
                result = model(video_list_src=video, video_list_tgt=video, mode='test')[0 if domain == 'source' else 1][0]
                columns['video-id'].extend([result['video_id']] * len(result['scores']))
                for key, value in [('t-start', result['segments'][:, 0]), ('t-end', result['segments'][:, 1]),
                                   ('label', result['labels']), ('score', result['scores'])]:
                    columns[key].extend(value.tolist())
        path = entry/(domain+'_val.npz'); np.savez(path, **{k:np.asarray(v) for k,v in columns.items()})
        outputs[domain] = dict(path=str(path), sha256=sha(path), manifest_sha256=sha(manifest), count=len(columns['score']))
    save(entry/'inference.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                    epoch=saved['epoch'], predictions=outputs, target_GT=False))


def evaluate(cfg, entry):
    import numpy as np
    from libs.evaluation import ANETdetection
    manifest = read(entry/'inference.json')
    for domain, subset in (('source', cfg['src_val_split']), ('target', cfg['tgt_val_split'])):
        record = manifest['predictions'][domain]
        assert sha(record['path']) == record['sha256']
        with np.load(record['path'], allow_pickle=False) as z:
            evaluator = ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'], subset,
                                     num_workers=2, action_subsampling='10_majority_classes')
            ap, mean, *_ = evaluator.evaluate({k:z[k] for k in z.files}, verbose=False)
        save(entry/'evaluation'/(domain+'.json'), dict(map_0_1_to_0_5=float(mean), map_by_tiou=ap.tolist(),
            prediction_sha256=record['sha256'], selected_epoch=manifest['epoch']))


def first_cuda_guard(entry):
    from tools.baseline_gpu import snapshot
    lease_path = Path(os.environ['PROPOSAL_FOREGROUND_LEASE'])
    assert sha(lease_path) == os.environ['PROPOSAL_FOREGROUND_LEASE_SHA256']
    lease = read(lease_path)
    assert lease['owner_pid'] == os.getppid() and lease['commit'] == code_identity()['commit']
    observed = snapshot(lease['index'])
    assert observed['uuid'] == lease['uuid'] == os.environ['CUDA_VISIBLE_DEVICES']
    assert observed['free_mib'] >= lease['minimum_free_mib'] >= 12288
    path = entry/('guard_%d.json' % os.getpid())
    save(path, dict(lease=str(lease_path), lease_sha256=sha(lease_path), owner_pid=os.getppid(),
                   child_pid=os.getpid(), observed=observed))
    return str(path)


def verify(root, gpus):
    """仅本方向六项GPU接线，不自动开正式矩阵，不停止任何其他实验。"""
    import fcntl
    from tools.baseline_gpu import snapshot
    active, pending, completed = {}, list(SHIFTS), []
    lock = (root/'verify_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(signum, frame):
        for task in active.values():
            if task['process'].poll() is None:
                os.killpg(task['process'].pid, signal.SIGTERM)
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        while pending or active:
            for shift, task in list(active.items()):
                if task['process'].poll() is None:
                    continue
                task['log'].close()
                if task['process'].returncode:
                    raise RuntimeError('GPU验收失败: ' + shift)
                report = read(root/'probes'/shift/'result.json')
                assert report['status'] == 'passed'
                completed.append(shift); del active[shift]
            occupied = {x['gpu'] for x in active.values()}
            for gpu in gpus:
                if not pending or gpu in occupied:
                    continue
                observed = snapshot(gpu)
                if observed['free_mib'] < 12288:
                    continue
                shift = pending.pop(0); entry = root/'probes'/shift
                lease = entry/'lease.json'
                save(lease, dict(owner_pid=os.getpid(), index=gpu, uuid=observed['uuid'],
                     minimum_free_mib=12288, commit=code_identity()['commit']))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=observed['uuid'], PYTHONDONTWRITEBYTECODE='1',
                    PYTHONHASHSEED='123', PYTHONPATH=str(CODE), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                    TMPDIR=str(root/'tmp'), PROPOSAL_FOREGROUND_LEASE=str(lease), PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                log = (entry/'probe.log').open('a')
                process = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__)), '--worker', 'probe',
                    '--entry', str(entry)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[shift] = dict(process=process, gpu=gpu, log=log)
            save(root/'state.json', dict(pid=os.getpid(), phase='GPU_verification_only', completed=completed,
                pending=pending, active={s:dict(pid=t['process'].pid, gpu=t['gpu']) for s,t in active.items()}))
            if pending or active:
                time.sleep(2)
        save(root/'acceptance.json', dict(status='passed', count=6, code=code_identity(),
            reports={s:sha(root/'probes'/s/'result.json') for s in SHIFTS}, formal_started=False))
    finally:
        for t in active.values():
            if t['process'].poll() is None:
                os.killpg(t['process'].pid, signal.SIGTERM)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--worker', choices=('probe', 'train', 'inference', 'evaluate'))
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.verify:
        verify(args.verify, [int(x) for x in args.gpus.split(',')])
    elif args.worker:
        entry = args.entry; cfg = read(entry/'config.json')
        os.chdir('/vepfs/vlms/SADA_PLUS')
        if args.worker == 'evaluate':
            evaluate(cfg, entry)
        else:
            guard = first_cuda_guard(entry)
            if args.worker == 'probe': probe(cfg, entry, guard)
            elif args.worker == 'train': train(cfg, entry)
            elif args.worker == 'inference': inference(cfg, entry, args.checkpoint or entry/'best.pt')
            save(entry/(args.worker+'_gpu.json'), dict(guard=guard, guard_sha256=sha(guard), config_sha256=sha(entry/'config.json')))
    else:
        parser.error('选择prepare/verify/worker，默认不启动训练')
