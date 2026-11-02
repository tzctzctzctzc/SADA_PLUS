"""只按已冻结源mAP重选EMA并补推理；不训练、不改原实验工件。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity

SCHEMA = 'sada-phase-source-ap-selection-v1'


def source_choice(rows):
    # target结果不属于此函数的输入；同分取较早轮次。
    assert sorted(r['epoch'] for r in rows) == [5, 10, 15, 20]
    return min(rows, key=lambda r: (-r['source_map'], r['epoch']))


def verify_plan(original):
    plan = read(original / 'monitor/source_ap_selection_plan.json')
    assert not plan['target_used_to_select']
    assert len(plan['choices']) == 6
    assert {c['shift'] for c in plan['choices']} == {'S%d' % i for i in range(1, 7)}
    for c in plan['choices']:
        legacy = original / 'formal' / c['shift'] / 'phase'
        audit = read(original / 'monitor/audits' / (c['shift'] + '_7bc7151.json'))
        assert audit['complete_sha256'] == sha(legacy / 'complete.json')
        rows = [read(p) for p in sorted((original / 'monitor/source_signals' / c['shift']).glob('epoch*/complete.json'))]
        chosen = source_choice(rows)
        assert chosen['epoch'] == c['epoch'] and chosen['source_map'] == c['source_map']
        assert sha(c['complete']) == c['complete_sha256']
        assert read(c['complete'])['checkpoint_sha256'] == c['checkpoint_sha256']
    return plan


def same_predictions(a, b):
    import numpy as np
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        assert set(x.files) == set(y.files)
        assert all(np.array_equal(x[k], y[k]) for k in x.files)


def infer(args):
    from tools.phase_runner_gpu import first_cuda_guard
    guard = first_cuda_guard()
    import torch
    import numpy as np
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    from tools.phase_boundary import PhaseBoundary
    e = args.entry
    selection = read(e / 'selection.json')
    cfg = read(e / 'config.json')
    assert sha(selection['checkpoint']) == selection['checkpoint_sha256']
    saved = torch.load(selection['checkpoint'], map_location='cpu')
    assert saved['config'] == cfg and saved['epoch'] == selection['epoch']
    for p, digest in [(cfg['prototype_capacity']['prepared_cache'], cfg['prototype_capacity']['prepared_cache_sha256'])]:
        assert sha(p) == digest
    os.chdir('/vepfs/vlms/SADA_PLUS')
    fix_random_seed(cfg['seed'], include_cuda=True)
    torch.cuda.manual_seed(cfg['seed'])
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').cuda()
    model.phase_boundary = PhaseBoundary(torch.zeros(10, 4, 512)).cuda()
    model.load_state_dict(saved['ema_state_dict'], strict=True)
    del saved
    model.eval()
    # 沿用原run_phase_boundary的完整forward、原NMS/voting及时间换算。
    for domain in ['source', 'target']:
        manifest = e / (domain + '_val_unlabeled.json')
        assert all('annotations' not in d for d in read(manifest)['database'].values())
        splits = cfg['src_val_split'] if domain == 'source' else cfg['tgt_val_split']
        ds = base.dataset(cfg, manifest, splits, False, True)
        columns = {k: [] for k in ['video-id', 't-start', 't-end', 'label', 'score']}
        with torch.no_grad():
            for i in range(len(ds)):
                video = base.move_to_device([ds[i]], 'cuda:0')
                assert not video[0]['segments'].numel() and not video[0]['labels'].numel()
                prediction = model(video_list_src=video, video_list_tgt=video, mode='test', domain=domain,
                                   curr_epoch=selection['epoch'])[0 if domain == 'source' else 1][0]
                columns['video-id'].extend([prediction['video_id']] * len(prediction['scores']))
                for k, z in [('t-start', prediction['segments'][:, 0]), ('t-end', prediction['segments'][:, 1]),
                             ('label', prediction['labels']), ('score', prediction['scores'])]:
                    columns[k].extend(z.cpu().tolist())
                print('SOURCE_AP_INFERENCE', selection['shift'], domain, i + 1, len(ds), flush=True)
        np.savez(e / (domain + '_val.npz'), **{k: np.asarray(v) for k, v in columns.items()})
    same_predictions(e / 'source_val.npz', Path(selection['complete']).parent / 'source_val.npz')
    save(e / 'gpu_complete.json', dict(guard=guard, guard_sha256=sha(guard), code=code_identity(),
         checkpoint_sha256=selection['checkpoint_sha256'], source_prediction_identity='all_arrays_exact',
         target_gt_used=False, training=False))


def evaluate(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    from libs.evaluation import ANETdetection
    e = args.entry
    cfg = read(e / 'config.json')
    selection = read(e / 'selection.json')
    annotation = cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation) == cfg['baseline']['annotation_sha256']
    results = {}
    for domain in ['source', 'target']:
        path = e / (domain + '_val.npz')
        with np.load(path, allow_pickle=False) as f:
            prediction = {k: f[k] for k in f.files}
        splits = cfg['src_val_split'] if domain == 'source' else cfg['tgt_val_split']
        evaluator = ANETdetection(annotation, splits, num_workers=2, action_subsampling='10_majority_classes')
        ap, mean, *_ = evaluator.evaluate(prediction, verbose=False)
        results[domain] = dict(map_0_1_to_0_5=float(mean), map_by_tiou=[float(x) for x in ap],
                               prediction_sha256=sha(path), prediction_count=len(prediction['score']))
    assert abs(results['source']['map_0_1_to_0_5'] - selection['source_map']) < 1e-7
    save(e / 'evaluation.json', dict(results=results, annotation_sha256=sha(annotation),
         selected_epoch=selection['epoch'], target_gt_scope='CPU offline evaluation only; never selection'))


def finish(e, original):
    s = read(e / 'selection.json')
    r = read(e / 'evaluation.json')
    assert abs(r['results']['source']['map_0_1_to_0_5'] - s['source_map']) < 1e-7
    for domain in ['source', 'target']:
        assert r['results'][domain]['prediction_sha256'] == sha(e / (domain + '_val.npz'))
    if (e / 'gpu_complete.json').exists():
        receipt = read(e / 'gpu_complete.json')
        assert sha(receipt['guard']) == receipt['guard_sha256']
        guard = read(receipt['guard'])
        assert sha(guard['lease']) == guard['lease_sha256']
        assert guard['owner_pid'] == read(guard['lease'])['owner_pid']
        assert guard['observed']['free_mib'] >= 12288
    reference = read(original / 'formal' / s['shift'] / 'phase/config.json')['phase_boundary_runtime']['reference']
    reference_map = read(Path(reference) / 'evaluation/target_val.json')['map_0.1_to_0.5']
    result = dict(schema=SCHEMA, shift=s['shift'], selected_epoch=s['epoch'], source_map=s['source_map'],
                  target_map=r['results']['target']['map_0_1_to_0_5'], reference_target_map=reference_map,
                  delta_pp=100 * (r['results']['target']['map_0_1_to_0_5'] - reference_map),
                  training=False, selection='source_AP_only', target_used_to_select=False,
                  artifacts={str(p): sha(p) for p in e.iterdir() if p.is_file() and p.suffix in ['.json', '.npz'] and p.name != 'complete.json'})
    save(e / 'complete.json', result)
    return result


def run(args):
    from tools.phase_runner_gpu import acquire
    identity = code_identity()
    original = args.original
    plan = verify_plan(original)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'plan.json', dict(schema=SCHEMA, source_plan=plan, source_plan_sha256=sha(original / 'monitor/source_ap_selection_plan.json'),
                                 original=str(original), code=identity, training=False))
    queue = []
    results = []
    for s in plan['choices']:
        e = root / s['shift']
        e.mkdir()
        old = original / 'formal' / s['shift'] / 'phase'
        save(e / 'selection.json', s)
        for name in ['config.json', 'source_val_unlabeled.json', 'target_val_unlabeled.json']:
            shutil.copyfile(old / name, e / name)
        if s['same_checkpoint_as_legacy']:
            import torch
            from tools.baseline_engine import tensor_digest
            assert sha(s['checkpoint']) == s['checkpoint_sha256']
            selected = torch.load(s['checkpoint'], map_location='cpu')
            legacy = torch.load(old / 'best.pt', map_location='cpu')
            assert selected['epoch'] == legacy['epoch'] == s['epoch']
            assert tensor_digest(selected['ema_state_dict']) == tensor_digest(legacy['ema_state_dict'])
            del selected, legacy
            same_predictions(old / 'source_val.npz', Path(s['complete']).parent / 'source_val.npz')
            for domain in ['source', 'target']:
                os.link(old / (domain + '_val.npz'), e / (domain + '_val.npz'))
            legacy_eval = {d: read(old / 'evaluation' / (d + '_val.json')) for d in ['source', 'target']}
            save(e / 'evaluation.json', dict(results={d: dict(map_0_1_to_0_5=x['map_0.1_to_0.5'],
                 prediction_sha256=x['prediction_sha256'], map_by_tiou=x['map_by_tiou']) for d, x in legacy_eval.items()},
                 reuse_complete=str(old / 'complete.json'), reuse_complete_sha256=sha(old / 'complete.json'), ema_identity='tensor_digest_exact'))
            results.append(finish(e, original))
        else:
            queue.append(s['shift'])
    active = {}
    cpu = []
    def state():
        save(root / 'state.json', dict(pid=os.getpid(), active={k: dict(pid=v['p'].pid, stage=v['stage'], gpu=v.get('gpu')) for k, v in active.items()},
             pending=queue, pending_cpu=cpu, completed=results, phase='complete' if len(results) == 6 else 'inference_and_CPU_evaluation'))
    def spawn(s, stage, card=None):
        e = root / s
        env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='')
        if card:
            env.update(CUDA_VISIBLE_DEVICES=card['uuid'], SADA_BASELINE_LEASE=card['lease'], SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        with (e / (stage + '.log')).open('x') as log:
            p = subprocess.Popen([sys.executable, '-u', '-B', __file__, '--stage', stage, '--entry', str(e)],
                                 cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(p=p, stage=stage, gpu=card['index'] if card else None)
    try:
        while queue or active or cpu:
            for s, t in list(active.items()):
                rc = t['p'].poll()
                if rc is None:
                    continue
                if t.get('ctx'):
                    t['ctx'].__exit__(None, None, None)
                del active[s]
                if rc:
                    save(root / 'error.json', dict(shift=s, stage=t['stage'], exit_code=rc))
                    raise RuntimeError('source-AP evaluation failed: ' + s + '/' + t['stage'])
                if t['stage'] == 'infer':
                    cpu.append(s)
                else:
                    results.append(finish(root / s, original))
            while cpu and sum(t['stage'] == 'evaluate' for t in active.values()) < 2:
                s = cpu.pop(0)
                active[s] = spawn(s, 'evaluate')
            for s in list(queue):
                found = acquire([g for g in range(4) if g not in {t.get('gpu') for t in active.values()}], Path('/vepfs/vlms/SADA_PLUS'), root, 12288, identity['commit'])
                if not found:
                    break
                ctx, card = found
                active[s] = dict(spawn(s, 'infer', card), ctx=ctx)
                queue.remove(s)
            state()
            time.sleep(2)
    finally:
        for t in active.values():
            if t['p'].poll() is None:
                os.killpg(t['p'].pid, signal.SIGTERM)
        for t in active.values():
            try:
                t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(t['p'].pid, signal.SIGKILL)
                t['p'].wait()
    assert len(results) == 6 and len({r['shift'] for r in results}) == 6
    save(root / 'final.json', dict(schema=SCHEMA, results=sorted(results, key=lambda r: r['shift']),
         mean_map=sum(r['target_map'] for r in results) / 6, mean_delta_pp=sum(r['delta_pp'] for r in results) / 6,
         selection='source_AP_only', training=False, target_used_to_select=False))
    state()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['run', 'infer', 'evaluate'], default='run')
    p.add_argument('--entry', type=Path)
    p.add_argument('--original', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    {'run': run, 'infer': infer, 'evaluate': evaluate}[a.stage](a)
