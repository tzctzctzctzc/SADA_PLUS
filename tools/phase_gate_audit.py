"""源验证集上的真实weak/strong门控重放；GPU无GT，CPU核验伪端点误差。"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity

SCHEMA = 'sada-phase-source-exact-gate-replay-v1'


def frozen_crop(video_id, length, maximum):
    width = min(length, maximum)
    value = int(hashlib.sha256(('phase-gate-source-v1/' + video_id).encode()).hexdigest()[:16], 16)
    return value % (length - width + 1), width


def traced_target_loss(*args):
    """只读观察原函数局部变量，保留原条件、损失归约及边界搜索的执行。"""
    from tools.phase_boundary import target_loss
    lines, start = inspect.getsourcelines(target_loss)
    def location(fragment):
        return next(start + i for i, line in enumerate(lines) if fragment in line)
    points = {key: location(text) for key, text in dict(
        begin="stats['target_proposals']+=1", crop='mapped=map_coordinates(box,view)',
        point='tt=interval_tokens', phase='if int(tclass.argmax', agreed='post=phase_posterior',
        refined='st=interval_tokens', mask='if mask.any():').items()}
    records = []
    current = None
    def trace(frame, event, arg):
        nonlocal current
        if frame.f_code is not target_loss.__code__:
            return None
        if event == 'line':
            d = frame.f_locals
            line = frame.f_lineno
            if line == points['begin']:
                if current is not None:
                    records.append(current)
                current = dict(video_index=d['vi'], box=d['box'].detach().cpu().tolist(), label=int(d['c']),
                               level=d['l'], index=d['j'], confidence=float(d['confidence']),
                               stride=float(d['model'].fpn_strides[d['l']]),
                               view=dict(d['view']), status='crop_rejected')
            elif current is not None:
                if line == points['crop']:
                    current['status'] = 'student_point_rejected'
                elif line == points['point']:
                    current['status'] = 'phase_class_rejected'
                elif line == points['phase']:
                    current['phase_label'] = int(d['tclass'].argmax(-1))
                elif line == points['agreed']:
                    current['status'] = 'refined_interval_invalid'
                elif line == points['refined']:
                    current['status'] = 'semantic_accepted'
                elif line == points['mask']:
                    from tools.phase_boundary import map_coordinates
                    current.update(mask=d['mask'].detach().cpu().tolist(), refined=d['refined'].detach().cpu().tolist(),
                                   student=map_coordinates(d['pred'].detach(), d['view'], inverse=True).cpu().tolist(),
                                   teacher_scale=d['tunc'].detach().cpu().tolist(), student_scale=d['sunc'].detach().cpu().tolist(),
                                   edge=[float(v) for v in d['edgeconf']], student_index=d['k'])
        elif event == 'return' and current is not None:
            records.append(current)
            current = None
        return trace
    old = sys.gettrace()
    try:
        sys.settrace(trace)
        result = target_loss(*args)
    finally:
        sys.settrace(old)
    _, stats = result
    assert len(records) == stats['target_proposals']
    assert sum('mask' in r for r in records) == stats['target_phase_used']
    for side, key in [(0, 'target_left_used'), (1, 'target_right_used')]:
        assert sum(int(r['mask'][side]) for r in records if 'mask' in r) == stats[key]
    return result, records


def gpu(args):
    from tools.phase_runner_gpu import first_cuda_guard
    guard = first_cuda_guard()
    import torch
    from libs.models.SADA import make_meta_arch
    from tools import baseline_engine as base
    from tools.phase_boundary import PhaseBoundary, strong_view, direct, target_loss
    e = args.entry
    job = read(e / 'job.json')
    assert sha(job['checkpoint']) == job['checkpoint_sha256']
    saved = torch.load(job['checkpoint'], map_location='cpu')
    cfg = saved['config']
    assert saved['epoch'] == job['epoch'] and cfg == read(e / 'config.json')
    manifest = e / 'source_val_unlabeled.json'
    assert all('annotations' not in d for d in read(manifest)['database'].values())
    os.chdir('/vepfs/vlms/SADA_PLUS')
    def model(weights):
        m = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').cuda()
        m.phase_boundary = PhaseBoundary(torch.zeros(10, 4, 512)).cuda()
        m.load_state_dict(weights, strict=True)
        return m
    teacher = model(saved['ema_state_dict']).eval()
    student = model(saved['state_dict']).train()
    del saved
    before = [base.tensor_digest(m.state_dict()) for m in [teacher, student]]
    ds = base.dataset(cfg, manifest, cfg['src_val_split'], False, True)
    maximum = cfg['dataset']['max_seq_len']
    rows = []
    crop_proof = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for begin in range(0, len(ds), 2):
            videos = []
            for idx in range(begin, min(begin + 2, len(ds))):
                v = ds[idx]
                assert not v['segments'].numel() and not v['labels'].numel()
                lo, width = frozen_crop(v['metadata']['video_id'], v['feats'].shape[-1], maximum)
                crop_proof.append(dict(video_id=v['metadata']['video_id'], lo=lo, width=width, metadata=v['metadata']))
                v['feats'] = v['feats'][:, lo:lo + width]
                videos.append(v)
            videos = base.move_to_device(videos, 'cuda:0')
            # 各epoch采用同一batch、step及随机流，检验模型变化而非抽样变化。
            step = begin // 2 + 1
            torch.manual_seed(123000 + step)
            torch.cuda.manual_seed(123000 + step)
            strong, views = [], []
            for index, v in enumerate(videos):
                x, view = strong_view(v, step, index)
                view['original_length'] = v['feats'].shape[-1]
                strong.append(x)
                views.append(view)
            weakdata = direct(teacher, videos)
            strongdata = direct(student, strong)
            (loss, stats), observed = traced_target_loss(student, teacher, student.phase_boundary, weakdata, strongdata, views, step)
            assert torch.isfinite(loss)
            for r in observed:
                proof = crop_proof[begin + r.pop('video_index')]
                r.update(video_id=proof['video_id'], crop_lo=proof['lo'], crop_length=proof['width'])
                rows.append(r)
            print('EXACT_GATE', job['shift'], job['epoch'], begin, len(ds), stats, flush=True)
    assert before == [base.tensor_digest(m.state_dict()) for m in [teacher, student]]
    save(e / 'observations.json', rows)
    save(e / 'crops.json', crop_proof)
    save(e / 'gpu_complete.json', dict(code=code_identity(), checkpoint_sha256=job['checkpoint_sha256'],
         observations_sha256=sha(e / 'observations.json'), crops_sha256=sha(e / 'crops.json'),
         original_target_loss_sha256=hashlib.sha256(inspect.getsource(target_loss).encode()).hexdigest(),
         guard=guard, guard_sha256=sha(guard), model_parameters_and_buffers_unchanged=True,
         target_gt_used=False, optimizer_steps=0, source_only=True, videos=len(ds),
         peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
         scope='same original target_loss and gates on fixed GT-free source-val crops; student train-mode, teacher eval; not replay of original target samples or training RNG'))


def iou(a, b):
    inter = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, a[1] - a[0] + b[1] - b[0] - inter)


def project_truth(annotations, metadata, crop_lo):
    out = []
    for g in annotations:
        if int(g['label_id']) >= 10:
            continue
        seconds = g['segment']
        if metadata['use_temporal_grid_feats']:
            segment = [(t * metadata['fps'] - .5 * metadata['feat_num_frames']) / metadata['feat_stride'] - crop_lo for t in seconds]
        else:
            segment = [t - crop_lo for t in seconds]
        out.append(dict(label=int(g['label_id']), segment=segment, duration_seconds=seconds[1] - seconds[0]))
    return out


def endpoint_rows(record, truth, stride):
    anchor = record['index'] * stride
    same = [g for g in truth if g['label'] == record['label'] and g['segment'][0] <= anchor <= g['segment'][1]]
    if not same or 'mask' not in record:
        return []
    g = min(same, key=lambda g: g['segment'][1] - g['segment'][0])
    duration = max(1e-9, g['segment'][1] - g['segment'][0])
    from tools.phase_boundary import map_coordinates
    result = []
    for side in range(2):
        t = g['segment'][side]
        mapped = map_coordinates(t, record['view'])
        raw = abs(record['box'][side] - t) / duration
        refined = abs(record['refined'][side] - t) / duration
        student = abs(record['student'][side] - t) / duration
        result.append(dict(video_id=record['video_id'], side=side, accepted=bool(record['mask'][side]),
             raw_teacher_error=raw, refined_teacher_error=refined, student_error=student,
             teacher_better=refined < student - 1e-8, teacher_worse=refined > student + 1e-8,
             refinement_helped=refined < raw - 1e-8, refinement_hurt=refined > raw + 1e-8,
             error_reduction=student - refined, weak_visible=0 <= t < record['crop_length'],
             strong_visible=0 <= mapped < record['view']['length'], teacher_scale=record['teacher_scale'][side],
             duration_group='<1s' if g['duration_seconds'] < 1 else '1-3s' if g['duration_seconds'] < 3 else '>=3s'))
    return result


def summarize(rows):
    import numpy as np
    if not rows:
        return dict(endpoints=0, videos=0)
    byvideo = {}
    for row in rows:
        byvideo.setdefault(row['video_id'], []).append(row)
    keys = ['teacher_better', 'teacher_worse', 'refinement_helped', 'refinement_hurt', 'error_reduction',
            'raw_teacher_error', 'refined_teacher_error', 'student_error']
    return dict(endpoints=len(rows), videos=len(byvideo),
                per_endpoint_mean={k: float(np.mean([r[k] for r in rows])) for k in keys},
                equal_video_mean={k: float(np.mean([np.mean([r[k] for r in v]) for v in byvideo.values()])) for k in keys},
                scale_error_correlation={k: float(np.corrcoef([r['teacher_scale'] for r in rows], [r[k] for r in rows])[0, 1])
                    if len(rows) > 1 and np.std([r['teacher_scale'] for r in rows]) > 0 and np.std([r[k] for r in rows]) > 0 else None
                    for k in ['raw_teacher_error', 'refined_teacher_error']})


def cpu(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    e = args.entry
    cfg = read(e / 'config.json')
    receipt = read(e / 'gpu_complete.json')
    assert sha(e / 'observations.json') == receipt['observations_sha256']
    assert sha(e / 'crops.json') == receipt['crops_sha256']
    annotation = cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation) == cfg['baseline']['annotation_sha256']
    db = read(annotation)['database']
    crops = {r['video_id']: r for r in read(e / 'crops.json')}
    assert set(crops) == set(read(e / 'source_val_unlabeled.json')['database'])
    assert all(db[v]['subset'] in cfg['src_val_split'] for v in crops)
    rows = read(e / 'observations.json')
    endpoint = []
    stages = {}
    class_overlap = {}
    for r in rows:
        proof = crops[r['video_id']]
        truth = project_truth(db[r['video_id']].get('annotations', []), proof['metadata'], proof['lo'])
        stages[r['status']] = stages.get(r['status'], 0) + 1
        correct = any(g['label'] == r['label'] and iou(g['segment'], r['box']) >= .1 for g in truth)
        group = class_overlap.setdefault(r['status'], dict(count=0, same_class_overlap_ge_point1=0))
        group['count'] += 1
        group['same_class_overlap_ge_point1'] += int(correct)
        endpoint += endpoint_rows(r, truth, r['stride'])
    groups = {'all_semantic_accepted_matched': endpoint,
              'gate_accepted': [r for r in endpoint if r['accepted']],
              'gate_rejected': [r for r in endpoint if not r['accepted']],
              'accepted_visible_in_both_views': [r for r in endpoint if r['accepted'] and r['weak_visible'] and r['strong_visible']],
              'accepted_GT_endpoint_outside_a_view': [r for r in endpoint if r['accepted'] and not (r['weak_visible'] and r['strong_visible'])]}
    for side in range(2):
        groups['accepted_side%d' % side] = [r for r in endpoint if r['accepted'] and r['side'] == side]
    for duration in ['<1s', '1-3s', '>=3s']:
        groups['accepted_duration_' + duration] = [r for r in endpoint if r['accepted'] and r['duration_group'] == duration]
    save(e / 'endpoint_errors.json', endpoint)
    save(e / 'quality.json', dict(schema=SCHEMA, shift=read(e / 'job.json')['shift'], epoch=read(e / 'job.json')['epoch'],
         stages=stages, class_overlap=class_overlap, groups={k: summarize(v) for k, v in groups.items()},
         gt_scope='source-val CPU diagnostic only', target_gt_used=False,
         limitation='same training function/gates but fixed source-val crops and random stream; not target accuracy or actual training sample replay; interval GT does not validate latent phase semantics'))


def run(args):
    from tools.phase_runner_gpu import acquire
    original, root = args.original, args.output
    root.mkdir(parents=True, exist_ok=False)
    identity = code_identity()
    choices = read(original / 'monitor/source_ap_selection_plan.json')['choices']
    jobs = []
    for c in choices:
        for epoch in sorted({5, 20, c['epoch']}):
            snapshot = original / 'monitor/source_signals' / c['shift'] / ('epoch%03d' % epoch)
            report = read(snapshot / 'complete.json')
            name = c['shift'] + '_e%03d' % epoch
            e = root / name
            e.mkdir()
            job = dict(shift=c['shift'], epoch=epoch, checkpoint=str(snapshot / 'checkpoint.pt'), checkpoint_sha256=report['checkpoint_sha256'],
                       source_complete=str(snapshot / 'complete.json'), source_complete_sha256=sha(snapshot / 'complete.json'))
            save(e / 'job.json', job)
            for f in ['config.json', 'source_val_unlabeled.json']:
                shutil.copyfile(snapshot / f, e / f)
            jobs.append(name)
    save(root / 'plan.json', dict(schema=SCHEMA, code=identity, jobs=jobs, checkpoint_rule='all six epochs5,20 plus source-AP selected if different',
         crop_rule='one GT-free stable video-hash contiguous crop up to original max_seq_len; same crop/batch/step/RNG across epochs',
         head_and_detector_training=False, optimizer_steps=0, target_data_used=False, original=str(original)))
    active, pending, cpuqueue, completed = {}, list(jobs), [], []
    def state():
        save(root / 'state.json', dict(pid=os.getpid(), pending=pending, pending_cpu=cpuqueue, completed=completed,
             active={k: dict(pid=v['p'].pid, stage=v['stage'], gpu=v.get('gpu')) for k, v in active.items()},
             phase='complete' if len(completed) == len(jobs) else 'source_gate_diagnostic'))
    def spawn(name, stage, card=None):
        e = root / name
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        if card:
            env.update(CUDA_VISIBLE_DEVICES=card['uuid'], SADA_BASELINE_LEASE=card['lease'], SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        with (e / (stage + '.log')).open('x') as log:
            p = subprocess.Popen([sys.executable, '-u', '-B', __file__, '--stage', stage, '--entry', str(e)], cwd=CODE, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(p=p, stage=stage, gpu=card['index'] if card else None)
    try:
        while pending or cpuqueue or active:
            for name, task in list(active.items()):
                rc = task['p'].poll()
                if rc is None:
                    continue
                if task.get('ctx'):
                    task['ctx'].__exit__(None, None, None)
                del active[name]
                if rc:
                    save(root / 'error.json', dict(job=name, stage=task['stage'], exit_code=rc))
                    raise RuntimeError('gate audit failed ' + name)
                if task['stage'] == 'gpu':
                    cpuqueue.append(name)
                else:
                    e = root / name
                    save(e / 'complete.json', dict(schema=SCHEMA, artifacts={str(p): sha(p) for p in e.iterdir() if p.suffix == '.json' and p.name != 'complete.json'}))
                    completed.append(name)
            while cpuqueue and sum(t['stage'] == 'cpu' for t in active.values()) < 2:
                name = cpuqueue.pop(0)
                active[name] = spawn(name, 'cpu')
            for name in list(pending):
                found = acquire([g for g in range(4) if g not in {t.get('gpu') for t in active.values()}], Path('/vepfs/vlms/SADA_PLUS'), root, 12288, identity['commit'])
                if not found:
                    break
                ctx, card = found
                active[name] = dict(spawn(name, 'gpu', card), ctx=ctx)
                pending.remove(name)
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
    save(root / 'final.json', dict(schema=SCHEMA, count=len(jobs), results={name: read(root / name / 'quality.json') for name in jobs},
         training=False, target_data_used=False))
    state()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['run', 'gpu', 'cpu'], default='run')
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    {'run': run, 'gpu': gpu, 'cpu': cpu}[args.stage](args)
