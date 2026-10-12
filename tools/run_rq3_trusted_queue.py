"""固定20%参与：逐shift准备、源域诊断、验收和单卡fresh实验。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_gpu import gpu_slot
from tools.run_rq3_matrix import CODE_ROOT, baseline_root, completed_baseline, load_entry_plan
from tools.score_rq3_vlm_candidates import check_code


DATA = Path('/vepfs/vlms/SADA_PLUS')
OLD_CANDIDATES = Path('/vepfs/vlms/rq3_shared_candidates_1b6e669_0928')
OLD_CACHES = Path('/vepfs/vlms/rq3_semantic_caches_434f630_0928')
RUNTIME = Path('/vepfs/vlms/cache/rq3_runtime_0928')
VIDEO = Path('/vepfs/vlms/data/EPIC_KITCHENS/videos')
RECOVERY = Path('/vepfs/vlms/data/EPIC_KITCHENS/rq2_frame_recovery_qwen_0924')
TEACHERS = {
    'egovideo_hod': ('/vepfs/vlms/envs/rq1-cu121/bin/python', 'ego_settings.json',
                     '/vepfs/vlms/rq2_features_staging_3b7/egovideo_hod/protocol.json'),
    'internvideo2_clip_1b': ('/vepfs/vlms/envs/sada_py39/bin/python', 'intern_settings.json',
                            '/vepfs/vlms/rq2_features_staging_3b7/internvideo2_clip_1b/protocol.json'),
    'qwen3_vl_8b': ('/vepfs/vlms/envs/rq1-cu121/bin/python', 'qwen_settings.json',
                     '/vepfs/vlms/rq2_qwen_staging_5a8041d_0925/qwen3_vl_8b/protocol.json'),
}
from tools.rq3_extra_teachers import EXTRA
TEACHERS.update({model: ('/vepfs/vlms/envs/rq1-cu121/bin/python', model + '_settings.json',
                         str(RUNTIME / (model + '_anchor.json'))) for model in EXTRA})


def ordered_rows(plan):
    """固定按shift、教师、策略排序，不按mAP筛选。"""
    return sorted(plan['entries'], key=lambda r: (int(r['shift'][1:]), r['teacher'],
                                                 ('hard', 'verify', 'prototype').index(r['strategy'])))


def entry_output_root(queue_root, stage, name):
    # matrix以输出根末级目录命名checkpoint；必须包含本次队列身份，避免与旧实验重名。
    return queue_root / (queue_root.name + '_' + stage + '_' + name)


def prepare_cache_overlay(destination, original, rows):
    """保留同候选跨教师审计；本次教师/shift的槽位只允许新打分填入。"""
    refreshed = {(row['teacher'], row['shift']) for row in rows}
    destination.mkdir(parents=True, exist_ok=False)
    for source in original.glob('*/*.json'):
        if (source.parent.name, source.stem) not in refreshed:
            target = destination / source.parent.name / source.name
            target.parent.mkdir(exist_ok=True)
            target.symlink_to(source.resolve())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch-size', type=int, choices=(2, 4, 8), default=2)
    parser.add_argument('--early-stop-patience', type=int, choices=(0, 15), default=0)
    parser.add_argument('--benchmark-batches', action='store_true', help='先实测batch4/8再固定吞吐更高者')
    parser.add_argument('--pair-students', action='store_true',
                        help='每两个shift准备后，同策略Hard/Prototype两学生共享一卡；Verify和评估独占')
    parser.add_argument('--pipeline-cpu', action='store_true',
                        help='只提前一个shift的目标帧CPU准备；不提前GPU查询或改变候选')
    parser.add_argument('--queue-slot', type=int, choices=(0, 1),
                        help='由两卡监控分配的互斥工作队列槽位')
    parser.add_argument('--prepared-root', type=Path, help='重验并复用完整候选/帧和教师缓存；不复用部分产物或训练')
    parser.add_argument('--refresh-teacher-caches', action='store_true',
                        help='仅复用精确候选帧；源域和目标域教师均用当前代码重新打分')
    parser.add_argument('--recovery-root', type=Path, default=RECOVERY)
    parser.add_argument('--recovery-manifest-sha256', default='5a8041d6355e1e5b9d9f1e268f903c58dd604efa906d780f69268d8996483b90')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('physical GPU index must be nonnegative')
    if args.pair_students and (args.batch_size != 2 or args.benchmark_batches):
        parser.error('student colocation only supports fixed batch2, without batch benchmark')
    if args.queue_slot is not None and os.environ.get('RQ3_QUEUE_SLOT') != str(args.queue_slot):
        parser.error('managed queue-slot must be supplied by the RQ3 monitor')
    check_code(args.expected_head)
    if sha256_file(args.recovery_root / 'manifest.json') != args.recovery_manifest_sha256:
        raise ValueError('exact frame recovery manifest hash differs')
    args.entry_plan = args.entry_plan.resolve()
    args.output_root = args.output_root.resolve()
    if args.prepared_root:
        args.prepared_root = args.prepared_root.resolve()
    load_entry_plan(args.entry_plan)
    plan = json.loads(args.entry_plan.read_text())
    fresh_matrix = plan['schema'] in ('rq3-revised-fresh-matrix-v1', 'rq3-expanded-fresh-matrix-v1')
    rows = ordered_rows(plan)
    if any(row['teacher'] not in TEACHERS for row in rows):
        raise ValueError('unsupported official RQ3 teacher')
    if any(row['teacher'] in EXTRA for row in rows) and not args.refresh_teacher_caches:
        raise ValueError('new teachers require fresh paired caches')
    import fcntl
    lock_name = ('.rq3_idle_monitor.lock' if args.queue_slot is None
                 else '.rq3_trusted_queue_slot%d.lock' % args.queue_slot)
    with (DATA / lock_name).open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output_root.mkdir(parents=True, exist_ok=False)
        logs = args.output_root / 'logs'
        logs.mkdir()
        cache_root = (prepare_cache_overlay(args.output_root / 'caches/target_proposal', OLD_CACHES, rows)
                      if args.refresh_teacher_caches else OLD_CACHES)
        (args.output_root / 'entry_plan.json').write_bytes(args.entry_plan.read_bytes())
        children = []
        stopped = False
        state = dict(pid=os.getpid(), gpu=args.gpu, max_gpus=1, expected_head=args.expected_head,
                     plan_sha256=sha256_file(args.entry_plan), total=len(rows), completed=0,
                     participation_fraction=.2, batch_size=args.batch_size,
                     confidence_policy='fixed-top20-percent-per-shift-v1', status='running')
        state.update(recovery_root=str(args.recovery_root), recovery_manifest_sha256=args.recovery_manifest_sha256)
        benchmark_pending = args.benchmark_batches
        state['queue_slot'] = args.queue_slot
        state['refresh_teacher_caches'] = args.refresh_teacher_caches
        state['early_stop_patience_epochs'] = args.early_stop_patience
        state['strategy_participation'] = dict(hard='all-common-candidates', verify=.2, prototype=.2)
        state['pair_students'] = args.pair_students
        pending_students = []
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        target_prefetch = {}

        def save(**values):
            state.update(values)
            temporary = args.output_root / 'state.tmp'
            temporary.write_text(json.dumps(state, sort_keys=True))
            temporary.replace(args.output_root / 'state.json')

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            for child in list(children):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        def run(command, name, python=None, selected=None, teacher=None):
            if stopped:
                raise InterruptedError('RQ3 queue stopped by user')
            check_code(args.expected_head)
            log = logs / (name + '.' + str(time.time_ns()) + '.log')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(CODE_ROOT))
            if teacher == 'internvideo2_clip_1b':
                env['PYTHONPATH'] = '/vepfs/vlms/cache/rq3_intern_legacy_deps_0928:' + str(CODE_ROOT)
            elif teacher == 'moss_vl_instruct_0708':
                # TorchCodec/系统FFmpeg需要较新ABI；不修改共享Conda环境。
                env['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1'
            elif teacher is None:
                env['PYTHONPATH'] += ':/vepfs/vlms/cache/rq3_preparation_deps_0928'
            if selected is not None:
                env.update(CUDA_VISIBLE_DEVICES=selected['uuid'], RQ3_GPU_INDEX=str(selected['index']),
                           RQ3_GPU_UUID=selected['uuid'])
            print('RQ3_TRUSTED_PHASE', name, str(log), flush=True)
            with log.open('x') as stream:
                # -m的sys.path首项来自cwd；必须优先固定代码，不能导入DATA/tools中的旧模块。
                child = subprocess.Popen([python or sys.executable, '-B', *command],
                                         cwd=CODE_ROOT if command[0] == '-m' else DATA, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                code = child.wait()
                children.remove(child)
            if code:
                raise RuntimeError('phase failed (%d): %s; see %s' % (code, name, log))

        def frames(shift, domain, scope, proposals):
            destination = args.output_root / (domain + '_' + scope) / shift
            if args.prepared_root:
                previous = args.prepared_root / (domain + '_' + scope) / shift
                if (previous / 'manifest.json').is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(previous, target_is_directory=True)
            run([str(CODE_ROOT / 'tools/prepare_rq3_candidates.py'), '--proposals', str(proposals),
                 '--baseline-root', str(baseline_root(Path('/vepfs/vlms'), shift)), '--shift', shift,
                 '--video-root', str(VIDEO), '--recovery-root', str(args.recovery_root),
                 '--recovery-manifest-sha256', args.recovery_manifest_sha256, '--output-root', str(destination),
                 '--domain', domain, '--evidence-scope', scope, '--decode-workers', '4', '--reuse-verified'],
                shift + '.' + domain + '.' + scope + '.frames')
            return destination / 'manifest.json'

        def finish_student(job):
            if not job['record'].is_file():
                raise ValueError('student runner exited without complete.json')
            save(completed=state['completed'] + 1, latest_complete=str(job['record']))
            print('RQ3_TRUSTED_ENTRY_COMPLETE', job['identity'], state['completed'], flush=True)

        def submit_student(command, name, record, strategy):
            job = dict(command=command, identity=name, record=record, strategy=strategy)
            if args.pair_students:
                pending_students.append(job)
                save(queued_students=len(pending_students))
            else:
                run(command, name + '.formal')
                finish_student(job)

        def paired_students(jobs):
            from tools.rq3_colocation import write_json, process_start, compute_pids, owners
            cohort = args.output_root / ('cohort_' + str(time.time_ns()))
            cohort.mkdir()
            group = []
            with gpu_slot(str(args.gpu), DATA, wait=True) as selected:
                try:
                    lease = dict(owner_pid=os.getpid(), owner_start=process_start(os.getpid()),
                                 selected=selected, members={}, entries=[j['identity'] for j in jobs])
                    for number, job in enumerate(jobs):
                        command = job['command']
                        if command[:2] != ['-m', 'tools.run_rq3_matrix']:
                            raise ValueError('colocation accepts only audited matrix commands')
                        env = dict(os.environ, PYTHONPATH=str(CODE_ROOT) + ':/vepfs/vlms/cache/rq3_preparation_deps_0928', PYTHONDONTWRITEBYTECODE='1',
                                   PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='',
                                   RQ3_COHORT_LEASE=str(cohort/'lease.json'), RQ3_COHORT_MEMBER=str(number))
                        log = logs / (job['identity'] + '.paired.' + str(time.time_ns()) + '.log')
                        with log.open('x') as stream:
                            child = subprocess.Popen([sys.executable, '-B', '-m', 'tools.rq3_colocation',
                                *command[2:]], env=env, cwd=CODE_ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
                        children.append(child)
                        group.append(child)
                        lease['members'][str(number)] = child.pid
                    write_json(cohort/'lease.json', lease)
                    print('RQ3_STUDENT_PAIR_STARTED', json.dumps(lease), flush=True)
                    while any(child.poll() is None for child in group):
                        if stopped:
                            raise InterruptedError('student pair stopped by user')
                        if any(child.poll() not in (None, 0) for child in group):
                            raise RuntimeError('student pair failed; partial outputs retained')
                        # admission期间不观察半登记状态；其余时间拒绝任何未知CUDA进程。
                        with (cohort/'admission.lock').open('a') as admission:
                            try:
                                fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                pass
                            else:
                                active = compute_pids(selected['uuid'])
                                if active != owners(cohort, active):
                                    raise RuntimeError('external CUDA process entered student cohort')
                        time.sleep(1)
                    if any(child.returncode != 0 for child in group):
                        raise RuntimeError('student pair exited unsuccessfully')
                    for job in jobs:
                        finish_student(job)
                    write_json(cohort/'cohort_finished.json', dict(diagnostic=False, entries=lease['entries']))
                finally:
                    for child in group:
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                    for child in group:
                        child.wait()
                        children.remove(child)

        def flush_students():
            from tools.rq3_colocation import choose_student_groups
            for group in choose_student_groups(pending_students):
                save(phase='training-evaluation', student_entries=[j['identity'] for j in group])
                check_code(args.expected_head)
                if len(group) == 2:
                    paired_students(group)
                else:
                    job = group[0]
                    run(job['command'], job['identity'] + '.formal')
                    finish_student(job)
            pending_students.clear()
            save(queued_students=0)

        def score(shift, model, kind, manifest):
            python, settings, frozen = TEACHERS[model]
            output = args.output_root / 'teachers' / kind / model / shift
            previous_cache = ((args.prepared_root / 'caches' / kind / model / (shift + '.json'))
                              if args.prepared_root and not args.refresh_teacher_caches else None)
            if previous_cache and previous_cache.is_file():
                from tools.rq3_protocol import rebuild_verified_cache, content_hash
                common = json.loads(manifest.read_text())
                payload = rebuild_verified_cache(previous_cache, manifest, model, shift)
                if payload['candidate_sha256'] != content_hash(common):
                    raise ValueError('reused RQ3 cache candidate hash differs')
                print('RQ3_TEACHER_REUSE_VERIFIED', str(previous_cache), flush=True)
            else:
                run(['-m', 'tools.score_rq3_vlm_candidates', '--model', model,
                 '--settings', str(RUNTIME / settings), '--frozen-protocol', frozen,
                 '--manifest', str(manifest), '--frame-root', str(manifest.parent / 'frames'),
                 '--output-root', str(output), '--expected-head', args.expected_head,
                 '--gpu', str(args.gpu), '--wait-gpu', '--full'],
                    shift + '.' + model + '.' + kind + '.score', python=python, teacher=model)
            from tools.build_rq3_cache import build_cache
            from tools.rq3_protocol import load_cache, validate_interface_audit, audit_cache_predictions
            if not previous_cache or not previous_cache.is_file():
                common = json.loads(manifest.read_text())
                protocol = json.loads((output / 'input_protocol.json').read_text())
                audit = validate_interface_audit(protocol, model)
                predictions = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
                payload = build_cache(common, predictions, protocol, model, shift, common['teacher_checkpoint_sha256'])
                audit_cache_predictions(payload, audit)
            cache = args.output_root / 'caches' / kind / model / (shift + '.json')
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open('x') as stream:
                json.dump(payload, stream, sort_keys=True)
            load_cache(cache, model, shift, common['target_video_ids'])
            return cache

        try:
            save(phase='baseline-input-audit')
            os.chdir(DATA)
            # 一次审计六个基线；各正式matrix仍独立重验其完整输入。
            baselines = {shift: completed_baseline(baseline_root(Path('/vepfs/vlms'), shift), shift)
                         for shift in sorted({row['shift'] for row in rows})}
            cpu_tested = set()
            shifts = sorted(baselines)
            for shift_number, shift in enumerate(shifts, 1):
                group = [row for row in rows if row['shift'] == shift]
                save(phase='prepare', shift=shift)
                baseline = baseline_root(Path('/vepfs/vlms'), shift)
                complete = json.loads((baseline / 'complete.json').read_text())
                proposals = args.output_root / 'proposals' / (shift + '.jsonl')
                proposals.parent.mkdir(exist_ok=True)
                previous = (args.prepared_root / 'proposals' / (shift + '.jsonl')) if args.prepared_root else None
                if previous and previous.is_file() and Path(str(previous) + '.meta.json').is_file():
                    proposals.symlink_to(previous)
                    Path(str(proposals) + '.meta.json').symlink_to(Path(str(previous) + '.meta.json'))
                else:
                    with gpu_slot(str(args.gpu), DATA, wait=True) as selected:
                        run([str(CODE_ROOT / 'tools/export_vlm_target_adapter_proposals.py'),
                         '--config', complete['so'], '--checkpoint', str(baselines[shift][1]),
                         '--trusted-matched-baseline-root', str(baseline), '--video-root', str(VIDEO),
                         '--include-missing-videos', '--output', str(proposals), '--domain', 'source',
                         '--split', 'val', '--device', 'cuda:0', '--weights', 'ema', '--levels', '0', '1', '2',
                         '--max-per-video', '30', '--max-proposals', '200', '--seed', '6'],
                            shift + '.source.proposals', selected=selected)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    work = {kind: (target_prefetch.pop(shift) if kind == 'target_local' and shift in target_prefetch
                                   else pool.submit(frames, shift, domain, scope, source))
                            for kind, domain, scope, source in (
                                ('target_local', 'target', 'local', OLD_CANDIDATES / 'proposals' / (shift + '.jsonl')),
                                ('source_local', 'source', 'local', proposals),
                                ('source_proposal', 'source', 'proposal', proposals))}
                    manifests = {kind: future.result() for kind, future in work.items()}
                if args.pipeline_cpu and shift_number < len(shifts):
                    next_shift = shifts[shift_number]
                    # 独立目录、原顺序与原SHA校验；后台仅CPU，训练RNG在独立进程。
                    target_prefetch[next_shift] = prefetch_pool.submit(
                        frames, next_shift, 'target', 'local',
                        OLD_CANDIDATES / 'proposals' / (next_shift + '.jsonl'))
                for row in group:
                    model, strategy = row['teacher'], row['strategy']
                    save(phase='teacher-calibration', teacher=model, strategy=strategy)
                    if args.refresh_teacher_caches and not (cache_root / model / (shift + '.json')).exists():
                        # Prototype必须与源proposal同版；Verify也保留统一候选快照供matrix审计。
                        score(shift, model, 'target_proposal', OLD_CANDIDATES / shift / 'manifest.json')
                    if strategy == 'hard':
                        # 不将旧Hard变成20%筛选策略；只扩模型身份/调度和统一早停。
                        subplan = args.output_root / ('plan_' + shift + '_' + model + '_hard.json')
                        subplan.write_text(json.dumps(dict(plan, entries=[row]), sort_keys=True))
                        name = shift + '_' + model + '_hard'
                        destination = entry_output_root(args.output_root, 'formal', name)
                        save(phase='training-evaluation')
                        command = ['-m', 'tools.run_rq3_matrix', '--revision', 'revised', '--entry-plan', str(subplan),
                             '--cache-root', str(cache_root), '--candidate-root', str(OLD_CANDIDATES),
                             '--verify-cache-root', str(args.output_root / 'caches/target_local'),
                             '--verify-candidate-root', str(args.output_root / 'target_local'),
                             '--calibration-root', str(args.output_root / 'calibrations'),
                             '--expected-head', args.expected_head, '--gpu', str(args.gpu), '--wait-gpu',
                             '--batch-size', str(args.batch_size), '--early-stop-patience', str(args.early_stop_patience),
                             '--output-root', str(destination), '--immediate-gpu', '--execute']
                        record = destination / model / 'hard' / (shift + '_seed6') / 'complete.json'
                        submit_student(command, name, record, strategy)
                        continue
                    source_kind = 'source_local' if strategy == 'verify' else 'source_proposal'
                    source_cache = score(shift, model, source_kind, manifests[source_kind])
                    if strategy == 'verify':
                        cache = score(shift, model, 'target_local', manifests['target_local'])
                        manifest = manifests['target_local']
                    else:
                        cache = cache_root / model / (shift + '.json')
                        manifest = OLD_CANDIDATES / shift / 'manifest.json'
                    from tools.rq3_confidence import calibrate, load_calibration
                    from tools.rq3_protocol import load_cache
                    report = calibrate(baselines[shift][0], source_cache, manifests[source_kind], strategy, None, None, .2)
                    calibration = args.output_root / 'calibrations' / strategy / model / (shift + '.json')
                    calibration.parent.mkdir(parents=True, exist_ok=True)
                    with calibration.open('x') as stream:
                        json.dump(report, stream, sort_keys=True)
                    payload, _ = load_cache(cache, model, shift)
                    load_calibration(calibration, baselines[shift][0], payload, json.loads(manifest.read_text()), strategy)
                    from tools.rq3_confidence import select_budget
                    print('RQ3_FIXED_PARTICIPATION', shift, model, strategy,
                          'target_selected', len(select_budget(payload['entries'])),
                          'source_selected', report['selected_count'],
                          'source_precision', report['empirical_precision'], flush=True)
                    if (model, strategy) not in cpu_tested:
                        save(phase='real-trainer-cpu-audit')
                        audit = args.output_root / ('cpu_' + model + '_' + strategy + '.json')
                        run(['-m', 'tools.audit_rq3_revised_cpu', '--strategy', strategy, '--teacher', model,
                             '--cache', str(cache), '--candidate-manifest', str(manifest),
                             '--confidence-calibration', str(calibration), '--report', str(audit)],
                            shift + '.' + model + '.' + strategy + '.cpu')
                        if json.loads(audit.read_text())['status'] != 'passed':
                            raise ValueError('real Trainer CPU audit did not pass')
                        cpu_tested.add((model, strategy))
                    subplan = args.output_root / ('plan_' + shift + '_' + model + '_' + strategy + '.json')
                    subplan.write_text(json.dumps(dict(plan, entries=[row]), sort_keys=True))
                    name = shift + '_' + model + '_' + strategy
                    base = ['-m', 'tools.run_rq3_matrix', '--revision', 'revised', '--entry-plan', str(subplan),
                            '--cache-root', str(cache_root), '--candidate-root', str(OLD_CANDIDATES),
                            '--verify-cache-root', str(args.output_root / 'caches/target_local'),
                            '--verify-candidate-root', str(args.output_root / 'target_local'),
                            '--calibration-root', str(args.output_root / 'calibrations'),
                            '--expected-head', args.expected_head, '--gpu', str(args.gpu), '--wait-gpu',
                            '--batch-size', str(args.batch_size)]
                    base += ['--early-stop-patience', str(args.early_stop_patience)]
                    if not fresh_matrix:
                        save(phase='formal-cpu-preflight')
                        run(base + ['--output-root', str(entry_output_root(args.output_root, 'audit', name)), '--audit-only'], name + '.preflight')
                    if benchmark_pending:
                        save(phase='batch-throughput-benchmark')
                        results = []
                        for batch in (4, 8):
                            bench_base = list(base)
                            bench_base[bench_base.index('--batch-size') + 1] = str(batch)
                            bench_root = entry_output_root(args.output_root, 'benchmark%d' % batch, name)
                            run(bench_base + ['--output-root', str(bench_root), '--immediate-gpu',
                                             '--execute', '--benchmark-only'], name + '.benchmark%d' % batch)
                            results.append(json.loads((bench_root / 'batch_benchmark.json').read_text()))
                        # 同数目的源/目标视频吞吐决定选择，不读取target mAP。
                        chosen = max(results, key=lambda r: min(m['source_target_videos_per_second'] for m in r['measurements']))
                        args.batch_size = chosen['batch_size']
                        (args.output_root / 'batch_selection.json').write_text(json.dumps(dict(
                            selected_batch=args.batch_size, criterion='max-worst-phase-videos-per-second',
                            benchmarks=results), sort_keys=True))
                        base[base.index('--batch-size') + 1] = str(args.batch_size)
                        save(batch_size=args.batch_size)
                        benchmark_pending = False
                    # 新完整矩阵直接进入execute：内部仍先实际train_da CPU预检/配对，再取得GPU锁。
                    save(phase='training-evaluation')
                    destination = entry_output_root(args.output_root, 'formal', name)
                    record = destination / model / strategy / (shift + '_seed6') / 'complete.json'
                    submit_student(base + ['--output-root', str(destination), '--immediate-gpu', '--execute'],
                                   name, record, strategy)
                if args.pair_students and (shift_number % 2 == 0 or shift_number == len(baselines)):
                    flush_students()
            save(status='complete', phase='complete')
        except BaseException as error:
            save(status='stopped' if stopped else 'failed', error=str(error))
            stop(None, None)
            raise
        finally:
            prefetch_pool.shutdown(wait=True, cancel_futures=True)
            for child in list(children):
                child.wait()


if __name__ == '__main__':
    main()
