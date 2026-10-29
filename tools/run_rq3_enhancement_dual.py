"""七组顺序不变，组内两个独立shift分别占GPU2/GPU1；不做同卡自有并发。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools import run_rq3_enhancement as single
from tools.rq3_enhancement import ORDER, SCHEMA, specification, assert_reference_pair
from tools.rq3_external_share import external_gpu_slot, audit_receipt
from tools.rq2_dense_protocol import sha256_file
from tools.rq3_protocol import audit_baseline, load_cache, validate_interface_audit, audit_cache_predictions
from tools.run_rq3_matrix import completed_baseline, supervision_audit
from tools.run_rq2_matched_control import check_code, checked_train_result, checked_evaluation
from tools.rq3_enhancement_recovery import inspect_partial,clone_partial,prefix_log

CODE = Path(__file__).resolve().parents[1]
read, save = single.read, single.save


def dispatch_pending(pending, active, gpus):
    used = {job['gpu'] for job in active.values()}
    return list(zip([gpu for gpu in gpus if gpu not in used], pending))


def ready_acceptance(stage, variant, head, input_sha):
    proof = read(stage/'acceptance_ready.json')
    report = read(stage/'acceptance.json')
    if (proof['code_commit'] != head or proof['inputs_sha256'] != input_sha or
        proof['variant'] != variant or proof['acceptance_sha256'] != sha256_file(stage/'acceptance.json') or
        report['status'] != 'passed' or report['specification'] != specification(variant)):
        raise ValueError('real-data acceptance identity changed')
    audit_receipt(report['gpu_admission'], head)


def worker(job_path, head):
    check_code(head)
    job = read(job_path)
    root, data = Path(job['root']), Path(job['data_root'])
    os.chdir(data)
    inputs = root/'inputs.json'
    if sha256_file(inputs) != job['inputs_sha256']:
        raise ValueError('shared immutable input changed')
    bundle = read(inputs)
    variant, shift, gpu = job['variant'], job['shift'], job['gpu']
    ref, baseline = bundle['references'][shift], bundle['baselines'][shift]
    snapshot = bundle['snapshots'][shift]
    for path, expected in snapshot.items():
        if sha256_file(path) != expected:
            raise ValueError('reference evidence changed: '+path)
    stage, entry = root/variant, root/variant/shift
    (entry/'logs').mkdir(parents=True, exist_ok=False)
    phase_path = entry/'worker_state.json'
    def phase(value):
        save(phase_path, dict(phase=value, pid=os.getpid(), gpu=gpu, variant=variant, shift=shift))
    def slot():
        return external_gpu_slot(gpu, data, root/'gpu_admissions', head)
    resume_plan = job.get('resume_plan')
    command_root = Path(resume_plan['origin_root']) if resume_plan else root
    name, run_root, common, command = single.entry_command(command_root,data,variant,shift,ref)
    if resume_plan:
        run_root,resume_checkpoint = clone_partial(resume_plan,entry)
        command += ['--resume',str(resume_checkpoint)]
        save(entry/'recovery.json',resume_plan)
    elif run_root.exists():
        raise FileExistsError('never overwrite an existing training root: '+str(run_root))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, single.stop)
    phase('preflight')
    single.run(command+['--rq2-audit-output', str(entry/'preflight.json')], entry/'logs/preflight.log', data)
    cfg = read(entry/'preflight.json')
    assert_reference_pair(cfg, ref)
    save(entry/'baseline_pair.json', single.baseline_pair(baseline, cfg))
    if shift == 'S1':
        phase('real_data_acceptance')
        smoke = list(command)
        smoke[smoke.index('-n')+1] = name+'_diagnostic'
        with slot() as card:
            single.run([sys.executable, '-B', str(CODE/'tools/smoke_rq3_enhancement.py'),
                        str(stage/'acceptance.json'), *smoke[2:]], entry/'logs/smoke.log', data, card)
        report = read(stage/'acceptance.json')
        if report['status'] != 'passed' or report['specification'] != specification(variant):
            raise ValueError('real-data acceptance failed')
        report['gpu_admission'] = card
        save(stage/'acceptance.json', report)
        save(stage/'acceptance_ready.json', dict(code_commit=head, variant=variant,
             inputs_sha256=job['inputs_sha256'], acceptance_sha256=sha256_file(stage/'acceptance.json')))
    else:
        phase('waiting_acceptance')
        while not (stage/'acceptance_ready.json').exists():
            os.kill(job['controller_pid'], 0)
            time.sleep(1)
    ready_acceptance(stage, variant, head, job['inputs_sha256'])
    if job.get('diagnostic_only'):
        phase('diagnostic_complete')
        return
    phase('training')
    with slot() as training_card:
        log = entry/'logs/resume.log' if resume_plan else entry/'logs/train.log'
        if resume_plan:
            checked_command=[sys.executable,'-B',str(CODE/'tools/run_rq3_resume_checked.py'),
                 str(entry/'resume_restore_proof.json'),str(resume_checkpoint),*command[2:]]
            single.run(checked_command,log,data,training_card)
        else:
            single.run(command, log, data, training_card)
    if resume_plan:
        with (entry/'logs/train.log').open('x') as merged:
            merged.write(prefix_log(resume_plan)+'\n# exact checkpoint continuation\n'+log.read_text())
    config, best = checked_train_result(run_root, entry/'logs/train.log', cfg)
    single.baseline_pair(baseline, read(config))
    phase('evaluation')
    evaluation = [sys.executable, '-B', str(CODE/'evaluate_da.py'), *common, '-n', name+'_eval',
                  '-r', str(best), '--rq2-formal', '--rq2-output-dir', str(entry/'evaluation')]
    with slot() as eval_card:
        single.run(evaluation, entry/'logs/evaluation.log', data, eval_card)
    checked_evaluation(entry/'evaluation', cfg, best)
    coverage = supervision_audit(config.parent/'benchmark_epoch_metrics.jsonl', cfg)
    save(entry/'supervision_audit.json', coverage)
    artifacts = [entry/'preflight.json', config, best, entry/'evaluation/manifest.json',
                 entry/'evaluation/source_val.json', entry/'evaluation/target_val.json']
    record = dict(schema=SCHEMA, variant=variant, specification=specification(variant),
         shift=shift, teacher='qwen3_vl_8b', strategy='enhanced_prototype', seed=6, batch_size=2,
         code_commit=head, run_root=str(run_root), training_gpu=training_card, evaluation_gpu=eval_card,
         reference_sha=snapshot, acceptance_sha256=sha256_file(stage/'acceptance.json'),
         artifacts={str(path):sha256_file(path) for path in artifacts})
    if resume_plan:
        record['recovery']=dict(sha256=sha256_file(entry/'recovery.json'),
             resume_log_sha256=sha256_file(entry/'logs/resume.log'),
             restore_proof_sha256=sha256_file(entry/'resume_restore_proof.json'),
             source_code_commit=resume_plan['origin_code_commit'],resumed_from_epoch=resume_plan['epoch'])
    save(entry/'complete.json',record)
    value = single.audit_entry(entry, ref, baseline)
    phase('complete')
    print('ENHANCEMENT_COMPLETE', variant, shift, value, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--migration', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--gpus', nargs=2, choices=('1','2'), default=['2','1'])
    parser.add_argument('--previous-root', type=Path)
    parser.add_argument('--resume-partial', type=Path, nargs='*',default=[])
    args = parser.parse_args()
    if args.job:
        return worker(args.job.absolute(), args.expected_head)
    if not args.migration or not args.output_root or len(set(args.gpus)) != 2:
        parser.error('requires fresh output, six references and distinct physical GPU2/GPU1')
    check_code(args.expected_head)
    migration = read(args.migration)
    if migration['teacher'] != 'qwen3_vl_8b' or sorted(r['shift'] for r in migration['references']) != ['S'+str(i) for i in range(1,7)]:
        raise ValueError('requires exact six Qwen references')
    root, data = args.output_root.absolute(), args.data_root.absolute()
    root.mkdir(parents=True, exist_ok=False)
    (root/'jobs').mkdir()
    os.chdir(data)
    state = dict(schema=SCHEMA, code_commit=args.expected_head, pid=os.getpid(),
                 phase='input_audit', completed=[], active={})
    save(root/'state.json', state)
    save(root/'plan.json', dict(order=list(ORDER), specifications=[specification(v) for v in ORDER],
         items=[dict(variant=v,shift='S'+str(i)) for v in ORDER for i in range(1,7)],
         references=migration['references'], target_based_selection=False, new_hardware_control=True,
         resource_policy=dict(physical_gpus=[int(g) for g in args.gpus], external_sharing=True,
                              own_cuda_per_gpu=1, minimum_free_mib=16384)))
    save(root/'environment.json',dict(python=sys.version,executable=sys.executable,
         code_commit=args.expected_head,data_root=str(data),
         resource_policy=read(root/'plan.json')['resource_policy']))
    active = {}
    def stop(signum, frame):
        raise SystemExit(128+signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        baselines, references, snapshots = {}, {}, {}
        for ref in migration['references']:
            shift, old = ref['shift'], Path(ref['entry'])
            cfg = read(old/'preflight.json')
            references[shift] = cfg
            baselines[shift] = completed_baseline(Path('/vepfs/vlms')/('rq2_matched_slowfast_%s_seed6_7067306_0927'%shift), shift)[0]
            audit_baseline(baselines[shift], cfg)
            payload, _ = load_cache(cfg['rq3_auxiliary']['cache_path'], 'qwen3_vl_8b', shift)
            audit_cache_predictions(payload, validate_interface_audit(payload['input_protocol'], 'qwen3_vl_8b'))
            paths = [old/'preflight.json', old/'complete.json']+[Path(cfg['rq3_auxiliary'][k]) for k in ('cache_path','manifest_path','calibration_path')]
            snapshots[shift] = {str(p):sha256_file(p) for p in paths}
            print('INPUT_AUDIT_PASS', shift, flush=True)
        save(root/'reference_sha.json', snapshots)
        save(root/'inputs.json', dict(references=references, baselines=baselines, snapshots=snapshots))
        inputs_sha = sha256_file(root/'inputs.json')
        previous,partial = {},{}
        if args.previous_root:
            old = args.previous_root.absolute()
            old_state = read(old/'state.json')
            if old_state['phase'] not in ('failed','stopped') or Path('/proc',str(old_state['pid'])).exists():
                raise ValueError('previous matrix must have exited before recovery')
            if read(old/'reference_sha.json') != snapshots:
                raise ValueError('recovery references differ from archived matrix')
            for job in old_state.get('active',{}).values():
                if Path('/proc',str(job['pid'])).exists():
                    raise ValueError('previous worker remains active')
            for item in old_state['completed']:
                key = item['variant']+'_'+item['shift']
                if key in previous:
                    raise ValueError('duplicated old complete identity')
                entry = Path(item['complete']).parent
                value = single.audit_entry(entry,references[item['shift']],baselines[item['shift']])
                previous[key] = dict(item,map=value,complete_sha256=sha256_file(entry/'complete.json'))
            for entry in args.resume_partial:
                plan = inspect_partial(entry.absolute())
                if Path(plan['origin_root'])!=old:
                    raise ValueError('partial entry belongs to another predecessor')
                cfg = read(entry/'preflight.json')
                key = cfg['rq3_enhancement']['variant']+'_'+cfg['rq2_formal']['shift']
                if key in previous or key in partial:
                    raise ValueError('partial conflicts with a completed or repeated identity')
                partial[key] = plan
            save(root/'predecessor.json',dict(previous_root=str(old),state_sha256=sha256_file(old/'state.json'),
                 completed=previous,partial=partial,old_code_commit=old_state['code_commit']))
        elif args.resume_partial:
            raise ValueError('partial continuation requires an audited predecessor')
        for variant in ORDER:
            stage = root/variant
            stage.mkdir()
            stage_results = {}
            stage_entries = {}
            pending = []
            for i in range(1,7):
                shift = 'S'+str(i)
                key = variant+'_'+shift
                if key in previous:
                    item = previous[key]
                    stage_results[shift] = item['map']
                    stage_entries[shift] = Path(item['complete']).parent
                    state['completed'].append(item)
                else:
                    pending.append(shift)
            # 已完成S1不重训；仅重做当前clean代码的无checkpoint验收。
            diagnostic_s1 = 'S1' in stage_results and bool(pending)
            if diagnostic_s1:pending.insert(0,'S1')
            if len(stage_results)==6:
                save(stage/'final_audit.json',dict(status='passed',count=6,mean_map=sum(stage_results.values())/6,
                     complete_sha={s:sha256_file(stage_entries[s]/'complete.json') for s in sorted(stage_entries)},
                     complete_paths={s:str(stage_entries[s]/'complete.json') for s in sorted(stage_entries)}))
                save(root/'state.json',state)
                continue
            while pending or active:
                dispatched = dispatch_pending(pending, state['active'], args.gpus)
                for gpu, shift in dispatched:
                    check_code(args.expected_head)
                    pending.remove(shift)
                    key = variant+'_'+shift
                    job_path = root/'jobs'/(key+'.json')
                    diagnostic = diagnostic_s1 and shift=='S1'
                    job = dict(root=str(root), data_root=str(data), variant=variant, shift=shift,
                         gpu=gpu, inputs_sha256=inputs_sha, controller_pid=os.getpid(),diagnostic_only=diagnostic)
                    if key in partial:job['resume_plan']=partial[key]
                    save(job_path,job)
                    env = dict(os.environ, PYTHONPATH=str(CODE), CUDA_VISIBLE_DEVICES='',
                               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
                    with (stage/(shift+'.worker.log')).open('x') as log:
                        child = subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                                '--job',str(job_path),'--expected-head',args.expected_head],
                                cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = child
                    state['active'][key] = dict(gpu=gpu, pid=child.pid, shift=shift, variant=variant,diagnostic_only=diagnostic)
                if dispatched:
                    state['phase'] = 'executing'
                    save(root/'state.json', state)
                for key, child in list(active.items()):
                    result = child.poll()
                    if result is None:
                        continue
                    if result:
                        raise RuntimeError('entry worker failed: %s exit=%d' % (key,result))
                    job = state['active'][key]
                    shift = job['shift']
                    entry = stage/shift
                    ready_acceptance(stage,variant,args.expected_head,inputs_sha)
                    if not job.get('diagnostic_only'):
                        value = single.audit_entry(entry,references[shift],baselines[shift])
                        state['completed'].append(dict(variant=variant,shift=shift,complete=str(entry/'complete.json'),map=value))
                        stage_results[shift] = value
                        stage_entries[shift] = entry
                        print('INDEPENDENT_AUDIT_COMPLETE',variant,shift,value,flush=True)
                    del active[key]
                    del state['active'][key]
                    save(root/'state.json',state)
                if active:
                    time.sleep(1)
            if sorted(stage_results) != ['S'+str(i) for i in range(1,7)]:
                raise ValueError('stage did not complete exact six shifts')
            save(stage/'final_audit.json',dict(status='passed',count=6,mean_map=sum(stage_results.values())/6,
                 complete_sha={s:sha256_file(stage_entries[s]/'complete.json') for s in sorted(stage_results)},
                 complete_paths={s:str(stage_entries[s]/'complete.json') for s in sorted(stage_results)}))
        if len(state['completed']) != 42:
            raise ValueError('matrix did not complete exact 42 items')
        state.update(phase='complete',active={})
        save(root/'state.json',state)
        save(root/'final_audit.json',dict(status='passed',count=42,results=state['completed']))
    except BaseException as error:
        for child in active.values():
            if child.poll() is None:
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in active.values():
            child.wait(timeout=40)
        state.update(phase='stopped' if isinstance(error,SystemExit) else 'failed',error=str(error))
        save(root/'state.json',state)
        raise


if __name__ == '__main__':
    main()
