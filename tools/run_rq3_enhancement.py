"""Qwen Prototype七组逐项增强；六shift全报告，不按target分数改变后续计划。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
from tools.rq3_enhancement import ORDER, SCHEMA, specification, assert_reference_pair
from tools.rq2_dense_protocol import sha256_file
from tools.rq3_gpu import gpu_slot
from tools.rq3_protocol import audit_baseline, load_cache, validate_interface_audit, audit_cache_predictions
from tools.run_rq3_matrix import completed_baseline, supervision_audit
from tools.run_rq2_matched_control import checked_train_result, checked_evaluation, check_code

CODE = Path(__file__).resolve().parents[1]
CHILD = None


def stop(signum, frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
        CHILD.wait()
    raise SystemExit(128 + (signum or 15))


def run(command, log, data, card=None):
    global CHILD
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES=card['uuid'] if card else '', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    env.pop('RQ3_GPU_INDEX', None)
    env.pop('RQ3_GPU_UUID', None)
    if card:
        env.update(RQ3_GPU_INDEX=str(card['index']), RQ3_GPU_UUID=card['uuid'])
    with log.open('x') as stream:
        CHILD = subprocess.Popen(command, cwd=data, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        code = CHILD.wait()
        CHILD = None
        if code:
            raise subprocess.CalledProcessError(code, command)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True))
    tmp.replace(path)


def baseline_pair(baseline, cfg):
    plain = deepcopy(cfg)
    plain.pop('rq3_enhancement', None)
    return audit_baseline(baseline, plain)


def audit_entry(root, reference, baseline):
    cfg = read(root/'preflight.json')
    assert_reference_pair(cfg, reference)
    record = read(root/'complete.json')
    if record['schema'] != SCHEMA or record['specification'] != specification(record['variant']):
        raise ValueError('complete variant changed')
    config, best = checked_train_result(Path(record['run_root']), root/'logs/train.log', cfg)
    saved = read(config)
    assert_reference_pair(saved, reference)
    if saved['rq3_enhancement'] != cfg['rq3_enhancement']:
        raise ValueError('saved enhancement differs')
    baseline_pair(baseline, saved)
    manifest = checked_evaluation(root/'evaluation', cfg, best)
    if manifest.get('rq3_enhancement') != cfg['rq3_enhancement'] or manifest['rq3_auxiliary'] != cfg['rq3_auxiliary']:
        raise ValueError('evaluation lost enhancement/evidence identity')
    expected = {str(p): sha256_file(p) for p in (root/'preflight.json', config, best,
               root/'evaluation/manifest.json', root/'evaluation/source_val.json', root/'evaluation/target_val.json')}
    if record['artifacts'] != expected:
        raise ValueError('completed artifacts changed')
    coverage = supervision_audit(config.parent/'benchmark_epoch_metrics.jsonl', cfg)
    if not coverage['effective'] or coverage != read(root/'supervision_audit.json'):
        raise ValueError('auxiliary never affected shared features')
    return read(root/'evaluation/target_val.json')['map_0.1_to_0.5']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--migration', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--gpu', choices=('0',), default='0')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    root, data = args.output_root.absolute(), args.data_root.absolute()
    migration = read(args.migration)
    if migration['teacher'] != 'qwen3_vl_8b' or sorted(r['shift'] for r in migration['references']) != ['S'+str(i) for i in range(1,7)]:
        raise ValueError('requires six unique migrated Qwen reference entries')
    check_code(args.expected_head)
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(data)
    registry = CODE/'configs/rq2/shift_registry.yaml'
    shifts = yaml.safe_load(registry.read_text())['shifts']
    state = dict(schema=SCHEMA, pid=os.getpid(), code_commit=args.expected_head, phase='input_audit',
                 plan=[dict(variant=v, shift='S'+str(i)) for v in ORDER for i in range(1,7)], completed=[], active=None)
    save(root/'state.json', state)
    save(root/'plan.json', dict(order=list(ORDER), specifications=[specification(v) for v in ORDER],
         references=migration['references'], target_based_selection=False, new_hardware_control=True))
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, stop)
    try:
        baselines, references, snapshots = {}, {}, {}
        for ref in migration['references']:
            shift = ref['shift']
            old = Path(ref['entry'])
            cfg = read(old/'preflight.json')
            references[shift] = cfg
            baseline = completed_baseline(Path('/vepfs/vlms')/('rq2_matched_slowfast_%s_seed6_7067306_0927'%shift), shift)[0]
            baselines[shift] = baseline
            audit_baseline(baseline, cfg)
            payload, _ = load_cache(cfg['rq3_auxiliary']['cache_path'], 'qwen3_vl_8b', shift)
            audit_cache_predictions(payload, validate_interface_audit(payload['input_protocol'], 'qwen3_vl_8b'))
            paths = [old/'preflight.json', old/'complete.json'] + [Path(cfg['rq3_auxiliary'][k]) for k in ('cache_path','manifest_path','calibration_path')]
            snapshots[shift] = {str(path): sha256_file(path) for path in paths}
        save(root/'reference_sha.json', snapshots)
        # 每一组先真实S1适配验收，再依次完整训练六shift；不以结果决定启用哪些组。
        for variant in ORDER:
            stage = root/variant
            stage.mkdir()
            stage_results = []
            for shift in sorted(references):
                check_code(args.expected_head)
                for path, digest in snapshots[shift].items():
                    if sha256_file(path) != digest:
                        raise ValueError('original evidence changed: '+path)
                entry = stage/shift
                (entry/'logs').mkdir(parents=True)
                ref = references[shift]
                aux = ref['rq3_auxiliary']
                name = 'rq3_enhance_%s_%s_%s'%(variant,shift,root.name)
                run_root = data/'model_results'/name
                if run_root.exists():
                    raise FileExistsError('never overwrite a partial or completed run: '+str(run_root))
                common = ['-c', str(CODE/shifts[shift]['config']), '-s','6','-i','0',
                          '--rq2-shift',shift,'--rq2-registry',str(registry),
                          '--rq2-reference-root',str(data/'data/ek100_slowfast/features')]
                command = [sys.executable,'-B',str(CODE/'train_da.py'),*common,'-n',name,
                           '--debug','--benchmark-audit-log','--rq2-regime','uda','--supervised-upper-bound','off',
                           '--rq2-target-manifest',ref['rq2_formal']['target_manifest']['path'],
                           '--rq3-strategy','prototype','--rq3-revision','revised','--rq3-teacher','qwen3_vl_8b',
                           '--rq3-cache',aux['cache_path'],'--rq3-candidate-manifest',aux['manifest_path'],
                           '--rq3-confidence-calibration',aux['calibration_path'],'--rq3-batch-size','2',
                           '--rq3-early-stop-patience','15' if aux.get('early_stopping') else '0',
                           '--rq3-enhancement',variant]
                state.update(phase='preflight',active=dict(variant=variant,shift=shift))
                save(root/'state.json',state)
                run(command+['--rq2-audit-output',str(entry/'preflight.json')],entry/'logs/preflight.log',data)
                cfg = read(entry/'preflight.json')
                assert_reference_pair(cfg,ref)
                save(entry/'baseline_pair.json', baseline_pair(baselines[shift],cfg))
                if args.preflight_only:
                    continue
                if shift == 'S1':
                    state['phase']='real_data_acceptance'
                    save(root/'state.json',state)
                    smoke = list(command)
                    smoke[smoke.index('-n')+1] = name+'_diagnostic'
                    with gpu_slot(args.gpu,data,wait=True) as card:
                        run([sys.executable,'-B',str(CODE/'tools/smoke_rq3_enhancement.py'),
                             str(stage/'acceptance.json'),*smoke[2:]], entry/'logs/smoke.log',data,card)
                    acceptance = read(stage/'acceptance.json')
                    if acceptance['specification'] != specification(variant) or acceptance['status'] != 'passed':
                        raise ValueError('invalid real-data acceptance')
                if not (stage/'acceptance.json').exists():
                    raise ValueError('no real-data acceptance')
                state['phase']='training'
                save(root/'state.json',state)
                with gpu_slot(args.gpu,data,wait=True) as training_card:
                    run(command,entry/'logs/train.log',data,training_card)
                config,best = checked_train_result(run_root,entry/'logs/train.log',cfg)
                baseline_pair(baselines[shift],read(config))
                state['phase']='evaluation'
                save(root/'state.json',state)
                evaluation=[sys.executable,'-B',str(CODE/'evaluate_da.py'),*common,'-n',name+'_eval',
                            '-r',str(best),'--rq2-formal','--rq2-output-dir',str(entry/'evaluation')]
                with gpu_slot(args.gpu,data,wait=True) as eval_card:
                    run(evaluation,entry/'logs/evaluation.log',data,eval_card)
                checked_evaluation(entry/'evaluation',cfg,best)
                coverage=supervision_audit(config.parent/'benchmark_epoch_metrics.jsonl',cfg)
                save(entry/'supervision_audit.json',coverage)
                artifacts=[entry/'preflight.json',config,best,entry/'evaluation/manifest.json',
                           entry/'evaluation/source_val.json',entry/'evaluation/target_val.json']
                save(entry/'complete.json',dict(schema=SCHEMA,variant=variant,specification=specification(variant),
                     shift=shift,teacher='qwen3_vl_8b',strategy='enhanced_prototype',seed=6,batch_size=2,
                     code_commit=args.expected_head,run_root=str(run_root),training_gpu=training_card,
                     evaluation_gpu=eval_card,reference_sha=snapshots[shift],
                     acceptance_sha256=sha256_file(stage/'acceptance.json'),
                     artifacts={str(path):sha256_file(path) for path in artifacts}))
                value=audit_entry(entry,ref,baselines[shift])
                state['completed'].append(dict(variant=variant,shift=shift,complete=str(entry/'complete.json'),map=value))
                save(root/'state.json',state)
                stage_results.append(value)
                print('ENHANCEMENT_COMPLETE',variant,shift,value,flush=True)
            if not args.preflight_only:
                save(stage/'final_audit.json',dict(status='passed',count=6,mean_map=sum(stage_results)/6,
                     complete_sha={s:sha256_file(stage/s/'complete.json') for s in sorted(references)}))
        state.update(phase='preflight_complete' if args.preflight_only else 'complete',active=None)
        save(root/'state.json',state)
        if not args.preflight_only:
            save(root/'final_audit.json',dict(status='passed',count=42,results=state['completed']))
    except BaseException as error:
        state.update(phase='stopped' if isinstance(error,SystemExit) else 'failed',error=str(error))
        save(root/'state.json',state)
        raise


if __name__ == '__main__':
    main()
