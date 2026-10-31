"""先用同一实例政策核验六方向源质量，未通过不启动student或读目标GT。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools import baseline_protocol as p
from tools import run_phase_alignment as owned
from tools.baseline_gpu import slot
from tools.instance_task_source import SCHEMA, SOURCE_POLICY, SOURCE_GATE, source_decision


def inputs(args, identity):
    references = {}; os.chdir(args.data_root)
    for shift in ['S'+str(i) for i in range(1,7)]:
        ref = args.control/'formal/control'/shift; cfg = p.read(ref/'config.json')
        if cfg['seed'] != 123 or cfg['loader']['batch_size'] != 2 or cfg['prototype_capacity']['variant'] != 'control':
            raise ValueError('不是冻结原Prototype参考')
        if cfg.get('phase_alignment') or cfg.get('endpoint_risk') or cfg.get('instance_task'):
            raise ValueError('原教师混入其它探索机制')
        old = args.candidate_root/'entries'/(shift+'_prototype')
        job = p.read(old/'job.json'); done = p.read(old/'inference_complete.json')
        if Path(job['reference']) != ref or done['status'] != 'passed' or not done['target_gt_input_empty'] or not done['no_training']:
            raise ValueError('旧raw缓存身份或GT隔离无效')
        if done['code_commit'] != 'c82256ce15357c254d47575b1d5bb97d67fce4fb' or done['job_sha256'] != p.sha(old/'job.json'):
            raise ValueError('不是精确旧采集政策与原job')
        for name, digest in job['reference_artifacts'].items():
            if p.sha(ref/name) != digest: raise ValueError('旧教师参考SHA改变: '+name)
        if done['outputs']['raw_pool.json'] != p.sha(old/'raw_pool.json'):
            raise ValueError('旧raw_pool改变')
        target = Path(cfg['baseline']['target_manifest'])
        if p.sha(target) != job['target_manifest_sha256'] or p.sha(target) != cfg['baseline']['target_manifest_sha256']:
            raise ValueError('旧缓存与当前无GT目标成员不同')
        target_rows = p.read(target)['database']
        if any(v.get('annotations') for v in target_rows.values()): raise ValueError('目标manifest含GT')
        raw = p.read(old/'raw_pool.json')
        if len(raw) != done['raw_candidates'] or done['videos'] != len(target_rows) or any(
            r['video_id'] not in target_rows or r['level'] not in SOURCE_POLICY['levels'] or
            r['score'] < cfg['model']['pl_threshold'] or r['interval'][1] <= r['interval'][0] for r in raw):
            raise ValueError('旧raw缓存不符合冻结政策')
        source = Path(cfg['dataset']['json_file'])
        if p.sha(source) != cfg['baseline']['source_manifest_sha256']: raise ValueError('源manifest漂移')
        db = p.read(source)['database']; source_ids = set(p.members(db, cfg['src_train_split']+cfg['src_val_split']))
        if source_ids != set(db) or not p.members(db, cfg['src_val_split']): raise ValueError('源manifest混域或无验证成员')
        references[shift] = dict(reference=str(ref), reference_artifacts=job['reference_artifacts'],
            pl_threshold=cfg['model']['pl_threshold'], old_raw_pool=str(old/'raw_pool.json'),
            old_raw_pool_sha256=p.sha(old/'raw_pool.json'), old_inference=str(old/'inference_complete.json'),
            old_inference_sha256=p.sha(old/'inference_complete.json'),
            target_manifest_sha256=p.sha(target), raw_candidates=len(raw),
            source_manifest_sha256=p.sha(source), source_validation_members=sorted(p.members(db,cfg['src_val_split'])))
    os.chdir(CODE)
    p.save(args.output_root/'inputs.json', dict(schema=SCHEMA, status='passed', code=identity,
        source_policy=SOURCE_POLICY, frozen_source_gate=SOURCE_GATE, references=references,
        target_gt_read=False, target_features_read=False, new_vlm_queries=False, student_training=False))


def state(root, phase, completed, pending):
    p.save(root/'state.json', dict(phase=phase, pid=os.getpid(), completed=completed, pending=pending,
        active={s:dict(pid=t['process'].pid,gpu=t['lease']['index']) for s,t in owned.ACTIVE.items()},
        formal_student_training=False, target_features_read=False, target_gt_read=False))


def main(args):
    identity = p.code_identity(); root = args.output_root; root.mkdir(exist_ok=False)
    inputs(args, identity)
    if args.preflight_only: state(root, 'cpu_preflight_passed', [], []); return
    pending = ['S'+str(i) for i in range(1,7)]; completed = []
    try:
        while pending or owned.ACTIVE:
            for shift, task in list(owned.ACTIVE.items()):
                if task['process'].poll() is None: continue
                if task['process'].returncode: raise RuntimeError(shift+'源实例检查技术失败，保留工件')
                entry = root/'source'/shift; result = p.read(entry/'result.json')
                if result['status'] != 'completed_source_instance_quality_check' or result['target_data_read']:
                    raise ValueError('不是合规源实例检查')
                for name, digest in result['artifacts'].items():
                    if p.sha(name) != digest: raise ValueError('源工件改变')
                completed.append(dict(shift=shift, result=str(entry/'result.json'), sha256=p.sha(entry/'result.json')))
                task['context'].__exit__(None,None,None); del owned.ACTIVE[shift]
                print('SOURCE_INSTANCE_CHECK_COMPLETE',shift,result['metrics'],flush=True)
            used = {t['lease']['index'] for t in owned.ACTIVE.values()}
            for shift in list(pending):
                launched = False
                for gpu in args.gpus:
                    if gpu in used: continue
                    context = slot(gpu,args.data_root,root,12288,identity['commit'])
                    try: lease = context.__enter__()
                    except BlockingIOError: continue
                    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                        CUDA_VISIBLE_DEVICES=lease['uuid'], OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                        SADA_BASELINE_LEASE=lease['lease'], SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                    with (root/(shift+'.log')).open('x') as log:
                        process = subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),
                            '--child',shift,'--output-root',str(root),'--control',str(args.control),
                            '--candidate-root',str(args.candidate_root),'--data-root',str(args.data_root)],
                            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    owned.ACTIVE[shift] = dict(process=process,context=context,lease=lease)
                    pending.remove(shift); used.add(gpu); launched=True
                    print('SOURCE_INSTANCE_CHECK_STARTED',shift,process.pid,gpu,flush=True); break
                if not launched: break
            state(root,'source_instance_quality_check',completed,pending); time.sleep(5)
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending))
        owned.terminate_owned(); state(root,'failed',completed,pending); raise
    decision = source_decision([p.read(r['result']) for r in completed]); decision['completed'] = completed
    p.save(root/'source_decision.json',decision)
    state(root,'source_passed_ready_for_student_implementation' if decision['status']=='passed' else 'source_rejected',completed,[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--candidate-root',type=Path,default=Path('/vepfs/vlms/candidate_refresh_offline_c82256c_1030'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--gpus',default='0,1,2,3'); parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--child',choices=['S'+str(i) for i in range(1,7)])
    args = parser.parse_args(); args.gpus=[int(i) for i in args.gpus.split(',')]
    for name in ('output_root','control','candidate_root','data_root'): setattr(args,name,getattr(args,name).resolve())
    signal.signal(signal.SIGTERM,owned.stop); signal.signal(signal.SIGINT,owned.stop)
    if args.child:
        from tools.instance_task_source import worker
        worker(args)
    else: main(args)
