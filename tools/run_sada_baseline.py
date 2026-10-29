"""独立官方修正基线：CPU冻结→batch测速→真实验收→六方向SO/UDA→全审。"""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
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
from tools.baseline_protocol import (SCHEMA, CODE, POLICY, audit_entry, batch_choice,
    code_identity, read, save, sanitized_inputs, sha, termination)
from tools.baseline_gpu import slot, snapshot

ACTIVE = {}


def stop(signum, frame):
    # 只停止本监督直接管理的自有子进程组；不对NVML宿主PID发信号。
    for task in list(ACTIVE.values()):
        if task['process'].poll() is None:
            os.killpg(task['process'].pid, signal.SIGTERM)
    raise SystemExit(128+signum)


def budget(database, cfg, batch):
    source = sum(v['subset'].lower() in cfg['src_train_split'] for v in database.values())
    target = sum(v['subset'].lower() in cfg['tgt_train_split'] for v in database.values())
    sb,tb = source//batch,target//batch
    if not min(sb,tb): raise ValueError('某域不足一个batch')
    return dict(source_videos=source,target_videos=target,source_batches=sb,target_batches=tb,
                iterations_per_epoch=max(sb,tb),batch_size=batch,drop_last=True)


def prepare(root, data, identity):
    import yaml
    registry = yaml.safe_load((CODE/'configs/rq2/shift_registry.yaml').read_text())['shifts']
    configs, inventories = {}, {}
    for shift,row in registry.items():
        original = CODE/row['config']
        raw = yaml.safe_load(original.read_text())
        annotation = (data/raw['dataset']['json_file']).resolve()
        features = (data/raw['dataset']['feat_folder']).resolve()
        db = read(annotation)['database']
        source,target = sanitized_inputs(db,raw['src_train_split'],raw['src_val_split'],raw['tgt_train_split'])
        source_file,target_file = root/'inputs'/shift/'source.json',root/'inputs'/shift/'target_unlabeled.json'
        save(source_file,source);save(target_file,target)
        cfg = deepcopy(raw)
        cfg.update(seed=6,init_rand_seed=6,gpu=0)
        cfg['dataset'].update(json_file=str(source_file),feat_folder=str(features))
        cfg['model'].update(input_dim=cfg['dataset']['input_dim'],num_classes=cfg['dataset']['num_classes'],
                            test_cfg=cfg['test_cfg'],max_seq_len=cfg['dataset']['max_seq_len'])
        cfg['train_cfg'].update(valid_freq=5,early_stop=-1,test_freq=-1)
        cfg['baseline'] = dict(schema=SCHEMA,shift=shift,policy=POLICY,code=identity,
            official_config=str(original),official_config_sha256=sha(original),
            annotation_for_offline_evaluation=str(annotation),annotation_sha256=sha(annotation),
            source_manifest_sha256=sha(source_file),target_manifest=str(target_file),
            target_manifest_sha256=sha(target_file),target_gt_used=False,
            source_target_budget='max-source-target-batches',initialization='official-independent-EMA',
            s6_provenance=row.get('config_provenance','official-release'),
            batch_comparison='new_batch_protocol_not_strictly_paired_to_old_batch2')
        configs[shift] = cfg
        members = {k for k,v in db.items() if any(v['subset'].lower() in raw[s] for s in
                   ['src_train_split','src_val_split','tgt_train_split','tgt_val_split'])}
        for vid in members:
            path = features/(vid+raw['dataset']['file_ext'])
            if not path.is_file(): raise FileNotFoundError(path)
            inventories[str(path)] = path
        save(root/'inputs'/shift/'membership.json',dict(
            source_train=sorted(k for k,v in db.items() if v['subset'].lower() in raw['src_train_split']),
            source_val=sorted(k for k,v in db.items() if v['subset'].lower() in raw['src_val_split']),
            target_train=sorted(target),
            target_val=sorted(k for k,v in db.items() if v['subset'].lower() in raw['tgt_val_split'])))
        cfg['baseline']['batch_budget_candidates'] = {str(b):budget(db,cfg,b) for b in (2,4)}
        save(root/'inputs'/shift/'template.json',cfg)
    def inventory(path):
        stat=path.stat()
        return dict(path=str(path),sha256=sha(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows=list(executor.map(inventory,inventories.values()))
    save(root/'inputs/feature_inventory.json',rows)
    save(root/'inputs/identity.json',dict(schema=SCHEMA,code=identity,feature_count=len(rows),
         feature_inventory_sha256=sha(root/'inputs/feature_inventory.json'),
         templates={s:sha(root/'inputs'/s/'template.json') for s in configs}))
    return configs


def runtime_config(template,batch,regime):
    cfg=deepcopy(template)
    cfg['loader']['batch_size']=batch
    cfg['baseline']['regime']=regime
    cfg['baseline']['epoch_budget']=cfg['baseline'].pop('batch_budget_candidates')[str(batch)]
    return cfg


def spawn(mode,cfg_path,entry,lease,data,report=None):
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',
             PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
             SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
             CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ['RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:
        env.pop(name,None)
    log=entry/(mode+'.log')
    stream=log.open('x')
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child',mode,'--config',str(cfg_path),
             '--entry',str(entry),'--data-root',str(data)]
    if report:command += ['--report',str(report)]
    process=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close()
    return process


def acquire(indices,data,root,minimum,commit):
    for index in indices:
        context=slot(index,data,root,minimum,commit)
        try:lease=context.__enter__()
        except BlockingIOError:continue
        return context,lease
    return None


def synchronous_probe(root,data,cfg,indices,minimum,label):
    entry=root/'probes'/label;entry.mkdir(parents=True,exist_ok=False)
    save(entry/'config.json',cfg)
    while not (found:=acquire(indices,data,root,minimum,cfg['baseline']['code']['commit'])):
        print('WAIT_FOR_PROBE_GPU',label,flush=True);time.sleep(15)
    context,lease=found
    process=spawn('probe',entry/'config.json',entry,lease,data,entry/'result.json')
    ACTIVE[label]=dict(process=process)
    try:
        code=process.wait()
        if code:raise RuntimeError('真实验收/测速失败: '+str(entry/'probe.log'))
        report=read(entry/'result.json')
        report['gpu_lease']=lease
        return report
    finally:
        ACTIVE.pop(label,None);context.__exit__(None,None,None)


def child(args):
    from tools.baseline_gpu import first_cuda_guard
    # 首次CUDA前独立核验，不能只相信父监督的状态。
    proof=first_cuda_guard()
    from tools.baseline_engine import probe,train,evaluate
    cfg=read(args.config)
    os.chdir(args.data_root)
    if args.child=='probe':probe(cfg,args.report)
    elif args.child=='train':train(cfg,args.entry)
    else:evaluate(cfg,args.entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(args.entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=sha(proof),
                                                config_sha256=sha(args.config)))


def main(args):
    root=args.output_root.resolve();data=args.data_root.resolve()
    identity=code_identity()
    root.mkdir(parents=True,exist_ok=False)
    save(root/'state.json',dict(phase='cpu_input_audit',pid=os.getpid(),completed=[],active={}))
    configs=prepare(root,data,identity)
    if args.prepare_only:
        save(root/'cpu_preflight.json',dict(status='passed',shifts=list(configs),code=identity))
        return
    indices=[int(x) for x in args.gpus.split(',')]
    reports={}
    for batch in (2,4):
        cfg=runtime_config(configs['S1'],batch,'uda')
        reports[batch]=synchronous_probe(root,data,cfg,indices,16384,'uda_batch%d'%batch)
    batch=batch_choice(reports)
    # batch只按两阶段吞吐冻结，不读取任何target mAP或按效果重选。
    minimum=max(12288,math.ceil(max(reports[batch]['peak_reserved_mib'],
        *[x['peak_reserved_mib'] for x in reports[batch]['phases'].values()])+4096))
    so=synchronous_probe(root,data,runtime_config(configs['S1'],batch,'so'),indices,minimum,'so_acceptance')
    if so['initial_state_sha'] != reports[batch]['initial_state_sha']:
        raise ValueError('SO/UDA初始模型不同')
    save(root/'acceptance.json',dict(status='passed',batch=batch,minimum_free_mib=minimum,
         batch_reports=reports,so=so,code=identity,inputs_sha256=sha(root/'inputs/identity.json'),
         report_sha256={str(p):sha(p) for p in (root/'probes').glob('*/result.json')}))
    jobs=[(shift,regime) for shift in configs for regime in ('so','uda')]
    save(root/'plan.json',dict(schema=SCHEMA,batch=batch,policy=POLICY,jobs=jobs,code=identity))
    pending=list(jobs);completed=[]
    try:
        while pending or ACTIVE:
            for key,task in list(ACTIVE.items()):
                code=task['process'].poll()
                if code is None:continue
                if code:raise RuntimeError('条目阶段失败: '+key+'/'+task['stage'])
                entry=task['entry']
                if task['stage']=='train':
                    termination(read(entry/'termination.json')['history'])
                    task['stage']='evaluate'
                    task['process']=spawn('evaluate',entry/'config.json',entry,task['lease'],data)
                else:
                    cfg=read(entry/'config.json')
                    artifacts=[entry/x for x in ('config.json','metrics.jsonl','termination.json',
                        'best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
                        'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')]
                    save(entry/'complete.json',dict(schema=SCHEMA,shift=task['shift'],regime=task['regime'],
                        code=identity,acceptance_sha256=sha(root/'acceptance.json'),
                        artifacts={str(p):sha(p) for p in artifacts}))
                    completed.append(audit_entry(entry))
                    task['context'].__exit__(None,None,None);del ACTIVE[key]
                    print('COMPLETE',key,completed[-1]['map'],flush=True)
            while pending:
                found=acquire(indices,data,root,minimum,identity['commit'])
                if found is None:break
                context,lease=found
                shift,regime=pending.pop(0);key=shift+'_'+regime
                entry=root/'formal'/shift/regime;entry.mkdir(parents=True,exist_ok=False)
                cfg=runtime_config(configs[shift],batch,regime);save(entry/'config.json',cfg)
                ACTIVE[key]=dict(context=context,lease=lease,entry=entry,shift=shift,regime=regime,
                    stage='train',process=spawn('train',entry/'config.json',entry,lease,data))
            save(root/'state.json',dict(schema=SCHEMA,phase='formal',pid=os.getpid(),batch=batch,
                completed=completed,pending=pending,
                active={k:dict(pid=v['process'].pid,gpu=v['lease']['index'],stage=v['stage']) for k,v in ACTIVE.items()}))
            time.sleep(5)
    except BaseException as error:
        for task in ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        save(root/'error.json',dict(type=type(error).__name__,error=str(error)))
        raise
    if {(x['shift'],x['regime']) for x in completed} != set(jobs):raise ValueError('不是精确12项')
    save(root/'final_audit.json',dict(status='passed',count=12,schema=SCHEMA,batch=batch,results=completed))
    save(root/'state.json',dict(phase='complete',completed=completed,active={},pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--gpus',default='0,3,1,2')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--child',choices=['probe','train','evaluate'])
    p.add_argument('--config',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--report',type=Path)
    args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.child:child(args)
    else:main(args)
